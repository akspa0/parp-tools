"""Spec 114 T015: architecture-generic direct geometry trainer (USER runs CUDA).

One trainer for the geometry bakeoff: the frozen ``direct_cnn_v112`` baseline and the
``mit_b0_regression`` candidate on the exact same frozen source-group split, same ``v112.1``
target, same one-output contract. It adds the observability the rejected bootstrap lacked and the
audit in research.md requires:

- flat AND tile-mean baselines computed in-run from validation truth;
- per-row border/interior metrics and the SC-002 border-vs-interior-p95 check at final evaluation;
- SC-001 recorded against both in-run baselines and the frozen Spec 112 run
  (``SPEC112_FROZEN_BEST_VAL_MAE``), so the comparison claim is self-contained;
- a schema-validated ``v50-model-stage-run-v1`` document (``model_stage_run.json``) with
  ``promotion_verdict=pending`` — only the user's visual gate can promote;
- optional AMP / OneCycle / gradient clipping flags addressing the bootstrap's audit finding.
  Defaults stay at bootstrap parity (constant LR, no AMP, no clip) so runs are comparable.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec103.v7_inputs import brush_mask_from_alpha
from harvester.spec118.object_loss import (
    OBJECT_MASK_ARRAY,
    object_mask_available,
    object_touched_rows,
)
from harvester.spec118.object_loss import (
    subset_metrics as object_subset_metrics,
)
from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.direct_geometry_model import (
    ARCHITECTURE_IDS,
    MIT_B0_HUB_ID,
    MIT_B0_ID,
    MIT_B0_LICENSE,
    build_geometry_model,
    load_pretrained_encoder,
)
from harvester.v50.feature_stores import (
    feature_channels_for_row,
    load_feature_stores,
    plan_entries,
    road_feature_binding,
    total_class_count,
)
from harvester.v50.height_relative_evaluate import (
    evaluate_height_model,
    render_fixed_model_preview,
    select_fixed_preview_rows,
)
from harvester.v50.height_relative_model import (
    TARGET_CONTRACT_VERSION,
    encode_relative_height,
    height_loss,
)
from harvester.v50.height_relative_train import (
    SOURCE_CHOICES,
    TrainerContractError,
    compute_tile_mean_baseline,
    curriculum_identity,
    require_new_output,
    select_training_rows,
    validate_curriculum_contract,
    validate_source_selection,
)
from harvester.v50.lr_schedule import (
    PCT_START_DEFAULT,
    make_onecycle_scheduler,
    warmup_complete,
    warmup_epochs_for,
)
from harvester.v50.model_stage_contract import (
    ContractViolationError,
    identity_for_path,
    sha256_file,
    validate_model_stage_run,
)
from harvester.v50.normal_guidance import normal_gradient_loss
from harvester.v50.spectral_guidance import multiscale_gradient_loss, radial_spectral_loss

STAGE = "direct_geometry"
OUTPUT_SIGNAL = "relative_height_257"
# research.md T003/T017-T018 record: the frozen Spec 112 comparison point every candidate must beat
# by SC-001's 5% relative margin on the same split.
SPEC112_FROZEN_BEST_VAL_MAE = 0.1492665126
SC001_RELATIVE_MARGIN = 0.05

LR_SCHEDULES = frozenset({"constant", "onecycle"})


def compute_flat_baseline(targets: list[np.ndarray]) -> float:
    """MAE of predicting a constant 0.5 field everywhere — the no-structure baseline."""
    if not targets:
        raise TrainerContractError("cannot compute a baseline over zero validation tiles")
    return float(np.mean([float(np.abs(t - 0.5).mean()) for t in targets]))


def check_sc002(records: list[dict]) -> dict:
    """SC-002: pooled held-out border MAE must not exceed the interior distribution's p95."""
    if not records:
        raise TrainerContractError("SC-002 requires at least one held-out row")
    border = float(np.mean([float(record["border_mae"]) for record in records]))
    interiors = np.array([float(record["interior_mae"]) for record in records], dtype=np.float64)
    p95 = float(np.percentile(interiors, 95))
    # "No worse than" includes exact equality; pooled-mean vs percentile float rounding must not
    # flip an identical distribution to a failure.
    passes = border <= p95 or math.isclose(border, p95, rel_tol=1e-9, abs_tol=1e-12)
    return {
        "border_mae": border,
        "interior_mae_p95": p95,
        "held_out_rows": len(records),
        "passes": passes,
    }


def evaluate_sc001(*, best_val_mae: float, tile_mean_baseline: float, flat_baseline: float) -> dict:
    """SC-001: best validation MAE must beat both in-run baselines AND the frozen Spec 112 run by
    at least 5% relative."""
    def _beats(reference: float) -> bool:
        return best_val_mae <= reference * (1.0 - SC001_RELATIVE_MARGIN)

    return {
        "best_val_mae": best_val_mae,
        "relative_margin_required": SC001_RELATIVE_MARGIN,
        "tile_mean_baseline": tile_mean_baseline,
        "flat_baseline": flat_baseline,
        "spec112_frozen_best_val_mae": SPEC112_FROZEN_BEST_VAL_MAE,
        "beats_tile_mean": _beats(tile_mean_baseline),
        "beats_flat": _beats(flat_baseline),
        "beats_spec112_frozen": _beats(SPEC112_FROZEN_BEST_VAL_MAE),
        "passes": (
            _beats(tile_mean_baseline) and _beats(flat_baseline) and _beats(SPEC112_FROZEN_BEST_VAL_MAE)
        ),
    }


def build_stage_run_summary(
    *,
    run_id: str,
    architecture: dict,
    pretrained_source: dict | None,
    curriculum: dict,
    checkpoint: dict,
    baselines: dict,
    metrics: dict,
    visual_evidence: dict,
    created_utc: str | None = None,
    promotion_verdict: str = "pending",
) -> dict:
    """Assemble and self-validate the published ``v50-model-stage-run-v1`` record."""
    summary = {
        "schema": "v50-model-stage-run-v1",
        "run_id": run_id,
        "created_utc": created_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": STAGE,
        "output_signal": OUTPUT_SIGNAL,
        "architecture": architecture,
        "pretrained_source": pretrained_source,
        "curriculum": curriculum,
        "upstream_models": [],
        "checkpoint": checkpoint,
        "baselines": baselines,
        "metrics": metrics,
        "visual_evidence": visual_evidence,
        "promotion_verdict": promotion_verdict,
    }
    try:
        validate_model_stage_run(summary)
    except ContractViolationError as exc:
        raise TrainerContractError(f"stage-run record violates its own contract: {exc}") from exc
    return summary


def build_direct_plan(
    *,
    architecture_identity: dict,
    pretrained_source: dict | None,
    source: str,
    index_rows: list[dict],
    selected_rows: list[int],
    batch_size: int,
    epochs: int,
    seed: int,
    lr: float,
    lr_schedule: str,
    amp: bool,
    amp_dtype: str,
    clip: float,
    spectral_weight: float = 0.0,
    multiscale_weight: float = 0.0,
) -> dict:
    """Machine-readable no-training preview printed before CUDA allocation."""
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    if lr_schedule not in LR_SCHEDULES:
        raise TrainerContractError(f"lr schedule must be one of {sorted(LR_SCHEDULES)}")
    if spectral_weight < 0 or multiscale_weight < 0:
        raise TrainerContractError("guidance weights must be >= 0")
    if amp_dtype not in ("fp16", "bf16"):
        raise TrainerContractError(f"amp_dtype must be 'fp16' or 'bf16', got {amp_dtype!r}")
    selected = [index_rows[i] for i in selected_rows]
    split_counts = {
        split: sum(str(row.get("split")) == split for row in selected) for split in ("train", "val")
    }
    source_counts: dict[str, int] = {}
    for row in selected:
        value = str(row.get("minimap_source", "unknown"))
        source_counts[value] = source_counts.get(value, 0) + 1
    return {
        "schema": "v114-direct-geometry-plan-v2",
        "stage": STAGE,
        "architecture": architecture_identity,
        "pretrained_source": pretrained_source,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "source_filter": source,
        "selected_rows": len(selected_rows),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "optimizer": {"name": "AdamW", "learning_rate": lr, "weight_decay": 1e-4},
        "lr_schedule": lr_schedule,
        "amp": amp,
        "amp_dtype": amp_dtype,
        "grad_clip": clip,
        "guidance": {
            "spectral_weight": spectral_weight,
            "multiscale_weight": multiscale_weight,
        },
        "train_steps_per_epoch": math.ceil(split_counts["train"] / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "training_target": "height_257 -> relative_height_257",
        "wdl_prior": False,
        "sc001_references": {
            "spec112_frozen_best_val_mae": SPEC112_FROZEN_BEST_VAL_MAE,
            "relative_margin": SC001_RELATIVE_MARGIN,
        },
    }


def apply_held_out_split(
    *,
    index_rows: list[dict],
    selected_rows: list[int],
    split_dir: Path,
) -> tuple[list[int], list[int], dict]:
    """Partition ``selected_rows`` by a Spec 116 US4 spatially-isolated held-out split.

    Consumes ``split_dir`` as an external, read-only artifact (never written back into the
    curriculum store, which stays immutable) via ``(map, tile_x, tile_y)`` lookup. Returns
    ``(train_rows, val_rows, split_manifest)``; raises ``TrainerContractError`` if the split is
    leaky (``verified_violation_count != 0`` -- FR-010).
    """
    from harvester.spec116.held_out_split import load_split

    split_manifest, split_rows = load_split(split_dir)
    if split_manifest["verified_violation_count"] != 0:
        raise TrainerContractError(
            f"--held-out-split has {split_manifest['verified_violation_count']} violations; "
            "refusing to train on a leaky split (FR-010)"
        )
    split_map: dict[tuple[str, int, int], str] = {
        (str(row["map"]), int(row["tile_x"]), int(row["tile_y"])): str(row["split"])
        for row in split_rows
    }
    train_rows: list[int] = []
    val_rows: list[int] = []
    for i in selected_rows:
        meta = index_rows[i]
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        label = split_map.get(key, "excluded")
        if label == "train":
            train_rows.append(i)
        elif label == "held_out":
            val_rows.append(i)
    return train_rows, val_rows, split_manifest


def main() -> int:
    import pyarrow.parquet as pq
    import torch
    import zarr
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser(description="Spec 114 direct geometry trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="dual-source curriculum store")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--run-id", required=True, help="immutable run identity, e.g. mit_b0-authored-v1")
    ap.add_argument("--architecture", required=True, choices=sorted(ARCHITECTURE_IDS))
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    ap.add_argument("--confirm-run", action="store_true",
                    help="launch CUDA training; without this flag only print the validated plan")
    ap.add_argument("--val-key", default="split")
    ap.add_argument("--val-value", default="val")
    ap.add_argument("--held-out-split", type=Path, default=None,
                    help="Spec 116 US4: a v50-held-out-split-v1 directory (from "
                         "spec116_build_held_out_split.py) whose spatially-isolated held_out "
                         "rows become validation, overriding --val-key/--val-value. Refuses a "
                         "leaky split (verified_violation_count != 0). Never mutates --store.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=sorted(LR_SCHEDULES))
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"],
                    help="AMP dtype: fp16 (default, needs GradScaler) or bf16 (native on Ampere+, "
                         "no scaler needed, better numerical stability for FFT losses)")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip; 0 disables (bootstrap parity)")
    ap.add_argument("--spectral-weight", type=float, default=0.0,
                    help="fractal radial log-power guidance weight; 0 keeps bootstrap parity")
    ap.add_argument("--multiscale-weight", type=float, default=0.0,
                    help="multi-octave gradient guidance weight; 0 keeps bootstrap parity")
    ap.add_argument("--workers", type=int, choices=[0], default=0)
    ap.add_argument("--cache-samples", action="store_true",
                    help="Hold every built sample in RAM after its first epoch. With --workers "
                         "pinned to 0 the Zarr decompression runs on the main thread while the GPU "
                         "idles (~31 ms/sample with the full signal set); caching removes that from "
                         "every epoch after the first. Results are bit-identical — sample "
                         "construction is deterministic — so runs stay comparable. Costs roughly "
                         "3.7 MB per sample of system RAM (~6 GB for the 1629-row authored set).")
    ap.add_argument("--val-tolerance", type=float, default=0.0,
                    help="Val MAE within this fraction of best still resets stale counter "
                         "(0.0 = strict, 0.01 = within 1%% of best counts as not stale). "
                         "Handles noisy val MAE when train loss is still decreasing.")
    ap.add_argument("--liquid-mask-weight", type=float, default=0.0,
                    help="Downweight point loss in liquid regions by this fraction "
                         "(0.0 = no masking, 1.0 = zero loss where liquid_mask=1). "
                         "The minimap input is blue water in liquid regions — the model "
                         "can't see underwater terrain, so penalizing it there is noise.")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--pct-start", type=float, default=PCT_START_DEFAULT,
                    help="OneCycleLR warmup fraction (torch default 0.3 = 30%% of steps). The "
                         "early-stopper is warmup-aware: it does not count stale epochs until the "
                         "warmup phase completes, so a long warmup can no longer kill a run before "
                         "the LR reaches its peak. For small datasets (e.g. 43 steps/epoch) a shorter "
                         "warmup (0.1) wastes less of a short run on under-LR steps.")
    ap.add_argument("--seed", type=int, default=114)
    ap.add_argument("--release", default="v50.1", type=validate_release)
    ap.add_argument("--normal-weight", type=float, default=0.0,
                    help="Weight for normal-derived GRADIENT supervision: constrain the predicted "
                         "height field's slope to match authored MCNR normals at real vertices. "
                         "Directly attacks spectral-bias blur, which point-wise L1 cannot see. "
                         "0 disables (parity). Requires normal_xyz + mcnr_mask_257.")
    ap.add_argument("--liquid-depth-aware", action="store_true",
                    help="Scale the liquid down-weight by WATER DEPTH instead of applying it flatly. "
                         "Terrain is visible through shallow water and invisible under deep ocean, so "
                         "a flat penalty throws away real learnable signal in rivers/shallows "
                         "(measured: depth p50 ~21 units vs p90 ~514). Shallow pixels keep ~full "
                         "loss weight; only deep water takes the full --liquid-mask-weight penalty.")
    ap.add_argument("--liquid-deep-threshold", type=float, default=100.0,
                    help="Depth (world units) at which the liquid penalty reaches its full value "
                         "under --liquid-depth-aware. Below this it ramps linearly from zero.")
    ap.add_argument("--object-mask-weight", type=float, default=0.0,
                    help="Spec 118: downweight point loss on VISIBLY object-covered pixels by this "
                         "fraction (0.0 = no masking, 1.0 = zero loss where "
                         "object_geometry_visible_mask_257=1). The mask marks only pixels where an "
                         "object actually pokes through the terrain, never the full footprint, so "
                         "a mostly-underground object barely reduces trainable land (FR-006/007). "
                         "Ground-truth signal admissible loss-side only (FR-014); inference inputs "
                         "are unchanged.")
    ap.add_argument("--brush-loss-weight", type=float, default=0.0,
                    help="V7 brush signal as a LOSS WEIGHT (not an input channel): upweight point "
                         "loss on alpha-transition strokes by this factor, so authored detail edges "
                         "carry more gradient than flat painted interiors. 0.5 means brush pixels "
                         "weigh 1.5x. 0 disables (parity with prior runs). Requires alpha_256.")
    ap.add_argument("--feature-store", type=Path, action="append", dest="feature_store", default=None,
                    metavar="FEATURE_STORE",
                    help="Spec 115/117/118: a v115-feature-map-v1 store from a bridge/materializer. "
                         "Concatenates the classifier's GENERATED per-pixel feature map onto the RGB "
                         "input (mit_b0_regression only), so height prediction gets a texture-vs-"
                         "terrain signal. Adds the road-region error metric. RGB-only without it. "
                         "REPEATABLE: pass --feature-store more than once to concatenate several "
                         "priors (e.g. the Spec 115 terrain-feature map AND the Spec 118 object map) "
                         "in CLI order — a later prior AUGMENTS the deconfounding, it does not evict "
                         "the earlier one. in_channels = 3 + sum(class_counts).")
    ap.add_argument("--label-store", type=Path, default=None,
                    help="A v115-terrain-feature-labels-v1 store: EXACT per-layer MCLY ground-truth "
                         "families. Loss-side only, never an inference input — which is precisely "
                         "why ground truth is admissible here, unlike --feature-store (an input, so "
                         "it must be predicted). Enables --flat-paint-weight/--brush-exclude-roads.")
    ap.add_argument("--flat-paint-weight", type=float, default=0.0,
                    help="UP-weight point loss on road-family pixels by this factor (0.5 => 1.5x). "
                         "Roads are painted FLAT, so their height target is already flat; the model "
                         "still sculpts them because the RGB edge is a strong relief cue. Weighting "
                         "the truth harder is the 'do not encode this as geometry' signal. "
                         "Requires --label-store. 0 disables (parity).")
    ap.add_argument("--brush-exclude-roads", action="store_true",
                    help="Withhold the --brush-loss-weight boost on road-family pixels. "
                         "brush_mask_from_alpha collapses the 4 MCLY layers with max(axis=2), so a "
                         "road edge and a cliff-detail edge are indistinguishable and the brush "
                         "term currently teaches the model to put DETAIL at road borders. "
                         "Requires --label-store.")
    ap.add_argument("--mit-hub-id", default=MIT_B0_HUB_ID)
    ap.add_argument("--mit-revision", default=None,
                    help="pinned HF revision; required with --mit-pretrained")
    ap.add_argument("--mit-sha256", default=None, help="pinned weights sha256 recorded in run identity")
    ap.add_argument("--mit-license", default=MIT_B0_LICENSE)
    ap.add_argument("--mit-pretrained", action="store_true",
                    help="USER-RUN: download pinned encoder weights (FR-013 optional ablation)")
    args = ap.parse_args()

    if args.mit_pretrained and (not args.mit_revision or not args.mit_sha256):
        raise SystemExit("--mit-pretrained requires --mit-revision and --mit-sha256 (FR-013 pinning)")
    pretrained_record = None
    if args.mit_pretrained:
        pretrained_record = {
            "hub_id": args.mit_hub_id,
            "revision": args.mit_revision,
            "sha256": args.mit_sha256,
            "license": args.mit_license,
        }

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
        validate_source_selection(attrs=dict(group.attrs), source=args.source)
        selected_rows = select_training_rows(index, args.source)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    split_manifest: dict | None = None
    if args.held_out_split is not None:
        try:
            train_rows, val_rows, split_manifest = apply_held_out_split(
                index_rows=index, selected_rows=selected_rows, split_dir=args.held_out_split,
            )
        except TrainerContractError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        train_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) != args.val_value]
        val_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) == args.val_value]
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(
            f"insufficient rows after --source {args.source}: train={len(train_rows)} val={len(val_rows)}"
        )

    # Optional EXACT per-layer MCLY ground-truth families, used only to shape the loss. Admissible
    # as ground truth precisely because it never reaches the model's input.
    label_group = None
    road_family_index = -1
    if args.label_store is not None:
        label_group = zarr.open_group(str(args.label_store), mode="r")
        label_attrs = dict(label_group.attrs)
        if label_attrs.get("schema") != "v115-terrain-feature-labels-v1":
            raise SystemExit(
                f"--label-store is not a v115-terrain-feature-labels-v1 store: {args.label_store}"
            )
        family_names = list(label_attrs.get("family_names", []))
        if "road" not in family_names:
            raise SystemExit(f"--label-store taxonomy has no 'road' family: {family_names}")
        road_family_index = family_names.index("road")
        # Row-aligned to the curriculum rather than indexed like the feature store; assert that
        # rather than trusting it, because a silent misalignment would weight the wrong pixels.
        label_rows = int(label_group["labels"].shape[0])
        if label_rows != len(index):
            raise SystemExit(
                f"--label-store has {label_rows} rows but the curriculum has {len(index)}; "
                "it must be materialized from this exact curriculum"
            )
    if (args.flat_paint_weight > 0 or args.brush_exclude_roads) and label_group is None:
        raise SystemExit("--flat-paint-weight/--brush-exclude-roads require --label-store")

    # Spec 115/117/118: optional generated feature-map inputs (repeatable). Loaded here so the
    # model is built with the right input-channel count and the dataset can concatenate them. Each
    # store is row-aligned to the curriculum via its index.parquet's source_row_index -- validated,
    # never assumed. Channel order is CLI order; total = sum of the stores' class_counts.
    feature_paths = list(args.feature_store) if args.feature_store else []
    if feature_paths and args.architecture != MIT_B0_ID:
        raise SystemExit(f"--feature-store requires --architecture {MIT_B0_ID}")
    try:
        feature_bindings = load_feature_stores(feature_paths, selected_rows=selected_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    feature_class_count = total_class_count(feature_bindings)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        model, model_identity = build_geometry_model(
            args.architecture,
            in_channels=3 + feature_class_count,
            pretrained_source=pretrained_record
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    use_normal_loss = args.normal_weight > 0
    if use_normal_loss and not ("normal_xyz" in group and "mcnr_mask_257" in group):
        print("WARNING: --normal-weight > 0 but normal_xyz/mcnr_mask_257 missing; "
              "normal gradient supervision disabled.", flush=True)
        use_normal_loss = False

    plan = build_direct_plan(
        architecture_identity=model_identity["architecture"],
        pretrained_source=model_identity["pretrained_source"],
        source=args.source,
        index_rows=index,
        selected_rows=selected_rows,
        batch_size=args.batch,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        clip=args.clip,
        spectral_weight=args.spectral_weight,
        multiscale_weight=args.multiscale_weight,
    )
    if args.held_out_split is not None:
        # build_direct_plan's split_counts reads the curriculum's own stale "split" column;
        # override with the actual rows the training loop below will use, and record the
        # external split's identity so a dry run's printed plan is never wrong about what will
        # actually train (FR-017: rebuilding this split invalidates absolute prior comparisons).
        plan["split_counts"] = {"train": len(train_rows), "val": len(val_rows)}
        plan["train_steps_per_epoch"] = math.ceil(max(len(train_rows), 1) / args.batch)
        plan["held_out_split"] = {
            "path": str((args.held_out_split / "split.json").resolve()),
            "sha256": sha256_file(args.held_out_split / "split.json"),
            "verified_violation_count": int(split_manifest["verified_violation_count"]),
            "absolute_comparison_to_prior_runs_invalid": True,
        }
    if use_normal_loss:
        # Loss-only, like the brush term: normals never enter the model input, so
        # deployment_inputs is unchanged.
        plan["normal_guidance"] = {
            "weight": args.normal_weight,
            "signal": "mcnr_gradient_supervision",
            "source_arrays": ["normal_xyz", "mcnr_mask_257"],
            "evaluated_at": "real MCNR vertices only (145/chunk quincunx)",
            "affects_inference_inputs": False,
        }
    if label_group is not None and (args.flat_paint_weight > 0 or args.brush_exclude_roads):
        plan["mcly_layer_guidance"] = {
            "flat_paint_weight": args.flat_paint_weight,
            "brush_excludes_roads": bool(args.brush_exclude_roads),
            "signal": "per_layer_mcly_ground_truth_families",
            "label_store": str(args.label_store),
            "taxonomy_revision": str(dict(label_group.attrs).get("taxonomy_revision")),
            "rule_set_sha256": str(dict(label_group.attrs).get("rule_set_sha256")),
            "ground_truth_admissible_because": "loss-side only; never an inference input",
            "affects_inference_inputs": False,
        }
    if args.liquid_mask_weight > 0:
        # Recorded so a run's liquid settings are recoverable from its artifacts. Without this the
        # plan captured brush and normals but silently omitted liquid, which made it impossible to
        # tell after the fact whether two runs differed by one knob or two.
        plan["liquid_mask"] = {
            "weight": args.liquid_mask_weight,
            "depth_aware": bool(args.liquid_depth_aware),
            "deep_threshold": args.liquid_deep_threshold if args.liquid_depth_aware else None,
            "source_arrays": ["liquid_mask"] + (
                ["liquid_height", "height_257"] if args.liquid_depth_aware else []
            ),
            "affects_inference_inputs": False,
            "note": "requested settings; the trainer warns and disables if source arrays are absent",
        }
    if args.object_mask_weight > 0:
        # Same recoverability rule as the liquid block: two runs must differ by exactly one knob,
        # visibly, in the plan.
        plan["object_mask"] = {
            "weight": args.object_mask_weight,
            "source_arrays": [OBJECT_MASK_ARRAY],
            "affects_inference_inputs": False,
            "note": "requested settings; the trainer warns and disables if source arrays are absent",
        }
    if args.cache_samples:
        # Report the RAM commitment up front: it scales with how many signals are switched on, and
        # the whole point is that the user can see the cost before starting a long run.
        # Reads args rather than the derived use_* flags: those are resolved further down, after
        # the dry-run exit, and an estimate that runs high is the safe direction for a RAM budget.
        per_sample = (3 + feature_class_count) * 256 * 256 * 4 + 257 * 257 * 4
        if args.liquid_mask_weight > 0 or args.brush_loss_weight > 0:
            per_sample += 257 * 257 * 4
        if use_normal_loss:
            per_sample += 257 * 257 * 3 * 4 + 257 * 257 * 4
        plan["sample_cache"] = {
            "enabled": True,
            "reason": "workers pinned to 0; Zarr decompression otherwise repeats every epoch",
            "bit_identical": True,
            "estimated_ram_gb": round(per_sample * len(selected_rows) / 1024**3, 2),
        }
    if args.brush_loss_weight > 0:
        # Loss-only: the brush signal never enters the model's input, so deployment_inputs is
        # deliberately unchanged. Recorded here so a run's plan states why its loss differs.
        plan["brush_loss"] = {
            "weight": args.brush_loss_weight,
            "signal": "v7_ch12_alpha_transition_strokes",
            "source_array": "alpha_256",
            "affects_inference_inputs": False,
        }
    if feature_bindings:
        plan["deployment_inputs"] = ["minimap_rgb", "generated_terrain_feature_map"]
        plan["feature_stores"] = plan_entries(feature_bindings)
        plan["feature_input_channels"] = 3 + feature_class_count
    plan["pct_start"] = args.pct_start if args.lr_schedule == "onecycle" else None
    plan["warmup_epochs"] = (
        warmup_epochs_for(args.pct_start, args.epochs, plan["train_steps_per_epoch"])
        if args.lr_schedule == "onecycle" else 0
    )
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned CUDA training.", flush=True)
        return 0
    try:
        require_new_output(args.output)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; user-run training refuses CPU.")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.mit_pretrained:
        load_pretrained_encoder(model, hub_id=args.mit_hub_id, revision=args.mit_revision)

    use_liquid_mask = args.liquid_mask_weight > 0
    has_liquid_mask = "liquid_mask" in group
    if use_liquid_mask and not has_liquid_mask:
        print("WARNING: --liquid-mask-weight > 0 but 'liquid_mask' not in store; "
              "liquid loss masking disabled.", flush=True)
        use_liquid_mask = False

    use_object_mask = args.object_mask_weight > 0
    if use_object_mask and not object_mask_available(group):
        print(f"WARNING: --object-mask-weight > 0 but '{OBJECT_MASK_ARRAY}' not in store; "
              "object loss masking disabled.", flush=True)
        use_object_mask = False

    # V7's brush imprint (spec 103 research-v7-contract ch 12), reused here as a LOSS WEIGHT rather
    # than an input channel. It marks alpha *transition* strokes -- paint edges, never solid painted
    # interiors -- which is where authored terrain detail actually lives. V7 used the same signal to
    # build its recovery mask (v7_losses.build_recovery_mask), so weighting loss by it is the
    # established use, not a new invention.
    use_depth_aware = args.liquid_depth_aware and use_liquid_mask
    if use_depth_aware and not ("liquid_height" in group and "height_257" in group):
        print("WARNING: --liquid-depth-aware needs 'liquid_height' and 'height_257'; "
              "falling back to a flat liquid penalty.", flush=True)
        use_depth_aware = False

    use_brush_mask = args.brush_loss_weight > 0
    if use_brush_mask and "alpha_256" not in group:
        print("WARNING: --brush-loss-weight > 0 but 'alpha_256' not in store; "
              "brush loss weighting disabled.", flush=True)
        use_brush_mask = False

    use_flat_paint = args.flat_paint_weight > 0 and label_group is not None
    # Only meaningful when there is a brush boost to withhold in the first place.
    exclude_roads_from_brush = args.brush_exclude_roads and use_brush_mask and label_group is not None

    class RowDataset(Dataset):
        def __init__(self, rows: list[int], *, cache: bool = False) -> None:
            self.rows = rows
            # `--workers` is pinned to 0 on Windows, so every Zarr read below happens on the main
            # thread with the GPU idle (measured: ~31 ms/sample for the full signal set, ~44 s per
            # authored epoch). Sample construction is deterministic — there is no augmentation — so
            # memoising the built tensors is bit-exact: the same values, decompressed once instead
            # of once per epoch. The training loop only ever does `.to(device)`, which copies, so
            # handing back the same CPU tensors cannot be corrupted by in-place writes.
            self._cache: dict[int, tuple] | None = {} if cache else None

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            if self._cache is not None:
                cached = self._cache.get(i)
                if cached is not None:
                    return cached
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            channels = torch.from_numpy(rgb).permute(2, 0, 1)
            feats = feature_channels_for_row(feature_bindings, row)
            if feats is not None:
                # Generated (sum(K), 256, 256) class probabilities from every prior, concatenated
                # onto RGB in CLI order. These are the classifiers' OUTPUTS, never ground-truth
                # labels (Spec 115 FR-007).
                channels = torch.cat([channels, torch.from_numpy(feats)], dim=0)
            target, tile_lo, tile_hi = encode_relative_height(
                np.asarray(group["height_257"][row])
            )
            # v112.1 denominator: the normal-gradient target scales by 1/denom, and denom is
            # per-tile, so it must travel with the row rather than be assumed constant.
            scale_t = torch.tensor(max(tile_hi - tile_lo, 1.0), dtype=torch.float32)
            if use_normal_loss:
                normals_t = torch.from_numpy(
                    np.asarray(group["normal_xyz"][row], dtype=np.float32)
                )
                nmask_t = torch.from_numpy(
                    np.asarray(group["mcnr_mask_257"][row]).astype(np.float32)
                )
            else:
                normals_t = torch.empty(0)
                nmask_t = torch.empty(0)

            def _to_target_grid(mask: np.ndarray) -> torch.Tensor:
                t = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
                t = torch.nn.functional.interpolate(t, size=target.shape, mode="nearest")
                return (t.squeeze(0).squeeze(0) > 0.1).float()

            # One combined per-pixel loss weight: liquid DOWN-weights (the minimap can't see
            # underwater terrain), brush UP-weights (authored detail edges deserve more gradient).
            point_weight = None
            if use_liquid_mask:
                liq = _to_target_grid(np.asarray(group["liquid_mask"][row], dtype=np.float32))
                penalty = args.liquid_mask_weight
                if use_depth_aware:
                    # Terrain IS visible through shallow water, so a flat penalty discards real
                    # signal. Ramp the penalty with depth: shallow keeps ~full loss weight, only
                    # deep water takes the full penalty.
                    surface = np.asarray(group["liquid_height"][row], dtype=np.float32)
                    terrain = np.asarray(group["height_257"][row], dtype=np.float32)
                    side = min(surface.shape[0], terrain.shape[0])
                    depth = np.zeros_like(terrain, dtype=np.float32)
                    depth[:side, :side] = surface[:side, :side] - terrain[:side, :side]
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    depth_norm = np.clip(depth / max(args.liquid_deep_threshold, 1e-6), 0.0, 1.0)
                    depth_t = torch.from_numpy(depth_norm)
                    if depth_t.shape != torch.Size(target.shape):
                        depth_t = torch.nn.functional.interpolate(
                            depth_t[None, None], size=target.shape, mode="bilinear",
                            align_corners=True,
                        ).squeeze(0).squeeze(0)
                    penalty = args.liquid_mask_weight * depth_t
                point_weight = 1.0 - penalty * liq
            if use_object_mask:
                # Spec 118 FR-006/007: visible-object pixels are down-weighted, never dropped;
                # the mask only ever covers the visible portion, so mostly-underground objects
                # keep nearly all of their tile's trainable land.
                obj = _to_target_grid(np.asarray(group[OBJECT_MASK_ARRAY][row], dtype=np.float32))
                obj_factor = 1.0 - args.object_mask_weight * obj
                point_weight = obj_factor if point_weight is None else point_weight * obj_factor
            road_t = None
            if label_group is not None:
                lab = np.asarray(label_group["labels"][row])
                ok = np.asarray(label_group["valid"][row])
                road_t = _to_target_grid(((lab == road_family_index) & ok).astype(np.float32))
            if use_flat_paint:
                # Roads are painted FLAT, so the target here is already flat and correct; the model
                # sculpts them anyway because the RGB edge reads as relief. Weighting the existing
                # truth harder is the "do not encode this as geometry" push.
                factor = 1.0 + args.flat_paint_weight * road_t
                point_weight = factor if point_weight is None else point_weight * factor
            if use_brush_mask:
                brush = brush_mask_from_alpha(np.asarray(group["alpha_256"][row]))
                if brush is not None:
                    brush_t = _to_target_grid(brush)
                    if exclude_roads_from_brush:
                        # max(axis=2) over the 4 MCLY layers made road edges and cliff-detail edges
                        # identical, so the boost was landing on exactly the pixels that must stay
                        # flat. Withhold it there and let the flat-paint term own those pixels.
                        brush_t = brush_t * (1.0 - road_t)
                    factor = 1.0 + args.brush_loss_weight * brush_t
                    point_weight = factor if point_weight is None else point_weight * factor
            weight_t = point_weight if point_weight is not None else torch.empty(0)
            built = (channels, torch.from_numpy(target), weight_t, normals_t, nmask_t, scale_t)
            if self._cache is not None:
                self._cache[i] = built
            return built

    val_targets = [encode_relative_height(np.asarray(group["height_257"][r]))[0] for r in val_rows]
    tile_mean_baseline = compute_tile_mean_baseline(val_targets)
    flat_baseline = compute_flat_baseline(val_targets)

    device = torch.device("cuda")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows, cache=args.cache_samples), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, generator=train_generator,
    )
    val_loader = DataLoader(
        RowDataset(val_rows, cache=args.cache_samples), batch_size=args.batch,
        num_workers=args.workers, pin_memory=True,
    )
    scheduler = None
    warmup_epochs = 0
    if args.lr_schedule == "onecycle":
        scheduler, warmup_epochs = make_onecycle_scheduler(
            opt, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=len(train_loader), pct_start=args.pct_start,
        )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = args.amp and args.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    fixed_preview_rows = select_fixed_preview_rows(val_rows, 8)

    args.output.mkdir(parents=True, exist_ok=True)
    identity = curriculum_identity(args.store)
    run_identity = {
        **release_identity(args.release),
        "model_variant": args.architecture,
        "parameter_count": model_identity["architecture"]["parameter_count"],
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "source_filter": args.source,
        "wdl_prior": False,
        "store": str(args.store.resolve()),
        "optimizer": plan["optimizer"],
        "loss": {
            "point": "smooth_l1",
            "gradient_l1_weight": 0.25,
            "liquid_mask_weight": args.liquid_mask_weight,
            "liquid_depth_aware": bool(use_depth_aware),
            "liquid_deep_threshold": args.liquid_deep_threshold if use_depth_aware else None,
            "object_mask_weight": args.object_mask_weight,
            "object_mask_signal": OBJECT_MASK_ARRAY if use_object_mask else None,
            "brush_loss_weight": args.brush_loss_weight,
            "normal_weight": args.normal_weight,
            "normal_signal": "mcnr_gradient_supervision" if use_normal_loss else None,
            "brush_signal": "v7_ch12_alpha_transition_strokes" if use_brush_mask else None,
            "spectral_weight": args.spectral_weight,
            "multiscale_weight": args.multiscale_weight,
        },
        "schedule": {
            "max_epochs": args.epochs, "batch_size": args.batch,
            "patience": args.patience, "val_tolerance": args.val_tolerance,
            "workers": args.workers, "seed": args.seed, "lr_schedule": args.lr_schedule,
            "pct_start": args.pct_start if args.lr_schedule == "onecycle" else None,
            "warmup_epochs": warmup_epochs,
            "amp": args.amp, "amp_dtype": args.amp_dtype, "grad_clip": args.clip,
        },
        "pretrained_source": model_identity["pretrained_source"],
        "held_out_split": plan.get("held_out_split"),
    }
    (args.output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    per_epoch: list[dict] = []
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y, weight, normals, nmask, scale in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                x_device = x.to(device)
                y_device = y.to(device)
                predicted = model(x_device)
                loss = height_loss(predicted, y_device)
                if weight.numel() > 0:
                    # Per-pixel weighted L1, carrying liquid down-weighting and/or brush
                    # up-weighting. Replaces the default point loss exactly as the liquid-only
                    # path did before, so runs with neither flag stay bit-comparable.
                    loss = (torch.abs(predicted - y_device) * weight.to(device)).mean()
                if use_normal_loss and normals.numel() > 0:
                    loss = loss + args.normal_weight * normal_gradient_loss(
                        predicted.float(), normals.to(device), nmask.to(device), scale.to(device)
                    )
                if args.spectral_weight > 0:
                    loss = loss + args.spectral_weight * radial_spectral_loss(
                        predicted.float(), y_device.float()
                    )
                if args.multiscale_weight > 0:
                    loss = loss + args.multiscale_weight * multiscale_gradient_loss(
                        predicted.float(), y_device.float()
                    )
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().item()))
        model.eval()
        val_absolute_error = 0.0
        val_elements = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
            for x, y, _liq, _n, _nm, _s in val_loader:
                y_device = y.to(device)
                absolute_error = torch.abs(model(x.to(device)) - y_device)
                val_absolute_error += float(absolute_error.sum().item())
                val_elements += absolute_error.numel()
        val_mae = val_absolute_error / val_elements
        train_loss = float(np.mean(train_losses))
        per_epoch.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        checkpoint = {**run_identity, "model": model.state_dict(), "epoch": epoch,
                      "val_mae": val_mae, "curriculum_identity": identity}
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if val_mae < best:
            best = val_mae
            stale = 0
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
            render_fixed_model_preview(
                model, group, index, fixed_preview_rows, device,
                args.output / "validation" / "best_previews" / f"epoch_{epoch:04d}.png",
                epoch=epoch, val_mae=val_mae, use_amp=args.amp,
                feature_bindings=feature_bindings,
            )
        elif args.val_tolerance > 0 and val_mae <= best * (1.0 + args.val_tolerance):
            stale = 0
        elif warmup_complete(epoch, warmup_epochs):
            # Past the OneCycleLR warmup: a non-improving epoch is genuinely stale.
            stale += 1
        # else: still inside warmup -- the LR is deliberately held low, so a flat
        # validation curve is the schedule's design, not a learning failure. Do not
        # penalize it (the prior bug: patience < warmup length killed runs mid-warmup).
        warmup_tag = " (warmup)" if not warmup_complete(epoch, warmup_epochs) else ""
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_mae={val_mae:.6f} "
            f"tile_mean={tile_mean_baseline:.6f} flat={flat_baseline:.6f} "
            f"best={best:.6f} stale={stale}/{args.patience}{warmup_tag}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"[early-stop] no improvement for {stale} epochs", flush=True)
            break

    best_record = min(per_epoch, key=lambda e: e["val_mae"])
    best_checkpoint = torch.load(
        args.output / "checkpoint_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    evaluation = evaluate_height_model(
        model, group, index, val_rows, device, args.output / "validation" / "final_best",
        batch_size=args.batch, workers=args.workers,
        checkpoint_epoch=int(best_checkpoint["epoch"]), use_amp=args.amp,
        feature_bindings=feature_bindings,
    )
    per_row = json.loads(
        (args.output / "validation" / "final_best" / "per_row_metrics.json").read_text(encoding="utf-8")
    )
    sc002 = check_sc002(per_row)
    sc001 = evaluate_sc001(
        best_val_mae=best_record["val_mae"],
        tile_mean_baseline=tile_mean_baseline,
        flat_baseline=flat_baseline,
    )

    # Spec 115 FR-008: road-region height error. The whole point of the feature channel is that
    # height error should drop specifically where the classifier says "road". Reported only when a
    # feature store is present (the RGB-only baseline has no road signal to region against); the
    # gate is a later comparison of two runs' road_region_mae, computed identically here.
    road_region_metrics = None
    road_binding = road_feature_binding(feature_bindings)
    if road_binding is not None:
        from harvester.v50.terrain_feature_labels import ROAD

        model.eval()
        road_abs = 0.0
        road_px = 0
        nonroad_abs = 0.0
        nonroad_px = 0
        with torch.no_grad():
            for row in val_rows:
                # The model consumes ALL priors concatenated, but the road mask is argmaxed from the
                # terrain-feature prior's OWN channels (its class ordering owns the ROAD id).
                all_feats = feature_channels_for_row(feature_bindings, row)
                road_feats = np.asarray(
                    road_binding.group["feature_map"][road_binding.row_to_position[row]],
                    dtype=np.float32,
                )
                rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
                channels = torch.cat(
                    [torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(all_feats)], dim=0
                ).unsqueeze(0).to(device)
                predicted = model(channels)[0].float().cpu().numpy()
                target, _, _ = encode_relative_height(np.asarray(group["height_257"][row]))
                abs_err = np.abs(predicted - target)
                # Road mask at the height grid: argmax of the feature map, nearest-resized 256->257.
                road256 = (road_feats.argmax(axis=0) == ROAD)
                road257 = np.asarray(
                    torch.nn.functional.interpolate(
                        torch.from_numpy(road256.astype(np.float32))[None, None],
                        size=target.shape, mode="nearest",
                    )[0, 0]
                ) > 0.5
                road_abs += float(abs_err[road257].sum())
                road_px += int(road257.sum())
                nonroad_abs += float(abs_err[~road257].sum())
                nonroad_px += int((~road257).sum())
        road_region_metrics = {
            "road_region_mae": (road_abs / road_px) if road_px else None,
            "nonroad_region_mae": (nonroad_abs / nonroad_px) if nonroad_px else None,
            "road_pixels": road_px,
            "nonroad_pixels": nonroad_px,
            "feature_store": str(road_binding.path.resolve()),
            "note": "FR-008: compare road_region_mae against the RGB-only baseline run on the same split",
        }
        rr = road_region_metrics["road_region_mae"]
        print(f"road-region MAE: {rr:.6f} (over {road_px:,} road px) | "
              f"non-road MAE: {road_region_metrics['nonroad_region_mae']:.6f}", flush=True)

    # Spec 118 FR-008: object-touched vs untouched region MAE on the same best checkpoint, so the
    # paired --object-mask-weight 0.0 vs >0 comparison reads the confound directly instead of an
    # aggregate dominated by flat untouched tiles. The ground-truth mask is evaluation/loss-side
    # only here (FR-014); it never enters the model input.
    object_region_metrics = None
    if use_object_mask:
        model.eval()
        obj_abs = 0.0
        obj_px = 0
        free_abs = 0.0
        free_px = 0
        with torch.no_grad():
            for row in val_rows:
                rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
                channels = torch.from_numpy(rgb).permute(2, 0, 1)
                feats = feature_channels_for_row(feature_bindings, row)
                if feats is not None:
                    channels = torch.cat([channels, torch.from_numpy(feats)], dim=0)
                predicted = model(channels.unsqueeze(0).to(device))[0].float().cpu().numpy()
                target, _, _ = encode_relative_height(np.asarray(group["height_257"][row]))
                abs_err = np.abs(predicted - target)
                obj257 = np.asarray(
                    torch.nn.functional.interpolate(
                        torch.from_numpy(
                            np.asarray(group[OBJECT_MASK_ARRAY][row], dtype=np.float32)
                        )[None, None],
                        size=target.shape, mode="nearest",
                    )[0, 0]
                ) > 0.5
                obj_abs += float(abs_err[obj257].sum())
                obj_px += int(obj257.sum())
                free_abs += float(abs_err[~obj257].sum())
                free_px += int((~obj257).sum())
        object_region_metrics = object_subset_metrics(
            obj_abs, obj_px, free_abs, free_px, weight=args.object_mask_weight
        )
        touched_rows = object_touched_rows(group, val_rows)
        object_region_metrics["object_touched_val_tiles"] = int(sum(touched_rows))
        object_region_metrics["val_tiles"] = len(val_rows)
        om = object_region_metrics["object_touched_region_mae"]
        if om is not None:
            print(f"object-touched region MAE: {om:.6f} (over {obj_px:,} object px) | "
                  f"untouched MAE: {object_region_metrics['object_untouched_region_mae']:.6f}",
                  flush=True)

    checkpoint_identity = identity_for_path(args.output / "checkpoint_best.pt")
    stage_run = build_stage_run_summary(
        run_id=args.run_id,
        architecture=model_identity["architecture"],
        pretrained_source=model_identity["pretrained_source"],
        curriculum=identity_for_path(
            args.store / "index.parquet", display_path=str(args.store.resolve())
        ),
        checkpoint={**checkpoint_identity, "best_epoch": int(best_checkpoint["epoch"])},
        baselines={
            "tile_mean": {"val_mae": tile_mean_baseline},
            "flat": {"val_mae": flat_baseline},
            "spec112_frozen": {
                "run_id": "direct_cnn_v112-authored-v1",
                "best_val_mae": SPEC112_FROZEN_BEST_VAL_MAE,
            },
        },
        metrics={
            "best_epoch": best_record["epoch"],
            "best_val_mae": best_record["val_mae"],
            "evaluator": evaluation,
            "sc001": sc001,
            "sc002": sc002,
            "road_region": road_region_metrics,
            "object_region": object_region_metrics,
            "structural_failure_epoch1_best": best_record["epoch"] == 1,
        },
        visual_evidence={
            "best_epoch_previews": "validation/best_previews",
            "error_quantiles": "validation/final_best/error_quantiles.png",
            "worst_cases": "validation/final_best/worst_cases.png",
            "per_row_metrics": "validation/final_best/per_row_metrics.json",
        },
    )
    (args.output / "model_stage_run.json").write_text(
        json.dumps(stage_run, indent=2), encoding="utf-8"
    )
    (args.output / "training_summary.json").write_text(
        json.dumps({"per_epoch_metrics": per_epoch, "model_stage_run": stage_run["run_id"]}, indent=2),
        encoding="utf-8",
    )
    if stage_run["metrics"]["structural_failure_epoch1_best"]:
        print("STRUCTURAL FAILURE: best epoch is epoch 1; this run is not a success.", flush=True)
        return 1
    print(
        f"best_epoch={best_record['epoch']} best_val_mae={best_record['val_mae']:.6f} "
        f"sc001={sc001['passes']} sc002={sc002['passes']} promotion=pending(user visual gate)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
