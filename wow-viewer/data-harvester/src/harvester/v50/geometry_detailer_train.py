"""Spec 114 T060: residual detailer trainer (USER runs CUDA).

Trains the detailer on GENERATED coarse outputs: minimap RGB + the frozen upstream checkpoint's
materialized ``coarse_relief`` -> one residual field; final relief = coarse + residual. The
coarse-only composition (residual ≡ 0) is the mandatory strong baseline, reported in-run from
validation truth — the detailer must beat it by ≥5% relative val MAE, alongside the flat,
tile-mean, and frozen Spec 112 references already owned by the coarse stage.

Split discipline: the derived coarse store's index is validated to align 1:1 with the selected
source rows, so the detailer reuses the exact frozen source-group split. The upstream checkpoint
never trained on val rows, so its val predictions are honest generated outputs.

Artifacts mirror the coarse trainer (plan, run identity, both checkpoints, fixed previews, per-row
metrics, quantile/worst sheets) plus ``model_stage_run.json`` with ``upstream_models`` naming the
coarse checkpoint — the detailer is replaceable without retraining anything else.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec118.object_loss import (
    OBJECT_MASK_ARRAY,
    object_mask_available,
)
from harvester.spec118.object_loss import (
    subset_metrics as object_subset_metrics,
)
from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.direct_geometry_materialize import COARSE_ARRAY, COARSE_STORE_SCHEMA
from harvester.v50.direct_geometry_train import apply_held_out_split
from harvester.v50.feature_stores import (
    feature_channels_for_row,
    load_feature_stores,
    plan_entries,
    total_class_count,
)
from harvester.v50.geometry_detailer_model import (
    DETAILER_ARCHITECTURE_ID,
    GeometryDetailerNet,
    compose_final,
    detailer_identity,
)
from harvester.v50.height_relative_evaluate import (
    compute_row_metrics,
    render_validation_sheet,
    select_error_quantile_rows,
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
    validate_model_stage_run,
)
from harvester.v50.spectral_guidance import (
    frequency_loss_2d,
    frequency_split_loss,
    laplacian_loss,
    multiscale_gradient_loss,
    radial_spectral_loss,
    sobel_edge_loss,
    transition_focus_loss,
)

STAGE = "direct_geometry"
OUTPUT_SIGNAL = "residual_relief_257"
SPEC112_FROZEN_BEST_VAL_MAE = 0.1492665126
DETAILER_RELATIVE_MARGIN = 0.05
LR_SCHEDULES = frozenset({"constant", "onecycle"})


def validate_coarse_store(
    *,
    attrs: dict,
    coarse_index_rows: list[dict],
    coarse_array_rows: int,
    selected: list[int],
    source: str,
) -> None:
    """Fail closed unless the derived store aligns 1:1 with the selected source rows."""
    if attrs.get("schema") != COARSE_STORE_SCHEMA:
        raise TrainerContractError(
            f"coarse store schema must be {COARSE_STORE_SCHEMA!r}, got {attrs.get('schema')!r}"
        )
    if attrs.get("source_filter") != source:
        raise TrainerContractError(
            f"coarse store was materialized with source filter {attrs.get('source_filter')!r}, "
            f"but this run selects {source!r}"
        )
    if coarse_array_rows != len(selected) or len(coarse_index_rows) != len(selected):
        raise TrainerContractError(
            f"coarse store row count ({coarse_array_rows} array / {len(coarse_index_rows)} index) "
            f"does not match the {len(selected)} selected source rows"
        )
    misaligned = [
        position
        for position, row in enumerate(coarse_index_rows)
        if int(row.get("source_row_index", -1)) != selected[position]
    ]
    if misaligned:
        raise TrainerContractError(
            f"coarse store index does not align with selected source rows at positions "
            f"{misaligned[:5]}; re-materialize from the same curriculum and source filter"
        )


def compute_coarse_baseline(coarse: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """Val MAE of the upstream composition alone — the strong baseline the detailer must beat."""
    if not coarse or len(coarse) != len(targets):
        raise TrainerContractError("coarse baseline requires one coarse field per validation target")
    return float(np.mean([float(np.abs(c - t).mean()) for c, t in zip(coarse, targets, strict=True)]))


def evaluate_detailer_gate(*, best_val_mae: float, coarse_baseline: float) -> dict:
    """The detailer promotion gate: ≥5% relative improvement over the coarse-only composition."""
    threshold = coarse_baseline * (1.0 - DETAILER_RELATIVE_MARGIN)
    return {
        "best_val_mae": best_val_mae,
        "coarse_only_baseline": coarse_baseline,
        "relative_margin_required": DETAILER_RELATIVE_MARGIN,
        "threshold": threshold,
        "beats_coarse_only": best_val_mae <= threshold,
        "passes": best_val_mae <= threshold,
    }


def build_detailer_plan(
    *,
    architecture: dict,
    upstream: dict,
    source: str,
    selected_rows: int,
    train_rows: int,
    val_rows: int,
    batch_size: int,
    epochs: int,
    seed: int,
    lr: float,
    lr_schedule: str,
    amp: bool,
    amp_dtype: str,
    clip: float,
    spectral_weight: float,
    multiscale_weight: float,
    frequency_2d_weight: float,
    laplacian_weight: float,
    edge_weight: float,
    transition_focus_weight: float,
    band_lf_weight: float,
    band_hf_weight: float,
    band_cutoff: float,
) -> dict:
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    if lr_schedule not in LR_SCHEDULES:
        raise TrainerContractError(f"lr schedule must be one of {sorted(LR_SCHEDULES)}")
    return {
        "schema": "v114-detailer-plan-v1",
        "stage": STAGE,
        "architecture": architecture,
        "upstream_coarse_checkpoint": upstream,
        "source_filter": source,
        "selected_rows": selected_rows,
        "split_counts": {"train": train_rows, "val": val_rows},
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
            "frequency_2d_weight": frequency_2d_weight,
            "laplacian_weight": laplacian_weight,
            "edge_weight": edge_weight,
            "transition_focus_weight": transition_focus_weight,
            "band_lf_weight": band_lf_weight,
            "band_hf_weight": band_hf_weight,
            "band_cutoff": band_cutoff,
        },
        "train_steps_per_epoch": math.ceil(train_rows / batch_size),
        "deployment_inputs": ["minimap_rgb", "generated_coarse_relief"],
        "training_target": "relative_height_257 - generated coarse (residual, v112.1 space)",
        "wdl_prior": False,
        "teacher_forced_truth_inputs": False,
    }


def build_detailer_stage_run(
    *,
    run_id: str,
    architecture: dict,
    upstream_identity: dict,
    curriculum: dict,
    checkpoint: dict,
    baselines: dict,
    metrics: dict,
    visual_evidence: dict,
    created_utc: str | None = None,
) -> dict:
    summary = {
        "schema": "v50-model-stage-run-v1",
        "run_id": run_id,
        "created_utc": created_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": STAGE,
        "output_signal": OUTPUT_SIGNAL,
        "architecture": architecture,
        "pretrained_source": None,
        "curriculum": curriculum,
        "upstream_models": [upstream_identity],
        "checkpoint": checkpoint,
        "baselines": baselines,
        "metrics": metrics,
        "visual_evidence": visual_evidence,
        "promotion_verdict": "pending",
    }
    try:
        validate_model_stage_run(summary)
    except ContractViolationError as exc:
        raise TrainerContractError(f"stage-run record violates its own contract: {exc}") from exc
    return summary


def render_detailer_epoch_preview(
    *,
    model,
    group,
    coarse_group,
    selected_rows: list[int],
    positions: list[int],
    index: list[dict],
    device,
    output: Path,
    epoch: int,
    val_mae: float,
    amp_enabled: bool,
    amp_dtype,
    feature_bindings=None,
) -> None:
    """Render a comprehensive per-epoch validation sheet showing ALL signals.

    Panels per tile (left to right):
      1. Minimap RGB (input)
      2. Coarse relief (frozen upstream input)
      3. Residual (detailer output, signed)
      4. Final composition (coarse + residual, clamped)
      5. Ground truth
      6. Signed error (final - truth)
      7. Absolute error
      8. Liquid mask (if available in store)
      9. Normal XYZ (if available in store, rendered as RGB)

    Called per-epoch when val_mae improves, so the user can visually track progress
    during training without waiting for the final evaluation pass.
    """
    import torch
    from PIL import Image, ImageDraw, ImageFont

    from harvester.v50.geometry_detailer_model import compose_final
    from harvester.v50.height_relative_evaluate import (
        _absolute_error_rgb,
        _gray_fixed,
        _signed_error_rgb,
        compute_row_metrics,
    )
    from harvester.v50.height_relative_model import encode_relative_height

    has_liquid = "liquid_mask" in group
    has_normals = "normal_xyz" in group

    labels = ["Minimap RGB", "Coarse relief", "Residual (signed)",
              "Final (coarse+res)", "Ground truth", "Signed error", "Abs error"]
    if has_liquid:
        labels.append("Liquid mask")
    if has_normals:
        labels.append("Normal XYZ")

    panel_size = 160
    label_width = 200
    header_height = 42
    row_height = panel_size + 4
    gap = 4
    width = label_width + (len(labels) * panel_size) + ((len(labels) - 1) * gap)
    height = header_height + (len(positions) * row_height)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 3), f"detailer fixed validation | epoch {epoch} | MAE {val_mae:.6f}",
              fill=(20, 20, 20), font=font)
    for column, label in enumerate(labels):
        x = label_width + column * (panel_size + gap)
        draw.text((x + 3, 23), label, fill=(30, 30, 30), font=font)

    model.eval()
    with torch.no_grad():
        for row_index, position in enumerate(positions):
            source_row = selected_rows[position]
            rgb = np.asarray(group["minimap_rgb"][source_row], dtype=np.uint8)
            coarse = np.asarray(coarse_group[COARSE_ARRAY][position], dtype=np.float32)
            truth, _, _ = encode_relative_height(np.asarray(group["height_257"][source_row]))
            rgb_channels = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            feats = feature_channels_for_row(feature_bindings or [], source_row)
            if feats is not None:
                rgb_channels = torch.cat([rgb_channels, torch.from_numpy(feats)], dim=0)
            tensor = rgb_channels.unsqueeze(0).to(device)
            coarse_t = torch.from_numpy(coarse).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                residual = model(tensor, coarse_t)
                final = compose_final(coarse_t, residual, clamp=True)
            predicted = final[0].float().cpu().numpy()
            residual_np = residual[0].float().cpu().numpy()
            row = index[source_row]
            metrics = compute_row_metrics(predicted, truth)
            y = header_height + row_index * row_height
            label_text = (
                f"row {source_row}  {row.get('map', '?')}\n"
                f"MAE {metrics['mae']:.4f}  base {metrics['tile_mean_baseline_mae']:.4f}\n"
                f"grad {metrics['gradient_mae']:.4f}  border {metrics['border_mae']:.4f}"
            )
            draw.multiline_text((5, y + 5), label_text, fill=(20, 20, 20), font=font, spacing=3)

            panels = [
                np.asarray(rgb, dtype=np.uint8),
                _gray_fixed(coarse),
                _signed_error_rgb(residual_np),
                _gray_fixed(predicted),
                _gray_fixed(truth),
                _signed_error_rgb(predicted - truth),
                _absolute_error_rgb(np.abs(predicted - truth)),
            ]
            if has_liquid:
                liq = np.asarray(group["liquid_mask"][source_row], dtype=np.float32)
                # Upscale to 257x257 if needed
                if liq.shape != truth.shape:
                    liq_t = torch.from_numpy(liq).unsqueeze(0).unsqueeze(0)
                    liq_t = torch.nn.functional.interpolate(liq_t, size=truth.shape, mode="nearest")
                    liq = liq_t.squeeze(0).squeeze(0).numpy()
                liq_rgb = np.stack([liq * 255, liq * 100, liq * 255], axis=-1).astype(np.uint8)
                panels.append(liq_rgb)
            if has_normals:
                nrm = np.asarray(group["normal_xyz"][source_row], dtype=np.float32)
                # Normals are (H, W, 3) in [-1, 1] — map to [0, 255]
                nrm_rgb = np.clip((nrm + 1.0) * 127.5, 0, 255).astype(np.uint8)
                if nrm_rgb.shape[:2] != truth.shape:
                    nrm_t = torch.from_numpy(nrm_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                    nrm_t = torch.nn.functional.interpolate(nrm_t, size=truth.shape, mode="nearest")
                    nrm_rgb = nrm_t.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                panels.append(nrm_rgb)

            for column, panel in enumerate(panels):
                image = Image.fromarray(panel, mode="RGB").resize(
                    (panel_size, panel_size), Image.Resampling.BILINEAR
                )
                x = label_width + column * (panel_size + gap)
                canvas.paste(image, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    import pyarrow.parquet as pq
    import torch
    import zarr
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser(description="Spec 114 residual detailer trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="dual-source curriculum store")
    ap.add_argument("--coarse-store", required=True, type=Path, help="materialized coarse store")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--val-key", default="split")
    ap.add_argument("--val-value", default="val")
    ap.add_argument("--held-out-split", type=Path, default=None,
                    help="Spec 116 US4: a v50-held-out-split-v1 directory whose spatially-isolated "
                         "held_out rows become validation, overriding --val-key/--val-value. "
                         "Refuses a leaky split. Never mutates --store.")
    ap.add_argument("--feature-store", type=Path, action="append", dest="feature_store", default=None,
                    metavar="FEATURE_STORE",
                    help="Spec 115/116/117/118: a v115-feature-map-v1 store (from a bridge/"
                         "materializer). Concatenates the classifier's GENERATED per-pixel feature "
                         "map onto RGB, so the detailer gets a texture-vs-terrain signal alongside "
                         "the coarse field. RGB-only without it. REPEATABLE: pass more than once to "
                         "concatenate several priors in CLI order (must match how the coarse "
                         "checkpoint's feature stores were ordered). in_channels = 3 + "
                         "sum(class_counts).")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=sorted(LR_SCHEDULES))
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"],
                    help="AMP dtype: fp16 (default, needs GradScaler) or bf16 (native on Ampere+, "
                         "no scaler needed, better numerical stability for FFT losses)")
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--spectral-weight", type=float, default=0.0)
    ap.add_argument("--multiscale-weight", type=float, default=0.0)
    ap.add_argument("--frequency-2d-weight", type=float, default=0.0,
                    help="V7 full 2D log-magnitude FFT L1 weight (directional structure)")
    ap.add_argument("--laplacian-weight", type=float, default=0.0,
                    help="V7 5-point Laplacian curvature L1 weight")
    ap.add_argument("--edge-weight", type=float, default=0.0,
                    help="V7 Sobel edge magnitude L1 weight")
    ap.add_argument("--transition-focus-weight", type=float, default=0.0,
                    help="V7 transition-focus weighted L1 weight (up-weights terrain transitions)")
    ap.add_argument("--band-lf-weight", type=float, default=0.0,
                    help="V25 LF band-split loss weight (low-frequency structure)")
    ap.add_argument("--band-hf-weight", type=float, default=0.0,
                    help="V25 HF band-split loss weight (high-frequency detail)")
    ap.add_argument("--band-cutoff", type=float, default=0.1,
                    help="V25 radial FFT cutoff fraction for LF/HF split (0.1 = lowest 10%%)")
    ap.add_argument("--workers", type=int, choices=[0], default=0)
    ap.add_argument("--val-tolerance", type=float, default=0.0,
                    help="Val MAE within this fraction of best still resets stale counter "
                         "(0.0 = strict, 0.01 = within 1%% of best counts as not stale). "
                         "Handles noisy val MAE when train loss is still decreasing.")
    ap.add_argument("--liquid-mask-weight", type=float, default=0.0,
                    help="Downweight point loss in liquid regions by this fraction "
                         "(0.0 = no masking, 1.0 = zero loss where liquid_mask=1). "
                         "The minimap input is blue water in liquid regions — the model "
                         "can't see underwater terrain, so penalizing it there is noise.")
    ap.add_argument("--object-mask-weight", type=float, default=0.0,
                    help="Spec 118: downweight point loss on VISIBLY object-covered pixels by this "
                         "fraction (0.0 = no masking, 1.0 = zero loss where "
                         "object_geometry_visible_mask_257=1). The mask marks only pixels where an "
                         "object actually pokes through the terrain, never the full footprint, so "
                         "a mostly-underground object barely reduces trainable land (FR-006/007). "
                         "Ground-truth signal admissible loss-side only (FR-014); the coarse-only "
                         "baseline stays unmasked so the relative gate remains honest.")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--pct-start", type=float, default=PCT_START_DEFAULT,
                    help="OneCycleLR warmup fraction (torch default 0.3 = 30%% of steps). The "
                         "early-stopper is warmup-aware: it does not count stale epochs until the "
                         "warmup phase completes. This matters most for the detailer, whose "
                         "zero-init residual head starts AT the coarse baseline and cannot improve "
                         "validation until the LR rises -- a patience shorter than the warmup "
                         "previously killed runs mid-warmup (best epoch 2, early-stop epoch 17). "
                         "For small datasets a shorter warmup (0.1) wastes less of a short run.")
    ap.add_argument("--seed", type=int, default=114)
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="Path to a checkpoint .pt file to initialize model weights from "
                         "(resumes training from that checkpoint's epoch).")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    coarse_group = zarr.open_group(str(args.coarse_store), mode="r")
    coarse_index = pq.read_table(args.coarse_store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
        validate_source_selection(attrs=dict(group.attrs), source=args.source)
        selected_rows = select_training_rows(index, args.source)
        validate_coarse_store(
            attrs=dict(coarse_group.attrs),
            coarse_index_rows=coarse_index,
            coarse_array_rows=int(coarse_group[COARSE_ARRAY].shape[0]),
            selected=selected_rows,
            source=args.source,
        )
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    positions = list(range(len(selected_rows)))
    held_out_split_manifest: dict | None = None
    if args.held_out_split is not None:
        try:
            train_rows, val_rows, held_out_split_manifest = apply_held_out_split(
                index_rows=index, selected_rows=selected_rows, split_dir=args.held_out_split,
            )
        except TrainerContractError as exc:
            raise SystemExit(str(exc)) from exc
        row_to_position = {row: position for position, row in enumerate(selected_rows)}
        train_positions = [row_to_position[r] for r in train_rows]
        val_positions = [row_to_position[r] for r in val_rows]
    else:
        train_positions = [p for p in positions if str(index[selected_rows[p]].get(args.val_key)) != args.val_value]
        val_positions = [p for p in positions if str(index[selected_rows[p]].get(args.val_key)) == args.val_value]
    if len(train_positions) < 32 or len(val_positions) < 8:
        raise SystemExit(
            f"insufficient rows: train={len(train_positions)} val={len(val_positions)}"
        )

    # Spec 115/116: optional generated feature-map input, concatenated onto RGB (never ground
    # truth). Loaded here so the model is built with the right input-channel count. Row-aligned to
    # the curriculum via its own index.parquet's source_row_index -- validated, never assumed.
    feature_paths = list(args.feature_store) if args.feature_store else []
    try:
        feature_bindings = load_feature_stores(feature_paths, selected_rows=selected_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    feature_class_count = total_class_count(feature_bindings)
    in_channels = 3 + feature_class_count

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = GeometryDetailerNet(in_channels=in_channels)
    if args.init_weights is not None:
        ck = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(ck["model"])
        print(f"Loaded init weights from {args.init_weights} (epoch {ck.get('epoch', '?')})", flush=True)
    architecture = detailer_identity(model, in_channels=in_channels)
    upstream_identity = {
        "path": str(coarse_group.attrs.get("checkpoint_path", "unknown")),
        "sha256": str(coarse_group.attrs.get("checkpoint_sha256", "0" * 64)),
    }
    plan = build_detailer_plan(
        architecture=architecture,
        upstream=upstream_identity,
        source=args.source,
        selected_rows=len(selected_rows),
        train_rows=len(train_positions),
        val_rows=len(val_positions),
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
        frequency_2d_weight=args.frequency_2d_weight,
        laplacian_weight=args.laplacian_weight,
        edge_weight=args.edge_weight,
        transition_focus_weight=args.transition_focus_weight,
        band_lf_weight=args.band_lf_weight,
        band_hf_weight=args.band_hf_weight,
        band_cutoff=args.band_cutoff,
    )
    if feature_bindings:
        plan["deployment_inputs"] = ["minimap_rgb", "generated_terrain_feature_map", "generated_coarse_relief"]
        plan["feature_stores"] = plan_entries(feature_bindings)
        plan["feature_input_channels"] = in_channels
    if held_out_split_manifest is not None:
        plan["split_counts"] = {"train": len(train_positions), "val": len(val_positions)}
        plan["held_out_split"] = {
            "path": str((args.held_out_split / "split.json").resolve()),
            "verified_violation_count": int(held_out_split_manifest["verified_violation_count"]),
            "absolute_comparison_to_prior_runs_invalid": True,
        }
    if args.object_mask_weight > 0:
        # Recoverability rule: two runs must differ by exactly one knob, visibly, in the plan.
        plan["object_mask"] = {
            "weight": args.object_mask_weight,
            "source_arrays": [OBJECT_MASK_ARRAY],
            "affects_inference_inputs": False,
            "note": "requested settings; the trainer warns and disables if source arrays are absent",
        }
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

    class DetailerDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            position = self.rows[i]
            source_row = selected_rows[position]
            rgb = np.asarray(group["minimap_rgb"][source_row], dtype=np.float32) / 255.0
            channels = torch.from_numpy(rgb).permute(2, 0, 1)
            feats = feature_channels_for_row(feature_bindings, source_row)
            if feats is not None:
                # Generated (sum(K), 256, 256) class probabilities from every prior, concatenated
                # onto RGB in CLI order. Classifiers' OUTPUTS, never ground-truth (Spec 115 FR-007).
                channels = torch.cat([channels, torch.from_numpy(feats)], dim=0)
            coarse = np.asarray(coarse_group[COARSE_ARRAY][position], dtype=np.float32)
            truth, _, _ = encode_relative_height(np.asarray(group["height_257"][source_row]))
            if use_liquid_mask:
                liq = np.asarray(group["liquid_mask"][source_row], dtype=np.float32)
                # liquid_mask is at 256x256; interpolate to 257x257 (nearest, binary)
                liq_t = torch.from_numpy(liq).unsqueeze(0).unsqueeze(0)
                liq_t = torch.nn.functional.interpolate(
                    liq_t, size=truth.shape, mode="nearest"
                ).squeeze(0).squeeze(0)
                # Binarize: 1 = liquid, 0 = terrain
                liq_t = (liq_t > 0.1).float()
            else:
                liq_t = torch.empty(0)
            if use_object_mask:
                # Strict visible-object mask at 257x257; same nearest-interp convention as liquid.
                obj = np.asarray(group[OBJECT_MASK_ARRAY][source_row], dtype=np.float32)
                obj_t = torch.from_numpy(obj).unsqueeze(0).unsqueeze(0)
                obj_t = torch.nn.functional.interpolate(
                    obj_t, size=truth.shape, mode="nearest"
                ).squeeze(0).squeeze(0)
                obj_t = (obj_t > 0.1).float()
            else:
                obj_t = torch.empty(0)
            return (
                channels,
                torch.from_numpy(coarse),
                torch.from_numpy(truth),
                liq_t,
                obj_t,
            )

    val_coarse = [np.asarray(coarse_group[COARSE_ARRAY][p], dtype=np.float32) for p in val_positions]
    val_targets = [
        encode_relative_height(np.asarray(group["height_257"][selected_rows[p]]))[0]
        for p in val_positions
    ]
    coarse_baseline = compute_coarse_baseline(val_coarse, val_targets)
    tile_mean_baseline = compute_tile_mean_baseline(val_targets)
    flat_baseline = float(np.mean([float(np.abs(t - 0.5).mean()) for t in val_targets]))

    device = torch.device("cuda")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        DetailerDataset(train_positions), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, generator=train_generator,
    )
    val_loader = DataLoader(
        DetailerDataset(val_positions), batch_size=args.batch,
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
    fixed_preview_positions = select_fixed_preview_rows(val_positions, 8)

    args.output.mkdir(parents=True, exist_ok=True)
    identity = curriculum_identity(args.store)
    run_identity = {
        **release_identity(args.release),
        "model_variant": DETAILER_ARCHITECTURE_ID,
        "parameter_count": architecture["parameter_count"],
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "source_filter": args.source,
        "wdl_prior": False,
        "store": str(args.store.resolve()),
        "coarse_store": str(args.coarse_store.resolve()),
        "upstream_coarse_checkpoint": upstream_identity,
        "optimizer": plan["optimizer"],
        "loss": {
            "point": "smooth_l1", "gradient_l1_weight": 0.25,
            "liquid_mask_weight": args.liquid_mask_weight,
            "object_mask_weight": args.object_mask_weight,
            "object_mask_signal": OBJECT_MASK_ARRAY if use_object_mask else None,
            "spectral_weight": args.spectral_weight,
            "multiscale_weight": args.multiscale_weight,
            "frequency_2d_weight": args.frequency_2d_weight,
            "laplacian_weight": args.laplacian_weight,
            "edge_weight": args.edge_weight,
            "transition_focus_weight": args.transition_focus_weight,
            "band_lf_weight": args.band_lf_weight,
            "band_hf_weight": args.band_hf_weight,
            "band_cutoff": args.band_cutoff,
            "on": "final composition (coarse + residual), unclamped",
        },
        "schedule": {
            "max_epochs": args.epochs, "batch_size": args.batch,
            "patience": args.patience, "val_tolerance": args.val_tolerance,
            "workers": args.workers, "seed": args.seed, "lr_schedule": args.lr_schedule,
            "pct_start": args.pct_start if args.lr_schedule == "onecycle" else None,
            "warmup_epochs": warmup_epochs,
            "amp": args.amp, "amp_dtype": args.amp_dtype, "grad_clip": args.clip,
        },
        "feature_stores": plan.get("feature_stores"),
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
        for rgb, coarse, truth, liq, obj in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                rgb_d = rgb.to(device)
                coarse_d = coarse.to(device)
                truth_d = truth.to(device)
                residual = model(rgb_d, coarse_d)
                final = compose_final(coarse_d, residual, clamp=False)
                loss = height_loss(final, truth_d)
                point_weight = None
                if use_liquid_mask and liq.numel() > 0:
                    # Downweight point loss in liquid regions: (1 - weight * mask)
                    point_weight = 1.0 - args.liquid_mask_weight * liq.to(device)
                if use_object_mask and obj.numel() > 0:
                    # Spec 118 FR-006/007: visible-object pixels are down-weighted, never dropped;
                    # the mask only ever covers the visible portion, so mostly-underground
                    # objects keep nearly all of their tile's trainable land.
                    obj_factor = 1.0 - args.object_mask_weight * obj.to(device)
                    point_weight = obj_factor if point_weight is None else point_weight * obj_factor
                if point_weight is not None:
                    loss = (torch.abs(final - truth_d) * point_weight).mean()
                if args.spectral_weight > 0:
                    loss = loss + args.spectral_weight * radial_spectral_loss(
                        final.float(), truth_d.float()
                    )
                if args.multiscale_weight > 0:
                    loss = loss + args.multiscale_weight * multiscale_gradient_loss(
                        final.float(), truth_d.float()
                    )
                if args.frequency_2d_weight > 0:
                    loss = loss + args.frequency_2d_weight * frequency_loss_2d(
                        final.float(), truth_d.float()
                    )
                if args.laplacian_weight > 0:
                    loss = loss + args.laplacian_weight * laplacian_loss(
                        final.float(), truth_d.float()
                    )
                if args.edge_weight > 0:
                    loss = loss + args.edge_weight * sobel_edge_loss(
                        final.float(), truth_d.float()
                    )
                if args.transition_focus_weight > 0:
                    loss = loss + args.transition_focus_weight * transition_focus_loss(
                        final.float(), truth_d.float()
                    )
                if args.band_lf_weight > 0 or args.band_hf_weight > 0:
                    lf_loss, hf_loss = frequency_split_loss(
                        final.float(), truth_d.float(), cutoff=args.band_cutoff
                    )
                    loss = loss + args.band_lf_weight * lf_loss + args.band_hf_weight * hf_loss
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
            for rgb, coarse, truth, _liq, _obj in val_loader:
                truth_d = truth.to(device)
                final = compose_final(
                    coarse.to(device), model(rgb.to(device), coarse.to(device)), clamp=True
                )
                absolute_error = torch.abs(final - truth_d)
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
            preview_dir = args.output / "validation" / "best_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            render_detailer_epoch_preview(
                model=model, group=group, coarse_group=coarse_group,
                selected_rows=selected_rows, positions=fixed_preview_positions,
                index=index, device=device,
                output=preview_dir / f"epoch_{epoch:04d}.png",
                epoch=epoch, val_mae=val_mae,
                amp_enabled=args.amp, amp_dtype=amp_dtype,
                feature_bindings=feature_bindings,
            )
        elif args.val_tolerance > 0 and val_mae <= best * (1.0 + args.val_tolerance):
            # Within tolerance band of best: model is still in a productive region.
            # Reset stale but don't update best (only strict improvement saves a checkpoint).
            stale = 0
        elif warmup_complete(epoch, warmup_epochs):
            # Past the OneCycleLR warmup: a non-improving epoch is genuinely stale.
            stale += 1
        # else: still inside warmup -- the LR is deliberately held low, so a flat
        # validation curve is the schedule's design, not a learning failure. The
        # detailer's zero-init residual head starts AT the coarse baseline and cannot
        # move until the LR rises; penalizing that as "stale" killed runs mid-warmup.
        warmup_tag = " (warmup)" if not warmup_complete(epoch, warmup_epochs) else ""
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_mae={val_mae:.6f} "
            f"coarse={coarse_baseline:.6f} tile_mean={tile_mean_baseline:.6f} "
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
    model.eval()

    # Final all-validation evaluation with per-row metrics and fixed-scale sheets.
    eval_dir = args.output / "validation" / "final_best"
    eval_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    samples_by_position: dict[int, dict] = {}
    obj_abs = 0.0
    obj_px = 0
    free_abs = 0.0
    free_px = 0
    with torch.no_grad():
        for position in val_positions:
            source_row = selected_rows[position]
            rgb = np.asarray(group["minimap_rgb"][source_row], dtype=np.uint8)
            coarse = np.asarray(coarse_group[COARSE_ARRAY][position], dtype=np.float32)
            truth, _, _ = encode_relative_height(np.asarray(group["height_257"][source_row]))
            rgb_channels = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            feats = feature_channels_for_row(feature_bindings or [], source_row)
            if feats is not None:
                rgb_channels = torch.cat([rgb_channels, torch.from_numpy(feats)], dim=0)
            tensor = rgb_channels.unsqueeze(0).to(device)
            coarse_t = torch.from_numpy(coarse).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                final = compose_final(coarse_t, model(tensor, coarse_t), clamp=True)[0]
            predicted = final.float().cpu().numpy()
            index_row = index[source_row]
            metrics = compute_row_metrics(predicted, truth)
            if use_object_mask:
                # Spec 118 FR-008: object-touched vs untouched region MAE on the best checkpoint,
                # so the paired with/without comparison reads the confound directly. Ground-truth
                # mask is evaluation/loss-side only here (FR-014).
                obj257 = np.asarray(
                    torch.nn.functional.interpolate(
                        torch.from_numpy(
                            np.asarray(group[OBJECT_MASK_ARRAY][source_row], dtype=np.float32)
                        )[None, None],
                        size=truth.shape, mode="nearest",
                    )[0, 0]
                ) > 0.5
                abs_err = np.abs(predicted - truth)
                obj_abs += float(abs_err[obj257].sum())
                obj_px += int(obj257.sum())
                free_abs += float(abs_err[~obj257].sum())
                free_px += int((~obj257).sum())
            records.append({
                "row_id": int(source_row),
                "map": str(index_row.get("map", "unknown")),
                "minimap_source": str(index_row.get("minimap_source", "unknown")),
                **metrics,
            })
            samples_by_position[position] = {
                "row_id": int(source_row),
                "label": f"row {source_row}  {index_row.get('map', '?')}",
                "rgb": rgb,
                "target": truth,
                "predicted": predicted,
                "metrics": metrics,
            }
    (eval_dir / "per_row_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    from harvester.v50.direct_geometry_train import check_sc002

    sc002 = check_sc002(records)
    gate = evaluate_detailer_gate(best_val_mae=best_record["val_mae"], coarse_baseline=coarse_baseline)
    quantile_rows = select_error_quantile_rows(records, 8)
    position_by_row_id = {selected_rows[p]: p for p in val_positions}
    render_validation_sheet(
        [samples_by_position[position_by_row_id[row]] for row in quantile_rows],
        eval_dir / "error_quantiles.png",
        title=f"detailer all-validation error quantiles | checkpoint epoch {best_checkpoint['epoch']}",
    )
    worst_rows = [
        int(r["row_id"]) for r in sorted(
            records, key=lambda r: (-float(r["mae"]), int(r["row_id"]))
        )[:8]
    ]
    render_validation_sheet(
        [samples_by_position[position_by_row_id[row]] for row in worst_rows],
        eval_dir / "worst_cases.png",
        title=f"detailer worst held-out rows | checkpoint epoch {best_checkpoint['epoch']}",
    )
    fixed_samples = [samples_by_position[p] for p in fixed_preview_positions]
    render_validation_sheet(
        fixed_samples,
        eval_dir / "fixed_rows.png",
        title=f"detailer fixed validation rows | checkpoint epoch {best_checkpoint['epoch']}",
    )

    aggregate = {
        key: float(np.mean([float(r[key]) for r in records]))
        for key in ("mae", "gradient_mae", "border_mae", "interior_mae",
                    "tile_mean_baseline_mae", "mae_delta_vs_baseline")
    }
    aggregate["val_rows"] = len(records)
    checkpoint_identity = identity_for_path(args.output / "checkpoint_best.pt")
    stage_run = build_detailer_stage_run(
        run_id=args.run_id,
        architecture=architecture,
        upstream_identity=upstream_identity,
        curriculum=identity_for_path(
            args.store / "index.parquet", display_path=str(args.store.resolve())
        ),
        checkpoint={**checkpoint_identity, "best_epoch": int(best_checkpoint["epoch"])},
        baselines={
            "coarse_only": {"val_mae": coarse_baseline},
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
            "evaluator": aggregate,
            "detailer_gate": gate,
            "sc002": sc002,
            "object_region": (
                object_subset_metrics(obj_abs, obj_px, free_abs, free_px,
                                      weight=args.object_mask_weight)
                if use_object_mask else None
            ),
            "structural_failure_epoch1_best": best_record["epoch"] == 1,
        },
        visual_evidence={
            "fixed_rows": "validation/final_best/fixed_rows.png",
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
        f"coarse_only={coarse_baseline:.6f} gate={gate['passes']} sc002={sc002['passes']} "
        f"promotion=pending(user visual gate)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
