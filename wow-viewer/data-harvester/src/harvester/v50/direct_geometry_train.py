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

from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.direct_geometry_model import (
    ARCHITECTURE_IDS,
    MIT_B0_HUB_ID,
    MIT_B0_LICENSE,
    build_geometry_model,
    load_pretrained_encoder,
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
from harvester.v50.model_stage_contract import (
    ContractViolationError,
    identity_for_path,
    validate_model_stage_run,
)

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
    clip: float,
) -> dict:
    """Machine-readable no-training preview printed before CUDA allocation."""
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    if lr_schedule not in LR_SCHEDULES:
        raise TrainerContractError(f"lr schedule must be one of {sorted(LR_SCHEDULES)}")
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
        "grad_clip": clip,
        "train_steps_per_epoch": math.ceil(split_counts["train"] / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "training_target": "height_257 -> relative_height_257",
        "wdl_prior": False,
        "sc001_references": {
            "spec112_frozen_best_val_mae": SPEC112_FROZEN_BEST_VAL_MAE,
            "relative_margin": SC001_RELATIVE_MARGIN,
        },
    }


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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=sorted(LR_SCHEDULES))
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip; 0 disables (bootstrap parity)")
    ap.add_argument("--workers", type=int, choices=[0], default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=114)
    ap.add_argument("--release", default="v50.1", type=validate_release)
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

    train_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) != args.val_value]
    val_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) == args.val_value]
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(
            f"insufficient rows after --source {args.source}: train={len(train_rows)} val={len(val_rows)}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        model, model_identity = build_geometry_model(
            args.architecture, pretrained_source=pretrained_record
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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
        clip=args.clip,
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

    class RowDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            target, _, _ = encode_relative_height(np.asarray(group["height_257"][row]))
            return torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(target)

    val_targets = [encode_relative_height(np.asarray(group["height_257"][r]))[0] for r in val_rows]
    tile_mean_baseline = compute_tile_mean_baseline(val_targets)
    flat_baseline = compute_flat_baseline(val_targets)

    device = torch.device("cuda")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, generator=train_generator,
    )
    val_loader = DataLoader(
        RowDataset(val_rows), batch_size=args.batch, num_workers=args.workers, pin_memory=True
    )
    scheduler = None
    if args.lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
        )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
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
        "loss": {"point": "smooth_l1", "gradient_l1_weight": 0.25},
        "schedule": {
            "max_epochs": args.epochs, "batch_size": args.batch, "patience": args.patience,
            "workers": args.workers, "seed": args.seed, "lr_schedule": args.lr_schedule,
            "amp": args.amp, "grad_clip": args.clip,
        },
        "pretrained_source": model_identity["pretrained_source"],
    }
    (args.output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    per_epoch: list[dict] = []
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss = height_loss(model(x.to(device)), y.to(device))
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
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp):
            for x, y in val_loader:
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
            )
        else:
            stale += 1
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_mae={val_mae:.6f} "
            f"tile_mean={tile_mean_baseline:.6f} flat={flat_baseline:.6f} "
            f"best={best:.6f} stale={stale}/{args.patience}",
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
