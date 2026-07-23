"""Spec 118 US3: from-scratch visible-object segmenter trainer (USER runs CUDA).

Predicts per-pixel visible-object class (none/doodad/building) from minimap RGB alone, supervised
by the strict US1 signal ``object_geometry_visible_source_257``. Deliberately mirrors the Spec 117
lattice trainer's shape: the smallest possible trainer, reusing (never reimplementing) the
already-validated curriculum/source/held-out-split machinery from
``height_relative_train``/``direct_geometry_train``.

``--held-out-split`` is REQUIRED with no ``--val-key``/``--val-value`` fallback (same contract as
Spec 117 FR-004: refuse a leaky or unspecified split, don't default away from one). Tiles with an
all-zero (object-free) mask are VALID NEGATIVES, not exclusions (spec Edge Cases: synthetic
terrain-only rows contain no objects by construction).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from harvester.spec118.object_contract import (
    CLASS_NAMES,
    STAGE,
    architecture_identity,
    build_object_stage_run,
    identity_for_path,
)
from harvester.spec118.object_segment_model import (
    ObjectSegmentNet,
    compute_class_weights,
    derive_class_target,
    per_class_iou_recall,
    visible_object_iou,
)
from harvester.v50.contracts import release_identity, require_store_release, validate_release
from harvester.v50.direct_geometry_train import apply_held_out_split
from harvester.v50.height_relative_train import (
    SOURCE_CHOICES,
    TrainerContractError,
    curriculum_identity,
    require_new_output,
    select_training_rows,
    validate_curriculum_contract,
    validate_source_selection,
)
from harvester.v50.lr_schedule import (
    make_onecycle_scheduler,
    warmup_complete,
)

SOURCE_ARRAY = "object_mask"
REQUIRED_ARRAYS = ("minimap_rgb", SOURCE_ARRAY)
LR_SCHEDULES = frozenset({"constant", "onecycle"})

# Research D-07: the first-cut gate thresholds (a gate against non-learning, not a ceiling).
GATE_MEDIAN_VISIBLE_IOU = 0.40
GATE_PER_CLASS_RECALL = 0.50


def require_object_arrays(group) -> None:
    """Fail closed when the store predates the Spec 118 catalog amendment (US1)."""
    missing = [name for name in REQUIRED_ARRAYS if name not in group]
    if missing:
        raise TrainerContractError(
            f"store is missing object-signal arrays {missing}; rebuild the store + curriculum after "
            "the object-mask catalog fix (docs/architecture/v50-clean-room-dataset-repo-audit-"
            "2026-07-15.md) so the alpha-painted object_mask is carried before training the segmenter"
        )


def build_object_plan(
    *,
    architecture: dict,
    source: str,
    train_rows: int,
    val_rows: int,
    object_touched_train: int,
    object_touched_val: int,
    batch_size: int,
    epochs: int,
    seed: int,
    lr: float,
    lr_schedule: str,
) -> dict:
    """Machine-readable no-training preview printed before CUDA allocation."""
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    if lr_schedule not in LR_SCHEDULES:
        raise TrainerContractError(f"lr schedule must be one of {sorted(LR_SCHEDULES)}")
    return {
        "schema": "v118-object-plan-v1",
        "stage": STAGE,
        "architecture": architecture,
        "source_filter": source,
        "split_counts": {"train": train_rows, "val": val_rows},
        "object_touched_tiles": {"train": object_touched_train, "val": object_touched_val},
        "class_names": list(CLASS_NAMES),
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "optimizer": {"name": "AdamW", "learning_rate": lr, "weight_decay": 1e-4},
        "lr_schedule": lr_schedule,
        "train_steps_per_epoch": math.ceil(max(train_rows, 1) / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "training_target": "object_mask (v18 placement footprint) -> object_class_2 (none/object)",
        "gate_thresholds": {
            "median_visible_object_iou": GATE_MEDIAN_VISIBLE_IOU,
            "per_class_recall": GATE_PER_CLASS_RECALL,
        },
        "no_gan_no_adversarial_no_generative_image": True,
    }


def main() -> int:
    import pyarrow.parquet as pq
    import torch
    import zarr
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser(description="Spec 118 visible-object segmenter trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="v50 curriculum store carrying object_mask")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--run-id", required=True, help="immutable run identity, e.g. objects-authored-v1")
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    ap.add_argument(
        "--held-out-split", required=True, type=Path,
        help="REQUIRED: a v50-held-out-split-v1 directory (spec116_build_held_out_split.py or "
             "equivalent). Refuses a leaky split (verified_violation_count != 0). No --val-key/"
             "--val-value fallback exists for this trainer.",
    )
    ap.add_argument("--confirm-run", action="store_true",
                     help="launch CUDA training; without this flag only print the validated plan")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=sorted(LR_SCHEDULES))
    ap.add_argument("--base", type=int, default=24,
                    help="ObjectSegmentNet width (from-scratch U-Net-lite; single-digit-hundred-K "
                         "params, SC-005)")
    ap.add_argument("--patience", type=int, default=30,
                    help="post-warmup epochs with no val visible-object-IoU improvement before "
                         "early stop. 0 = never stop (run the full --epochs).")
    ap.add_argument("--pct-start", type=float, default=0.1,
                    help="OneCycleLR warmup fraction; the early-stopper is warmup-aware (Spec 117 "
                         "scheduling fix) and never counts stale epochs during warmup.")
    ap.add_argument("--seed", type=int, default=118)
    ap.add_argument("--release", default="v50.1", type=validate_release)
    ap.add_argument("--workers", type=int, choices=[0], default=0)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index)
        validate_source_selection(attrs=dict(group.attrs), source=args.source)
        require_object_arrays(group)
        selected_rows = select_training_rows(index, args.source)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        train_rows, val_rows, split_manifest = apply_held_out_split(
            index_rows=index, selected_rows=selected_rows, split_dir=args.held_out_split,
        )
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(f"insufficient rows: train={len(train_rows)} val={len(val_rows)}")

    train_targets = [derive_class_target(np.asarray(group[SOURCE_ARRAY][row])) for row in train_rows]
    val_targets = [derive_class_target(np.asarray(group[SOURCE_ARRAY][row])) for row in val_rows]
    touched_train = int(sum(bool((t > 0).any()) for t in train_targets))
    touched_val = int(sum(bool((t > 0).any()) for t in val_targets))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ObjectSegmentNet(base=args.base)
    identity = architecture_identity(
        model, architecture_id="object_segment_net",
        config={"class": "ObjectSegmentNet", "arch": "object_segment_net_v1",
                "base": args.base, "input": "3x256x256", "output": "3x256x256"},
    )
    plan = build_object_plan(
        architecture=identity, source=args.source,
        train_rows=len(train_rows), val_rows=len(val_rows),
        object_touched_train=touched_train, object_touched_val=touched_val,
        batch_size=args.batch, epochs=args.epochs, seed=args.seed,
        lr=args.lr, lr_schedule=args.lr_schedule,
    )
    plan["held_out_split"] = {
        "path": str((args.held_out_split / "split.json").resolve()),
        "verified_violation_count": int(split_manifest["verified_violation_count"]),
    }
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

    class RowDataset(Dataset):
        def __init__(self, rows: list[int], targets: list[np.ndarray]) -> None:
            self.rows = rows
            self.targets = targets

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(self.targets[i]),
            )

    class_weights = compute_class_weights(train_targets)
    device = torch.device("cuda")
    model = model.to(device)
    class_weights = class_weights.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows, train_targets), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, generator=train_generator,
    )
    scheduler = None
    warmup_epochs = 0
    if args.lr_schedule == "onecycle":
        scheduler, warmup_epochs = make_onecycle_scheduler(
            opt, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=len(train_loader), pct_start=args.pct_start,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    identity_doc = curriculum_identity(args.store)
    run_identity = {
        **release_identity(args.release),
        "model_variant": "object_segment_net",
        "parameter_count": identity["parameter_count"],
        "source_filter": args.source,
        "store": str(args.store.resolve()),
        "optimizer": plan["optimizer"],
        "loss": {"point": "class_weighted_cross_entropy", "class_weights": class_weights.tolist()},
        "schedule": {
            "max_epochs": args.epochs, "batch_size": args.batch,
            "patience": args.patience, "workers": args.workers, "seed": args.seed,
            "lr_schedule": args.lr_schedule,
            "pct_start": args.pct_start if args.lr_schedule == "onecycle" else None,
            "warmup_epochs": warmup_epochs,
        },
        "held_out_split": plan["held_out_split"],
    }
    (args.output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")

    def _val_metrics() -> dict:
        model.eval()
        class_totals: dict[str, dict[str, float]] = {
            name: {"intersection": 0.0, "union": 0.0, "support": 0.0, "hits": 0.0}
            for name in CLASS_NAMES
        }
        visible_ious: list[float] = []
        with torch.no_grad():
            for row, target in zip(val_rows, val_targets, strict=True):
                rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
                x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                predicted = model(x).squeeze(0).argmax(0).cpu().numpy()
                per_class = per_class_iou_recall(predicted, target)
                for class_id, name in enumerate(CLASS_NAMES):
                    p = predicted == class_id
                    t = target == class_id
                    class_totals[name]["intersection"] += float((p & t).sum())
                    class_totals[name]["union"] += float((p | t).sum())
                    class_totals[name]["support"] += float(t.sum())
                    class_totals[name]["hits"] += float((p & t).sum())
                if (target > 0).any():
                    iou = visible_object_iou(predicted, target)
                    if iou is not None:
                        visible_ious.append(iou)
        per_class = {
            name: {
                "iou": (totals["intersection"] / totals["union"]) if totals["union"] else None,
                "recall": (totals["hits"] / totals["support"]) if totals["support"] else None,
            }
            for name, totals in class_totals.items()
        }
        return {
            "per_class": per_class,
            "median_visible_object_iou": float(np.median(visible_ious)) if visible_ious else 0.0,
            "object_touched_val_tiles": len(visible_ious),
        }

    per_epoch: list[dict] = []
    best = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits, y.to(device), weight=class_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().item()))
        metrics = _val_metrics()
        selection = metrics["median_visible_object_iou"]
        train_loss = float(np.mean(train_losses))
        per_epoch.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        checkpoint = {**run_identity, "model": model.state_dict(), "epoch": epoch,
                      "object_config": {"base": args.base},
                      "val_metrics": metrics, "curriculum_identity": identity_doc}
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if selection > best:
            best = selection
            stale = 0
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
        elif warmup_complete(epoch, warmup_epochs):
            stale += 1
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"visible_iou={selection:.4f} best={best:.4f} stale={stale}/{args.patience}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"[early-stop] no improvement for {stale} epochs", flush=True)
            break

    best_record = max(per_epoch, key=lambda e: e["median_visible_object_iou"])
    final_metrics = _val_metrics()
    gate_passes = (
        final_metrics["median_visible_object_iou"] >= GATE_MEDIAN_VISIBLE_IOU
        and all(
            (final_metrics["per_class"][name]["recall"] or 0.0) >= GATE_PER_CLASS_RECALL
            for name in ("object",)
        )
    )
    checkpoint_identity = identity_for_path(args.output / "checkpoint_best.pt")
    stage_run = build_object_stage_run(
        run_id=args.run_id,
        architecture=identity,
        curriculum=identity_for_path(
            args.store / "index.parquet", display_path=str(args.store.resolve())
        ),
        checkpoint={**checkpoint_identity, "best_epoch": int(best_record["epoch"])},
        baselines={"majority_class": {"median_visible_object_iou": 0.0}},
        metrics={
            "best_epoch": best_record["epoch"],
            "final": final_metrics,
            "gate": {
                "thresholds": plan["gate_thresholds"],
                "passes": bool(gate_passes),
                "note": "research D-07 first-cut gate against non-learning; promotion stays pending",
            },
        },
        visual_evidence={},
    )
    (args.output / "model_stage_run.json").write_text(
        json.dumps(stage_run, indent=2), encoding="utf-8"
    )
    (args.output / "training_summary.json").write_text(
        json.dumps({"per_epoch_metrics": per_epoch, "model_stage_run": stage_run["run_id"]}, indent=2),
        encoding="utf-8",
    )
    print(
        f"best_epoch={best_record['epoch']} "
        f"visible_iou={final_metrics['median_visible_object_iou']:.4f} "
        f"gate={gate_passes} promotion=pending(user gate)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
