"""Spec 115 follow-on: per-cell liquid classifier trainer (dry run by default; USER runs CUDA).

Class balance on the real corpus is none 56.5% / river 2.9% / ocean 40.7% — far better conditioned
than the Spec 115 road target (0.26%), but river is still the rare class and is the whole point of
the model: "is there water" is already covered by the existing per-pixel ``water`` class, whereas
river-vs-ocean is a genuine type judgement.

So the promotion metric is **river IoU/recall**, never overall accuracy: predicting none+ocean
everywhere scores ~97% accuracy while never identifying a single river. The majority-class baseline
is computed in-run to make that failure mode explicit.

Classes with zero support in the corpus (magma and slime here) keep their contract ordinals but get
zero loss weight — never invented, and their absence is reported rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.v50.contracts import require_store_release
from harvester.v50.height_relative_train import (
    TrainerContractError,
    require_new_output,
    select_training_rows,
    validate_curriculum_contract,
)
from harvester.v50.liquid_cell_labels import (
    CHUNKS_PER_AXIS,
    CLASS_COUNT,
    CLASS_NAMES,
    RIVER,
    TAXONOMY_REVISION,
    labels_from_liquid_type_grid,
    labels_from_mcnk_flags,
)
from harvester.v50.liquid_cell_model import (
    LIQUID_CELL_ARCHITECTURE_ID,
    build_liquid_cell_model,
)
from harvester.v50.model_stage_contract import (
    identity_for_path,
    sha256_file,
    validate_model_stage_run,
)

STAGE = "terrain_features"  # nearest valid stage in the published contract enum
# Grid-suffixed at use so a 16-grid and a 128-grid run never claim the same output signal.
OUTPUT_SIGNAL_PREFIX = "liquid_cell_map"
MAX_CLASS_WEIGHT = 15.0


def compute_class_weights(
    class_counts: dict[str, int], *, max_weight: float = MAX_CLASS_WEIGHT
) -> list[float]:
    """Capped inverse-frequency weights in class-ordinal order; absent classes get weight 0."""
    total = sum(class_counts.get(name, 0) for name in CLASS_NAMES)
    if total <= 0:
        raise TrainerContractError("label set reports zero labelled cells")
    present = sum(1 for name in CLASS_NAMES if class_counts.get(name, 0) > 0)
    weights: list[float] = []
    for name in CLASS_NAMES:
        count = class_counts.get(name, 0)
        if count <= 0:
            # Never supervise a class the corpus does not contain.
            weights.append(0.0)
            continue
        weights.append(min(total / (present * count), max_weight))
    return weights


def confusion_metrics(confusion: np.ndarray) -> dict:
    """Per-class IoU/precision/recall plus macro summaries, ignoring zero-support classes."""
    per_class: dict[str, dict[str, float]] = {}
    ious: list[float] = []
    for ordinal in range(CLASS_COUNT):
        true_positive = float(confusion[ordinal, ordinal])
        predicted = float(confusion[:, ordinal].sum())
        actual = float(confusion[ordinal, :].sum())
        union = predicted + actual - true_positive
        iou = true_positive / union if union > 0 else 0.0
        per_class[CLASS_NAMES[ordinal]] = {
            "iou": iou,
            "precision": true_positive / predicted if predicted > 0 else 0.0,
            "recall": true_positive / actual if actual > 0 else 0.0,
            "support": actual,
        }
        if actual > 0:
            ious.append(iou)
    total = float(confusion.sum())
    return {
        "per_class": per_class,
        "macro_iou": float(np.mean(ious)) if ious else 0.0,
        "cell_accuracy": float(np.trace(confusion) / total) if total > 0 else 0.0,
    }


def build_training_plan(
    *,
    architecture_identity: dict,
    source: str,
    index_rows: list[dict],
    selected_rows: list[int],
    class_weights: list[float],
    class_counts: dict[str, int],
    cell_grid: int,
    batch_size: int,
    epochs: int,
    seed: int,
    lr: float,
) -> dict:
    if batch_size < 1 or epochs < 1:
        raise TrainerContractError("batch size and epochs must both be positive")
    selected = [index_rows[i] for i in selected_rows]
    split_counts = {
        split: sum(str(row.get("split")) == split for row in selected) for split in ("train", "val")
    }
    map_counts: dict[str, int] = {}
    for row in selected:
        name = str(row.get("map", "unknown"))
        map_counts[name] = map_counts.get(name, 0) + 1
    total_cells = sum(class_counts.values())
    return {
        "schema": "v115-liquid-cell-plan-v1",
        "stage": STAGE,
        "architecture": architecture_identity,
        "taxonomy_revision": TAXONOMY_REVISION,
        "classes": list(CLASS_NAMES),
        "source_filter": source,
        "selected_rows": len(selected_rows),
        "split_counts": split_counts,
        "map_counts": dict(sorted(map_counts.items())),
        "cell_grid": [cell_grid, cell_grid],
        "label_cell_counts": class_counts,
        "label_cell_fraction": {
            name: (count / total_cells if total_cells else 0.0)
            for name, count in class_counts.items()
        },
        "class_weights": {CLASS_NAMES[i]: w for i, w in enumerate(class_weights)},
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "learning_rate": lr,
        "train_steps_per_epoch": math.ceil(max(split_counts["train"], 1) / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "output_signal": f"{OUTPUT_SIGNAL_PREFIX}_{cell_grid}",
        "promotion_metric": "river_iou (NOT cell accuracy: none+ocean everywhere scores ~97%)",
    }


def row_labels(group, row: int, cell_grid: int) -> np.ndarray:
    """The single label-resolution path, shared by counting, training, and the baseline.

    At the 16x16 chunk grid the authoritative source is ``mcnk_flags_16`` (the format's own
    per-chunk liquid bits). Finer grids cannot be expressed by chunk flags at all, so they come from
    per-pixel ``liquid_type_256`` — which agrees with the flags to ~99.7% at 16x16, so the two are
    corroborated rather than arbitrary alternatives.
    """
    if cell_grid == CHUNKS_PER_AXIS and "mcnk_flags_16" in group:
        return labels_from_mcnk_flags(np.asarray(group["mcnk_flags_16"][row]))
    return labels_from_liquid_type_grid(np.asarray(group["liquid_type_256"][row]), cell_grid)


def collect_label_counts(group, rows: list[int], cell_grid: int) -> dict[str, int]:
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for row in rows:
        labels = row_labels(group, row, cell_grid)
        for ordinal in range(CLASS_COUNT):
            counts[CLASS_NAMES[ordinal]] += int(np.count_nonzero(labels == ordinal))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 115 per-cell liquid classifier training (dry run by default)"
    )
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--release", default="v50.1")
    ap.add_argument("--source", default="authored", help="authored | synthetic | all")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=115)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-key", default="split")
    ap.add_argument("--val-value", default="val")
    ap.add_argument("--max-class-weight", type=float, default=MAX_CLASS_WEIGHT)
    ap.add_argument("--cell-grid", type=int, default=CHUNKS_PER_AXIS,
                    help="Prediction grid per tile. 16 = MCNK chunk grid (mcnk_flags_16). "
                         "128 = the real quad grid (129 outer vertices per axis -> 128 quads), "
                         "labelled from per-pixel liquid_type_256 since chunk flags cannot "
                         "express sub-chunk water. Hashed into the architecture identity.")
    ap.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    import torch
    import zarr
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for required in ("mcnk_flags_16", "minimap_rgb"):
        if required not in group:
            raise SystemExit(f"store is missing {required!r}: {args.store}")

    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
        selected_rows = select_training_rows(index, args.source)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    train_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) != args.val_value]
    val_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) == args.val_value]
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(
            f"insufficient rows after --source {args.source}: "
            f"train={len(train_rows)} val={len(val_rows)}"
        )

    class_counts = collect_label_counts(group, train_rows, args.cell_grid)
    class_weights = compute_class_weights(class_counts, max_weight=args.max_class_weight)
    model, model_identity = build_liquid_cell_model(base=args.base, cell_grid=args.cell_grid)
    plan = build_training_plan(
        architecture_identity=model_identity["architecture"],
        source=args.source,
        index_rows=index,
        selected_rows=selected_rows,
        class_weights=class_weights,
        class_counts=class_counts,
        cell_grid=args.cell_grid,
        batch_size=args.batch,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class RowDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            labels = row_labels(group, row, args.cell_grid)
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(labels.astype(np.int64)),
            )

    device = torch.device("cuda")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, generator=generator,
    )
    val_loader = DataLoader(
        RowDataset(val_rows), batch_size=args.batch, shuffle=False, num_workers=args.workers
    )

    # Majority-class baseline: predict the most common class everywhere. This is the degenerate
    # solution that scores high cell accuracy while finding zero rivers.
    val_counts = collect_label_counts(group, val_rows, args.cell_grid)
    majority = int(np.argmax([val_counts[name] for name in CLASS_NAMES]))
    majority_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for ordinal, name in enumerate(CLASS_NAMES):
        majority_confusion[ordinal, majority] = val_counts[name]
    baseline_metrics = confusion_metrics(majority_confusion)

    args.output.mkdir(parents=True, exist_ok=True)
    best_river_iou = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for rgb, labels in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss = criterion(model(rgb), labels)
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            steps += 1

        model.eval()
        confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        with torch.no_grad():
            for rgb, labels in val_loader:
                predicted = model(rgb.to(device)).argmax(dim=1).cpu().numpy().ravel()
                np.add.at(confusion, (labels.numpy().ravel(), predicted), 1)

        metrics = confusion_metrics(confusion)
        river_iou = metrics["per_class"][CLASS_NAMES[RIVER]]["iou"]
        river_recall = metrics["per_class"][CLASS_NAMES[RIVER]]["recall"]
        history.append({
            "epoch": epoch,
            "train_loss": running / max(steps, 1),
            "river_iou": river_iou,
            "river_recall": river_recall,
            "macro_iou": metrics["macro_iou"],
            "cell_accuracy": metrics["cell_accuracy"],
        })
        print(
            f"epoch {epoch:3d} loss={running / max(steps,1):.4f} river_iou={river_iou:.4f} "
            f"river_recall={river_recall:.4f} macro_iou={metrics['macro_iou']:.4f} "
            f"acc={metrics['cell_accuracy']:.4f} best={max(best_river_iou,0):.4f}",
            flush=True,
        )

        if river_iou > best_river_iou:
            best_river_iou = river_iou
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_variant": LIQUID_CELL_ARCHITECTURE_ID,
                    "taxonomy_revision": TAXONOMY_REVISION,
                    "num_classes": CLASS_COUNT,
                    "base": args.base,
                    "epoch": epoch,
                    "river_iou": river_iou,
                    "metrics": metrics,
                },
                args.output / "checkpoint_best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch} (patience {args.patience})", flush=True)
                break

    checkpoint_path = args.output / "checkpoint_best.pt"
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    baseline_river_iou = baseline_metrics["per_class"][CLASS_NAMES[RIVER]]["iou"]
    run_record = {
        "schema": "v50-model-stage-run-v1",
        "run_id": args.run_id or args.output.name,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": STAGE,
        "output_signal": f"{OUTPUT_SIGNAL_PREFIX}_{args.cell_grid}",
        "architecture": model_identity["architecture"],
        "curriculum": identity_for_path(
            args.store / "index.parquet", display_path=str(args.store.resolve())
        ),
        "upstream_models": [],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "best_epoch": int(best["epoch"]),
        },
        "baselines": {
            "majority_class": {
                "class": CLASS_NAMES[majority],
                "river_iou": baseline_river_iou,
                "macro_iou": baseline_metrics["macro_iou"],
                "cell_accuracy": baseline_metrics["cell_accuracy"],
            }
        },
        "metrics": {
            "best_epoch": int(best["epoch"]),
            "best_river_iou": float(best_river_iou),
            "beats_majority_baseline": bool(best_river_iou > baseline_river_iou),
            "evaluator": best["metrics"],
            "history": history,
            "label_cell_counts": class_counts,
        },
        "visual_evidence": {"note": "per-cell liquid map; see history for river IoU/recall"},
        "promotion_verdict": "pending",
    }
    validate_model_stage_run(run_record)
    (args.output / "model_stage_run.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print(f"best epoch {best_epoch} river_iou={best_river_iou:.4f} "
          f"(majority baseline {baseline_river_iou:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
