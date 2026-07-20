"""Spec 115: terrain-feature classifier trainer (dry run by default; USER runs the CUDA pass).

The training problem is dominated by one fact measured on the real corpus: road is **0.26%** of
labelled pixels (471,556 of ~179M) while terrain is 94.7%. A model that predicts "terrain"
everywhere scores ~95% pixel accuracy and finds zero roads — which would be worthless for the
geometry deconfounding this feature exists to enable. So:

- the loss is class-weighted (inverse-frequency, computed in-run from the real label store, capped);
- the promotion metric is **road IoU / recall**, not pixel accuracy;
- the in-run baseline is the majority class, making the "predicts terrain everywhere" degenerate
  solution explicit and impossible to mistake for success.

Invalid pixels (``valid == False`` in the label store) are excluded from both loss and metrics; they
are never coerced into a class. Rows the label builder excluded are dropped entirely.
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
from harvester.v50.model_stage_contract import (
    identity_for_path,
    sha256_file,
    validate_model_stage_run,
)
from harvester.v50.terrain_feature_labels import (
    CLASS_COUNT,
    FAMILY_NAMES,
    ROAD,
    TAXONOMY_REVISION,
    UNKNOWN,
)
from harvester.v50.terrain_feature_model import (
    TERRAIN_FEATURE_ARCHITECTURE_ID,
    build_terrain_feature_model,
)

STAGE = "terrain_features"
OUTPUT_SIGNAL = "terrain_feature_map_256"
# Capped inverse frequency. Measured on the v1 run, per-class PRECISION tracks this weight almost
# monotonically: weight 0.21 -> precision 0.998, 4.87 -> 0.986, 22.97 -> 0.603, 50.0 -> 0.125. A cap
# of 50 drove road to ~8x over-prediction (recall 0.808 but precision 0.125). 15 keeps the rare
# class strongly emphasised while staying inside the range where precision had not yet collapsed.
MAX_CLASS_WEIGHT = 15.0


def compute_class_weights(
    label_counts: dict[str, int],
    *,
    max_weight: float = MAX_CLASS_WEIGHT,
    supervise_unknown: bool = False,
) -> list[float]:
    """Capped inverse-frequency weights, in family-ordinal order.

    ``unknown`` is given weight 0 by default and masked out of the loss entirely (see
    ``supervise_unknown``). It is not a terrain class -- it means "no rule matched", and on the real
    corpus it resolves to genuine placeholder/void art (Black.blp, Checkers.blp) at 0.05% of pixels.
    Supervising it at the rare-class weight made the v1 run spend capacity learning a non-class and
    put it in direct competition with road for the same over-predicted pixels (unknown landed at
    precision 0.127 / recall 0.893, the same pathology as road).
    """
    total = sum(label_counts.get(name, 0) for name in FAMILY_NAMES)
    if total <= 0:
        raise TrainerContractError("label store reports zero labelled pixels")
    weights: list[float] = []
    for index, name in enumerate(FAMILY_NAMES):
        if index == UNKNOWN and not supervise_unknown:
            weights.append(0.0)
            continue
        count = label_counts.get(name, 0)
        if count <= 0:
            # An absent class gets neutral weight, never an infinite one.
            weights.append(1.0)
            continue
        weights.append(min(total / (CLASS_COUNT * count), max_weight))
    return weights


def confusion_metrics(confusion: np.ndarray) -> dict:
    """Per-class IoU / precision / recall plus macro summaries from a KxK confusion matrix."""
    per_class: dict[str, dict[str, float]] = {}
    ious: list[float] = []
    for family in range(CLASS_COUNT):
        true_positive = float(confusion[family, family])
        predicted = float(confusion[:, family].sum())
        actual = float(confusion[family, :].sum())
        union = predicted + actual - true_positive
        iou = true_positive / union if union > 0 else 0.0
        per_class[FAMILY_NAMES[family]] = {
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
        "pixel_accuracy": float(np.trace(confusion) / total) if total > 0 else 0.0,
    }


def build_training_plan(
    *,
    architecture_identity: dict,
    source: str,
    index_rows: list[dict],
    selected_rows: list[int],
    class_weights: list[float],
    label_counts: dict[str, int],
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
    return {
        "schema": "v115-terrain-feature-plan-v1",
        "stage": STAGE,
        "architecture": architecture_identity,
        "taxonomy_revision": TAXONOMY_REVISION,
        "families": list(FAMILY_NAMES),
        "source_filter": source,
        "selected_rows": len(selected_rows),
        "split_counts": split_counts,
        "map_counts": dict(sorted(map_counts.items())),
        "label_pixel_counts": label_counts,
        "class_weights": {FAMILY_NAMES[i]: w for i, w in enumerate(class_weights)},
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "learning_rate": lr,
        "train_steps_per_epoch": math.ceil(max(split_counts["train"], 1) / batch_size),
        "deployment_inputs": ["minimap_rgb"],
        "output_signal": OUTPUT_SIGNAL,
        "promotion_metric": "road_iou (NOT pixel accuracy: road is ~0.26% of pixels)",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 115 terrain-feature classifier training (dry run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="curriculum Zarr store")
    ap.add_argument("--labels", required=True, type=Path, help="terrain-feature label store")
    ap.add_argument("--output", required=True, type=Path, help="run output directory")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--release", default="v50.1")
    ap.add_argument("--source", default="authored", help="authored | synthetic | all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=115)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-key", default="split")
    ap.add_argument("--val-value", default="val")
    ap.add_argument("--max-class-weight", type=float, default=MAX_CLASS_WEIGHT)
    ap.add_argument(
        "--supervise-unknown", action="store_true",
        help="train on the 'unknown' class too; default masks it out of loss and metrics because it "
             "is an absence-of-information marker (void/placeholder art), not a terrain class",
    )
    ap.add_argument("--confirm-run", action="store_true", help="required to launch CUDA training")
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

    labels_group = zarr.open_group(str(args.labels), mode="r")
    for required in ("labels", "valid", "included"):
        if required not in labels_group:
            raise SystemExit(f"label store missing {required!r}: {args.labels}")
    recorded_revision = str(dict(labels_group.attrs).get("taxonomy_revision", ""))
    if recorded_revision != TAXONOMY_REVISION:
        raise SystemExit(
            f"label store taxonomy revision {recorded_revision!r} != code {TAXONOMY_REVISION!r}; "
            "rebuild the labels or check out the matching code"
        )

    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
        selected_rows = select_training_rows(index, args.source)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    included = np.asarray(labels_group["included"][:], dtype=bool)
    if included.shape[0] != len(index):
        raise SystemExit("label store row count does not match curriculum index")
    selected_rows = [i for i in selected_rows if included[i]]

    train_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) != args.val_value]
    val_rows = [i for i in selected_rows if str(index[i].get(args.val_key)) == args.val_value]
    if len(train_rows) < 32 or len(val_rows) < 8:
        raise SystemExit(
            f"insufficient labelled rows after --source {args.source}: "
            f"train={len(train_rows)} val={len(val_rows)}"
        )

    report_path = args.labels / "label_build_report.json"
    if not report_path.exists():
        raise SystemExit(f"label store has no label_build_report.json: {args.labels}")
    label_report = json.loads(report_path.read_text(encoding="utf-8"))
    label_counts = label_report["family_pixels"]
    class_weights = compute_class_weights(
        label_counts, max_weight=args.max_class_weight, supervise_unknown=args.supervise_unknown
    )

    model, model_identity = build_terrain_feature_model(base=args.base)
    plan = build_training_plan(
        architecture_identity=model_identity["architecture"],
        source=args.source,
        index_rows=index,
        selected_rows=selected_rows,
        class_weights=class_weights,
        label_counts=label_counts,
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

    supervise_unknown = args.supervise_unknown

    def row_supervision_mask(label: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Pixels that count toward loss and metrics: valid, and (by default) not the unknown class."""
        mask = valid
        if not supervise_unknown:
            mask = mask & (label != UNKNOWN)
        return mask

    class RowDataset(Dataset):
        def __init__(self, rows: list[int]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            rgb = np.asarray(group["minimap_rgb"][row], dtype=np.float32) / 255.0
            label = np.asarray(labels_group["labels"][row], dtype=np.int64)
            valid = np.asarray(labels_group["valid"][row], dtype=bool)
            # Ignore-index folds masked pixels out of the loss without relabelling them: invalid
            # pixels always, and 'unknown' too unless --supervise-unknown.
            label = np.where(row_supervision_mask(label, valid), label, -1)
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(label),
            )

    device = torch.device("cuda")
    model = model.to(device)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        RowDataset(train_rows), batch_size=args.batch, shuffle=True,
        num_workers=args.workers, generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        RowDataset(val_rows), batch_size=args.batch, shuffle=False, num_workers=args.workers
    )

    # Majority-class baseline over the validation split: the degenerate "always terrain" solution.
    # Uses the SAME supervision mask as training/eval so the comparison is apples-to-apples.
    majority_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for row in val_rows:
        label = np.asarray(labels_group["labels"][row], dtype=np.int64)
        valid = np.asarray(labels_group["valid"][row], dtype=bool)
        actual = label[row_supervision_mask(label, valid)]
        counts = np.bincount(actual, minlength=CLASS_COUNT)
        majority_confusion[:, 1] += counts  # predict TERRAIN (ordinal 1) everywhere
    baseline_metrics = confusion_metrics(majority_confusion)

    args.output.mkdir(parents=True, exist_ok=True)
    best_road_iou = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for rgb, label in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(rgb)
                loss = criterion(logits, label)
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
            for rgb, label in val_loader:
                rgb = rgb.to(device, non_blocking=True)
                predicted = model(rgb).argmax(dim=1).cpu().numpy().ravel()
                actual = label.numpy().ravel()
                keep = actual >= 0
                np.add.at(confusion, (actual[keep], predicted[keep]), 1)

        metrics = confusion_metrics(confusion)
        road_iou = metrics["per_class"][FAMILY_NAMES[ROAD]]["iou"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(steps, 1),
                "road_iou": road_iou,
                "macro_iou": metrics["macro_iou"],
                "pixel_accuracy": metrics["pixel_accuracy"],
            }
        )
        print(
            f"epoch {epoch:3d} loss={running / max(steps,1):.4f} "
            f"road_iou={road_iou:.4f} macro_iou={metrics['macro_iou']:.4f} "
            f"acc={metrics['pixel_accuracy']:.4f} best={max(best_road_iou,0):.4f}",
            flush=True,
        )

        if road_iou > best_road_iou:
            best_road_iou = road_iou
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_variant": TERRAIN_FEATURE_ARCHITECTURE_ID,
                    "taxonomy_revision": TAXONOMY_REVISION,
                    "num_classes": CLASS_COUNT,
                    "base": args.base,
                    "epoch": epoch,
                    "road_iou": road_iou,
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
    baseline_road_iou = baseline_metrics["per_class"][FAMILY_NAMES[ROAD]]["iou"]
    run_record = {
        "schema": "v50-model-stage-run-v1",
        "run_id": args.run_id or args.output.name,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": STAGE,
        "output_signal": OUTPUT_SIGNAL,
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
            "majority_class_terrain": {
                "road_iou": baseline_road_iou,
                "macro_iou": baseline_metrics["macro_iou"],
                "pixel_accuracy": baseline_metrics["pixel_accuracy"],
            }
        },
        "metrics": {
            "best_epoch": int(best["epoch"]),
            "best_road_iou": float(best_road_iou),
            "beats_majority_baseline": bool(best_road_iou > baseline_road_iou),
            "evaluator": best["metrics"],
            "history": history,
            "label_store": {
                "path": str(args.labels.resolve()),
                "taxonomy_revision": TAXONOMY_REVISION,
            },
        },
        "visual_evidence": {"note": "run v50_infer_terrain_features.py for OOD review sheets"},
        "promotion_verdict": "pending",
    }
    validate_model_stage_run(run_record)
    (args.output / "model_stage_run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps(run_record["metrics"], indent=2), flush=True)
    print(f"best epoch {best_epoch} road_iou={best_road_iou:.4f} "
          f"(majority baseline {baseline_road_iou:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
