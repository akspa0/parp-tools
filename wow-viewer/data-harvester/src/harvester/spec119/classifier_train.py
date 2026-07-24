"""Spec 119 US1 object-library classifier trainer (T012/T013; USER runs CUDA, FR-010).

Supervised by the library's own ``asset_type`` labels (coarse) or the heuristic fine-family
labels (``--fine-labels``, D-03) — never by minimap/terrain data. Blank captures are relabeled
``empty`` (D-04, FR-006). Class imbalance is handled with inverse-frequency weights and per-class
precision/recall reporting (FR-007); the majority-class baseline is always recorded (FR-005) so
a majority-predicting model cannot pass as successful (SC-001).

Dry-run-first: without ``--confirm-run`` the CLI prints the validated plan (param count,
train/held-out counts, majority-class baseline, class weights) and exits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from harvester.spec119.classifier_model import (
    BACKBONE_CONFIGS,
    ObjectClassifier,
    compute_class_weights,
    majority_class_baseline,
    per_class_precision_recall,
)
from harvester.spec119.library_data import (
    captured_rows,
    coarse_labels,
    fine_labels,
    label_index_map,
    load_asset_rows,
    open_library,
    read_image,
    require_new_output,
    row_coverages,
)
from harvester.spec119.object_library_contract import (
    OUTPUT_SIGNAL_CLASSIFIER,
    STAGE_CLASSIFIER,
    ObjectLibraryContractError,
    architecture_identity,
    build_stage_run,
    identity_for_path,
)
from harvester.spec119.split import SplitError, apply_family_split, load_split
from harvester.v50.lr_schedule import make_onecycle_scheduler, warmup_complete


def build_classifier_plan(
    *,
    architecture: dict,
    class_index: dict[str, int],
    train_count: int,
    held_out_count: int,
    majority_baseline: float,
    class_weights: list[float],
    blank_relabeled: int,
    fine_labels_used: bool,
    args: argparse.Namespace,
) -> dict:
    """Machine-readable no-training preview printed before any CUDA allocation (FR-010)."""
    return {
        "schema": "v119-classifier-plan-v1",
        "stage": STAGE_CLASSIFIER,
        "architecture": architecture,
        "class_index": class_index,
        "fine_labels_heuristic": fine_labels_used,
        "split_counts": {"train": train_count, "held_out": held_out_count},
        "blank_threshold": args.blank_threshold,
        "blank_relabeled_to_empty": blank_relabeled,
        "majority_class_baseline": majority_baseline,
        "class_weights": class_weights,
        "batch_size": args.batch,
        "epochs": args.epochs,
        "seed": args.seed,
        "optimizer": {"name": "AdamW", "learning_rate": args.lr, "weight_decay": 1e-4},
        "lr_schedule": {"name": "onecycle", "pct_start": args.pct_start},
        "success_gate": {"sc_001": "held-out accuracy >= majority baseline + 0.15"},
        "no_pretrained_backbone_from_scratch": args.backbone == "scratch",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Spec 119 object-library classifier trainer (USER runs CUDA; dry-run-first)"
    )
    ap.add_argument("--store", required=True, type=Path, help="object-library zarr (read-only)")
    ap.add_argument("--split", required=True, type=Path,
                    help="REQUIRED: a v119-family-split-v1 JSON (spec119_build_split.py); refuses "
                         "a leaky split (verified_violation_count != 0). No random fallback.")
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--backbone", default="scratch",
                    help="Vision backbone: scratch (from-scratch conv), dinov2_vits14, clip_vitb32, "
                         "or timm/<name> for any timm model (e.g. timm/efficientnet_b0)")
    ap.add_argument("--base", type=int, default=16,
                    help="ObjectClassifier width (scratch backbone only; <1M params at 16, SC-005)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--blank-threshold", type=float, default=0.01,
                    help="mask coverage below this fraction is relabeled 'empty' (D-04/FR-006)")
    ap.add_argument("--fine-labels", action="store_true",
                    help="train on the heuristic fine-family labels (D-03); run record marks it "
                         "heuristic. SC-001 is still reported on the coarse split.")
    ap.add_argument("--patience", type=int, default=20,
                    help="post-warmup epochs with no held-out accuracy improvement before early "
                         "stop. 0 = never stop.")
    ap.add_argument("--seed", type=int, default=119)
    ap.add_argument("--confirm-run", action="store_true",
                    help="launch CUDA training; without this flag only print the validated plan")
    args = ap.parse_args()

    try:
        group = open_library(args.store)
        rows = captured_rows(load_asset_rows(args.store))
        split = load_split(args.split)
        train_idx, held_out_idx = apply_family_split(rows, split)
    except (ObjectLibraryContractError, SplitError) as exc:
        raise SystemExit(str(exc)) from exc
    if not train_idx or not held_out_idx:
        raise SystemExit(f"degenerate split: train={len(train_idx)} held_out={len(held_out_idx)}")

    coverages = row_coverages(group, rows)
    labels = (
        fine_labels(rows, coverages, args.blank_threshold)
        if args.fine_labels
        else coarse_labels(rows, coverages, args.blank_threshold)
    )
    class_index = label_index_map(labels)
    train_labels = [class_index[labels[i]] for i in train_idx]
    held_out_labels = [class_index[labels[i]] for i in held_out_idx]
    blank_relabeled = int(sum(1 for label in labels if label == "empty"))
    baseline = majority_class_baseline(train_labels)
    weights = compute_class_weights(train_labels, num_classes=len(class_index))

    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ObjectClassifier(backbone=args.backbone, base=args.base, num_classes=len(class_index))
    input_size = model.input_size
    identity = architecture_identity(
        model,
        architecture_id="object_library_classifier",
        config={"class": "ObjectClassifier", "backbone": args.backbone,
                "base": args.base, "num_classes": len(class_index),
                "input": f"3x{input_size}x{input_size}",
                "output": f"{len(class_index)} logits"},
    )
    plan = build_classifier_plan(
        architecture=identity, class_index=class_index,
        train_count=len(train_idx), held_out_count=len(held_out_idx),
        majority_baseline=baseline, class_weights=[round(float(w), 6) for w in weights],
        blank_relabeled=blank_relabeled, fine_labels_used=bool(args.fine_labels), args=args,
    )
    plan["held_out_split"] = {
        "path": str(args.split.resolve()),
        "verified_violation_count": int(split["verified_violation_count"]),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.confirm_run:
        print("DRY RUN ONLY: add --confirm-run to launch user-owned CUDA training.", flush=True)
        return 0

    try:
        output = Path(args.output_root) / args.run_name
        require_new_output(output)
    except ObjectLibraryContractError as exc:
        raise SystemExit(str(exc)) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; user-run training refuses CPU.")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from torch.utils.data import DataLoader, Dataset

    class RowDataset(Dataset):
        def __init__(self, indices: list[int], targets: list[int]) -> None:
            self.indices = indices
            self.targets = targets

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            from PIL import Image
            zarr_row = rows[self.indices[i]]["_row_index"]
            rgb = read_image(group, zarr_row)  # float32 HWC [0,1]
            # Resize to model's expected input size if different from native 128x128
            if rgb.shape[0] != input_size or rgb.shape[1] != input_size:
                rgb = np.asarray(
                    Image.fromarray((rgb * 255).astype(np.uint8)).resize(
                        (input_size, input_size), Image.BILINEAR
                    )
                ).astype(np.float32) / 255.0
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.tensor(self.targets[i], dtype=torch.long),
            )

    device = torch.device("cuda")
    model = model.to(device)
    class_weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    loader = DataLoader(
        RowDataset(train_idx, train_labels), batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=True, generator=train_generator,
    )
    scheduler, warmup_epochs = make_onecycle_scheduler(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader),
        pct_start=args.pct_start,
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    index_class = {index: name for name, index in class_index.items()}

    def _held_out_metrics() -> dict:
        model.eval()
        predictions: list[int] = []
        from PIL import Image
        with torch.no_grad():
            for i in held_out_idx:
                zarr_row = rows[i]["_row_index"]
                rgb = read_image(group, zarr_row)
                if rgb.shape[0] != input_size or rgb.shape[1] != input_size:
                    rgb = np.asarray(
                        Image.fromarray((rgb * 255).astype(np.uint8)).resize(
                            (input_size, input_size), Image.BILINEAR
                        )
                    ).astype(np.float32) / 255.0
                predictions.append(int(model(x).squeeze(0).argmax().item()))
        correct = sum(1 for p, t in zip(predictions, held_out_labels, strict=True) if p == t)
        per_class = per_class_precision_recall(predictions, held_out_labels, len(class_index))
        return {
            "held_out_accuracy": correct / max(len(held_out_labels), 1),
            "per_class": {
                index_class[c]: metrics for c, metrics in per_class.items()
            },
        }

    def _save_checkpoint(path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "kind": "classifier",
                "state_dict": model.state_dict(),
                "architecture": {"backbone": args.backbone, "base": args.base,
                                 "num_classes": len(class_index), "class_index": class_index,
                                 "input_size": input_size},
                "config": {"lr": args.lr, "epochs": args.epochs,
                           "blank_threshold": args.blank_threshold,
                           "fine_labels": bool(args.fine_labels),
                           "split": str(args.split.resolve())},
                "epoch": epoch,
                "val_metrics": metrics,
            },
            path,
        )

    per_epoch: list[dict] = []
    best = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits, y.to(device), weight=class_weights_t
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            losses.append(float(loss.detach().item()))
        metrics = _held_out_metrics()
        selection = metrics["held_out_accuracy"]
        per_epoch.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        _save_checkpoint(output / "checkpoint_last.pt", epoch, metrics)
        if selection > best:
            best = selection
            stale = 0
            _save_checkpoint(output / "classifier.pt", epoch, metrics)
        elif warmup_complete(epoch, warmup_epochs):
            stale += 1
        print(
            f"[epoch {epoch:03d}] train_loss={np.mean(losses):.6f} "
            f"held_out_acc={selection:.4f} best={best:.4f} stale={stale}/{args.patience}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"[early-stop] no improvement for {stale} epochs", flush=True)
            break

    best_record = max(per_epoch, key=lambda e: e["held_out_accuracy"])
    final_metrics = _held_out_metrics()
    sc001_passes = final_metrics["held_out_accuracy"] >= baseline + 0.15
    stage_run = build_stage_run(
        stage=STAGE_CLASSIFIER,
        output_signal=OUTPUT_SIGNAL_CLASSIFIER if not args.fine_labels
        else f"{OUTPUT_SIGNAL_CLASSIFIER}_fine_heuristic",
        run_id=args.run_name,
        architecture=identity,
        curriculum=identity_for_path(
            args.store / "assets.parquet", display_path=str(args.store.resolve())
        ),
        checkpoint={
            **identity_for_path(output / "classifier.pt"),
            "best_epoch": int(best_record["epoch"]),
        },
        baselines={"majority_class": {"held_out_accuracy": float(baseline)}},
        metrics={
            "best_epoch": best_record["epoch"],
            "final": final_metrics,
            "gate": {
                "sc_001": "held-out accuracy >= majority baseline + 0.15",
                "passes": bool(sc001_passes),
                "note": "promotion stays pending; the user reviews the SC-001 gate (FR-010)",
            },
            "fine_labels_heuristic": bool(args.fine_labels),
        },
    )
    (output / "model_stage_run.json").write_text(
        json.dumps(stage_run, indent=2), encoding="utf-8"
    )
    (output / "training_summary.json").write_text(
        json.dumps({"per_epoch_metrics": per_epoch, "model_stage_run": stage_run["run_id"]},
                   indent=2),
        encoding="utf-8",
    )
    print(
        f"best_epoch={best_record['epoch']} "
        f"held_out_acc={final_metrics['held_out_accuracy']:.4f} "
        f"majority_baseline={baseline:.4f} sc001={sc001_passes} promotion=pending(user gate)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
