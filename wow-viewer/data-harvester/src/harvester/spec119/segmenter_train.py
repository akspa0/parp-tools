"""Spec 119 US2 object-library segmenter trainer (T018/T019; USER runs CUDA, FR-010).

Supervised by the library's own ``capture_mask`` (FR-002): per-pixel object-vs-background, the
"reproduce the renderer's silhouette from RGB alone" learnability test. Blank captures are
EXCLUDED from training (D-04, FR-006): an all-background target teaches the model the
all-background trivial baseline that SC-002 must beat. The all-foreground and all-background
trivial IoU baselines are always recorded (SC-002); held-out IoU is stratified by ground-truth
mask-coverage bucket so thin/extreme-aspect failures are measured, not hidden.

Dry-run-first: without ``--confirm-run`` the CLI prints the validated plan (param count, trivial
baselines, exclusion count, train/held-out counts) and exits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from harvester.spec119.library_data import (
    CAPTURE_MASK,
    captured_rows,
    load_asset_rows,
    open_library,
    read_image,
    require_new_output,
    row_coverages,
)
from harvester.spec119.object_library_contract import (
    OUTPUT_SIGNAL_SEGMENTER,
    STAGE_SEGMENTER,
    ObjectLibraryContractError,
    architecture_identity,
    build_stage_run,
    identity_for_path,
    segmentation_target,
)
from harvester.spec119.segmenter_model import (
    ObjectSegmenter,
    binary_iou,
    per_coverage_bucket_iou,
    trivial_iou_baselines,
)
from harvester.spec119.split import SplitError, apply_family_split, load_split
from harvester.v50.lr_schedule import make_onecycle_scheduler, warmup_complete


def build_segmenter_plan(
    *,
    architecture: dict,
    train_count: int,
    held_out_count: int,
    exclusion_count: int,
    trivial_baselines: dict,
    args: argparse.Namespace,
) -> dict:
    """Machine-readable no-training preview printed before any CUDA allocation (FR-010)."""
    return {
        "schema": "v119-segmenter-plan-v1",
        "stage": STAGE_SEGMENTER,
        "architecture": architecture,
        "split_counts": {"train": train_count, "held_out": held_out_count},
        "blank_threshold": args.blank_threshold,
        "blank_excluded_count": exclusion_count,
        "trivial_baselines": trivial_baselines,
        "batch_size": args.batch,
        "epochs": args.epochs,
        "seed": args.seed,
        "optimizer": {"name": "AdamW", "learning_rate": args.lr, "weight_decay": 1e-4},
        "lr_schedule": {"name": "onecycle", "pct_start": args.pct_start},
        "success_gate": {"sc_002": "held-out IoU >= better trivial baseline + 0.20"},
        "no_pretrained_backbone_from_scratch": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Spec 119 object-library segmenter trainer (USER runs CUDA; dry-run-first)"
    )
    ap.add_argument("--store", required=True, type=Path, help="object-library zarr (read-only)")
    ap.add_argument("--split", required=True, type=Path,
                    help="REQUIRED: a v119-family-split-v1 JSON (spec119_build_split.py); refuses "
                         "a leaky split (verified_violation_count != 0). No random fallback.")
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base", type=int, default=16,
                    help="ObjectSegmenter width (from scratch; <1M params at 16, SC-005)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--blank-threshold", type=float, default=0.01,
                    help="mask coverage below this fraction is EXCLUDED from training (D-04/FR-006)")
    ap.add_argument("--patience", type=int, default=20,
                    help="post-warmup epochs with no held-out IoU improvement before early stop. "
                         "0 = never stop.")
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

    coverages = row_coverages(group, rows)
    # D-04: blank captures are excluded from BOTH sides (a blank held-out row would score the
    # trivial all-background prediction as a perfect IoU=1 and inflate the metric).
    kept_train = [i for i in train_idx if coverages[i] >= args.blank_threshold]
    kept_held_out = [i for i in held_out_idx if coverages[i] >= args.blank_threshold]
    exclusion_count = (len(train_idx) - len(kept_train)) + (len(held_out_idx) - len(kept_held_out))
    if not kept_train or not kept_held_out:
        raise SystemExit(
            f"no non-blank rows after exclusion: train={len(kept_train)} "
            f"held_out={len(kept_held_out)} (blank_threshold={args.blank_threshold})"
        )

    held_out_targets = [
        segmentation_target(np.asarray(group[CAPTURE_MASK][rows[i]["_row_index"]]))
        for i in kept_held_out
    ]
    trivial = trivial_iou_baselines(held_out_targets)

    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ObjectSegmenter(base=args.base)
    identity = architecture_identity(
        model,
        architecture_id="object_library_segmenter",
        config={"class": "ObjectSegmenter", "arch": "object_library_segmenter_v1",
                "base": args.base, "input": "3x128x128", "output": "1x128x128"},
    )
    plan = build_segmenter_plan(
        architecture=identity, train_count=len(kept_train),
        held_out_count=len(kept_held_out), exclusion_count=exclusion_count,
        trivial_baselines=trivial, args=args,
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
        def __init__(self, indices: list[int]) -> None:
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            zarr_row = rows[self.indices[i]]["_row_index"]
            rgb = read_image(group, zarr_row)
            target = segmentation_target(np.asarray(group[CAPTURE_MASK][zarr_row]))
            return (
                torch.from_numpy(rgb).permute(2, 0, 1),
                torch.from_numpy(target.astype(np.float32)).unsqueeze(0),
            )

    device = torch.device("cuda")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    loader = DataLoader(
        RowDataset(kept_train), batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=True, generator=train_generator,
    )
    scheduler, warmup_epochs = make_onecycle_scheduler(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader),
        pct_start=args.pct_start,
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    held_out_coverages = [coverages[i] for i in kept_held_out]

    def _held_out_metrics() -> dict:
        model.eval()
        ious: list[float] = []
        with torch.no_grad():
            for i, target in zip(kept_held_out, held_out_targets, strict=True):
                rgb = read_image(group, rows[i]["_row_index"])
                x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                prediction = (model(x).squeeze(0).squeeze(0).cpu().numpy() > 0.5)
                ious.append(binary_iou(prediction, target))
        return {
            "held_out_iou": float(np.mean(ious)),
            "per_coverage_bucket": per_coverage_bucket_iou(ious, held_out_coverages),
        }

    def _save_checkpoint(path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "kind": "segmenter",
                "state_dict": model.state_dict(),
                "architecture": {"base": args.base},
                "config": {"lr": args.lr, "epochs": args.epochs,
                           "blank_threshold": args.blank_threshold,
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
            prediction = model(x.to(device))
            loss = torch.nn.functional.binary_cross_entropy(prediction, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            losses.append(float(loss.detach().item()))
        metrics = _held_out_metrics()
        selection = metrics["held_out_iou"]
        per_epoch.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        _save_checkpoint(output / "checkpoint_last.pt", epoch, metrics)
        if selection > best:
            best = selection
            stale = 0
            _save_checkpoint(output / "segmenter.pt", epoch, metrics)
        elif warmup_complete(epoch, warmup_epochs):
            stale += 1
        print(
            f"[epoch {epoch:03d}] train_loss={np.mean(losses):.6f} "
            f"held_out_iou={selection:.4f} best={best:.4f} stale={stale}/{args.patience}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(f"[early-stop] no improvement for {stale} epochs", flush=True)
            break

    best_record = max(per_epoch, key=lambda e: e["held_out_iou"])
    final_metrics = _held_out_metrics()
    better_trivial = max(trivial["all_foreground"], trivial["all_background"])
    sc002_passes = final_metrics["held_out_iou"] >= better_trivial + 0.20
    stage_run = build_stage_run(
        stage=STAGE_SEGMENTER,
        output_signal=OUTPUT_SIGNAL_SEGMENTER,
        run_id=args.run_name,
        architecture=identity,
        curriculum=identity_for_path(
            args.store / "assets.parquet", display_path=str(args.store.resolve())
        ),
        checkpoint={
            **identity_for_path(output / "segmenter.pt"),
            "best_epoch": int(best_record["epoch"]),
        },
        baselines={"trivial": trivial},
        metrics={
            "best_epoch": best_record["epoch"],
            "final": final_metrics,
            "blank_excluded_count": int(exclusion_count),
            "gate": {
                "sc_002": "held-out IoU >= better trivial baseline + 0.20",
                "passes": bool(sc002_passes),
                "note": "promotion stays pending; the user reviews the SC-002 gate (FR-010)",
            },
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
        f"held_out_iou={final_metrics['held_out_iou']:.4f} "
        f"better_trivial={better_trivial:.4f} sc002={sc002_passes} promotion=pending(user gate)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
