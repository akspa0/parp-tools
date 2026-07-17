"""Canonical v50 owner for RGB-to-WDL prior training (user-run CUDA trainer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

from harvester.spec103.prefab_curation import resolve_manifest_rows, validate_source_group_split
from harvester.spec103.wdl_prior_model import (
    INPUT_CONTRACT,
    MODEL_VARIANT_WDL_PRIOR,
    TARGET_CONTRACT,
    WDL_OUTER_SIZE,
    WdlPriorNet,
    build_wdl_target,
    normalize_minimap_rgb,
)
from harvester.v50.contracts import release_identity, require_store_release, validate_release


class PriorDataset(Dataset):
    def __init__(self, group, rows: list[int]) -> None:
        self.group, self.rows = group, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return normalize_minimap_rgb(self.group["minimap_rgb"][row]), torch.from_numpy(build_wdl_target(self.group["height_257"][row]))


def filter_deployable_rows(group, rows: list[int], *, min_rgb_mean: float, max_object_coverage: float) -> tuple[list[int], dict[str, int]]:
    """Keep only tiles whose visible minimap can honestly supervise RGB-only inference."""
    has_object_mask = "object_precise_mask" in group
    kept: list[int] = []
    dropped_dark = dropped_object = 0
    for row in rows:
        if float(np.asarray(group["minimap_rgb"][row], dtype=np.float32).mean()) < min_rgb_mean:
            dropped_dark += 1
            continue
        if has_object_mask and float((np.asarray(group["object_precise_mask"][row]) > 0.5).mean()) > max_object_coverage:
            dropped_object += 1
            continue
        kept.append(row)
    return kept, {"dropped_dark": dropped_dark, "dropped_object": dropped_object}


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval(); composite_losses = []; point_l1s = []
    with torch.no_grad():
        for x, y in loader:
            predicted = model(x.to(device))
            target = y.to(device)
            composite_losses.append(float(wdl_loss(predicted, target).item()))
            point_l1s.append(float(torch.nn.functional.l1_loss(predicted, target).item()))
    if not composite_losses:
        return {"composite": float("inf"), "point_l1": float("inf"), "world_mae": float("inf")}
    point_l1 = float(np.mean(point_l1s))
    return {"composite": float(np.mean(composite_losses)), "point_l1": point_l1, "world_mae": point_l1 * 4000.0}


def wdl_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute lattice loss plus topology matching; normalized L1 alone hid noise."""
    point = torch.nn.functional.smooth_l1_loss(predicted, target)
    split = WDL_OUTER_SIZE * WDL_OUTER_SIZE
    pred_outer = predicted[:, :split].view(-1, WDL_OUTER_SIZE, WDL_OUTER_SIZE)
    target_outer = target[:, :split].view(-1, WDL_OUTER_SIZE, WDL_OUTER_SIZE)
    gradient = torch.nn.functional.l1_loss(pred_outer[:, 1:, :], target_outer[:, 1:, :]) + torch.nn.functional.l1_loss(pred_outer[:, :, 1:], target_outer[:, :, 1:])
    return point + 0.25 * gradient


def main() -> int:
    ap = argparse.ArgumentParser(description="v50 RGB-only spatial WDL prior trainer (USER runs CUDA)")
    ap.add_argument("--store", required=True, type=Path, help="compact representative paired store")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", default=None, help="complete source-group column when no manifest partition exists")
    ap.add_argument("--val-value", default=None)
    ap.add_argument("--curation-manifest", type=Path, default=None,
                    help="existing Spec 103 representative-pattern curation manifest (or its directory). "
                         "Only its kept train/val rows are read; no full-corpus training.")
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many epochs without a strictly better held-out composite loss (0 disables)")
    ap.add_argument("--min-rgb-mean", type=float, default=25.0,
                    help="reject dark/water placeholder minimaps before training (uint8 mean, default 25)")
    ap.add_argument("--max-object-coverage", type=float, default=0.0,
                    help="reject tiles with more occluding object-mask coverage (default 0, clean terrain only)")
    ap.add_argument("--include-pathological", action="store_true",
                    help="retain curation rows labelled pathological (off by default for first RGB/WDL proof)")
    ap.add_argument("--min-train-rows", type=int, default=32); ap.add_argument("--min-val-rows", type=int, default=8)
    ap.add_argument("--release", default="v50.1", type=validate_release,
                    help="must match the v50 store release (default: v50.1)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; user-run training refuses CPU.")
    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if "minimap_rgb" not in group or "height_257" not in group:
        raise SystemExit("store must contain minimap_rgb and height_257")
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    if not index:
        raise SystemExit("store index is empty")
    split_mode = "manifest_partition"
    if args.curation_manifest is not None:
        manifest_path = args.curation_manifest / "curation_manifest.parquet" if args.curation_manifest.is_dir() else args.curation_manifest
        manifest_rows = pq.read_table(manifest_path).to_pylist()
        if not args.include_pathological:
            for row in manifest_rows:
                if str(row.get("difficulty_bucket", "")) == "pathological":
                    row["keep"] = False
        train_rows, val_rows, split_mode = resolve_manifest_rows(
            index, manifest_rows, val_key=args.val_key or "map", val_value=args.val_value or ""
        )
    else:
        if args.val_key is None or args.val_value is None or args.val_key not in index[0]:
            raise SystemExit("--val-key and --val-value are required when no curation manifest supplies partitions")
        split_mode = f"holdout:{args.val_key}={args.val_value}"
        train_rows = [i for i, row in enumerate(index) if str(row.get(args.val_key)) != str(args.val_value)]
        val_rows = [i for i, row in enumerate(index) if str(row.get(args.val_key)) == str(args.val_value)]
        validate_source_group_split(index, train_rows, val_rows)
    # A v50 mixed store is filtered before selection. Re-filtering it here was
    # the old source of a surprising second row drop after the user had already
    # approved a small curated dataset.
    if str(group.attrs.get("schema", "")) == "v50-mixed-curriculum-v1":
        train_filter = val_filter = {"dropped_dark": 0, "dropped_object": 0}
        print(f"[filter] v50 store was filtered before selection; train={len(train_rows)} val={len(val_rows)}", flush=True)
    else:
        train_rows, train_filter = filter_deployable_rows(group, train_rows, min_rgb_mean=args.min_rgb_mean, max_object_coverage=args.max_object_coverage)
        val_rows, val_filter = filter_deployable_rows(group, val_rows, min_rgb_mean=args.min_rgb_mean, max_object_coverage=args.max_object_coverage)
        print(f"[filter] train={len(train_rows)} {train_filter}; val={len(val_rows)} {val_filter}; pathological={'included' if args.include_pathological else 'excluded'}", flush=True)
    if len(train_rows) < args.min_train_rows or len(val_rows) < args.min_val_rows:
        raise SystemExit(f"insufficient deployable rows after filters: train={len(train_rows)} (min {args.min_train_rows}) val={len(val_rows)} (min {args.min_val_rows})")
    device = torch.device("cuda"); model = WdlPriorNet().to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = DataLoader(PriorDataset(group, train_rows), batch_size=min(args.batch, len(train_rows)), shuffle=True, num_workers=args.workers, pin_memory=True)
    val = DataLoader(PriorDataset(group, val_rows), batch_size=min(args.batch, len(val_rows)), num_workers=args.workers, pin_memory=True)
    args.output.mkdir(parents=True, exist_ok=True); best = float("inf"); best_epoch = 0; stale_epochs = 0
    run_identity = {**release_identity(args.release), "model_variant": MODEL_VARIANT_WDL_PRIOR,
                    "store": str(args.store.resolve())}
    (args.output / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train:
            opt.zero_grad(set_to_none=True); loss = wdl_loss(model(x.to(device)), y.to(device)); loss.backward(); opt.step()
        metrics = evaluate(model, val, device)
        checkpoint = {**run_identity, "input_contract": INPUT_CONTRACT, "target_contract": TARGET_CONTRACT,
                      "model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                      "store": str(args.store.resolve()), "split": {"key": args.val_key, "value": args.val_value}}
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if metrics["composite"] < best:
            best = metrics["composite"]; best_epoch = epoch; stale_epochs = 0; torch.save(checkpoint, args.output / "checkpoint_best.pt")
        else:
            stale_epochs += 1
        print(f"[epoch {epoch:03d}] val_composite={metrics['composite']:.6f} point_l1={metrics['point_l1']:.6f} world_mae={metrics['world_mae']:.2f} best={best:.6f} stale={stale_epochs}/{args.patience}", flush=True)
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"[early-stop] best epoch={best_epoch} val_composite={best:.6f}; no improvement for {stale_epochs} epochs", flush=True)
            break
    (args.output / "training_summary.json").write_text(json.dumps({**run_identity, "best_val_composite": best, "best_epoch": best_epoch, "stale_epochs": stale_epochs, "patience": args.patience, "train_rows": len(train_rows), "val_rows": len(val_rows), "split": {"mode": split_mode, "key": args.val_key, "value": args.val_value, "curation_manifest": str(args.curation_manifest) if args.curation_manifest else None}}, indent=2), encoding="utf-8")
    return 0
