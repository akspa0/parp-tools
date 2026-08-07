#!/usr/bin/env python3
"""Train a shadow→height model (Spec 134 US3).

Takes ``terrain_shadow_256`` (single-channel, 256x256) as input and predicts
``height_257`` (257x257 relative height under the v112.1 target contract) as output.

The model reuses the ``direct_cnn_v112`` architecture (HeightRelativeNet) with
``in_channels=1``. The single shadow channel replaces the 3-channel minimap RGB input,
so the model learns the physical relationship between terrain shadow and terrain height
without the confounding texture signal.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v60_train_shadow_height.py \\
        --store ../output/datasets/v60/v60.1/unified.zarr \\
        --output ../output/runs/shadow-height-v1 \\
        --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.height_relative_model import (  # noqa: E402
    HEIGHT_GRID,
    HeightRelativeNet,
    encode_relative_height,
    height_loss,
)
from harvester.v50.model_stage_contract import (  # noqa: E402
    sha256_json,
)

SHADOW_SIZE = 256
TARGET_RELEASE = "v60"
BASELINE_VAL_MAE = 0.1492665126  # SPEC112_FROZEN_BEST_VAL_MAE


class ShadowRowDataset(Dataset):
    """One row per tile: terrain_shadow_256 (1x256x256) -> height_257 (257x257)."""

    def __init__(self, stores: list[Path], split: str = "train", val_fraction: float = 0.15):
        import pyarrow.parquet as pq
        import zarr

        # Accept one or more per-build stores (or a unified store). Each store is a
        # (build, map) Zarr store with terrain_shadow_256 + height_257 arrays.
        self._stores: list[tuple[zarr.Group, list[dict]]] = []
        for store in stores:
            group = zarr.open_group(str(store), mode="r")
            if "terrain_shadow_256" not in group:
                print(f"  WARNING: {store} has no terrain_shadow_256; skipping", flush=True)
                continue
            if "height_257" not in group:
                print(f"  WARNING: {store} has no height_257; skipping", flush=True)
                continue
            index = pq.read_table(store / "index.parquet").to_pylist()
            self._stores.append((group, index))

        if not self._stores:
            raise ValueError("no store has both terrain_shadow_256 and height_257")

        # Flatten all rows across all stores.
        self._rows: list[tuple[int, int]] = []  # (store_idx, row_id)
        for store_idx, (_, index) in enumerate(self._stores):
            for row_id in range(len(index)):
                self._rows.append((store_idx, row_id))

        # Deterministic split: last val_fraction rows by flattened order (spatial holdout).
        n = len(self._rows)
        val_count = max(1, int(n * val_fraction))
        if split == "train":
            self.row_ids = list(range(n - val_count))
        else:
            self.row_ids = list(range(n - val_count, n))

        print(f"  ShadowRowDataset ({split}): {len(self.row_ids)} rows across {len(self._stores)} stores", flush=True)

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat_idx = self.row_ids[idx]
        store_idx, row_id = self._rows[flat_idx]
        group = self._stores[store_idx][0]
        shadow = np.asarray(group["terrain_shadow_256"][row_id], dtype=np.float32)
        height = np.asarray(group["height_257"][row_id], dtype=np.float32)

        # Shadow: (256, 256) -> (1, 256, 256)
        shadow_t = torch.from_numpy(shadow).unsqueeze(0)

        # Target: relative height under v112.1 contract (min-max normalized, [0, 1]).
        target_norm, _, _ = encode_relative_height(height)
        target_t = torch.from_numpy(target_norm)

        return shadow_t, target_t


def _tile_mean_mae(dataset: ShadowRowDataset) -> float:
    """Compute the tile-mean baseline: predict the tile's mean height everywhere."""
    errors: list[float] = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        mean = target.mean().item()
        errors.append(torch.abs(target - mean).mean().item())
    return float(np.mean(errors)) if errors else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a shadow→height model (Spec 134 US3)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--store", type=Path,
                        help="a single v60 Zarr store with terrain_shadow_256 and height_257")
    source.add_argument("--store-dir", type=Path,
                        help="a directory of per-build v60 Zarr stores (e.g. the .v60-work dir); "
                             "all *.zarr subdirs are used")
    parser.add_argument("--output", required=True, type=Path,
                        help="output run directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="fraction of tiles held out as a trailing spatial block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="build dataset + model, print shapes, exit (no CUDA)")
    args = parser.parse_args()

    if args.store is not None:
        if not args.store.exists():
            raise SystemExit(f"store not found: {args.store}")
        stores = [args.store]
    else:
        if not args.store_dir.exists():
            raise SystemExit(f"store dir not found: {args.store_dir}")
        stores = sorted(args.store_dir.glob("*.zarr"))
        if not stores:
            raise SystemExit(f"no *.zarr stores found in {args.store_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {len(stores)} store(s)", flush=True)
    train_ds = ShadowRowDataset(stores, split="train", val_fraction=args.val_fraction)
    val_ds = ShadowRowDataset(stores, split="val", val_fraction=args.val_fraction)

    model = HeightRelativeNet(base=32, in_channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: direct_cnn_v112 (1-channel), {n_params:,} params", flush=True)

    if args.dry_run:
        shadow_t, target_t = train_ds[0]
        out = model(shadow_t.unsqueeze(0))
        print(f"Dry-run: input={tuple(shadow_t.shape)} output={tuple(out.shape)} "
              f"target={tuple(target_t.shape)}", flush=True)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("WARNING: no CUDA device; training on CPU will be very slow", flush=True)

    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader),
        pct_start=0.1,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    best_epoch = -1
    identity = sha256_json({
        "architecture": "direct_cnn_v112",
        "input_channels": 1,
        "input_signal": "terrain_shadow_256",
        "target_contract": "v112.1",
        "store": str(args.store),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
    })
    (args.output / "run_identity.json").write_text(
        json.dumps({
            "release": TARGET_RELEASE,
            "architecture": "direct_cnn_v112",
            "input_channels": 1,
            "input_signal": "terrain_shadow_256",
            "target_contract": "v112.1",
            "config_sha256": identity,
            "baseline_val_mae": BASELINE_VAL_MAE,
        }, indent=2), encoding="utf-8")

    print(f"Baseline (tile-mean) val_mae: {BASELINE_VAL_MAE:.4f}", flush=True)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for shadow_t, target_t in train_loader:
            shadow_t = shadow_t.to(device)
            target_t = target_t.to(device)
            optimizer.zero_grad()
            pred = model(shadow_t)
            loss = height_loss(pred, target_t)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * shadow_t.size(0)

        model.eval()
        val_mae = 0.0
        n_val = 0
        with torch.no_grad():
            for shadow_t, target_t in val_loader:
                shadow_t = shadow_t.to(device)
                target_t = target_t.to(device)
                pred = model(shadow_t)
                val_mae += torch.abs(pred - target_t).mean().item() * shadow_t.size(0)
                n_val += shadow_t.size(0)
        val_mae /= max(1, n_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss/max(1,len(train_ds)):.4f}  "
                  f"val_mae={val_mae:.4f}  ({time.time()-start:.0f}s)", flush=True)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), args.output / "checkpoint_best.pt")

    torch.save(model.state_dict(), args.output / "checkpoint_last.pt")

    beats = best_val_mae < BASELINE_VAL_MAE * 0.95
    summary = {
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "baseline_val_mae": BASELINE_VAL_MAE,
        "beats_baseline": beats,
        "beats_by_5pct": beats,
        "device": str(device),
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - start,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[DONE] best_val_mae={best_val_mae:.4f} (epoch {best_epoch})", flush=True)
    print(f"       baseline={BASELINE_VAL_MAE:.4f}", flush=True)
    print(f"       beats baseline by 5%: {beats}", flush=True)
    print(f"       -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())