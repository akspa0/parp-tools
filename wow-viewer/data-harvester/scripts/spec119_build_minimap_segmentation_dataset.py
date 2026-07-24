#!/usr/bin/env python3
"""Spec 119 — Ground-Truth Minimap Object Segmentation Dataset & Trainer.

Materializes exact ground-truth 256x256 binary object segmentation masks directly from
`placements.parquet` and `minimap_rgb_authored` in 0_5_3_3368-Azeroth.zarr.
Trains a UNet MinimapObjectSegmenter to predict precise object footprints vs terrain background.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import zarr

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec120.obb_contract import (
    ADT_TILE_SIZE_YARDS,
    DEFAULT_TILE_PIXELS,
    YARDS_PER_PIXEL,
    world_to_tile_pixels,
)


def build_minimap_segmentation_target(
    placements_data: dict[str, list[Any]],
    tile_x: int,
    tile_y: int,
    spatial_index: dict[tuple[int, int], list[int]],
    tile_pixels: int = 256,
) -> np.ndarray:
    """Construct exact 256x256 binary object segmentation mask from placements.parquet."""
    mask = np.zeros((tile_pixels, tile_pixels), dtype=np.uint8)

    pos_x = placements_data.get("posX", placements_data.get("position_x", []))
    pos_y = placements_data.get("posY", placements_data.get("position_y", []))
    scales = placements_data.get("scale", [1.0] * len(pos_x))
    bb_min_x = placements_data.get("bbMinX", [0.0] * len(pos_x))
    bb_max_x = placements_data.get("bbMaxX", [0.0] * len(pos_x))
    bb_min_y = placements_data.get("bbMinY", [0.0] * len(pos_x))
    bb_max_y = placements_data.get("bbMaxY", [0.0] * len(pos_x))

    candidate_rows = []
    for dtx in (-1, 0, 1):
        for dty in (-1, 0, 1):
            candidate_rows.extend(spatial_index.get((tile_x + dtx, tile_y + dty), []))

    for i in candidate_rows:
        wx = float(pos_x[i])
        wy = float(pos_y[i])
        px, py = world_to_tile_pixels(wx, wy, tile_x, tile_y, tile_pixels)

        if -16 <= px <= tile_pixels + 16 and -16 <= py <= tile_pixels + 16:
            scale_val = float(scales[i]) if i < len(scales) else 1.0
            extent_x = abs(float(bb_max_x[i]) - float(bb_min_x[i])) if i < len(bb_max_x) else 0.0
            extent_y = abs(float(bb_max_y[i]) - float(bb_min_y[i])) if i < len(bb_min_y) else 0.0

            if extent_x <= 0.01:
                extent_x = 10.0 * scale_val
            if extent_y <= 0.01:
                extent_y = 10.0 * scale_val

            w_px = max(4.0, extent_x / YARDS_PER_PIXEL)
            h_px = max(4.0, extent_y / YARDS_PER_PIXEL)

            x0 = max(0, int(math.floor(px - w_px / 2.0)))
            x1 = min(tile_pixels, int(math.ceil(px + w_px / 2.0)))
            y0 = max(0, int(math.floor(py - h_px / 2.0)))
            y1 = min(tile_pixels, int(math.ceil(py + h_px / 2.0)))

            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255

    return mask


class MinimapUNetSegmenter(nn.Module):
    """256x256 U-Net for Minimap Terrain Object Segmentation."""

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        b = base
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, kernel_size=3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(b, b * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(b * 2, b * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(b * 4, b * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(b * 2, b, kernel_size=3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        y = self.conv1(torch.cat([self.up1(x3), x2], dim=1))
        y = self.conv2(torch.cat([self.up2(y), x1], dim=1))
        return self.outc(y)


class MinimapDataset(Dataset):

    def __init__(
        self,
        rgb_arr: zarr.Array,
        tile_tuples: list[tuple[int, int, int]],
        placements_data: dict[str, list[Any]],
        spatial_index: dict[tuple[int, int], list[int]],
    ):
        self.rgb_arr = rgb_arr
        self.tile_tuples = tile_tuples
        self.placements_data = placements_data
        self.spatial_index = spatial_index

    def __len__(self) -> int:
        return len(self.tile_tuples)

    def __getitem__(self, idx: int):
        zarr_row, tx, ty = self.tile_tuples[idx]
        rgb_np = np.asarray(self.rgb_arr[zarr_row])  # (256, 256, 3) uint8
        mask_np = build_minimap_segmentation_target(
            self.placements_data, tx, ty, self.spatial_index
        )  # (256, 256) uint8

        x = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ground-Truth Minimap Object Segmentation UNet.")
    parser.add_argument(
        "--zarr-store",
        type=Path,
        default=Path("../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr"),
        help="Path to 0_5_3_3368-Azeroth.zarr store.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../output/spec119/minimap_unet_runs"),
        help="Output directory for model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--confirm-run", action="store_true", help="Confirm GPU training run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Spec 119 Ground-Truth Minimap Object Segmentation UNet Trainer ===")
    print(f"Zarr Store: {args.zarr_store.resolve()}")

    zarr_grp = zarr.open_group(args.zarr_store, mode="r")
    rgb_arr = zarr_grp["minimap_rgb_authored"]
    p_tbl = pq.read_table(args.zarr_store / "placements.parquet")
    idx_tbl = pq.read_table(args.zarr_store / "index.parquet")

    placements_data = p_tbl.to_pydict()

    # Build spatial index
    pos_x = placements_data.get("posX", [])
    pos_y = placements_data.get("posY", [])
    spatial_index: dict[tuple[int, int], list[int]] = {}
    for i in range(len(pos_x)):
        tx = int(math.floor((17066.666666666668 - float(pos_x[i])) / ADT_TILE_SIZE_YARDS))
        ty = int(math.floor((17066.666666666668 - float(pos_y[i])) / ADT_TILE_SIZE_YARDS))
        spatial_index.setdefault((tx, ty), []).append(i)

    xs = idx_tbl["tile_x"].to_pylist()
    ys = idx_tbl["tile_y"].to_pylist()

    # Filter for active land tiles with placements
    tile_tuples = []
    for zarr_row, (tx, ty) in enumerate(zip(xs, ys)):
        if (tx, ty) in spatial_index and len(spatial_index[(tx, ty)]) >= 5:
            tile_tuples.append((zarr_row, tx, ty))

    print(f"Found {len(tile_tuples)} active land tiles with ground-truth object placement targets.")

    # Spatial split (80% train, 20% val)
    np.random.seed(42)
    np.random.shuffle(tile_tuples)
    val_count = max(1, int(len(tile_tuples) * 0.2))
    train_tuples = tile_tuples[val_count:]
    val_tuples = tile_tuples[:val_count]

    print(f"Dataset Split: {len(train_tuples)} Train Tiles | {len(val_tuples)} Val Tiles")

    if not args.confirm_run:
        print("[DRY RUN ONLY] Add --confirm-run to launch GPU training.")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing MinimapUNetSegmenter on {device}...")

    model = MinimapUNetSegmenter(in_channels=3, base=32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_ds = MinimapDataset(rgb_arr, train_tuples, placements_data, spatial_index)
    val_ds = MinimapDataset(rgb_arr, val_tuples, placements_data, spatial_index)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = args.output_dir / "minimap_unet_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = (preds * y).sum(dim=(1, 2, 3))
                union = ((preds + y) > 0).float().sum(dim=(1, 2, 3))
                iou = (intersection + 1e-6) / (union + 1e-6)
                val_iou += iou.sum().item()

        val_loss /= len(val_ds)
        val_iou /= len(val_ds)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                },
                best_ckpt_path,
            )

    print(f"\n[TRAINING COMPLETE] Saved best MinimapUNetSegmenter model to {best_ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
