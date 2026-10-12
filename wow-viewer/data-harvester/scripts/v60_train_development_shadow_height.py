"""Train Shadow-to-Height Model directly on Authentic Development Maps (Spec 264).

Trains `HeightRelativeNet` on authentic `original_development` non-museum tiles:
Input:  1-channel bare terrain shadow dS (256x256)
Target: authentic ADT height_257 in yards (normalized under v112.1 relative height contract)

Usage:
    uv run python scripts/v60_train_development_shadow_height.py --epochs 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.v50.height_relative_model import (
    HeightRelativeNet,
    decode_relative_height,
    encode_relative_height,
    height_loss,
)
from harvester.v60.development_ground_truth import DevelopmentGroundTruthExtractor
from harvester.v60.minimap_shadow_stripper import MinimapShadowStripper


class AuthenticDevelopmentDataset(Dataset):
    """Dataset of authentic non-museum development tiles with ground truth elevation."""

    def __init__(
        self,
        maps_dir: Path,
        textures_dir: Path,
        split: str = "train",
        val_fraction: float = 0.15,
        max_tiles: int = 200,
    ):
        self.extractor = DevelopmentGroundTruthExtractor(maps_dir, textures_dir)
        self.stripper = MinimapShadowStripper(inpaint_scales=2, iterations=15)

        # Scan all sculpted tiles
        print(f"Scanning authentic development tiles in {maps_dir.name}...", flush=True)
        all_sculpted = self.extractor.scan_all_sculpted_tiles()
        print(f"Found {len(all_sculpted)} sculpted tiles.", flush=True)

        # Load samples that have matching minimap textures
        self.samples = []
        for tx, ty in all_sculpted[:max_tiles]:
            png_path = textures_dir / f"development_{tx}_{ty}.png"
            if not png_path.is_file():
                continue
            tile = self.extractor.load_tile(tx, ty)
            if tile is None or np.all(tile.minimap_rgb == 0):
                continue
            if tile.relief_span < 2.0:  # Skip virtually flat tiles
                continue
            self.samples.append(tile)

        print(f"Loaded {len(self.samples)} valid non-flat tiles with minimap PNGs.", flush=True)

        n = len(self.samples)
        val_count = max(2, int(n * val_fraction))
        if split == "train":
            self.items = self.samples[: n - val_count]
        else:
            self.items = self.samples[n - val_count :]

        print(f"Dataset split '{split}': {len(self.items)} tiles.", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float]:
        tile = self.items[idx]
        # Extract stripped shadow
        shadow, _ = self.stripper.strip_and_inpaint(
            tile.minimap_rgb,
            object_mask=tile.building_mask,
            normalize_albedo=True,
        )

        # Input shadow: (1, 256, 256) float32
        shadow_t = torch.from_numpy(shadow).unsqueeze(0).float()

        # Target relative height under v112.1 contract: [0, 1]
        norm_h, min_z, max_z = encode_relative_height(tile.height_257)
        target_t = torch.from_numpy(norm_h).float()

        return {
            "shadow": shadow_t,
            "target": target_t,
            "min_z": float(min_z),
            "max_z": float(max_z),
            "tile_x": tile.tile_x,
            "tile_y": tile.tile_y,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train shadow->height model on authentic development maps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-model", default="../output/models/shadow_height_development.pt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    maps_dir = repo_root / "test_data" / "original_development" / "World" / "Maps" / "development"
    textures_dir = repo_root / "test_data" / "original_development" / "World" / "Textures" / "Minimap"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = AuthenticDevelopmentDataset(maps_dir, textures_dir, split="train")
    val_ds = AuthenticDevelopmentDataset(maps_dir, textures_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = HeightRelativeNet(base=32, in_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_mae_yards = float("inf")
    best_val_r2 = -float("inf")

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            x = batch["shadow"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = height_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_ds)
        dt = time.time() - t0

        # Evaluation on held-out validation tiles
        model.eval()
        val_mae_yards = []
        val_r2_list = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["shadow"].to(device)
                pred_norm = model(x).cpu().numpy()[0]
                target_norm = batch["target"].numpy()[0]
                min_z = batch["min_z"].item()
                max_z = batch["max_z"].item()

                pred_z = decode_relative_height(pred_norm, min_z, max_z)
                target_z = decode_relative_height(target_norm, min_z, max_z)

                mae_yards = float(np.mean(np.abs(pred_z - target_z)))
                val_mae_yards.append(mae_yards)

                ss_tot = np.sum((target_z - np.mean(target_z)) ** 2)
                ss_res = np.sum((target_z - pred_z) ** 2)
                r2 = float(1.0 - (ss_res / max(1e-5, ss_tot)))
                val_r2_list.append(r2)

        mean_mae = float(np.mean(val_mae_yards))
        mean_r2 = float(np.mean(val_r2_list))

        if mean_mae < best_val_mae_yards:
            best_val_mae_yards = mean_mae
            best_val_r2 = mean_r2
            out_p = Path(args.out_model)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_mae_yards": best_val_mae_yards,
                    "best_val_r2": best_val_r2,
                    "epoch": epoch,
                },
                out_p,
            )
            saved_tag = "[SAVED]"
        else:
            saved_tag = ""

        if epoch % 5 == 0 or epoch == 1 or saved_tag:
            print(
                f"Epoch {epoch:2d}/{args.epochs:2d} ({dt:.1f}s): "
                f"Train Loss: {train_loss:.4f} | "
                f"Val MAE: {mean_mae:5.2f} yds | "
                f"Val R^2: {mean_r2:5.2f} {saved_tag}",
                flush=True,
            )

    print("\n=================================================================")
    print("Training Complete on Authentic Development Tiles!")
    print(f"Best Validation MAE: {best_val_mae_yards:.2f} yards")
    print(f"Best Validation R^2: {best_val_r2:.4f}")
    print(f"Saved Checkpoint:    {args.out_model}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
