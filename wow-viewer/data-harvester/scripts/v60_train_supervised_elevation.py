"""Train Supervised Neural Elevation U-Net on Authentic Development Maps (Spec 264).

Trains `SupervisedElevationUNet` directly on authentic non-museum development maps:
Input:  3-channel minimap RGB (256x256)
Target: authentic ADT height_257 in physical yards (257x257)

Usage:
    uv run python scripts/v60_train_supervised_elevation.py --epochs 40 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.v60.development_elevation_dataset import (
    DevelopmentElevationDataset,
    build_or_load_development_corpus,
)
from harvester.v60.supervised_elevation_model import (
    SupervisedElevationUNet,
    elevation_composite_loss,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Supervised Elevation U-Net on authentic development maps")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=48, help="Base channel width of UNet")
    parser.add_argument("--out-model", default="../output/models/supervised_elevation_development.pt")
    parser.add_argument("--cache-dataset", default="../output/datasets/development_elevation_corpus.npz")
    parser.add_argument("--rebuild-cache", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    maps_dir = repo_root / "test_data" / "original_development" / "World" / "Maps" / "development"
    textures_dir = repo_root / "test_data" / "original_development" / "World" / "Textures" / "Minimap"
    cache_path = (Path(__file__).resolve().parent.parent / args.cache_dataset).resolve()
    out_model_path = (Path(__file__).resolve().parent.parent / args.out_model).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=================================================================")
    print(f"Spec 264: Supervised Neural Elevation Training (PyTorch)")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print(f"=================================================================")

    # 1. Build / load dataset
    train_samples, val_samples = build_or_load_development_corpus(
        cache_path=cache_path,
        maps_dir=maps_dir,
        textures_dir=textures_dir,
        force_rebuild=args.rebuild_cache,
    )

    # Compute global dataset elevation stats
    all_z = [s["z"] for s in train_samples]
    global_mean = float(np.mean([np.mean(z) for z in all_z]))
    global_std = float(np.mean([np.std(z) for z in all_z]))
    print(f"Global Training Elevation: Mean = {global_mean:.2f} yds, Std = {global_std:.2f} yds")

    train_ds = DevelopmentElevationDataset(train_samples, augment=True)
    val_ds = DevelopmentElevationDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # 2. Instantiate Model
    model = SupervisedElevationUNet(
        in_channels=3,
        base_channels=args.base_channels,
        global_elevation_mean=global_mean,
        global_elevation_std=global_std,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SupervisedElevationUNet initialized with {total_params:,} trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_mae = float("inf")
    best_val_pearson = -float("inf")

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1_yards = 0.0
        t0 = time.time()

        for batch in train_loader:
            x = batch["rgb"].to(device)
            target = batch["z"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss, metrics = elevation_composite_loss(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x)
            train_l1_yards += metrics["l1_yards"] * len(x)

        train_loss /= len(train_ds)
        train_l1_yards /= len(train_ds)
        scheduler.step()
        dt = time.time() - t0

        # Evaluate on held-out validation tiles
        model.eval()
        val_maes = []
        val_rmses = []
        val_pearsons = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["rgb"].to(device)
                pred_z = model(x).cpu().numpy()[0]
                gt_z = batch["z"].numpy()[0]

                diff = np.abs(pred_z - gt_z)
                mae = float(np.mean(diff))
                rmse = float(np.sqrt(np.mean(diff**2)))

                c_matrix = np.corrcoef(pred_z.flatten(), gt_z.flatten())
                pearson_r = float(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0

                val_maes.append(mae)
                val_rmses.append(rmse)
                val_pearsons.append(pearson_r)

        mean_val_mae = float(np.mean(val_maes))
        mean_val_rmse = float(np.mean(val_rmses))
        mean_val_pearson = float(np.mean(val_pearsons))

        is_best = mean_val_mae < best_val_mae
        if is_best:
            best_val_mae = mean_val_mae
            best_val_pearson = mean_val_pearson
            out_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": args.base_channels,
                    "global_elevation_mean": global_mean,
                    "global_elevation_std": global_std,
                    "best_val_mae": best_val_mae,
                    "best_val_pearson": best_val_pearson,
                    "epoch": epoch,
                },
                out_model_path,
            )
            saved_tag = "[SAVED]"
        else:
            saved_tag = ""

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(
                f"Epoch {epoch:2d}/{args.epochs:2d} ({dt:.1f}s): "
                f"Train Loss: {train_loss:.3f} (L1: {train_l1_yards:.2f} yds) | "
                f"Val MAE: {mean_val_mae:5.2f} yds | "
                f"Val RMSE: {mean_val_rmse:5.2f} yds | "
                f"Val Pearson r: {mean_val_pearson:.4f} {saved_tag}",
                flush=True,
            )

    print("\n=================================================================")
    print("Training Complete on Authentic Development Maps!")
    print(f"Best Validation MAE:       {best_val_mae:.2f} yards")
    print(f"Best Validation Pearson r: {best_val_pearson:.4f}")
    print(f"Saved Checkpoint:          {out_model_path}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
