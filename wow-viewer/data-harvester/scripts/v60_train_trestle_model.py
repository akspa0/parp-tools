"""Train Trestle Elevation UNet on Authentic Development Maps (Spec 266).

Modern successor to the February 2026 v7 WDL trestle model.
Supervises residual carving ΔZ and global bounds [Z_min, Z_max] on top of
coarse WDL lattices transferred from Wrath 3.0.1.8303 Northrend prototype matches.

Usage:
    uv run python scripts/v60_train_trestle_model.py --epochs 35 --batch-size 8
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

from harvester.v60.trestle_dataset import (
    TrestleElevationDataset,
    build_or_load_trestle_corpus,
)
from harvester.v60.trestle_elevation_model import (
    TrestleElevationUNet,
    trestle_composite_loss,
)


def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two flat arrays."""
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x < 1e-5 or std_y < 1e-5:
        return 0.0
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Trestle Elevation UNet (Spec 266)")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel width")
    parser.add_argument("--trestle-corpus", default="../output/datasets/trestle_elevation_corpus.npz")
    parser.add_argument("--base-corpus", default="../output/datasets/development_elevation_corpus.npz")
    parser.add_argument("--synth-wdl", default="../output/development_synthesized_wdl.npz")
    parser.add_argument("--out-model", default="../output/models/trestle_elevation_v1.pt")
    args = parser.parse_args()

    trestle_corpus_path = (Path(__file__).resolve().parent.parent / args.trestle_corpus).resolve()
    base_corpus_path = (Path(__file__).resolve().parent.parent / args.base_corpus).resolve()
    synth_wdl_path = (Path(__file__).resolve().parent.parent / args.synth_wdl).resolve()
    out_model_path = (Path(__file__).resolve().parent.parent / args.out_model).resolve()
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("Spec 266: Trestle Elevation UNet Training (Modern v7 Successor)")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} (VRAM: {vram_gb:.1f} GB)")
    print("=================================================================")

    # 1. Load Dataset
    train_samples, val_samples = build_or_load_trestle_corpus(
        cache_path=trestle_corpus_path,
        base_corpus_path=base_corpus_path,
        synth_wdl_path=synth_wdl_path,
        force_rebuild=False,
    )

    train_ds = TrestleElevationDataset(train_samples, augment=True)
    val_ds = TrestleElevationDataset(val_samples, augment=False)

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
    model = TrestleElevationUNet(
        in_channels=6,
        base_channels=args.base_channels,
    ).to(device)

    print(f"Model Parameters: {model.parameter_count:,} (AC-002: < 10M)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_mae = float("inf")
    best_val_r = 0.0

    # 3. Training Loop
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_l1s = []

        for batch in train_loader:
            features = batch["features"].to(device)    # (B, 6, 256, 256)
            z_trestle = batch["z_trestle"].to(device)  # (B, 257, 257)
            z_gt = batch["z_gt"].to(device)            # (B, 257, 257)
            bounds_gt = batch["bounds"].to(device)      # (B, 2)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_delta_z, pred_bounds = model(features)
                loss, metrics = trestle_composite_loss(
                    pred_delta_z=pred_delta_z,
                    pred_bounds=pred_bounds,
                    z_trestle=z_trestle,
                    gt_z=z_gt,
                    gt_bounds=bounds_gt,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(metrics["total_loss"])
            epoch_l1s.append(metrics["final_l1"])

        scheduler.step()

        # 4. Validation Loop
        model.eval()
        val_maes = []
        val_rs = []
        val_span_ratios = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                z_trestle = batch["z_trestle"].to(device)
                z_gt = batch["z_gt"].to(device)

                pred_delta_z, _ = model(features)
                z_final = z_trestle + pred_delta_z

                z_final_np = z_final.squeeze(0).cpu().numpy()
                z_gt_np = z_gt.squeeze(0).cpu().numpy()

                mae = float(np.mean(np.abs(z_final_np - z_gt_np)))
                r = compute_pearson_r(z_final_np, z_gt_np)

                span_pred = float(np.ptp(z_final_np))
                span_gt = max(1.0, float(np.ptp(z_gt_np)))
                span_ratio = span_pred / span_gt

                val_maes.append(mae)
                val_rs.append(r)
                val_span_ratios.append(span_ratio)

        mean_val_mae = float(np.mean(val_maes))
        mean_val_r = float(np.mean(val_rs))
        mean_span_ratio = float(np.mean(val_span_ratios))

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {np.mean(epoch_losses):.2f} | "
            f"Train L1: {np.mean(epoch_l1s):.2f} yds | "
            f"Val MAE: {mean_val_mae:.2f} yds | "
            f"Val Pearson r: {mean_val_r:.4f} | "
            f"Span Ratio: {mean_span_ratio:.2f}"
        )

        if mean_val_mae < best_val_mae:
            best_val_mae = mean_val_mae
            best_val_r = mean_val_r
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": best_val_mae,
                    "val_r": best_val_r,
                    "base_channels": args.base_channels,
                    "model_name": "TrestleElevationUNet",
                },
                out_model_path,
            )

    elapsed = time.time() - t_start
    print("=================================================================")
    print(f"Spec 266 Trestle Model Training Complete in {elapsed:.1f}s!")
    print(f"Best Validation MAE:       {best_val_mae:.2f} yards (Target <= 25.0 yds)")
    print(f"Best Validation Pearson r: {best_val_r:.4f} (Target >= 0.75)")
    print(f"Saved Checkpoint:          {out_model_path}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
