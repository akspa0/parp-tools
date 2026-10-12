"""Train Stage 2 Residual Elevation Refiner on Authentic Development Maps (Spec 265).

Trains `ResidualElevationRefiner` directly on the residual error field:
    Input:  6-channel feature tensor (Minimap RGB [3] + Initial Elevation [1] + Surface Normals [2])
    Target: ΔZ = Z_gt - Z_initial (in physical world yards)
    Final:  Z_final = Z_initial + ΔZ_pred

Usage:
    uv run python scripts/v60_train_residual_refiner.py --epochs 40 --batch-size 8
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

from harvester.v60.residual_elevation_dataset import (
    ResidualElevationDataset,
    build_or_load_residual_corpus,
)
from harvester.v60.residual_elevation_model import (
    ResidualElevationRefiner,
    residual_composite_loss,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage 2 Residual Elevation Refiner")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=40, help="Base channel width of refiner U-Net")
    parser.add_argument("--stage1-checkpoint", default="../output/models/supervised_elevation_development.pt")
    parser.add_argument("--base-dataset", default="../output/datasets/development_elevation_corpus.npz")
    parser.add_argument("--cache-residual", default="../output/datasets/residual_elevation_corpus.npz")
    parser.add_argument("--out-model", default="../output/models/residual_refiner_development.pt")
    parser.add_argument("--rebuild-cache", action="store_true", default=False)
    args = parser.parse_args()

    stage1_ckpt_path = (Path(__file__).resolve().parent.parent / args.stage1_checkpoint).resolve()
    base_corpus_path = (Path(__file__).resolve().parent.parent / args.base_dataset).resolve()
    residual_cache_path = (Path(__file__).resolve().parent.parent / args.cache_residual).resolve()
    out_model_path = (Path(__file__).resolve().parent.parent / args.out_model).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=================================================================")
    print(f"Spec 265: Stage 2 Cascaded Residual Elevation Refinement Training")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print(f"=================================================================")

    # 1. Build / Load Residual Corpus
    res_train, res_val = build_or_load_residual_corpus(
        residual_cache_path=residual_cache_path,
        base_corpus_path=base_corpus_path,
        stage1_checkpoint_path=stage1_ckpt_path,
        device=str(device),
        force_rebuild=args.rebuild_cache,
    )

    # Compute delta stats across training set
    all_deltas = [s["delta_z"] for s in res_train]
    delta_mean = float(np.mean([np.mean(d) for d in all_deltas]))
    delta_std = float(np.mean([np.std(d) for d in all_deltas]))
    print(f"Residual Training Stats: Mean Delta Z = {delta_mean:.2f} yds, Std Delta Z = {delta_std:.2f} yds")

    train_ds = ResidualElevationDataset(res_train, augment=True)
    val_ds = ResidualElevationDataset(res_val, augment=False)

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

    # 2. Instantiate Stage 2 Model
    model = ResidualElevationRefiner(
        in_channels=6,
        base_channels=args.base_channels,
        delta_std=delta_std,
        delta_mean=delta_mean,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResidualElevationRefiner initialized with {total_params:,} trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_mae = float("inf")
    best_val_pearson = -float("inf")

    print("\nStarting Stage 2 training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_final_l1 = 0.0
        t0 = time.time()

        for batch in train_loader:
            feats = batch["features"].to(device)
            target_delta = batch["delta_z"].to(device)
            z_init = batch["z_init"].to(device)
            z_gt = batch["z_gt"].to(device)

            optimizer.zero_grad()
            pred_delta = model(feats)
            loss, metrics = residual_composite_loss(pred_delta, target_delta, z_init, z_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(feats)
            train_final_l1 += metrics["final_l1"] * len(feats)

        train_loss /= len(train_ds)
        train_final_l1 /= len(train_ds)
        scheduler.step()
        dt = time.time() - t0

        # Evaluation on held-out validation tiles
        model.eval()
        val_final_maes = []
        val_final_rmses = []
        val_final_pearsons = []

        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                pred_delta = model(feats).cpu().numpy()[0]
                z_init = batch["z_init"].numpy()[0]
                gt_z = batch["z_gt"].numpy()[0]

                # Combined reconstruction: Z_final = Z_init + pred_delta
                z_final = z_init + pred_delta

                diff = np.abs(z_final - gt_z)
                mae = float(np.mean(diff))
                rmse = float(np.sqrt(np.mean(diff**2)))

                c_matrix = np.corrcoef(z_final.flatten(), gt_z.flatten())
                pearson_r = float(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0

                val_final_maes.append(mae)
                val_final_rmses.append(rmse)
                val_final_pearsons.append(pearson_r)

        mean_val_mae = float(np.mean(val_final_maes))
        mean_val_rmse = float(np.mean(val_final_rmses))
        mean_val_pearson = float(np.mean(val_final_pearsons))

        is_best = mean_val_mae < best_val_mae
        if is_best:
            best_val_mae = mean_val_mae
            best_val_pearson = mean_val_pearson
            out_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": args.base_channels,
                    "delta_std": delta_std,
                    "delta_mean": delta_mean,
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
                f"Train Loss: {train_loss:.3f} (Combined L1: {train_final_l1:.2f} yds) | "
                f"Val Combined MAE: {mean_val_mae:5.2f} yds | "
                f"Val RMSE: {mean_val_rmse:5.2f} yds | "
                f"Val Pearson r: {mean_val_pearson:.4f} {saved_tag}",
                flush=True,
            )

    print("\n=================================================================")
    print("Stage 2 Training Complete!")
    print(f"Best Combined Validation MAE:       {best_val_mae:.2f} yards (was 81.07 yds)")
    print(f"Best Combined Validation Pearson r: {best_val_pearson:.4f}")
    print(f"Saved Refiner Checkpoint:           {out_model_path}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
