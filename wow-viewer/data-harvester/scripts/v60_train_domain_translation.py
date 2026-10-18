#!/usr/bin/env python3
"""Train Lightweight Domain Translation Model: Textured Minimap -> Synthetic DXT1 Hillshade (Spec 274).

Usage:
    # Dry-run verification (1 mini-epoch, 2 batches on CPU/GPU)
    uv run python scripts/v60_train_domain_translation.py --dry-run

    # Full training run
    uv run python scripts/v60_train_domain_translation.py \
        --dataset output/datasets/v60_clean_pairs.zarr \
        --epochs 60 \
        --batch-size 8 \
        --lr 2e-4 \
        --checkpoint-dir models/domain_translation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.clean_pair_dataset import CleanPairDataset
from harvester.v60.domain_translation_model import (
    DomainTranslationLoss,
    DomainTranslationUNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_domain_translation")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DomainTranslationLoss,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    running_losses: dict[str, float] = {
        "loss_total": 0.0,
        "loss_l1": 0.0,
        "loss_grad": 0.0,
        "loss_ssim": 0.0,
    }
    steps = 0

    for batch in loader:
        real_rgb = batch["real_minimap"].to(device)
        target_hillshade = batch["synthetic_hillshade"].to(device)

        optimizer.zero_grad()
        pred = model(real_rgb)
        loss, metrics = criterion(pred, target_hillshade)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in metrics.items():
            running_losses[k] += v
        steps += 1

        if max_batches and steps >= max_batches:
            break

    return {k: v / max(1, steps) for k, v in running_losses.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: DomainTranslationLoss,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    running_losses: dict[str, float] = {
        "loss_total": 0.0,
        "loss_l1": 0.0,
        "loss_grad": 0.0,
        "loss_ssim": 0.0,
    }
    steps = 0

    for batch in loader:
        real_rgb = batch["real_minimap"].to(device)
        target_hillshade = batch["synthetic_hillshade"].to(device)

        pred = model(real_rgb)
        _, metrics = criterion(pred, target_hillshade)

        for k, v in metrics.items():
            running_losses[k] += v
        steps += 1

        if max_batches and steps >= max_batches:
            break

    return {k: v / max(1, steps) for k, v in running_losses.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("output/datasets/v60_clean_pairs.zarr"),
        help="Path to clean paired Zarr dataset",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/domain_translation"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute 1 mini-epoch check without saving or full run",
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    logger.info(f"Target execution device: {device}")

    # Build model
    model = DomainTranslationUNet(in_channels=3, out_channels=3).to(device)
    param_count = model.count_parameters()
    logger.info(f"Initialized DomainTranslationUNet with {param_count:,} parameters (Budget: <= 10M)")
    assert param_count <= 10_000_000, f"Model parameters {param_count} exceed 10M budget!"

    criterion = DomainTranslationLoss(l1_weight=1.0, grad_weight=0.5, ssim_weight=0.3, in_channels=3).to(device)

    # Check dataset existence
    if not args.dataset.exists():
        if args.dry_run:
            logger.warning(f"Dataset {args.dataset} not found; creating in-memory synthetic mini-batch for dry-run validation.")
            # Mock dry-run step
            mock_real = torch.rand(2, 3, 256, 256, device=device)
            mock_tgt = torch.rand(2, 3, 256, 256, device=device)
            pred = model(mock_real)
            loss, metrics = criterion(pred, mock_tgt)
            loss.backward()
            logger.info(f"Synthetic dry-run forward/backward passed! Loss metrics: {metrics}")
            print(json.dumps({"dry_run": True, "status": "PASS", "param_count": param_count, "metrics": metrics}, indent=2))
            return 0
        else:
            logger.error(f"Dataset {args.dataset} does not exist! Run scripts/v60_batch_export_clean_pairs.py first.")
            return 1

    train_ds = CleanPairDataset(args.dataset, split="train", augment=True)
    val_ds = CleanPairDataset(args.dataset, split="val", augment=False)
    logger.info(f"Loaded dataset: {len(train_ds)} train tiles, {len(val_ds)} val tiles")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("Executing dry-run epoch (max 2 batches)...")
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, max_batches=2)
        val_metrics = evaluate(model, val_loader, criterion, device, max_batches=2)
        report = {
            "dry_run": True,
            "param_count": param_count,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "status": "PASS",
        }
        print(json.dumps(report, indent=2))
        return 0

    best_val_loss = float("inf")
    logger.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] ({elapsed:.1f}s) "
            f"Train L1: {train_metrics['loss_l1']:.4f} SSIM: {train_metrics['loss_ssim']:.4f} | "
            f"Val L1: {val_metrics['loss_l1']:.4f} SSIM: {val_metrics['loss_ssim']:.4f} Total: {val_metrics['loss_total']:.4f}"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            ckpt_path = args.checkpoint_dir / "domain_translation_unet_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "param_count": param_count,
                },
                ckpt_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {ckpt_path.resolve()} (Val Total: {best_val_loss:.4f})")

    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
