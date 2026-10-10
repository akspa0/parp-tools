"""Train Minimap-to-WDL Neural Model on Development Ground Truth & Zarr Stores (Spec 268).

Trains MinimapToWdlNet with decoupled shape/span heads to synthesize authentic
17x17 WDL elevation lattices with physical vertical relief scale [50, 600+] yards
for minimap-only tiles (such as development_17_14).

Usage:
    uv run python scripts/train_minimap_to_wdl.py --epochs 35 --batch-size 32
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

from harvester.v60.minimap_to_wdl_model import (
    MinimapToWdlNet,
    MultiCorpusWdlDataset,
    minimap_wdl_composite_loss,
)


def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two flat arrays."""
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x < 1e-5 or std_y < 1e-5:
        return 0.0
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MinimapToWdlNet on Multi-Corpus Datasets")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel width")
    parser.add_argument(
        "--dev-corpus",
        type=Path,
        default=Path("../output/datasets/development_elevation_corpus.npz"),
        help="Path to development elevation corpus",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("../output/datasets/v50/v50.1"),
        help="Directory containing 0_5_3 Zarr datasets",
    )
    parser.add_argument(
        "--out-checkpoint",
        type=Path,
        default=Path("checkpoints/minimap_to_wdl_best.pt"),
        help="Path to save best model checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("Spec 268: Minimap to WDL Neural Elevation Model Training")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} (VRAM: {vram_gb:.1f} GB)")
    print("=================================================================")

    # 1. Resolve corpora paths
    dev_corpus_path = (Path(__file__).resolve().parent.parent / args.dev_corpus).resolve()
    base_ds_dir = (Path(__file__).resolve().parent.parent / args.datasets_dir).resolve()
    candidate_stores = [
        base_ds_dir / "0_5_3_3368-Azeroth.zarr",
        base_ds_dir / "0_5_3_3368-Kalimdor.zarr",
    ]
    valid_stores = [p for p in candidate_stores if p.exists()]

    # 2. Build multi-corpus dataset
    full_ds = MultiCorpusWdlDataset(
        dev_corpus_path=dev_corpus_path if dev_corpus_path.is_file() else None,
        zarr_store_paths=valid_stores,
        augment=False,
    )
    total_samples = len(full_ds)
    print(f"Total Available Samples across corpora: {total_samples}")
    if total_samples == 0:
        print("Error: No training samples loaded.", file=sys.stderr)
        return 1

    rng = np.random.RandomState(42)
    shuffled_idx = rng.permutation(total_samples)
    val_count = max(32, int(total_samples * 0.15))
    train_idx = shuffled_idx[val_count:].tolist()
    val_idx = shuffled_idx[:val_count].tolist()

    print(f"Split: {len(train_idx)} train samples, {len(val_idx)} validation samples")

    train_ds = MultiCorpusWdlDataset(
        dev_corpus_path=dev_corpus_path if dev_corpus_path.is_file() else None,
        zarr_store_paths=valid_stores,
        indices=train_idx,
        augment=True,
    )
    val_ds = MultiCorpusWdlDataset(
        dev_corpus_path=dev_corpus_path if dev_corpus_path.is_file() else None,
        zarr_store_paths=valid_stores,
        indices=val_idx,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Instantiate Model
    model = MinimapToWdlNet(
        in_channels=6,
        base_channels=args.base_channels,
    ).to(device)

    print(f"Model Parameters: {model.parameter_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_ckpt_path = (Path(__file__).resolve().parent.parent / args.out_checkpoint).resolve()
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_val_r = 0.0

    # 4. Training Loop
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_shapes = []
        train_spans = []

        for batch in train_loader:
            features = batch["features"].to(device)  # (B, 4, 256, 256)
            gt_h17 = batch["h17"].to(device)        # (B, 17, 17)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_h17, pred_shape, pred_span, pred_base = model(features)
                loss, metrics = minimap_wdl_composite_loss(
                    pred_h17=pred_h17,
                    pred_shape=pred_shape,
                    pred_span=pred_span,
                    pred_base=pred_base,
                    gt_h17=gt_h17,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(metrics["total_loss"])
            train_shapes.append(metrics["shape_l1"])
            train_spans.append(metrics["span_loss"])

        scheduler.step()

        # 5. Validation Loop
        model.eval()
        val_maes = []
        val_rs = []
        val_spans = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                gt_h17 = batch["h17"].to(device)

                pred_h17, _, pred_span, _ = model(features)

                preds_np = pred_h17.cpu().numpy()
                gts_np = gt_h17.cpu().numpy()
                spans_np = pred_span.cpu().numpy()

                for b in range(preds_np.shape[0]):
                    p_b = preds_np[b]
                    g_b = gts_np[b]
                    mae = float(np.mean(np.abs(p_b - g_b)))
                    r = compute_pearson_r(p_b, g_b)

                    val_maes.append(mae)
                    val_rs.append(r)
                    val_spans.append(float(spans_np[b]))

        mean_val_mae = float(np.mean(val_maes))
        mean_val_r = float(np.mean(val_rs))
        mean_pred_span = float(np.mean(val_spans))

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {np.mean(train_losses):.2f} | "
            f"Shape L1: {np.mean(train_shapes):.4f} | "
            f"Span Loss: {np.mean(train_spans):.1f} | "
            f"Val MAE: {mean_val_mae:.2f} yds | "
            f"Val r: {mean_val_r:.4f} | "
            f"Mean Pred Span: {mean_pred_span:.1f} yds",
            flush=True,
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
                    "mean_span": mean_pred_span,
                    "base_channels": args.base_channels,
                    "in_channels": 6,
                    "model_name": "MinimapToWdlNet_v2",
                },
                out_ckpt_path,
            )

    elapsed = time.time() - t_start
    print("=================================================================")
    print(f"Spec 268 Multi-Corpus WDL Model Training Complete in {elapsed:.1f}s!")
    print(f"Best Validation MAE:       {best_val_mae:.2f} yards")
    print(f"Best Validation Pearson r: {best_val_r:.4f}")
    print(f"Saved Checkpoint:          {out_ckpt_path}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
