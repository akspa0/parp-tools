"""Train V14 Model R1: Terrain Reconstruction.

Predicts height (257×257), hole mask (16×16), and liquid mask (256×256)
from the residual image (real − synthetic minimap).

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/train_r1.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from harvester.r1_dataset import R1Dataset  # noqa: E402
from harvester.r1_model import R1UNet  # noqa: E402

DEFAULT_SHARD_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "datasets" / "d1_reharvest" / "shards"
)
DEFAULT_VALIDATION_JSON = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "datasets" / "full_shard_batch_staged_native"
    / "manifests" / "validation_selection.json"
)
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V14 Model R1")
    p.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    p.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION_JSON)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-interval", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    return p.parse_args()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Sobel edge filter on single-channel heightmap for edge-aware loss."""
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    gx = nn.functional.conv2d(nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), kernel_x)
    gy = nn.functional.conv2d(nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), kernel_y)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available()
        else "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ---- datasets ----
    train_ds = R1Dataset(args.shard_root, args.validation_json, split="train",
                         max_samples=args.max_samples, seed=args.seed)
    val_ds = R1Dataset(args.shard_root, args.validation_json, split="val",
                       max_samples=None, seed=args.seed)
    train_ds._ensure_index()
    val_ds._ensure_index()
    n_train = len(train_ds._eligible)
    n_val = len(val_ds._eligible)
    print(f"Training: {n_train}  Validation: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False,
                            pin_memory=(device.type == "cuda"))

    # ---- model ----
    model = R1UNet().to(device)
    print(f"Parameters: {model.count_parameters():,}")

    criterion_bce = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    start_epoch = 1
    best_val = float("inf")
    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("val_height", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ---- training ----
    log_entries: list[dict] = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_h = 0.0
        epoch_sobel = 0.0
        epoch_hol = 0.0
        epoch_liq = 0.0
        t0 = time.perf_counter()

        for _batch_idx, (residual, h_gt, hol_gt, liq_gt) in enumerate(train_loader):
            residual = residual.to(device, non_blocking=True)
            h_gt = h_gt.to(device, non_blocking=True)
            hol_gt = hol_gt.to(device, non_blocking=True)
            liq_gt = liq_gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                pred_h, pred_hol, pred_liq = model(residual)
                loss_h = nn.functional.l1_loss(pred_h, h_gt)
                loss_sobel = nn.functional.l1_loss(_sobel_edges(pred_h), _sobel_edges(h_gt))
                loss_hol = criterion_bce(pred_hol, hol_gt)
                loss_liq = criterion_bce(pred_liq, liq_gt)
                loss = loss_h + 0.1 * loss_sobel + 0.5 * loss_hol + 0.5 * loss_liq

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_h += loss_h.item()
            epoch_sobel += loss_sobel.item()
            epoch_hol += loss_hol.item()
            epoch_liq += loss_liq.item()

        scheduler.step()

        n_batches = len(train_loader)
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"h={epoch_h / n_batches:.4f}  edge={epoch_sobel / n_batches:.4f}  "
            f"hole={epoch_hol / n_batches:.4f}  liq={epoch_liq / n_batches:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
        )

        entry = {"epoch": epoch, "train_h": epoch_h / n_batches,
                 "train_sobel": epoch_sobel / n_batches,
                 "train_holes": epoch_hol / n_batches,
                 "train_liquid": epoch_liq / n_batches, "lr": lr_now}

        if epoch % args.val_interval == 0 and n_val > 0:
            v = run_validation(model, val_loader, device, criterion_bce)
            entry.update(v)
            print(f"        val | h={v['val_h']:.4f}  hole={v['val_hole']:.4f}  liq={v['val_liq']:.4f}")
            if v["val_h"] < best_val:
                best_val = v["val_h"]
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "val_height": best_val},
                    args.checkpoint_dir / "r1_best.pt",
                )
                print("        saved best checkpoint")

        log_entries.append(entry)

    torch.save(
        {"epoch": args.epochs, "model_state_dict": model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        args.checkpoint_dir / "r1_final.pt",
    )
    (args.log_dir / "r1_training_log.json").write_text(json.dumps(log_entries, indent=2))
    print(f"Done. checkpoint: {args.checkpoint_dir / 'r1_final.pt'}")


@torch.no_grad()
def run_validation(model, loader, device, criterion_bce):
    model.eval()
    total_h = 0.0
    total_hol = 0.0
    total_liq = 0.0
    n = 0
    for residual, h_gt, hol_gt, liq_gt in loader:
        residual = residual.to(device, non_blocking=True)
        h_gt = h_gt.to(device, non_blocking=True)
        hol_gt = hol_gt.to(device, non_blocking=True)
        liq_gt = liq_gt.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_h, pred_hol, pred_liq = model(residual)
            v_h = nn.functional.l1_loss(pred_h, h_gt).item()
            v_hol = criterion_bce(pred_hol, hol_gt).item()
            v_liq = criterion_bce(pred_liq, liq_gt).item()
        bs = residual.size(0)
        total_h += v_h * bs
        total_hol += v_hol * bs
        total_liq += v_liq * bs
        n += bs
    return {"val_h": total_h / n, "val_hole": total_hol / n, "val_liq": total_liq / n}


if __name__ == "__main__":
    main()
