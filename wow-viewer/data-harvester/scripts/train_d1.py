"""Train V14 Model D1: Tileset Decomposition.

D1 decomposes a 256×256 minimap into two tileset-layer colour
contributions and two alpha masks, supervised by MCAL ground truth.

Per-layer colour targets are proportionally split from the minimap
using compositor alpha weights.  Object pixels are masked from loss.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/train_d1.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from harvester.audit import audit_shards, format_audit_terminal  # noqa: E402
from harvester.d1_model import D1UNet  # noqa: E402
from harvester.dataset import D1Dataset, _build_shard_index  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V14 Model D1")
    p.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    p.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION_JSON)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-interval", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--allow-dropouts", action="store_true")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--alpha-weight", type=float, default=1.0)
    return p.parse_args()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted L1 loss — weight=0 pixels contribute nothing."""
    diff = (pred - target).abs().mean(dim=1, keepdim=True)
    return (diff * weight).sum() / (weight.sum() + 1e-8)


# ---------------------------------------------------------------------------
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

    # ---- shard index ----
    train_paths_all, val_paths_all = _build_shard_index(args.shard_root, args.validation_json)

    # ---- audit (skip if dropouts allowed) ----
    if not (args.skip_audit or args.allow_dropouts):
        report = audit_shards(train_paths_all, val_paths_all)
        print(format_audit_terminal(report))
        if report.dropouts_train:
            print("ERROR: training dropouts.  Use --allow-dropouts or fix harvest.")
            sys.exit(1)

    # ---- datasets ----
    train_ds = D1Dataset(args.shard_root, args.validation_json, split="train",
                         max_samples=args.max_samples, seed=args.seed)
    train_ds.strict = not args.allow_dropouts
    val_ds = D1Dataset(args.shard_root, args.validation_json, split="val",
                       max_samples=args.max_val_samples, seed=args.seed)
    train_ds._ensure_index()
    val_ds._ensure_index()
    n_train = len(train_ds._eligible)
    n_val = len(val_ds._eligible)
    print(f"Training: {n_train}  (dropouts excluded: {len(train_ds._dropout_indices)})")
    print(f"Validation: {n_val}")

    # ---- chain of custody ----
    manifest = {
        "schema_version": "chain-of-custody.v1",
        "model": "D1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hyperparameters": {
            "batch_size": args.batch_size, "epochs": args.epochs,
            "lr": args.lr, "seed": args.seed, "max_samples": args.max_samples,
            "alpha_weight": args.alpha_weight, "amp": not args.no_amp,
        },
        "train_eligible": n_train, "val_eligible": n_val,
    }
    (args.log_dir / "d1_chain_of_custody.json").write_text(json.dumps(manifest, indent=2))

    # ---- loaders ----
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False,
                            pin_memory=(device.type == "cuda"))

    # ---- model ----
    model = D1UNet().to(device)
    print(f"Parameters: {model.count_parameters():,}")

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
        best_val = ckpt.get("val_tileset", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ---- training ----
    log_entries: list[dict] = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_tileset = 0.0
        epoch_alpha = 0.0
        t0 = time.perf_counter()

        for _batch_idx, (inputs, t1_gt, t2_gt, a1_gt, a2_gt, weight) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            t1_gt = t1_gt.to(device, non_blocking=True)
            t2_gt = t2_gt.to(device, non_blocking=True)
            a1_gt = a1_gt.to(device, non_blocking=True)
            a2_gt = a2_gt.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                pred_t1, pred_t2, pred_a1, pred_a2 = model(inputs)
                loss_tileset = _masked_l1(pred_t1, t1_gt, weight) + _masked_l1(pred_t2, t2_gt, weight)
                loss_alpha = _masked_l1(pred_a1, a1_gt, weight) + _masked_l1(pred_a2, a2_gt, weight)
                loss = loss_tileset + args.alpha_weight * loss_alpha

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_tileset += loss_tileset.item()
            epoch_alpha += loss_alpha.item()

        scheduler.step()

        n_batches = len(train_loader)
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"ts={epoch_tileset / n_batches:.4f}  "
            f"a={epoch_alpha / n_batches:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
        )

        entry = {"epoch": epoch, "train_tileset": epoch_tileset / n_batches,
                 "train_alpha": epoch_alpha / n_batches, "lr": lr_now}

        if epoch % args.val_interval == 0 and n_val > 0:
            v_ts, v_a = run_validation(model, val_loader, device, args)
            entry["val_tileset"] = v_ts
            entry["val_alpha"] = v_a
            print(f"        val | ts={v_ts:.4f}  a={v_a:.4f}")
            if v_ts < best_val:
                best_val = v_ts
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "val_tileset": best_val},
                    args.checkpoint_dir / "d1_best.pt",
                )
                print("        saved best checkpoint")

        log_entries.append(entry)

    torch.save(
        {"epoch": args.epochs, "model_state_dict": model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        args.checkpoint_dir / "d1_final.pt",
    )
    (args.log_dir / "d1_training_log.json").write_text(json.dumps(log_entries, indent=2))
    print(f"Done. checkpoint: {args.checkpoint_dir / 'd1_final.pt'}")


@torch.no_grad()
def run_validation(model, loader, device, args):
    model.eval()
    total_ts = 0.0
    total_a = 0.0
    n = 0
    for inputs, t1_gt, t2_gt, a1_gt, a2_gt, weight in loader:
        inputs = inputs.to(device, non_blocking=True)
        t1_gt = t1_gt.to(device, non_blocking=True)
        t2_gt = t2_gt.to(device, non_blocking=True)
        a1_gt = a1_gt.to(device, non_blocking=True)
        a2_gt = a2_gt.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
            pred_t1, pred_t2, pred_a1, pred_a2 = model(inputs)
            v_ts = (_masked_l1(pred_t1, t1_gt, weight).item()
                    + _masked_l1(pred_t2, t2_gt, weight).item())
            v_a = (_masked_l1(pred_a1, a1_gt, weight).item()
                   + _masked_l1(pred_a2, a2_gt, weight).item())
        bs = inputs.size(0)
        total_ts += v_ts * bs
        total_a += v_a * bs
        n += bs
    return total_ts / n if n else 0, total_a / n if n else 0


if __name__ == "__main__":
    main()
