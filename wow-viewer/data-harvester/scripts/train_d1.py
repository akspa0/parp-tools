"""Train V14 Model D1: Tileset Decomposition.

Decomposes a 256x256 minimap into two texture-layer contributions and two alpha masks.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/train_d1.py

Paths default to the staged native shard directory and its validation selection JSON.
Before training, a full signal audit is run across all shards.  If required D1
signals are missing from any training shard, training is REFUSED.  A chain-of-custody
manifest is saved alongside the training log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure src/ is on the path so 'harvester' package is importable
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from harvester.audit import (  # noqa: E402
    audit_shards,
    audit_to_json,
    format_audit_terminal,
)
from harvester.d1_model import D1UNet  # noqa: E402
from harvester.dataset import D1Dataset, _build_shard_index  # noqa: E402

# ----------------------------------------------------------------------------
# Default paths (relative within wow-viewer/data-harvester)
# ----------------------------------------------------------------------------
DEFAULT_SHARD_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "output"
    / "datasets"
    / "full_shard_batch_staged_native"
    / "shards"
)
DEFAULT_VALIDATION_JSON = (
    Path(__file__).resolve().parent.parent.parent
    / "output"
    / "datasets"
    / "full_shard_batch_staged_native"
    / "manifests"
    / "validation_selection.json"
)
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V14 Model D1: Tileset Decomposition")
    p.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    p.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION_JSON)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-interval", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--skip-audit", action="store_true", help="Skip the pre-training signal audit (dangerous)."
    )
    p.add_argument(
        "--allow-dropouts",
        action="store_true",
        help="Allow training on shards with missing D1 signals (dangerous).",
    )
    return p.parse_args()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shard_path_to_manifest_entry(p: Path) -> dict[str, Any]:
    """Convert a shard path to a compact manifest entry."""
    from harvester.audit import _parse_shard_path

    build, map_name, tx, ty = _parse_shard_path(p)
    return {"build": build, "map": map_name, "tile_x": tx, "tile_y": ty, "path": str(p)}


def _write_chain_of_custody(
    log_dir: Path,
    train_paths: list[Path],
    val_paths: list[Path],
    audit_dict: dict[str, Any],
    args: argparse.Namespace,
    n_train_eligible: int,
    n_val_eligible: int,
) -> Path:
    """Write the chain-of-custody manifest JSON."""
    manifest = {
        "schema_version": "chain-of-custody.v1",
        "model": "D1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "amp_enabled": not args.no_amp,
        },
        "split": {
            "train": {
                "total_shards": len(train_paths),
                "d1_eligible": n_train_eligible,
                "shards": [_shard_path_to_manifest_entry(p) for p in train_paths],
            },
            "val": {
                "total_shards": len(val_paths),
                "d1_eligible": n_val_eligible,
                "shards": [_shard_path_to_manifest_entry(p) for p in val_paths],
            },
        },
        "signal_audit": audit_dict,
    }
    manifest_path = log_dir / "d1_chain_of_custody.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Build shard index (path lists, no eligibility filtering yet)
    # ------------------------------------------------------------------
    train_paths_all, val_paths_all = _build_shard_index(args.shard_root, args.validation_json)

    # ------------------------------------------------------------------
    # SIGNAL AUDIT (guard)
    # ------------------------------------------------------------------
    if not args.skip_audit:
        print("Running pre-training signal audit...")
        report = audit_shards(train_paths_all, val_paths_all)
        print(format_audit_terminal(report, model_name="D1"))
        audit_dict = audit_to_json(report)
        print()

        if report.dropouts_train and not args.allow_dropouts:
            print(
                "ERROR: Training shards have missing D1 signals. "
                "Fix the harvest pipeline and re-harvest before training."
            )
            print(f"  {len(report.dropouts_train)} training dropout(s) — see audit above.")
            print(
                "  To override (DANGEROUS), use --allow-dropouts, but the model will "
                "train on incomplete data."
            )
            sys.exit(1)

        if report.dropouts_val:
            print(
                f"WARNING: {len(report.dropouts_val)} validation shard(s) have missing "
                f"D1 signals. They will be excluded from validation."
            )
    else:
        audit_dict = {"skipped": True}

    # ------------------------------------------------------------------
    # Datasets (eligibility-filtered)
    # ------------------------------------------------------------------
    train_ds = D1Dataset(args.shard_root, args.validation_json, split="train")
    train_ds.strict = not args.allow_dropouts
    val_ds = D1Dataset(args.shard_root, args.validation_json, split="val")

    train_ds._ensure_index()
    val_ds._ensure_index()
    n_train = len(train_ds._eligible)
    n_val = len(val_ds._eligible)

    n_train_dropouts = len(train_ds._dropout_indices)
    n_val_dropouts = len(val_ds._dropout_indices)
    print(
        f"Training samples:   {n_train} eligible"
        + (f"  ({n_train_dropouts} dropouts excluded)" if n_train_dropouts else "")
    )
    print(
        f"Validation samples: {n_val} eligible"
        + (f"  ({n_val_dropouts} dropouts excluded)" if n_val_dropouts else "")
    )

    if n_train == 0:
        print("ERROR: No eligible D1 training samples. Cannot train.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Chain of Custody
    # ------------------------------------------------------------------
    manifest_path = _write_chain_of_custody(
        args.log_dir,
        train_ds.all_paths,
        val_ds.all_paths,
        audit_dict,
        args,
        n_train,
        n_val,
    )
    print(f"Chain of custody:   {manifest_path}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    model = D1UNet().to(device)
    n_params = model.count_parameters()
    print(f"Model parameters:   {n_params:,}")

    criterion_l1 = nn.L1Loss()
    criterion_bce = nn.BCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_entries: list[dict] = []
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_bce = 0.0
        t0 = time.perf_counter()

        for _batch_idx, (inputs, t1, t2, a1, a2) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            t1 = t1.to(device, non_blocking=True)
            t2 = t2.to(device, non_blocking=True)
            a1 = a1.to(device, non_blocking=True)
            a2 = a2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                pred_t1, pred_t2, pred_a1, pred_a2 = model(inputs)
                loss_l1 = criterion_l1(pred_t1, t1) + criterion_l1(pred_t2, t2)
                loss_bce = criterion_bce(pred_a1, a1) + criterion_bce(pred_a2, a2)
                loss = loss_l1 + loss_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_l1 += loss_l1.item()
            epoch_bce += loss_bce.item()

        scheduler.step()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_bce = epoch_bce / n_batches
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f}  l1={avg_l1:.4f}  bce={avg_bce:.4f}  "
            f"lr={lr_now:.2e}  time={elapsed:.1f}s"
        )

        entry: dict = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_l1": avg_l1,
            "train_bce": avg_bce,
            "lr": lr_now,
        }

        # Validation
        if epoch % args.val_interval == 0 and n_val > 0:
            val_metrics = run_validation(model, val_loader, device, args)
            entry.update(val_metrics)
            print(
                f"        val | "
                f"l1={val_metrics['val_l1']:.4f}  "
                f"bce={val_metrics['val_bce']:.4f}  "
                f"acc_a1={val_metrics['val_alpha1_acc']:.3f}  "
                f"acc_a2={val_metrics['val_alpha2_acc']:.3f}"
            )

            if val_metrics["val_l1"] < best_val_l1:
                best_val_l1 = val_metrics["val_l1"]
                checkpoint_path = args.checkpoint_dir / "d1_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_l1": best_val_l1,
                    },
                    checkpoint_path,
                )
                print(f"        saved best checkpoint -> {checkpoint_path}")

        log_entries.append(entry)

    # Final checkpoint
    final_path = args.checkpoint_dir / "d1_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_path,
    )
    print(f"Final checkpoint saved -> {final_path}")

    # Save epoch log
    log_path = args.log_dir / "d1_training_log.json"
    log_path.write_text(json.dumps(log_entries, indent=2), encoding="utf-8")
    print(f"Training log saved -> {log_path}")


@torch.no_grad()
def run_validation(
    model: D1UNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()

    total_l1 = 0.0
    total_bce = 0.0
    total_acc_a1 = 0.0
    total_acc_a2 = 0.0
    n = 0

    for inputs, t1, t2, a1, a2 in loader:
        inputs = inputs.to(device, non_blocking=True)
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        a1 = a1.to(device, non_blocking=True)
        a2 = a2.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
            pred_t1, pred_t2, pred_a1, pred_a2 = model(inputs)

        batch_l1 = (
            nn.functional.l1_loss(pred_t1, t1).item() + nn.functional.l1_loss(pred_t2, t2).item()
        )
        batch_bce = (
            nn.functional.binary_cross_entropy(pred_a1, a1).item()
            + nn.functional.binary_cross_entropy(pred_a2, a2).item()
        )

        acc_a1 = ((pred_a1 > 0.5).float() == (a1 > 0.5).float()).float().mean().item()
        acc_a2 = ((pred_a2 > 0.5).float() == (a2 > 0.5).float()).float().mean().item()

        bs = inputs.size(0)
        total_l1 += batch_l1 * bs
        total_bce += batch_bce * bs
        total_acc_a1 += acc_a1 * bs
        total_acc_a2 += acc_a2 * bs
        n += bs

    return {
        "val_l1": total_l1 / n if n else 0.0,
        "val_bce": total_bce / n if n else 0.0,
        "val_alpha1_acc": total_acc_a1 / n if n else 0.0,
        "val_alpha2_acc": total_acc_a2 / n if n else 0.0,
    }


if __name__ == "__main__":
    main()
