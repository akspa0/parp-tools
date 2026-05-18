"""Train V16 terrain model — minimap → terrain mesh from Zarr stores.

Uses V16Dataset (consolidated Zarr) with V15Model architecture (ConvNeXt V2 Nano
encoder + U-Net decoder + liquid head). The model is the same; only the data
pipeline changed (Zarr instead of individual NPZ shards).

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/train_v16.py --builds 3_3_5_12340
    uv run python scripts/train_v16.py --builds 3_3_5_12340 4_0_0_11927
    uv run python scripts/train_v16.py --run-name my_experiment --builds 3_3_5_12340
    uv run python scripts/train_v16.py --resume-checkpoint models/v16/runs/<run>/checkpoints/v16_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image as _PILImage  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from harvester.v15_model import V15Model  # noqa: E402
from harvester.v16_dataset import V16Dataset  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT = _PROJECT_ROOT / "models"
DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V16 terrain model (Zarr-based)")
    p.add_argument("--dataset-dir", type=Path, default=DATASET_ROOT,
                    help="Root directory containing .zarr stores")
    p.add_argument("--builds", nargs="+", default=["3_3_5_12340"],
                    help="Build keys to include in training")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1,
                    help="Fraction of data held out for validation")
    p.add_argument("--val-interval", type=int, default=10)
    p.add_argument("--val-snapshots", type=int, default=5,
                    help="Number of validation tiles to export as images per val run")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", action="store_false", dest="augment")
    p.add_argument("--run-name", type=str, default=None,
                    help="Name for this run (auto-generated from timestamp if omitted)")
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    return p.parse_args()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = MODELS_ROOT / "v16" / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    val_dir = run_dir / "validation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "v16",
        "run_name": run_name,
        "dataset_dir": str(args.dataset_dir),
        "builds": args.builds,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "val_interval": args.val_interval,
        "val_snapshots": args.val_snapshots,
        "no_amp": args.no_amp,
        "augment": args.augment,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "started_at": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Run: {run_name}")
    print(f"Output: {run_dir}")

    seed_all(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available()
        else "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    train_ds = V16Dataset(
        dataset_dir=args.dataset_dir,
        builds=args.builds,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=args.augment,
    )
    val_ds = V16Dataset(
        dataset_dir=args.dataset_dir,
        builds=args.builds,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=False,
    )
    print(f"Training: {len(train_ds)}  Validation: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = V15Model().to(device)
    n = model.count_parameters()
    print(f"Parameters: {n:,}")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5),
        ],
        milestones=[5],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    start_epoch = 1
    best_val = float("inf")
    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("val_height", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best val_h={best_val:.4f}")

    log_entries: list[dict] = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_h = 0.0
        epoch_n = 0.0
        epoch_a = 0.0
        epoch_ho = 0.0
        epoch_lq = 0.0
        epoch_mc = 0.0
        t0 = time.perf_counter()

        for batch in train_loader:
            inp = batch["input"].to(device, non_blocking=True)
            hgt = batch["height"].to(device, non_blocking=True)
            nrm = batch["normals"].to(device, non_blocking=True)
            nm_mask = batch["normal_mask"].to(device, non_blocking=True)
            alp = batch["alpha"].to(device, non_blocking=True)
            hol = batch["holes"].to(device, non_blocking=True)
            liq = batch["liquid"].to(device, non_blocking=True)
            mly = batch["mcly_ids"].to(device, non_blocking=True)
            mlm = batch["mcly_mask"].to(device, non_blocking=True)
            wgt = batch["weight"].to(device, non_blocking=True)
            has_n = batch["has_normals"].to(device, non_blocking=True).float()
            has_a = batch["has_alpha"].to(device, non_blocking=True).float()
            has_ho = batch["has_holes"].to(device, non_blocking=True).float()
            has_lq = batch["has_liquid"].to(device, non_blocking=True).float()
            has_mc = batch["has_mcly"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                pred_h, pred_n, pred_a, pred_ho, pred_lq, pred_mc = model(inp)

                loss_h = _weighted_l1(pred_h, hgt, wgt)
                loss_n = _cosine_loss(pred_n, nrm, nm_mask) * has_n.mean()
                loss_a = _weighted_l1(pred_a, alp, wgt) * has_a.mean()
                loss_ho = _weighted_l1(pred_ho, hol, wgt) * has_ho.mean()
                loss_lq = _weighted_l1(pred_lq, liq, wgt) * has_lq.mean()

                B = pred_mc.size(0)
                pred_mc_r = pred_mc.view(B, 4, 16, 16, 16).permute(0, 1, 4, 2, 3)
                loss_mc = torch.nn.functional.cross_entropy(
                    pred_mc_r.reshape(-1, 16),
                    mly.permute(0, 3, 1, 2).reshape(-1),
                    reduction="none",
                ).view(B, 4, 16, 16)
                mc_mask = mlm.permute(0, 3, 1, 2)
                n_active = mc_mask.sum() + 1e-8
                loss_mc = (loss_mc * mc_mask).sum() / n_active * has_mc.mean()

                loss = loss_h + 2.0 * loss_n + loss_a + loss_ho + loss_lq + 0.3 * loss_mc

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_h += loss_h.item()
            epoch_n += loss_n.item()
            epoch_a += loss_a.item()
            epoch_ho += loss_ho.item()
            epoch_lq += loss_lq.item()
            epoch_mc += loss_mc.item()

        scheduler.step()

        n_bt = len(train_loader)
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"h={epoch_h / n_bt:.4f} n={epoch_n / n_bt:.4f} "
            f"a={epoch_a / n_bt:.4f} ho={epoch_ho / n_bt:.4f} "
            f"lq={epoch_lq / n_bt:.4f} mc={epoch_mc / n_bt:.4f} "
            f"lr={lr_now:.2e} {elapsed:.1f}s"
        )

        entry = {
            "epoch": epoch, "train_h": epoch_h / n_bt, "train_n": epoch_n / n_bt,
            "train_a": epoch_a / n_bt, "train_ho": epoch_ho / n_bt,
            "train_lq": epoch_lq / n_bt, "train_mc": epoch_mc / n_bt, "lr": lr_now,
        }

        if epoch % args.val_interval == 0 and len(val_ds) > 0:
            v = _validate(model, val_loader, device, args.no_amp)
            entry.update(v)
            print(
                f"        val | h={v['val_h']:.4f} n={v['val_n']:.4f} "
                f"a={v['val_a']:.4f} lq={v['val_lq']:.4f} mc={v['val_mc']:.4f}"
            )
            if v["val_h"] < best_val:
                best_val = v["val_h"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_height": best_val,
                    },
                    ckpt_dir / "v16_best.pt",
                )
                print(f"        *** new best val_h={best_val:.4f}")

            snap_dir = val_dir / f"epoch_{epoch:04d}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            _save_val_snapshots(model, val_loader, device, snap_dir, args.val_snapshots, epoch)

        log_entries.append(entry)
        (run_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2))

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / "v16_final.pt",
    )
    config["finished_at"] = datetime.now().isoformat()
    config["best_val_height"] = best_val
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Done. Run dir: {run_dir}")


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs().mean(dim=1, keepdim=True)
    if diff.shape[2:] != weight.shape[2:]:
        weight = F.interpolate(weight, size=diff.shape[2:], mode="bilinear", align_corners=False)
    return (diff * weight).sum() / (weight.sum() + 1e-8)


def _cosine_loss(pred: torch.Tensor, target: torch.Tensor, normal_mask: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=1)
    cos_sim = F.cosine_similarity(pred_n, target, dim=1)
    mask = normal_mask.squeeze(1)
    if cos_sim.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(1), size=cos_sim.shape[-2:], mode="nearest").squeeze(1)
    return ((1.0 - cos_sim) * mask).sum() / (mask.sum() + 1e-8)


@torch.no_grad()
def _save_val_snapshots(model, loader, device, out_dir, n_samples, epoch):
    """Save side-by-side PNG comparisons for the first N validation tiles."""
    model.eval()
    count = 0
    for batch in loader:
        for i in range(batch["input"].size(0)):
            if count >= n_samples:
                break
            inp = batch["input"][i:i + 1].to(device)
            hgt = batch["height"][i].squeeze().cpu().numpy()
            nrm = batch["normals"][i].permute(1, 2, 0).cpu().numpy()
            nm_mask = batch["normal_mask"][i].squeeze().cpu().numpy()
            alp = batch["alpha"][i].permute(1, 2, 0).cpu().numpy()
            liq_gt = batch["liquid"][i].squeeze().cpu().numpy()
            wgt = batch["weight"][i].squeeze().cpu().numpy()

            pred_h_raw, pred_n, pred_a, _pred_ho, pred_lq, _pred_mc = model(inp)
            pred_h = pred_h_raw.squeeze().cpu().numpy()
            pred_n = F.normalize(pred_n, dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_a = pred_a.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_lq = pred_lq.squeeze().cpu().numpy()

            tile_dir = out_dir / f"tile_{count:02d}"
            tile_dir.mkdir(parents=True, exist_ok=True)

            _snap_save(hgt, hgt.min(), hgt.max(), tile_dir / "height_gt.png")
            _snap_save(pred_h, pred_h.min(), pred_h.max(), tile_dir / "height_pred.png")
            _snap_save((nrm + 1) / 2, 0, 1, tile_dir / "normals_gt.png")
            _snap_save((pred_n + 1) / 2, 0, 1, tile_dir / "normals_pred.png")
            _snap_save(alp[:, :, 0], 0, 1, tile_dir / "alpha_gt_ch0.png")
            _snap_save(pred_a[:, :, 0], 0, 1, tile_dir / "alpha_pred_ch0.png")
            _snap_save(wgt, 0, 1, tile_dir / "object_weight.png")
            _snap_save(nm_mask, 0, 1, tile_dir / "normal_mask.png")

            has_lq = bool(batch["has_liquid"][i]) if "has_liquid" in batch else liq_gt.max() > 0.5
            if has_lq:
                _snap_save(liq_gt, 0, 1, tile_dir / "liquid_gt.png")
                _snap_save(pred_lq, 0, 1, tile_dir / "liquid_pred.png")

            valid = nm_mask > 0.5
            if valid.any():
                pred_n_v = pred_n[valid]
                nrm_v = nrm[valid]
                cos_pp = (pred_n_v * nrm_v).sum(axis=-1) / (
                    np.linalg.norm(pred_n_v, axis=-1) * np.linalg.norm(nrm_v, axis=-1) + 1e-8
                )
                n_cos = float(cos_pp.mean())
            else:
                n_cos = 0.0
            tile_metrics = {
                "epoch": epoch, "tile": count,
                "height_l1": float(np.abs(pred_h - hgt).mean()),
                "normals_cosine": n_cos,
                "alpha_l1": float(np.abs(pred_a - alp).mean()),
                "liquid_l1": float(np.abs(pred_lq - liq_gt).mean()),
            }
            (tile_dir / "metrics.json").write_text(json.dumps(tile_metrics, indent=2))
            count += 1
        if count >= n_samples:
            break
    model.train()


def _snap_save(arr: np.ndarray, lo: float, hi: float, path: Path) -> None:
    rng = hi - lo if abs(hi - lo) > 1e-8 else 1.0
    arr = np.clip((arr - lo) / rng, 0, 1)
    if arr.ndim == 2:
        img = (arr * 255).astype(np.uint8)
        _PILImage.fromarray(img, "L").save(path)
    else:
        img = (arr * 255).astype(np.uint8)
        _PILImage.fromarray(img, "RGB").save(path)


@torch.no_grad()
def _validate(model, loader, device, no_amp: bool = False):
    model.eval()
    total_h = 0.0
    total_n = 0.0
    total_a = 0.0
    total_lq = 0.0
    total_mc = 0.0
    n = 0
    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        hgt = batch["height"].to(device, non_blocking=True)
        nrm = batch["normals"].to(device, non_blocking=True)
        nm_mask = batch["normal_mask"].to(device, non_blocking=True)
        alp = batch["alpha"].to(device, non_blocking=True)
        liq = batch["liquid"].to(device, non_blocking=True)
        mly = batch["mcly_ids"].to(device, non_blocking=True)
        mlm = batch["mcly_mask"].to(device, non_blocking=True)
        wgt = batch["weight"].to(device, non_blocking=True)
        has_n = batch["has_normals"].to(device, non_blocking=True).float()
        has_a = batch["has_alpha"].to(device, non_blocking=True).float()
        has_lq = batch["has_liquid"].to(device, non_blocking=True).float()
        has_mc = batch["has_mcly"].to(device, non_blocking=True).float()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not no_amp)):
            pred_h, pred_n, pred_a, _, pred_lq, pred_mc = model(inp)
            total_h += _weighted_l1(pred_h, hgt, wgt).item()
            total_n += (_cosine_loss(pred_n, nrm, nm_mask) * has_n.mean()).item()
            total_a += (_weighted_l1(pred_a, alp, wgt) * has_a.mean()).item()
            total_lq += (_weighted_l1(pred_lq, liq, wgt) * has_lq.mean()).item()
            B = pred_mc.size(0)
            pred_mc_r = pred_mc.view(B, 4, 16, 16, 16).permute(0, 1, 4, 2, 3)
            loss_mc = torch.nn.functional.cross_entropy(
                pred_mc_r.reshape(-1, 16),
                mly.permute(0, 3, 1, 2).reshape(-1),
                reduction="none",
            ).view(B, 4, 16, 16)
            mc_mask = mlm.permute(0, 3, 1, 2)
            n_active = mc_mask.sum() + 1e-8
            total_mc += ((loss_mc * mc_mask).sum() / n_active * has_mc.mean()).item()
        n += 1
    if n == 0:
        return {"val_h": 0.0, "val_n": 0.0, "val_a": 0.0, "val_lq": 0.0, "val_mc": 0.0}
    return {
        "val_h": total_h / n,
        "val_n": total_n / n,
        "val_a": total_a / n,
        "val_lq": total_lq / n,
        "val_mc": total_mc / n,
    }


if __name__ == "__main__":
    main()
