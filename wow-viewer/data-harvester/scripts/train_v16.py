"""Train V16 terrain model — minimap → terrain mesh from Zarr stores.

Uses V16Dataset (consolidated Zarr) with V16Model architecture (ConvNeXt V2 Nano
encoder + U-Net decoder + liquid-mask head). The model is the same; only the data
pipeline changed (Zarr instead of individual NPZ shards).

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/train_v16.py --builds 3_3_5_12340
    uv run python scripts/train_v16.py --builds 3_3_5_12340 4_0_0_11927
    uv run python scripts/train_v16.py --run-name my_experiment --builds 3_3_5_12340
    uv run python scripts/train_v16.py --resume-checkpoint models/v16/runs/<run>/checkpoints/v16_best.pt
    uv run python scripts/train_v16.py --run-name my_experiment --resume-from auto
    uv run python scripts/train_v16.py --target-vram-gb 8 --gpu-duty-cycle 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image as _PILImage  # noqa: E402
from PIL import ImageDraw as _PILImageDraw  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.data import Sampler  # noqa: E402

from harvester.v16_model import V16Model  # noqa: E402
from harvester.v16_dataset import V16Dataset  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT = _PROJECT_ROOT / "models"
DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_PANEL_SIZE = 256


class _DeterministicEpochSampler(Sampler[int]):
    """Deterministic no-replacement sampler with per-epoch order evidence."""

    def __init__(self, n: int, seed: int, order_log_path: Path | None = None) -> None:
        self._n = int(n)
        self._seed = int(seed)
        self._epoch = 0
        self._order_log_path = order_log_path

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        order = torch.randperm(self._n, generator=g).tolist()
        if self._order_log_path is not None:
            payload = {
                "epoch": self._epoch,
                "num_samples": self._n,
                "order_sha256": hashlib.sha256(np.asarray(order, dtype=np.int32).tobytes()).hexdigest(),
                "order": order,
            }
            with self._order_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        return iter(order)

    def __len__(self) -> int:
        return self._n


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
    p.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader worker processes alive between epochs (effective when --num-workers > 0)",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker (effective when --num-workers > 0)",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument(
        "--target-vram-gb",
        type=float,
        default=0.0,
        help="Soft VRAM target for guidance logs (0 disables guidance)",
    )
    p.add_argument(
        "--gpu-duty-cycle",
        type=float,
        default=100.0,
        help="Approximate max GPU active duty cycle percentage via step throttling (1-100)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1,
                    help="Fraction of data held out for validation")
    p.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Run scalar validation every N epochs (best checkpoint updates only when validation runs)",
    )
    p.add_argument("--val-snapshots", type=int, default=5,
                    help="Number of validation tiles to export as images per val run")
    p.add_argument(
        "--val-snapshot-interval",
        type=int,
        default=10,
        help="Export validation snapshot images every N epochs (0 disables snapshot export)",
    )
    p.add_argument(
        "--val-overview-columns",
        type=int,
        default=12,
        help="Column count for labeled validation overview image",
    )
    p.add_argument(
        "--train-max-tiles",
        type=int,
        default=0,
        help="If >0, randomly curate this many training tiles from the split",
    )
    p.add_argument(
        "--val-max-tiles",
        type=int,
        default=0,
        help="If >0, randomly curate this many validation tiles from the split",
    )
    p.add_argument(
        "--curation-seed",
        type=int,
        default=None,
        help="Seed for train/val subset curation (defaults to --seed)",
    )
    p.add_argument(
        "--include-placeholder-map-tiles",
        action="store_true",
        help="Allow curation to include rows with placeholder map labels (e.g. map=memory)",
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", action="store_false", dest="augment")
    p.add_argument("--run-name", type=str, default=None,
                    help="Name for this run (auto-generated from timestamp if omitted)")
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument(
        "--resume-from",
        type=str,
        choices=["none", "auto", "last", "best"],
        default="none",
        help="Resume mode from run checkpoints directory when --resume-checkpoint is not provided",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (useful for CPU-only or limited toolchains)",
    )
    return p.parse_args()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _is_placeholder_map_name(name: Any) -> bool:
    s = str(name or "").strip().lower()
    return s in {"", "memory", "<memory>", "unknown", "<unknown>"}


def _to_curation_row(entry: dict[str, Any], split: str, subset_pos: int) -> dict[str, Any]:
    return {
        "split": split,
        "subset_pos": subset_pos,
        "build": entry.get("_build"),
        "tile_id": _as_int(entry.get("tile_id")),
        "map": entry.get("map"),
        "tile_x": _as_int(entry.get("tile_x")),
        "tile_y": _as_int(entry.get("tile_y")),
        "height_mean": float(entry.get("height_mean", 0.0)),
        "height_std": float(entry.get("height_std", 0.0)),
    }


def _curate_split(
    ds: V16Dataset,
    split: str,
    max_tiles: int,
    seed: int,
    evidence_dir: Path,
    include_placeholder_map_tiles: bool,
) -> dict[str, Any]:
    full_indices = list(ds._indices)
    n_full = len(full_indices)
    candidate_indices = full_indices
    dropped_placeholder = 0
    if not include_placeholder_map_tiles:
        candidate_indices = []
        for gi in full_indices:
            entry = ds._index_entries[gi]
            if _is_placeholder_map_name(entry.get("map")):
                dropped_placeholder += 1
                continue
            candidate_indices.append(gi)

    if not candidate_indices:
        raise RuntimeError(
            f"{split} curation has zero candidates after filtering. "
            "Use --include-placeholder-map-tiles to bypass this guard or repair/rebuild the affected dataset."
        )

    n_candidates = len(candidate_indices)
    if max_tiles > 0 and max_tiles < n_candidates:
        rng = np.random.RandomState(seed)
        chosen_local = sorted(rng.choice(n_candidates, size=max_tiles, replace=False).tolist())
        mode = "random_subset_no_replace"
    else:
        chosen_local = list(range(n_candidates))
        mode = "all_tiles"

    chosen_global = [candidate_indices[i] for i in chosen_local]
    ds._indices = chosen_global

    rows = []
    for i, global_idx in enumerate(chosen_global):
        rows.append(_to_curation_row(ds._index_entries[global_idx], split=split, subset_pos=i))

    jsonl_path = evidence_dir / f"{split}_selection.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "split": split,
        "mode": mode,
        "seed": int(seed),
        "available_tiles": n_full,
        "candidate_tiles_after_filters": n_candidates,
        "dropped_placeholder_map_tiles": dropped_placeholder,
        "selected_tiles": len(chosen_global),
        "selection_jsonl": str(jsonl_path),
        "selection_sha256": digest,
    }


def _resolve_resume_checkpoint(args: argparse.Namespace, ckpt_dir: Path) -> Path | None:
    if args.resume_checkpoint is not None:
        return args.resume_checkpoint

    mode = str(args.resume_from or "none")
    if mode == "none":
        return None

    best_path = ckpt_dir / "v16_best.pt"
    last_path = ckpt_dir / "v16_last.pt"
    if mode == "best":
        return best_path if best_path.exists() else None
    if mode == "last":
        return last_path if last_path.exists() else None

    # auto
    if last_path.exists():
        return last_path
    if best_path.exists():
        return best_path
    return None


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_val: float,
    log_entries: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_height": float(best_val),
            "training_log": log_entries,
        },
        path,
    )


def main() -> None:
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = MODELS_ROOT / "v16" / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    val_dir = run_dir / "validation"
    evidence_dir = run_dir / "evidence"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "v16",
        "run_name": run_name,
        "dataset_dir": str(args.dataset_dir),
        "builds": args.builds,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "target_vram_gb": args.target_vram_gb,
        "gpu_duty_cycle": args.gpu_duty_cycle,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "val_interval": args.val_interval,
        "val_snapshots": args.val_snapshots,
        "val_snapshot_interval": args.val_snapshot_interval,
        "val_overview_columns": args.val_overview_columns,
        "train_max_tiles": args.train_max_tiles,
        "val_max_tiles": args.val_max_tiles,
        "curation_seed": args.curation_seed,
        "include_placeholder_map_tiles": args.include_placeholder_map_tiles,
        "no_amp": args.no_amp,
        "augment": args.augment,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "resume_from": args.resume_from,
        "no_compile": args.no_compile,
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
    print(
        f"DataLoader: workers={args.num_workers} "
        f"persistent_workers={bool(args.persistent_workers and args.num_workers > 0)} "
        f"prefetch_factor={(args.prefetch_factor if args.num_workers > 0 else 'n/a')}"
    )
    if device.type == "cuda":
        print(
            f"GPU budget controls: target_vram_gb={args.target_vram_gb:.2f} "
            f"gpu_duty_cycle={args.gpu_duty_cycle:.1f}%"
        )

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
    curation_seed = args.curation_seed if args.curation_seed is not None else args.seed
    train_cur = _curate_split(
        train_ds,
        "train",
        args.train_max_tiles,
        curation_seed + 101,
        evidence_dir,
        include_placeholder_map_tiles=args.include_placeholder_map_tiles,
    )
    val_cur = _curate_split(
        val_ds,
        "val",
        args.val_max_tiles,
        curation_seed + 202,
        evidence_dir,
        include_placeholder_map_tiles=args.include_placeholder_map_tiles,
    )

    curation_manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "builds": args.builds,
        "val_fraction": args.val_fraction,
        "split_seed": args.seed,
        "curation_seed": curation_seed,
        "train": train_cur,
        "val": val_cur,
    }
    (evidence_dir / "curation_manifest.json").write_text(json.dumps(curation_manifest, indent=2), encoding="utf-8")
    config["curation_manifest"] = str(evidence_dir / "curation_manifest.json")
    config["train_selected_tiles"] = int(train_cur["selected_tiles"])
    config["val_selected_tiles"] = int(val_cur["selected_tiles"])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(
        "Curated tiles: "
        f"train={train_cur['selected_tiles']}/{train_cur['available_tiles']} "
        f"val={val_cur['selected_tiles']}/{val_cur['available_tiles']}"
    )

    train_order_log = evidence_dir / "train_epoch_orders.jsonl"
    train_sampler = _DeterministicEpochSampler(len(train_ds), seed=args.seed, order_log_path=train_order_log)
    if args.prefetch_factor < 1:
        raise RuntimeError("--prefetch-factor must be >= 1")
    if args.val_interval < 1:
        raise RuntimeError("--val-interval must be >= 1")
    if args.val_snapshot_interval < 0:
        raise RuntimeError("--val-snapshot-interval must be >= 0")
    _loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        _loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        _loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        **_loader_kwargs,
    )

    model = V16Model().to(device)
    n = model.count_parameters()
    print(f"Parameters: {n:,}")
    can_compile = hasattr(torch, "compile") and not args.no_compile and device.type == "cuda"
    if can_compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as ex:
            print(f"torch.compile: disabled (compile failed: {ex})")
    else:
        print("torch.compile: disabled")

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
    log_entries: list[dict[str, Any]] = []
    resume_path = _resolve_resume_checkpoint(args, ckpt_dir)
    config["resume_resolved_checkpoint"] = str(resume_path) if resume_path is not None else None
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    if resume_path is not None:
        if not resume_path.exists():
            raise RuntimeError(f"Requested resume checkpoint does not exist: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val_height", ckpt.get("val_height", float("inf"))))
        if isinstance(ckpt.get("training_log"), list):
            log_entries = ckpt["training_log"]
        else:
            log_path = run_dir / "training_log.json"
            if log_path.exists():
                try:
                    loaded = json.loads(log_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        log_entries = loaded
                except Exception:
                    log_entries = []
        if start_epoch > args.epochs:
            print(
                f"Resume checkpoint epoch={ckpt['epoch']} already reached/exceeded requested "
                f"--epochs={args.epochs}; nothing to do."
            )
            config["finished_at"] = datetime.now().isoformat()
            config["best_val_height"] = best_val
            config["resume_noop"] = True
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))
            return
        print(f"Resumed from {resume_path} -> start_epoch={start_epoch}, best_val_h={best_val:.4f}")
    else:
        if train_order_log.exists():
            train_order_log.unlink()
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_h = 0.0
        epoch_n = 0.0
        epoch_a = 0.0
        epoch_ho = 0.0
        epoch_lq = 0.0
        epoch_mc = 0.0
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch in train_loader:
            step_t0 = time.perf_counter()
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

                loss_h = _weighted_l1_per_sample(pred_h, hgt, wgt).mean()
                loss_n = _masked_mean(_cosine_loss_per_sample(pred_n, nrm, nm_mask), has_n)
                loss_a = _masked_mean(_weighted_l1_per_sample(pred_a, alp, wgt), has_a)
                loss_ho = _masked_mean(_weighted_l1_per_sample(pred_ho, hol, wgt), has_ho)
                loss_lq = _masked_mean(_weighted_l1_per_sample(pred_lq, liq, wgt), has_lq)

                B = pred_mc.size(0)
                pred_mc_r = pred_mc.view(B, 4, 16, 16, 16).permute(0, 1, 4, 2, 3)
                loss_mc = torch.nn.functional.cross_entropy(
                    pred_mc_r.reshape(-1, 16),
                    mly.permute(0, 3, 1, 2).reshape(-1),
                    reduction="none",
                ).view(B, 4, 16, 16)
                mc_mask = mlm.permute(0, 3, 1, 2)
                loss_mc = _masked_mean(
                    (loss_mc * mc_mask).sum(dim=(1, 2, 3)) / (mc_mask.sum(dim=(1, 2, 3)) + 1e-8),
                    has_mc,
                )

                loss = loss_h + 2.0 * loss_n + loss_a + loss_ho + loss_lq + 0.3 * loss_mc

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Thermal guardrail: approximate duty-cycle throttling.
            if device.type == "cuda" and args.gpu_duty_cycle < 100.0:
                duty = max(1.0, min(100.0, float(args.gpu_duty_cycle)))
                active_s = time.perf_counter() - step_t0
                target_total_s = active_s / (duty / 100.0)
                sleep_s = max(0.0, target_total_s - active_s)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

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
        peak_alloc_gb = None
        peak_reserved_gb = None
        if device.type == "cuda":
            peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024.0 ** 3)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"h={epoch_h / n_bt:.4f} n={epoch_n / n_bt:.4f} "
            f"a={epoch_a / n_bt:.4f} ho={epoch_ho / n_bt:.4f} "
            f"lq={epoch_lq / n_bt:.4f} mc={epoch_mc / n_bt:.4f} "
            f"lr={lr_now:.2e} {elapsed:.1f}s"
        )
        if peak_reserved_gb is not None:
            print(
                f"        cuda_mem | alloc_peak={peak_alloc_gb:.2f}GB "
                f"reserved_peak={peak_reserved_gb:.2f}GB"
            )
            if args.target_vram_gb > 0:
                target = float(args.target_vram_gb)
                if peak_reserved_gb < target * 0.70:
                    suggested_bs = max(
                        args.batch_size + 1,
                        int(round(args.batch_size * (target / max(peak_reserved_gb, 1e-6)))),
                    )
                    print(
                        f"        tuning | below target_vram_gb={target:.2f}; "
                        f"consider batch-size ~{suggested_bs}"
                    )
                elif peak_reserved_gb > target * 1.05:
                    print(
                        f"        tuning | above target_vram_gb={target:.2f}; "
                        f"consider reducing batch-size"
                    )

        entry = {
            "epoch": epoch, "train_h": epoch_h / n_bt, "train_n": epoch_n / n_bt,
            "train_a": epoch_a / n_bt, "train_ho": epoch_ho / n_bt,
            "train_lq": epoch_lq / n_bt,
            "train_mc": epoch_mc / n_bt, "lr": lr_now,
        }
        if peak_alloc_gb is not None and peak_reserved_gb is not None:
            entry["cuda_peak_alloc_gb"] = peak_alloc_gb
            entry["cuda_peak_reserved_gb"] = peak_reserved_gb

        if epoch % args.val_interval == 0 and len(val_ds) > 0:
            v = _validate(model, val_loader, device, args.no_amp)
            entry.update(v)
            print(
                f"        val | h={v['val_h']:.4f} n={v['val_n']:.4f} "
                f"a={v['val_a']:.4f} lq={v['val_lq']:.4f} mc={v['val_mc']:.4f}"
            )
            if v["val_h"] < best_val:
                best_val = v["val_h"]
                _save_training_checkpoint(
                    ckpt_dir / "v16_best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val=best_val,
                    log_entries=log_entries + [entry],
                )
                print(f"        *** new best val_h={best_val:.4f}")

            if (
                args.val_snapshots > 0
                and args.val_snapshot_interval > 0
                and epoch % args.val_snapshot_interval == 0
            ):
                snap_dir = val_dir / f"epoch_{epoch:04d}"
                snap_dir.mkdir(parents=True, exist_ok=True)
                _save_val_snapshots(
                    model,
                    val_loader,
                    device,
                    snap_dir,
                    args.val_snapshots,
                    epoch,
                    args.val_overview_columns,
                )

        log_entries.append(entry)
        _save_training_checkpoint(
            ckpt_dir / "v16_last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val=best_val,
            log_entries=log_entries,
        )
        print("        checkpoint | wrote v16_last.pt")
        (run_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2))

    _save_training_checkpoint(
        ckpt_dir / "v16_final.pt",
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_val=best_val,
        log_entries=log_entries,
    )
    config["finished_at"] = datetime.now().isoformat()
    config["best_val_height"] = best_val
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Done. Run dir: {run_dir}")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)


def _weighted_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs().mean(dim=1, keepdim=True)
    if diff.shape[2:] != weight.shape[2:]:
        weight = F.interpolate(weight, size=diff.shape[2:], mode="bilinear", align_corners=False)
    numer = (diff * weight).sum(dim=(1, 2, 3))
    denom = weight.sum(dim=(1, 2, 3)) + 1e-8
    return numer / denom


def _cosine_loss_per_sample(pred: torch.Tensor, target: torch.Tensor, normal_mask: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=1)
    cos_sim = F.cosine_similarity(pred_n, target, dim=1)
    mask = normal_mask.squeeze(1)
    if cos_sim.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(1), size=cos_sim.shape[-2:], mode="nearest").squeeze(1)
    numer = ((1.0 - cos_sim) * mask).sum(dim=(1, 2))
    denom = mask.sum(dim=(1, 2)) + 1e-8
    return numer / denom


def _to_rgb_panel(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    rng = hi - lo if abs(hi - lo) > 1e-8 else 1.0
    arr_n = np.clip((arr - lo) / rng, 0, 1)
    if arr_n.ndim == 2:
        panel = np.repeat((arr_n[..., None] * 255).astype(np.uint8), 3, axis=2)
    else:
        panel = (arr_n * 255).astype(np.uint8)
    return panel


def _draw_label(panel: np.ndarray, text: str) -> np.ndarray:
    img = _PILImage.fromarray(panel, "RGB")
    drw = _PILImageDraw.Draw(img)
    drw.rectangle([(0, 0), (img.width, 17)], fill=(0, 0, 0))
    drw.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(img)


def _save_labeled_overview(rows: list[dict[str, Any]], out_path: Path, cols: int) -> None:
    if not rows:
        return

    max_cols = max(len(row["panels"]) for row in rows)
    n_cols = max(cols, max_cols, 1)
    panel_w = _PANEL_SIZE
    panel_h = _PANEL_SIZE
    row_h = panel_h + 20
    total_rows = len(rows)
    canvas = _PILImage.new("RGB", (n_cols * panel_w, total_rows * row_h), (25, 25, 25))
    draw = _PILImageDraw.Draw(canvas)

    for r, row in enumerate(rows):
        y0 = r * row_h
        title = row["title"]
        draw.rectangle([(0, y0), (canvas.width, y0 + 19)], fill=(12, 12, 12))
        draw.text((4, y0 + 4), title, fill=(240, 240, 240))
        for c, panel in enumerate(row["panels"]):
            x0 = c * panel_w
            canvas.paste(_PILImage.fromarray(panel, "RGB"), (x0, y0 + 20))

    canvas.save(out_path)


def _batch_value(batch: dict[str, Any], key: str, idx: int, default: Any) -> Any:
    value = batch.get(key)
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        if idx < value.shape[0]:
            item = value[idx]
            return item.item() if torch.is_tensor(item) and item.ndim == 0 else item
        return default
    if isinstance(value, (list, tuple)):
        return value[idx] if idx < len(value) else default
    return value


@torch.no_grad()
def _save_val_snapshots(model, loader, device, out_dir, n_samples, epoch, overview_cols):
    """Save per-tile PNGs plus one labeled validation overview image."""
    model.eval()
    count = 0
    overview_rows: list[dict[str, Any]] = []
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

            map_name = str(_batch_value(batch, "meta_map", i, "unknown"))
            tile_id = int(_batch_value(batch, "meta_tile_id", i, -1))
            tile_x = int(_batch_value(batch, "meta_tile_x", i, -1))
            tile_y = int(_batch_value(batch, "meta_tile_y", i, -1))

            h_lo = min(float(hgt.min()), float(pred_h.min()))
            h_hi = max(float(hgt.max()), float(pred_h.max()))
            panels = [
                _draw_label(_to_rgb_panel(batch["input"][i].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0), "input/minimap"),
                _draw_label(_to_rgb_panel(hgt, h_lo, h_hi), "height gt"),
                _draw_label(_to_rgb_panel(pred_h, h_lo, h_hi), "height pred"),
                _draw_label(_to_rgb_panel((nrm + 1.0) / 2.0, 0.0, 1.0), "normals gt"),
                _draw_label(_to_rgb_panel((pred_n + 1.0) / 2.0, 0.0, 1.0), "normals pred"),
                _draw_label(_to_rgb_panel(alp[:, :, 0], 0.0, 1.0), "alpha gt ch0"),
                _draw_label(_to_rgb_panel(pred_a[:, :, 0], 0.0, 1.0), "alpha pred ch0"),
                _draw_label(_to_rgb_panel(liq_gt, 0.0, 1.0), "liquid mask gt"),
                _draw_label(_to_rgb_panel(pred_lq, 0.0, 1.0), "liquid mask pred"),
                _draw_label(_to_rgb_panel(wgt, 0.0, 1.0), "terrain weight"),
            ]
            overview_rows.append(
                {
                    "title": (
                        f"tile {count:02d} map={map_name} tile_id={tile_id} "
                        f"xy=({tile_x},{tile_y}) h_l1={tile_metrics['height_l1']:.4f}"
                    ),
                    "panels": panels,
                }
            )
            count += 1
        if count >= n_samples:
            break
    _save_labeled_overview(overview_rows, out_dir / "validation_overview.png", int(overview_cols))
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
            total_h += _weighted_l1_per_sample(pred_h, hgt, wgt).mean().item()
            total_n += _masked_mean(_cosine_loss_per_sample(pred_n, nrm, nm_mask), has_n).item()
            total_a += _masked_mean(_weighted_l1_per_sample(pred_a, alp, wgt), has_a).item()
            total_lq += _masked_mean(_weighted_l1_per_sample(pred_lq, liq, wgt), has_lq).item()
            B = pred_mc.size(0)
            pred_mc_r = pred_mc.view(B, 4, 16, 16, 16).permute(0, 1, 4, 2, 3)
            loss_mc = torch.nn.functional.cross_entropy(
                pred_mc_r.reshape(-1, 16),
                mly.permute(0, 3, 1, 2).reshape(-1),
                reduction="none",
            ).view(B, 4, 16, 16)
            mc_mask = mlm.permute(0, 3, 1, 2)
            total_mc += _masked_mean(
                (loss_mc * mc_mask).sum(dim=(1, 2, 3)) / (mc_mask.sum(dim=(1, 2, 3)) + 1e-8),
                has_mc,
            ).item()
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
