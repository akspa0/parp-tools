"""Spec 077 Phase 4 (US3) height-only training script.

Tiny training entry point that loads the teacher-prior dataset, runs
``V161HeightModel`` (= ``V18HeightModel``) for a bounded number of
steps, and writes previews showing the prior input, the predicted
height, the ground truth, and the loss weight.

Optimizations ported from ``train_v16_1_common.py`` (the V18 "chef's
kiss" stack):

* **AMP** (mixed precision) via ``torch.amp.GradScaler`` with a
  ``torch.amp.autocast`` forward pass.
* **torch.compile** with a graceful fallback when the toolchain
  rejects compilation (CPU-only, missing Triton, etc.).
* **Gradient clipping** at ``max_norm=1.0`` (post-AMP unscale).
* **AdamW** with explicit ``weight_decay`` instead of plain Adam.
* **``set_to_none=True``** in ``optimizer.zero_grad`` to skip the
  zero-fill allocation.
* **``non_blocking=True``** on every ``.to(device, ...)`` transfer.
* **Multi-scale masked L1 loss** at 257 / 128 / 64 / 32 / 16 px —
  shares the V18 multi-resolution trick that stabilizes small-object
  detail.
* **Optional spectral / gradient / normal-consistency losses** behind
  explicit weights (default 0 so the first proof is a pure height
  lane; FR-013 forbids joint training, but auxiliary smoothness on the
  height head is allowed by FR-014).
* **Early stopping** with patience + min-improvement.
* **Resume support** that reloads model + optimizer + scaler + step
  counter from a checkpoint.
* **Labeled preview panels** with text strips (prior / truth /
  pred / weight) instead of the 4-row free-form preview.
* **DataLoader** with ``num_workers`` / ``prefetch_factor`` /
  ``persistent_workers`` for overlap of I/O with compute.
* **Optional VRAM autotune** that probes a batch-size ladder against
  ``--target-vram-gb`` and writes a ``batch_autotune.json`` evidence
  file. No-op on CPU.
* **Deterministic seed** for reproducible smoke runs.
* **Per-step timing** with throughput (tiles/sec) reporting.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, default_collate

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.height_only_prior_dataset import HeightOnlyPriorDataset  # noqa: E402
from harvester.v18_models import V18HeightModel  # noqa: E402

_PANEL_SIZE = 256
_PANEL_LABEL_HEIGHT = 18
_DEFAULT_AUTOTUNE_BATCH_CANDIDATES = (8, 12, 16, 20, 24, 32, 40, 48, 56, 64)


# ---------------------------------------------------------------------------
# small utility helpers (port of _seed_all / _resolve_* from v18 common)
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable.")
        return torch.device("cuda")
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return int(requested)
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def _resolve_persistent_workers(requested: bool | None, num_workers: int) -> bool:
    if num_workers <= 0:
        return False
    if requested is None:
        return True
    return bool(requested)


# ---------------------------------------------------------------------------
# preview / panel helpers (port of _save_horizontal_panel + _draw_text_strip)
# ---------------------------------------------------------------------------

def _draw_text_strip(img: Image.Image, text: str, height: int) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), color=(0, 0, 0))
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (canvas.width, height - 1)], fill=(14, 14, 14))
    draw.text((4, 3), str(text), fill=(240, 240, 240))
    return canvas


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0).astype(np.uint8)


def _compose_horizontal_panel(panels: list[tuple[str, torch.Tensor]]) -> Image.Image:
    images: list[Image.Image] = []
    for label, tensor in panels:
        arr = _to_uint8_hwc(tensor)
        img = Image.fromarray(arr)
        if img.size != (_PANEL_SIZE, _PANEL_SIZE):
            img = img.resize((_PANEL_SIZE, _PANEL_SIZE), Image.Resampling.BILINEAR)
        img = _draw_text_strip(img, label, _PANEL_LABEL_HEIGHT)
        images.append(img)
    canvas = Image.new("RGB", (_PANEL_SIZE * len(images), _PANEL_SIZE + _PANEL_LABEL_HEIGHT), color=(0, 0, 0))
    for idx, img in enumerate(images):
        canvas.paste(img, (idx * _PANEL_SIZE, 0))
    return canvas


def _save_horizontal_panel(panels: list[tuple[str, torch.Tensor]], out_path: Path) -> None:
    _compose_horizontal_panel(panels).save(out_path)


# ---------------------------------------------------------------------------
# loss helpers (multi-scale + optional gradient / normal-consistency terms)
# ---------------------------------------------------------------------------

def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_map * mask).sum() / mask.sum().clamp_min(1e-8)


def _gradient_magnitude_257(x: torch.Tensor) -> torch.Tensor:
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    return 0.5 * (dx + dy)


def _multi_scale_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    ms_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-scale masked L1 at 257 / 128 / 64 / 32 / 16 px.

    ``ms_weight`` is the per-scale weight; total loss is ``ms_weight * 5`` scales
    unless ``ms_weight <= 0`` in which case a single 257-px L1 is returned.
    """
    metrics: dict[str, float] = {}
    if ms_weight <= 0.0:
        loss = _masked_mean((pred - target).abs(), mask)
        metrics["height"] = float(loss.item())
        metrics["height_mask_cov"] = float(mask.mean().item())
        return loss, metrics
    total = pred.new_zeros(())
    for scale in (257, 128, 64, 32, 16):
        if scale == 257:
            p, t, m = pred, target, mask
        else:
            p = F.interpolate(pred, size=(scale, scale), mode="bilinear", align_corners=False)
            t = F.interpolate(target, size=(scale, scale), mode="bilinear", align_corners=False)
            m = F.interpolate(mask, size=(scale, scale), mode="nearest")
        scale_loss = _masked_mean((p - t).abs(), m)
        total = total + ms_weight * scale_loss
        metrics[f"l1_{scale}"] = float(scale_loss.item())
    metrics["height"] = float(total.item())
    metrics["height_mask_cov"] = float(mask.mean().item())
    return total, metrics


def compute_height_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    ms_weight: float,
    grad_weight: float,
    nc_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Aggregate the height-only loss with optional auxiliary smoothness terms.

    ``pred``, ``target``, and ``weight`` are all ``(B, 1, 257, 257)``.
    """
    loss, metrics = _multi_scale_l1(pred, target, weight, ms_weight)
    if grad_weight > 0.0:
        pred_grad = _gradient_magnitude_257(pred)
        target_grad = _gradient_magnitude_257(target)
        grad_loss = _masked_mean((pred_grad - target_grad).abs(), weight)
        loss = loss + grad_weight * grad_loss
        metrics["grad_loss"] = float(grad_loss.item())
    if nc_weight > 0.0:
        # Smoothness prior: encourage locally constant height (z-up) inside
        # the train mask. This is a self-supervised normal-consistency proxy
        # without ever predicting normals as a separate head.
        diff_x = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
        diff_y = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
        diff_x = F.pad(diff_x, (0, 1, 0, 0), mode="replicate")
        diff_y = F.pad(diff_y, (0, 0, 0, 1), mode="replicate")
        flatness = 0.5 * (diff_x + diff_y)
        nc_loss = _masked_mean(flatness, weight)
        loss = loss + nc_weight * nc_loss
        metrics["nc_loss"] = float(nc_loss.item())
    return loss, metrics


# ---------------------------------------------------------------------------
# VRAM autotune (port of _autotune_batch_size, height-only subset)
# ---------------------------------------------------------------------------

def _autotune_batch_size(
    *,
    train_ds: HeightOnlyPriorDataset,
    device: torch.device,
    args: argparse.Namespace,
    evidence_dir: Path,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "autotune_batch_size", False)):
        return None
    if device.type != "cuda":
        print("Autotune: skipped because device is not CUDA.", flush=True)
        return None
    if float(args.target_vram_gb) <= 0.0:
        print("Autotune: skipped because --target-vram-gb <= 0.", flush=True)
        return None
    if len(train_ds) <= 0:
        print("Autotune: skipped because train dataset is empty.", flush=True)
        return None

    base_batch_size = int(args.batch_size)
    candidates = list(getattr(args, "autotune_batch_candidates", _DEFAULT_AUTOTUNE_BATCH_CANDIDATES)) or list(_DEFAULT_AUTOTUNE_BATCH_CANDIDATES)
    candidates = sorted({c for c in candidates if int(c) >= 1})

    safety_factor = float(getattr(args, "autotune_safety_factor", 0.0))
    if safety_factor <= 0.0:
        safety_factor = 0.72 if (hasattr(torch, "compile") and not args.no_compile) else 0.82
    effective_target_vram_gb = float(args.target_vram_gb) * float(safety_factor)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    chosen_batch_size = base_batch_size
    reached_limit = False
    results: list[dict[str, Any]] = []

    print(
        "Autotune: probing batch-size ladder "
        f"{candidates} against target_vram_gb={float(args.target_vram_gb):.2f} "
        f"(effective_target={effective_target_vram_gb:.2f}GB, safety_factor={safety_factor:.2f})",
        flush=True,
    )

    def _cleanup() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

    warmup_steps = max(1, int(getattr(args, "autotune_probe_warmup_steps", 2)))
    measure_steps = max(1, int(getattr(args, "autotune_probe_measure_steps", 3)))
    total_steps = warmup_steps + measure_steps

    for candidate in candidates:
        probe_batch_size = min(int(candidate), len(train_ds))
        probe_model = None
        probe_optimizer = None
        probe_scaler = None
        peak_alloc_gb = None
        peak_reserved_gb = None
        measured_alloc_gb = None
        measured_reserved_gb = None
        status = "ok"
        fits_target = False
        _cleanup()
        try:
            probe_model = V18HeightModel().to(device)
            if bool(hasattr(torch, "compile") and not args.no_compile):
                try:
                    probe_model = torch.compile(probe_model)
                except Exception:
                    pass
            probe_optimizer = torch.optim.AdamW(
                probe_model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            probe_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            probe_model.train()
            measured_allocs: list[float] = []
            measured_reserveds: list[float] = []

            for step in range(total_steps):
                probe_batch = default_collate(
                    [train_ds[(step * probe_batch_size + idx) % len(train_ds)] for idx in range(probe_batch_size)]
                )
                prior = probe_batch["input_prior"].to(device, non_blocking=True)[:, :3, :, :]
                target = probe_batch["height_257"].to(device, non_blocking=True)
                weight = probe_batch["weight_257"].to(device, non_blocking=True)
                probe_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = probe_model(prior)
                    loss, _, _ = compute_height_loss(
                        pred, target, weight,
                        ms_weight=args.multiscale_weight,
                        grad_weight=args.gradient_weight,
                        nc_weight=args.normal_consistency_weight,
                    )
                probe_scaler.scale(loss).backward()
                probe_scaler.unscale_(probe_optimizer)
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=1.0)
                probe_scaler.step(probe_optimizer)
                probe_scaler.update()
                del probe_batch, prior, target, weight, pred, loss

                torch.cuda.synchronize(device)
                current_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                current_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024.0 ** 3)
                peak_alloc_gb = current_alloc_gb if peak_alloc_gb is None else max(peak_alloc_gb, current_alloc_gb)
                peak_reserved_gb = current_reserved_gb if peak_reserved_gb is None else max(peak_reserved_gb, current_reserved_gb)
                if step + 1 == warmup_steps:
                    torch.cuda.reset_peak_memory_stats(device)
                elif step >= warmup_steps:
                    measured_allocs.append(current_alloc_gb)
                    measured_reserveds.append(current_reserved_gb)

            if measured_allocs:
                measured_alloc_gb = max(measured_allocs)
            if measured_reserveds:
                measured_reserved_gb = max(measured_reserveds)

            torch.cuda.synchronize(device)
            decision_reserved_gb = measured_reserved_gb if measured_reserved_gb is not None else peak_reserved_gb
            fits_target = bool(decision_reserved_gb is not None and decision_reserved_gb <= effective_target_vram_gb)
            if fits_target:
                chosen_batch_size = int(candidate)
            else:
                status = "above_target"
                reached_limit = True
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                status = "oom"
                reached_limit = True
            else:
                raise
        finally:
            del probe_scaler
            del probe_optimizer
            del probe_model
            _cleanup()

        result = {
            "candidate_batch_size": int(candidate),
            "probe_batch_size": int(probe_batch_size),
            "status": status,
            "fits_target": bool(fits_target),
            "peak_alloc_gb": peak_alloc_gb,
            "peak_reserved_gb": peak_reserved_gb,
            "measured_alloc_gb": measured_alloc_gb,
            "measured_reserved_gb": measured_reserved_gb,
            "warmup_steps": int(warmup_steps),
            "measure_steps": int(measure_steps),
        }
        results.append(result)
        print(
            f"Autotune: batch-size {candidate} -> status={status} "
            f"peak_alloc={peak_alloc_gb if peak_alloc_gb is None else f'{peak_alloc_gb:.2f}GB'} "
            f"peak_reserved={peak_reserved_gb if peak_reserved_gb is None else f'{peak_reserved_gb:.2f}GB'}",
            flush=True,
        )
        if reached_limit:
            break

    payload = {
        "enabled": True,
        "target_vram_gb": float(args.target_vram_gb),
        "effective_target_vram_gb": effective_target_vram_gb,
        "autotune_safety_factor": safety_factor,
        "compile_enabled_for_run": bool(hasattr(torch, "compile") and not args.no_compile),
        "original_batch_size": int(base_batch_size),
        "chosen_batch_size": int(chosen_batch_size),
        "candidate_results": results,
    }
    (evidence_dir / "batch_autotune.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.batch_size = int(chosen_batch_size)
    print(f"Autotune: selected batch-size={args.batch_size}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# main training routine
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spec 077 height-only terrain training on teacher priors."
    )
    parser.add_argument("--prior", type=Path, required=True,
                        help="Path to <build>.zarr teacher-prior store.")
    parser.add_argument("--v18", type=Path, default=None,
                        help="Path to the source <build>.zarr V18 store for height_257 target.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for checkpoints, metrics, and preview PNGs.")
    parser.add_argument("--run-name", type=str, default="height_only_prior_smoke")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--val-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-tiles", type=int, default=64,
                        help="Cap dataset size to keep smoke runs bounded.")
    parser.add_argument("--no-weight", action="store_true", default=False,
                        help="Disable terrain-valid weighting (use a constant 1.0).")
    parser.add_argument("--seed", type=int, default=42)
    # perf / safety knobs (port from train_v16_1_common)
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader workers; -1 = auto (0 on CPU, 2..8 on CUDA).")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm; 0 disables clipping.")
    parser.add_argument("--multiscale-weight", type=float, default=0.2,
                        help="Per-scale weight in the multi-scale L1 loss. <=0 falls back to single 257-px L1.")
    parser.add_argument("--gradient-weight", type=float, default=0.0,
                        help="Weight for the Sobel gradient loss term. Default 0 (off).")
    parser.add_argument("--normal-consistency-weight", type=float, default=0.0,
                        help="Weight for the height-flatness (normal-consistency proxy) term. Default 0 (off).")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop training if val loss does not improve for N steps. 0 disables.")
    parser.add_argument("--early-stop-min-improvement", type=float, default=1e-4)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--autotune-batch-size", action="store_true", default=False)
    parser.add_argument("--target-vram-gb", type=float, default=0.0)
    parser.add_argument("--autotune-batch-candidates", type=int, nargs="*", default=None)
    parser.add_argument("--autotune-safety-factor", type=float, default=0.0)
    parser.add_argument("--autotune-probe-warmup-steps", type=int, default=2)
    parser.add_argument("--autotune-probe-measure-steps", type=int, default=3)
    return parser.parse_args(argv)


def _build_dataloader(
    dataset: HeightOnlyPriorDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    if num_workers <= 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.prior.exists():
        print(f"Prior store not found: {args.prior}", file=sys.stderr)
        return 2
    if args.v18 is not None and not args.v18.exists():
        print(f"V18 store not found: {args.v18}", file=sys.stderr)
        return 2

    _seed_all(args.seed)
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Optional VRAM autotune needs a model on the device; build dataset first
    # so the probe can index into it.
    dataset = HeightOnlyPriorDataset(
        prior_path=args.prior,
        v18_path=args.v18,
        include_weight=not args.no_weight,
        height_norm=True,
    )
    if args.max_tiles and len(dataset) > args.max_tiles:
        keep = list(range(args.max_tiles))
        dataset = HeightOnlyPriorDataset(
            prior_path=args.prior,
            v18_path=args.v18,
            tile_filter=keep,
            include_weight=not args.no_weight,
            height_norm=True,
        )
    if len(dataset) == 0:
        print("Dataset is empty; nothing to train.", file=sys.stderr)
        return 2

    if args.autotune_batch_size:
        _autotune_batch_size(
            train_ds=dataset,
            device=device,
            args=args,
            evidence_dir=args.output_dir,
        )

    num_workers = _resolve_num_workers(args.num_workers, device)
    persistent_workers = _resolve_persistent_workers(args.persistent_workers, num_workers)
    if num_workers > 0 and args.prefetch_factor < 1:
        raise RuntimeError("--prefetch-factor must be >= 1")

    loader = _build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
    )

    model = V18HeightModel().to(device)
    compile_status = "disabled"
    can_compile = hasattr(torch, "compile") and not args.no_compile and device.type == "cuda"
    if can_compile:
        try:
            model = torch.compile(model)
            compile_status = "enabled"
        except Exception as ex:  # noqa: BLE001
            compile_status = f"disabled (compile failed: {ex})"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    best_val = float("inf")
    early_stop_counter = 0
    early_stop_triggered = False
    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"Resume: loaded {args.resume_checkpoint} from step {start_step}", flush=True)

    print(f"torch.compile: {compile_status}", flush=True)
    print(
        f"DataLoader: num_workers={num_workers} prefetch_factor={args.prefetch_factor} "
        f"persistent_workers={persistent_workers}",
        flush=True,
    )

    def _train_one_batch(batch: dict, model: torch.nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
        prior = batch["input_prior"].to(device, non_blocking=True)[:, :3, :, :]
        target = batch["height_257"].to(device, non_blocking=True)
        weight = batch["weight_257"].to(device, non_blocking=True)
        if args.no_weight:
            weight = torch.ones_like(weight)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            pred = model(prior)
            loss, metrics = compute_height_loss(
                pred,
                target,
                weight,
                ms_weight=args.multiscale_weight,
                grad_weight=args.gradient_weight,
                nc_weight=args.normal_consistency_weight,
            )
        if use_amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
        return loss, metrics

    def _validate_batch(batch: dict, model: torch.nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
        prior = batch["input_prior"].to(device, non_blocking=True)[:, :3, :, :]
        target = batch["height_257"].to(device, non_blocking=True)
        weight = batch["weight_257"].to(device, non_blocking=True)
        if args.no_weight:
            weight = torch.ones_like(weight)
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = model(prior)
                loss, metrics = compute_height_loss(
                    pred, target, weight,
                    ms_weight=args.multiscale_weight,
                    grad_weight=args.gradient_weight,
                    nc_weight=args.normal_consistency_weight,
                )
        return loss, metrics

    metrics_log: list[dict] = []
    val_log: list[dict] = []
    preview_batch = None
    train_state = _make_epoch_iterator(loader)
    model.train()
    t0 = time.perf_counter()
    tiles_seen = 0
    for step in range(start_step, start_step + args.steps):
        batch = _next_batch_or_reset(train_state, loader)
        if batch is None:
            break
        loss, metrics = _train_one_batch(batch, model)
        tiles_seen += int(batch["input_prior"].shape[0])
        elapsed = time.perf_counter() - t0
        rate = tiles_seen / max(elapsed, 1e-9)
        entry = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "tiles_per_sec": rate,
            "tile_id": int(batch["meta_tile_id"][0]) if torch.is_tensor(batch["meta_tile_id"]) else int(batch["meta_tile_id"]),
        }
        for k, v in metrics.items():
            entry[k] = float(v)
        metrics_log.append(entry)
        print(
            f"step {step}: loss={float(loss):.4f} "
            f"tiles/s={rate:.2f} tile_id={entry['tile_id']}",
            flush=True,
        )
        if preview_batch is None:
            preview_batch = batch

    # Validation pass
    val_tiles_seen = 0
    val_t0 = time.perf_counter()
    model.eval()
    val_state = _make_epoch_iterator(loader)
    with torch.no_grad():
        for v in range(args.val_steps):
            batch = _next_batch_or_reset(val_state, loader)
            if batch is None:
                break
            loss, metrics = _validate_batch(batch, model)
            val_tiles_seen += int(batch["input_prior"].shape[0])
            entry = {
                "step": v,
                "val_loss": float(loss.detach().cpu()),
                "tile_id": int(batch["meta_tile_id"][0]) if torch.is_tensor(batch["meta_tile_id"]) else int(batch["meta_tile_id"]),
            }
            for k, val in metrics.items():
                entry[k] = float(val)
            val_log.append(entry)
            current_val = float(loss.detach().cpu())
            improved = current_val < (best_val - args.early_stop_min_improvement)
            if improved:
                best_val = current_val
                early_stop_counter = 0
            elif args.early_stop_patience > 0:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    early_stop_triggered = True
                    print(
                        f"Early stop: no val improvement for {early_stop_counter} steps",
                        flush=True,
                    )
                    break
            print(
                f"val {v}: val_loss={current_val:.4f} tile_id={entry['tile_id']} "
                f"best_val={best_val:.4f}",
                flush=True,
            )
    val_elapsed = time.perf_counter() - val_t0
    val_rate = val_tiles_seen / max(val_elapsed, 1e-9)

    # Preview: use the first captured batch; render a labeled 4-panel image.
    preview_path = args.output_dir / f"{args.run_name}_preview.png"
    if preview_batch is not None:
        prior_chw = preview_batch["input_prior"][:3].cpu()
        truth = preview_batch["height_257"][0].cpu()
        weight = preview_batch["weight_257"][0].cpu()
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = model(preview_batch["input_prior"].to(device, non_blocking=True)[:, :3, :, :])
        pred_chw = pred[0].cpu()
        prior_rgb = (prior_chw * 255).clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        # Normalize truth/pred/weight to [0,1] for display
        def _norm(x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            lo, hi = float(x.min()), float(x.max())
            if hi - lo < 1e-8:
                return torch.zeros_like(x)
            return (x - lo) / (hi - lo)
        truth_disp = _norm(truth[None, :, :]).squeeze(0)
        pred_disp = _norm(pred_chw[0:1]).squeeze(0)
        weight_disp = _norm(weight[None, :, :]).squeeze(0)
        # Replicate single-channel to 3ch for the panel composer
        truth_rgb = truth_disp.unsqueeze(0).repeat(3, 1, 1)
        pred_rgb = pred_disp.unsqueeze(0).repeat(3, 1, 1)
        weight_rgb = weight_disp.unsqueeze(0).repeat(3, 1, 1)
        prior_t = torch.from_numpy(prior_rgb).permute(2, 0, 1).float() / 255.0
        # Resize to panel size
        prior_t = F.interpolate(prior_t.unsqueeze(0), size=(_PANEL_SIZE, _PANEL_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        truth_rgb = F.interpolate(truth_rgb.unsqueeze(0), size=(_PANEL_SIZE, _PANEL_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        pred_rgb = F.interpolate(pred_rgb.unsqueeze(0), size=(_PANEL_SIZE, _PANEL_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        weight_rgb = F.interpolate(weight_rgb.unsqueeze(0), size=(_PANEL_SIZE, _PANEL_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        _save_horizontal_panel(
            [
                ("prior RGB", prior_t),
                ("height truth", truth_rgb),
                ("height pred", pred_rgb),
                ("loss weight", weight_rgb),
            ],
            preview_path,
        )

    metrics_payload = {
        "run_name": args.run_name,
        "schema": "spec-077-height-only-prior",
        "step_count": args.steps,
        "val_step_count": args.val_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "device": str(device),
        "compile_status": compile_status,
        "amp_enabled": use_amp,
        "grad_clip": float(args.grad_clip),
        "multiscale_weight": float(args.multiscale_weight),
        "gradient_weight": float(args.gradient_weight),
        "normal_consistency_weight": float(args.normal_consistency_weight),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_triggered": early_stop_triggered,
        "best_val": best_val if best_val != float("inf") else None,
        "model_parameter_count": sum(p.numel() for p in (model._orig_mod.parameters() if hasattr(model, "_orig_mod") else model.parameters())),
        "num_workers": num_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": persistent_workers,
        "train_metrics": metrics_log,
        "val_metrics": val_log,
        "val_tiles_per_sec": val_rate,
        "preview_path": str(preview_path),
        "seed": args.seed,
    }
    metrics_path = args.output_dir / f"{args.run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    ckpt_path = args.output_dir / f"{args.run_name}_model.pt"
    state = {
        "state_dict": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else None,
        "step": start_step + args.steps,
        "best_val": best_val,
        "args": vars(args),
    }
    torch.save(state, ckpt_path)
    print(f"Wrote smoke training output to {args.output_dir}", flush=True)
    return 0


def _make_epoch_iterator(loader: DataLoader) -> list:
    """Return a one-cell list whose slot holds the active loader iterator.

    The cell is mutable so callers can transparently recycle the iterator
    on epoch boundaries without rebuilding the persistent DataLoader
    workers. The companion helper is :func:`_next_batch_or_reset`.
    """
    return [iter(loader)]


def _next_batch_or_reset(state: list, loader: DataLoader) -> dict | None:
    """Pop the next batch from *state*; rebuild the iterator on exhaustion.

    Returns ``None`` only when the loader itself is empty.
    """
    while True:
        try:
            return next(state[0])
        except StopIteration:
            # Generator is closed; create a fresh one. This still reuses
            # the underlying DataLoader workers when ``persistent_workers``
            # is enabled, because PyTorch reuses the worker pool.
            try:
                state[0] = iter(loader)
            except StopIteration:
                return None


if __name__ == "__main__":
    sys.exit(main_with_args())
