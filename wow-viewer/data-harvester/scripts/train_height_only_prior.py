"""Spec 077 Phase 4 (US3) height-only training script.

Tiny training entry point that loads the teacher-prior dataset, runs
``V161HeightModel`` (= ``V18HeightModel``) for epoch-based training, and
writes previews showing the prior input, the predicted height, the
ground truth, and the loss weight. ``--steps`` is retained only as an
optional smoke/resume cap; ``--epochs`` is the real training contract.

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
* **Optional V18 normal guidance** that derives normals from predicted
  height and compares them to `normal_xyz`; this is an auxiliary loss,
  not a normal output head.
* **Early stopping** with epoch patience + min-improvement.
* **Resume support** that reloads model + optimizer + scaler + epoch +
  step counters from a checkpoint.
* **Latest/best checkpointing** via ``*_latest.pt`` and ``*_best.pt``;
  ``*_model.pt`` is kept as a compatibility alias to latest.
* **Labeled preview panels** with text strips. Per-epoch validation
  previews show raw minimap, object mask/confidence, object-suppressed
  prior, truth, prediction, error, and loss weight.
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
from torch.utils.data import ConcatDataset, DataLoader, Subset, default_collate

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.height_only_prior_dataset import HeightOnlyPriorDataset  # noqa: E402
from harvester.height_to_normal import analytic_normals_from_height  # noqa: E402
from harvester.v16_curation import load_curation_keys  # noqa: E402
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
    if sys.platform.startswith("win"):
        # Windows DataLoader workers use spawn and must pickle the dataset.
        # This trainer keeps large Zarr-backed arrays in memory, so auto
        # multiprocessing can fail with truncated pickle data.
        return 0
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


def _compose_panel_grid(rows: list[list[tuple[str, torch.Tensor]]]) -> Image.Image:
    row_images = [_compose_horizontal_panel(row) for row in rows if row]
    if not row_images:
        return Image.new("RGB", (_PANEL_SIZE, _PANEL_SIZE + _PANEL_LABEL_HEIGHT), color=(0, 0, 0))
    width = max(img.width for img in row_images)
    height = sum(img.height for img in row_images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for img in row_images:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas


def _norm_for_display(x: torch.Tensor, *, lo: float | None = None, hi: float | None = None) -> torch.Tensor:
    x = x.detach().float().cpu()
    x_lo = float(x.min()) if lo is None else float(lo)
    x_hi = float(x.max()) if hi is None else float(hi)
    if x_hi - x_lo < 1e-8:
        return torch.zeros_like(x)
    return ((x - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)


def _gray3(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0).repeat(3, 1, 1)
    if x.ndim == 3 and x.shape[0] == 3:
        return x
    raise ValueError(f"Cannot convert tensor with shape {tuple(x.shape)} to display RGB")


def _save_deconstruction_preview(
    *,
    batch: dict,
    pred: torch.Tensor,
    out_path: Path,
    max_samples: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[tuple[str, torch.Tensor]]] = []
    sample_count = min(int(max_samples), int(batch["input_prior"].shape[0]))
    tile_ids = batch.get("meta_tile_id")
    for idx in range(sample_count):
        tile_id = int(tile_ids[idx]) if torch.is_tensor(tile_ids) else int(tile_ids[idx] if isinstance(tile_ids, (list, tuple)) else tile_ids)
        raw = batch.get("raw_minimap_rgb", batch["input_prior"][:, :3])[idx].cpu()
        prior = batch["input_prior"][idx, :3].cpu()
        mask = batch.get("teacher_object_mask", batch["input_prior"][:, 3:4])[idx].cpu()
        confidence = batch.get("teacher_object_confidence", batch["input_prior"][:, 4:5])[idx].cpu()
        truth = batch["height_257"][idx].cpu()
        pred_i = pred[idx].detach().cpu()
        weight = batch["weight_257"][idx].cpu()
        lo = min(float(truth.min()), float(pred_i.min()))
        hi = max(float(truth.max()), float(pred_i.max()))
        error = (pred_i - truth).abs()
        rows.append(
            [
                (f"raw tile {tile_id}", raw),
                ("object mask", _gray3(mask.clamp(0.0, 1.0))),
                ("confidence", _gray3(confidence.clamp(0.0, 1.0))),
                ("suppressed prior", prior),
                ("height truth", _gray3(_norm_for_display(truth.squeeze(0), lo=lo, hi=hi))),
                ("height pred", _gray3(_norm_for_display(pred_i.squeeze(0), lo=lo, hi=hi))),
                ("abs error", _gray3(_norm_for_display(error.squeeze(0)))),
                ("loss weight", _gray3(_norm_for_display(weight.squeeze(0), lo=0.0, hi=1.0))),
            ]
        )
    _compose_panel_grid(rows).save(out_path)


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
    normal_guidance_weight: float = 0.0,
    target_normals: torch.Tensor | None = None,
    normal_guidance_mask: torch.Tensor | None = None,
    normal_guidance_spacing: float = 1.0,
    hard_error_weight: float = 0.0,
    hard_error_power: float = 1.0,
    hard_error_max_multiplier: float = 4.0,
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
    if normal_guidance_weight > 0.0 and target_normals is not None:
        pred_normals = analytic_normals_from_height(pred, spacing=float(normal_guidance_spacing))
        target_normals = F.normalize(target_normals, dim=1, eps=1e-8)
        cos = (pred_normals * target_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        if normal_guidance_mask is None:
            guidance_mask = weight
        else:
            guidance_mask = normal_guidance_mask * weight
        normal_loss = _masked_mean(1.0 - cos, guidance_mask)
        loss = loss + normal_guidance_weight * normal_loss
        metrics["normal_guidance_loss"] = float(normal_loss.item())
        metrics["normal_guidance_mask_cov"] = float(guidance_mask.mean().item())
    if hard_error_weight > 0.0:
        abs_err = (pred - target).abs()
        with torch.no_grad():
            mean_err = _masked_mean(abs_err.detach(), weight).clamp_min(1e-8)
            hard_multiplier = (abs_err.detach() / mean_err).clamp_min(0.0).pow(float(hard_error_power))
            hard_multiplier = hard_multiplier.clamp(1.0, float(hard_error_max_multiplier))
            hard_mask = weight * hard_multiplier
        hard_loss = _masked_mean(abs_err, hard_mask)
        loss = loss + hard_error_weight * hard_loss
        metrics["hard_error_loss"] = float(hard_loss.item())
        metrics["hard_error_weight_mean"] = float(hard_multiplier.mean().item())
        metrics["hard_error_weight_max"] = float(hard_multiplier.max().item())
    return loss, metrics


# ---------------------------------------------------------------------------
# VRAM autotune (port of _autotune_batch_size, height-only subset)
# ---------------------------------------------------------------------------

def _autotune_batch_size(
    *,
    train_ds,
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
    requested_candidates = getattr(args, "autotune_batch_candidates", None)
    candidates = list(requested_candidates or _DEFAULT_AUTOTUNE_BATCH_CANDIDATES)
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
                target_normals = probe_batch["normal_xyz"].to(device, non_blocking=True) if "normal_xyz" in probe_batch else None
                normal_mask = probe_batch["normal_mask"].to(device, non_blocking=True) if "normal_mask" in probe_batch else None
                probe_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = probe_model(prior)
                    loss, _ = compute_height_loss(
                        pred, target, weight,
                        ms_weight=args.multiscale_weight,
                        grad_weight=args.gradient_weight,
                        nc_weight=args.normal_consistency_weight,
                        normal_guidance_weight=args.normal_guidance_weight,
                        target_normals=target_normals,
                        normal_guidance_mask=normal_mask,
                        normal_guidance_spacing=args.normal_guidance_spacing,
                        hard_error_weight=args.hard_error_weight,
                        hard_error_power=args.hard_error_power,
                        hard_error_max_multiplier=args.hard_error_max_multiplier,
                    )
                probe_scaler.scale(loss).backward()
                probe_scaler.unscale_(probe_optimizer)
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=1.0)
                probe_scaler.step(probe_optimizer)
                probe_scaler.update()
                del probe_batch, prior, target, weight, target_normals, normal_mask, pred, loss

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
    parser.add_argument("--prior", type=Path, nargs="+", required=True,
                        help="One or more <build>.zarr teacher-prior stores.")
    parser.add_argument("--v18", type=Path, nargs="*", default=None,
                        help="Matching source <build>.zarr V18 stores for height_257 targets.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for checkpoints, metrics, and preview PNGs.")
    parser.add_argument("--run-name", type=str, default="height_only_prior_smoke")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of full dataset epochs. Use this for real training.")
    parser.add_argument("--steps", type=int, default=0,
                        help="Optional total optimizer-step cap for smoke/backcompat. 0 = run --epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="Optional train batches per epoch cap. 0 = full train split.")
    parser.add_argument("--val-steps", type=int, default=0,
                        help="Optional validation batches per epoch. 0 = full validation split.")
    parser.add_argument("--val-fraction", type=float, default=0.10,
                        help="Deterministic validation fraction cut from the curated dataset.")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--preview-every-epochs", type=int, default=1,
                        help="Write validation deconstruction preview every N epochs. 0 disables epoch previews.")
    parser.add_argument("--preview-samples", type=int, default=2,
                        help="Number of validation samples per deconstruction preview grid.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-learning-rate", type=float, default=0.0,
                        help="If >0, override optimizer LR after loading --resume-checkpoint.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-plateau-patience", type=int, default=8,
                        help="Reduce LR after N validation epochs without improvement. 0 disables.")
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5,
                        help="Multiplicative LR decay used by validation plateau scheduler.")
    parser.add_argument("--lr-plateau-min-delta", type=float, default=1e-4,
                        help="Minimum validation-loss improvement for plateau scheduler.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6,
                        help="Lower bound for validation plateau LR scheduler.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-tiles", type=int, default=0,
                        help="Optional combined dataset cap. 0 means use all curated tiles; set this for smoke runs.")
    parser.add_argument("--min-train-tiles", type=int, default=0,
                        help="Fail fast if the final combined dataset has fewer than this many tiles.")
    parser.add_argument("--curation-manifest", type=Path, default=None,
                        help="Optional V18 curation manifest directory/file; only kept (build,tile_id) rows are trained.")
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
    parser.add_argument("--normal-guidance-weight", type=float, default=0.0,
                        help="Weight for auxiliary V18 normal guidance. No normal head is predicted; normals are derived from predicted height.")
    parser.add_argument("--normal-guidance-spacing", type=float, default=1.0,
                        help="Spacing used when deriving analytic normals from predicted height for normal guidance.")
    parser.add_argument("--hard-error-weight", type=float, default=0.0,
                        help="Auxiliary training-only focal L1 weight that emphasizes high absolute-error height pixels. Validation loss never uses this.")
    parser.add_argument("--hard-error-power", type=float, default=1.0,
                        help="Exponent for training-only hard-error pixel weights.")
    parser.add_argument("--hard-error-max-multiplier", type=float, default=4.0,
                        help="Clamp for training-only hard-error pixel weights.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop training if val loss does not improve for N epochs. 0 disables.")
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
    dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    *,
    shuffle: bool,
) -> DataLoader:
    if num_workers <= 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _resolve_v18_paths(prior_paths: list[Path], v18_paths: list[Path] | None) -> list[Path | None] | None:
    if not v18_paths:
        return [None] * len(prior_paths)
    if len(v18_paths) != len(prior_paths):
        return None
    return list(v18_paths)


def _build_training_dataset(args: argparse.Namespace) -> ConcatDataset | HeightOnlyPriorDataset | None:
    prior_paths = list(args.prior)
    v18_paths = _resolve_v18_paths(prior_paths, args.v18)
    if v18_paths is None:
        print(
            f"--v18 must provide exactly one path per --prior path; got {len(args.v18 or [])} v18 paths for {len(prior_paths)} priors.",
            file=sys.stderr,
        )
        return None

    curation_keys = load_curation_keys(args.curation_manifest) if args.curation_manifest is not None else None
    remaining = int(args.max_tiles) if args.max_tiles else None
    datasets: list[HeightOnlyPriorDataset] = []
    for prior_path, v18_path in zip(prior_paths, v18_paths, strict=True):
        build = prior_path.stem.replace(".zarr", "")
        tile_filter = None
        if curation_keys is not None:
            tile_filter = sorted(tile_id for key_build, tile_id in curation_keys if key_build == build)
            if not tile_filter:
                print(f"Curation manifest has no kept rows for build {build}: {args.curation_manifest}", file=sys.stderr)
                continue
        ds = HeightOnlyPriorDataset(
            prior_path=prior_path,
            v18_path=v18_path,
            tile_filter=tile_filter,
            include_weight=not args.no_weight,
            height_norm=True,
        )
        if remaining is not None:
            if remaining <= 0:
                break
            if len(ds) > remaining:
                ds = HeightOnlyPriorDataset(
                    prior_path=prior_path,
                    v18_path=v18_path,
                    tile_filter=[int(ds.tile_meta[i].get("tile_id", i)) for i in range(remaining)] if getattr(ds, "tile_meta", None) else list(range(remaining)),
                    include_weight=not args.no_weight,
                    height_norm=True,
                )
            remaining -= len(ds)
        datasets.append(ds)

    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def _dataset_source_summaries(dataset: ConcatDataset | HeightOnlyPriorDataset) -> list[dict[str, object]]:
    sources = list(dataset.datasets) if isinstance(dataset, ConcatDataset) else [dataset]
    summaries: list[dict[str, object]] = []
    for ds in sources:
        tile_ids: list[int] = []
        if getattr(ds, "tile_meta", None):
            tile_ids = [int(row.get("tile_id", idx)) for idx, row in enumerate(ds.tile_meta)]
        summaries.append(
            {
                "prior_path": str(getattr(ds, "prior_path", "")),
                "v18_path": str(getattr(ds, "v18_path", "")),
                "tile_count": len(ds),
                "first_tile_ids": tile_ids[:8],
                "last_tile_ids": tile_ids[-8:] if tile_ids else [],
            }
        )
    return summaries


def _split_train_val(dataset, *, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    n = len(dataset)
    if n <= 1:
        return Subset(dataset, list(range(n))), Subset(dataset, list(range(n)))
    val_count = int(round(n * max(0.0, min(0.9, float(val_fraction)))))
    val_count = max(1, min(n - 1, val_count))
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    val_indices = perm[:val_count]
    train_indices = perm[val_count:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _mean_metric(rows: list[dict[str, float]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if key in row]
    return float(sum(vals) / len(vals)) if vals else None


def _extract_first_tile_id(batch: dict) -> int:
    value = batch["meta_tile_id"]
    if torch.is_tensor(value):
        return int(value[0])
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _model_state_dict(model: torch.nn.Module) -> dict:
    return model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()


def _load_model_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(state_dict)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    epoch: int,
    global_step: int,
    best_val: float,
    args: argparse.Namespace,
    history: list[dict],
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> Path:
    state = {
        "state_dict": _model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "step": int(global_step),
        "best_val": float(best_val),
        "history": history,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    torch.save(state, tmp_path)
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(tmp_path, path)
            return path
        except OSError as ex:
            last_error = ex
            # Windows can transiently lock recently-read .pt files with
            # ERROR_USER_MAPPED_FILE (1224) or sharing violations. Keep the
            # run alive and retry before falling back to a step checkpoint.
            if getattr(ex, "winerror", None) not in (5, 32, 1224):
                break
            time.sleep(0.25 * (attempt + 1))

    fallback_path = path.with_name(
        f"{path.stem}_epoch{int(epoch):04d}_step{int(global_step):07d}_{int(time.time())}{path.suffix}"
    )
    try:
        os.replace(tmp_path, fallback_path)
        print(
            f"Checkpoint warning: could not replace {path} ({last_error}); wrote fallback {fallback_path}",
            flush=True,
        )
        return fallback_path
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for prior_path in args.prior:
        if not prior_path.exists():
            print(f"Prior store not found: {prior_path}", file=sys.stderr)
            return 2
    for v18_path in args.v18 or []:
        if not v18_path.exists():
            print(f"V18 store not found: {v18_path}", file=sys.stderr)
            return 2

    v18_paths = _resolve_v18_paths(list(args.prior), args.v18)
    if v18_paths is None:
        print(
            f"--v18 must provide exactly one path per --prior path; got {len(args.v18 or [])} v18 paths for {len(args.prior)} priors.",
            file=sys.stderr,
        )
        return 2

    _seed_all(args.seed)
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Optional VRAM autotune needs a model on the device; build dataset first
    # so the probe can index into it.
    dataset = _build_training_dataset(args)
    if dataset is None or len(dataset) == 0:
        print("Dataset is empty; nothing to train.", file=sys.stderr)
        return 2
    source_summaries = _dataset_source_summaries(dataset)
    print(f"Dataset: combined_tile_count={len(dataset)} source_count={len(source_summaries)}", flush=True)
    for idx, summary in enumerate(source_summaries, start=1):
        print(
            f"  source {idx}: tiles={summary['tile_count']} prior={summary['prior_path']} "
            f"first_tile_ids={summary['first_tile_ids']} last_tile_ids={summary['last_tile_ids']}",
            flush=True,
        )
    if int(args.min_train_tiles) > 0 and len(dataset) < int(args.min_train_tiles):
        print(
            f"Dataset has only {len(dataset)} tiles, below --min-train-tiles {args.min_train_tiles}. "
            "Check curation manifest and teacher-prior tile counts.",
            file=sys.stderr,
        )
        return 2

    train_dataset, val_dataset = _split_train_val(dataset, val_fraction=args.val_fraction, seed=args.seed)
    if len(train_dataset) <= 0 or len(val_dataset) <= 0:
        print("Train/validation split is empty; nothing to train.", file=sys.stderr)
        return 2

    full_steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, int(args.batch_size))))
    train_batches_per_epoch = full_steps_per_epoch
    if int(args.steps_per_epoch) > 0:
        train_batches_per_epoch = min(train_batches_per_epoch, int(args.steps_per_epoch))
    val_batches_per_epoch = int(math.ceil(len(val_dataset) / max(1, int(args.batch_size))))
    if int(args.val_steps) > 0:
        val_batches_per_epoch = min(val_batches_per_epoch, int(args.val_steps))
    print(
        f"Split: train_tiles={len(train_dataset)} val_tiles={len(val_dataset)} "
        f"steps_per_epoch={train_batches_per_epoch}/{full_steps_per_epoch} "
        f"val_batches={val_batches_per_epoch}",
        flush=True,
    )

    if args.autotune_batch_size:
        _autotune_batch_size(
            train_ds=train_dataset,
            device=device,
            args=args,
            evidence_dir=args.output_dir,
        )

        full_steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, int(args.batch_size))))
        train_batches_per_epoch = full_steps_per_epoch
        if int(args.steps_per_epoch) > 0:
            train_batches_per_epoch = min(train_batches_per_epoch, int(args.steps_per_epoch))
        val_batches_per_epoch = int(math.ceil(len(val_dataset) / max(1, int(args.batch_size))))
        if int(args.val_steps) > 0:
            val_batches_per_epoch = min(val_batches_per_epoch, int(args.val_steps))

    num_workers = _resolve_num_workers(args.num_workers, device)
    persistent_workers = _resolve_persistent_workers(args.persistent_workers, num_workers)
    if num_workers > 0 and args.prefetch_factor < 1:
        raise RuntimeError("--prefetch-factor must be >= 1")

    train_loader = _build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
        shuffle=True,
    )
    val_loader = _build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
        shuffle=False,
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
    scheduler = None
    if int(args.lr_plateau_patience) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_plateau_factor),
            patience=int(args.lr_plateau_patience),
            threshold=float(args.lr_plateau_min_delta),
            threshold_mode="abs",
            min_lr=float(args.min_learning_rate),
        )
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    start_epoch = 0
    best_val = float("inf")
    history: list[dict] = []
    early_stop_counter = 0
    early_stop_triggered = False
    if args.resume_checkpoint is not None:
        # This checkpoint is produced by this script and includes CLI args
        # containing pathlib.Path values. PyTorch 2.6 defaults to
        # weights_only=True, which rejects those metadata objects.
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            _load_model_state_dict(model, ckpt["state_dict"])
        elif "model_state_dict" in ckpt:
            _load_model_state_dict(model, ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if float(args.resume_learning_rate) > 0.0:
            for group in optimizer.param_groups:
                group["lr"] = float(args.resume_learning_rate)
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        history = list(ckpt.get("history", []))
        print(f"Resume: loaded {args.resume_checkpoint} from epoch {start_epoch} step {start_step}", flush=True)

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
        target_normals = batch["normal_xyz"].to(device, non_blocking=True) if "normal_xyz" in batch else None
        normal_mask = batch["normal_mask"].to(device, non_blocking=True) if "normal_mask" in batch else None
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
                normal_guidance_weight=args.normal_guidance_weight,
                target_normals=target_normals,
                normal_guidance_mask=normal_mask,
                normal_guidance_spacing=args.normal_guidance_spacing,
                hard_error_weight=args.hard_error_weight,
                hard_error_power=args.hard_error_power,
                hard_error_max_multiplier=args.hard_error_max_multiplier,
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
        target_normals = batch["normal_xyz"].to(device, non_blocking=True) if "normal_xyz" in batch else None
        normal_mask = batch["normal_mask"].to(device, non_blocking=True) if "normal_mask" in batch else None
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
                    normal_guidance_weight=args.normal_guidance_weight,
                    target_normals=target_normals,
                    normal_guidance_mask=normal_mask,
                    normal_guidance_spacing=args.normal_guidance_spacing,
                    hard_error_weight=0.0,
                )
        return loss, metrics

    metrics_log: list[dict] = []
    val_log: list[dict] = []
    preview_batch = None
    validation_preview_dir = args.output_dir / f"{args.run_name}_validation_previews"
    validation_preview_paths: list[str] = []
    latest_path = args.output_dir / f"{args.run_name}_latest.pt"
    best_path = args.output_dir / f"{args.run_name}_best.pt"
    compat_model_path = args.output_dir / f"{args.run_name}_model.pt"
    latest_checkpoint_written_path = latest_path
    best_checkpoint_written_path = best_path
    compat_checkpoint_written_path = compat_model_path
    global_step = int(start_step)
    requested_additional_steps = int(args.steps)
    step_limit = (start_step + requested_additional_steps) if requested_additional_steps > 0 else None
    extra_epochs_for_step_cap = int(math.ceil(requested_additional_steps / max(1, train_batches_per_epoch))) + 1
    effective_epochs = int(args.epochs)
    if step_limit is not None:
        effective_epochs = max(effective_epochs, int(start_epoch) + extra_epochs_for_step_cap)

    t0 = time.perf_counter()
    tiles_seen = 0
    val_tiles_seen = 0
    val_elapsed_total = 0.0

    for epoch_idx in range(int(start_epoch), effective_epochs):
        if step_limit is not None and global_step >= step_limit:
            break

        epoch_number = epoch_idx + 1
        epoch_train_rows: list[dict[str, float]] = []
        model.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch_idx > train_batches_per_epoch:
                break
            if step_limit is not None and global_step >= step_limit:
                break
            loss, metrics = _train_one_batch(batch, model)
            batch_size = int(batch["input_prior"].shape[0])
            tiles_seen += batch_size
            elapsed = time.perf_counter() - t0
            rate = tiles_seen / max(elapsed, 1e-9)
            entry = {
                "epoch": epoch_number,
                "batch": batch_idx,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "tiles_per_sec": rate,
                "batch_size": batch_size,
                "tile_id": _extract_first_tile_id(batch),
            }
            for k, v in metrics.items():
                entry[k] = float(v)
            metrics_log.append(entry)
            epoch_train_rows.append(entry)
            if global_step == start_step or (int(args.log_interval) > 0 and global_step % int(args.log_interval) == 0):
                print(
                    f"epoch {epoch_number}/{effective_epochs} batch {batch_idx}/{train_batches_per_epoch} "
                    f"step {global_step}: loss={float(loss.detach()):.4f} "
                    f"tiles/s={rate:.2f} first_tile_id={entry['tile_id']}",
                    flush=True,
                )
            if preview_batch is None:
                preview_batch = batch
            global_step += 1

        model.eval()
        epoch_val_rows: list[dict[str, float]] = []
        epoch_preview_batch = None
        val_t0 = time.perf_counter()
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader, start=1):
                if val_batch_idx > val_batches_per_epoch:
                    break
                if epoch_preview_batch is None:
                    epoch_preview_batch = batch
                loss, metrics = _validate_batch(batch, model)
                batch_size = int(batch["input_prior"].shape[0])
                val_tiles_seen += batch_size
                entry = {
                    "epoch": epoch_number,
                    "batch": val_batch_idx,
                    "step": global_step,
                    "val_loss": float(loss.detach().cpu()),
                    "batch_size": batch_size,
                    "tile_id": _extract_first_tile_id(batch),
                }
                for k, val in metrics.items():
                    entry[k] = float(val)
                val_log.append(entry)
                epoch_val_rows.append(entry)
        val_elapsed_total += time.perf_counter() - val_t0

        preview_path_for_epoch = None
        if (
            int(args.preview_every_epochs) > 0
            and epoch_preview_batch is not None
            and epoch_number % int(args.preview_every_epochs) == 0
        ):
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_amp):
                    preview_pred = model(epoch_preview_batch["input_prior"].to(device, non_blocking=True)[:, :3, :, :])
            preview_path_for_epoch = validation_preview_dir / f"epoch_{epoch_number:04d}.png"
            _save_deconstruction_preview(
                batch=epoch_preview_batch,
                pred=preview_pred,
                out_path=preview_path_for_epoch,
                max_samples=max(1, int(args.preview_samples)),
            )
            validation_preview_paths.append(str(preview_path_for_epoch))

        epoch_train_loss = _mean_metric(epoch_train_rows, "loss")
        epoch_val_loss = _mean_metric(epoch_val_rows, "val_loss")
        lr_before = _current_lr(optimizer)
        improved = False
        if epoch_val_loss is not None:
            improved = epoch_val_loss < (best_val - float(args.early_stop_min_improvement))
            if improved:
                best_val = float(epoch_val_loss)
                early_stop_counter = 0
            elif int(args.early_stop_patience) > 0:
                early_stop_counter += 1
            if scheduler is not None:
                scheduler.step(float(epoch_val_loss))
        lr_after = _current_lr(optimizer)

        epoch_record = {
            "epoch": epoch_number,
            "global_step": global_step,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "learning_rate": lr_after,
            "learning_rate_before_scheduler": lr_before,
            "learning_rate_changed": bool(abs(lr_after - lr_before) > 1e-12),
            "train_batches": len(epoch_train_rows),
            "val_batches": len(epoch_val_rows),
            "best_val": best_val if best_val != float("inf") else None,
            "improved": improved,
            "preview_path": str(preview_path_for_epoch) if preview_path_for_epoch is not None else None,
        }
        history.append(epoch_record)
        latest_checkpoint_written_path = _save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch_number,
            global_step=global_step,
            best_val=best_val,
            args=args,
            history=history,
            scheduler=scheduler,
        )
        compat_checkpoint_written_path = _save_checkpoint(
            compat_model_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch_number,
            global_step=global_step,
            best_val=best_val,
            args=args,
            history=history,
            scheduler=scheduler,
        )
        if improved or not best_path.exists():
            best_checkpoint_written_path = _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                epoch=epoch_number,
                global_step=global_step,
                best_val=best_val,
                args=args,
                history=history,
                scheduler=scheduler,
            )
        print(
            f"epoch {epoch_number}: train_loss={epoch_train_loss if epoch_train_loss is not None else 'n/a'} "
            f"val_loss={epoch_val_loss if epoch_val_loss is not None else 'n/a'} "
            f"best_val={best_val if best_val != float('inf') else 'n/a'} "
            f"lr={lr_after:.3e} step={global_step} preview={preview_path_for_epoch if preview_path_for_epoch is not None else 'n/a'}",
            flush=True,
        )

        if int(args.early_stop_patience) > 0 and early_stop_counter >= int(args.early_stop_patience):
            early_stop_triggered = True
            print(f"Early stop: no val improvement for {early_stop_counter} epochs", flush=True)
            break

    if not latest_path.exists():
        latest_checkpoint_written_path = _save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            epoch=int(start_epoch),
            global_step=global_step,
            best_val=best_val,
            args=args,
            history=history,
            scheduler=scheduler,
        )
        compat_checkpoint_written_path = _save_checkpoint(
            compat_model_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            epoch=int(start_epoch),
            global_step=global_step,
            best_val=best_val,
            args=args,
            history=history,
            scheduler=scheduler,
        )
    val_rate = val_tiles_seen / max(val_elapsed_total, 1e-9)

    # Preview: use the first captured batch; render a labeled 4-panel image.
    preview_path = args.output_dir / f"{args.run_name}_preview.png"
    if preview_batch is not None:
        prior_chw = preview_batch["input_prior"][0, :3].cpu()
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
        truth_disp = _norm(truth.squeeze(0))
        pred_disp = _norm(pred_chw.squeeze(0))
        weight_disp = _norm(weight.squeeze(0))
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
        "epoch_count": len(history),
        "requested_epochs": int(args.epochs),
        "requested_steps": int(args.steps),
        "step_count": len(metrics_log),
        "global_step": int(global_step),
        "val_step_count": len(val_log),
        "steps_per_epoch": int(train_batches_per_epoch),
        "full_steps_per_epoch": int(full_steps_per_epoch),
        "val_batches_per_epoch": int(val_batches_per_epoch),
        "learning_rate": args.learning_rate,
        "resume_learning_rate": float(args.resume_learning_rate),
        "current_learning_rate": _current_lr(optimizer),
        "lr_plateau_enabled": scheduler is not None,
        "lr_plateau_patience": int(args.lr_plateau_patience),
        "lr_plateau_factor": float(args.lr_plateau_factor),
        "lr_plateau_min_delta": float(args.lr_plateau_min_delta),
        "min_learning_rate": float(args.min_learning_rate),
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "source_count": len(args.prior),
        "prior_paths": [str(path) for path in args.prior],
        "v18_paths": [str(path) if path is not None else None for path in v18_paths],
        "curation_manifest": str(args.curation_manifest) if args.curation_manifest is not None else None,
        "dataset_tile_count": len(dataset),
        "train_tile_count": len(train_dataset),
        "val_tile_count": len(val_dataset),
        "val_fraction": float(args.val_fraction),
        "dataset_sources": source_summaries,
        "device": str(device),
        "compile_status": compile_status,
        "amp_enabled": use_amp,
        "grad_clip": float(args.grad_clip),
        "multiscale_weight": float(args.multiscale_weight),
        "gradient_weight": float(args.gradient_weight),
        "normal_consistency_weight": float(args.normal_consistency_weight),
        "normal_guidance_weight": float(args.normal_guidance_weight),
        "normal_guidance_spacing": float(args.normal_guidance_spacing),
        "hard_error_weight": float(args.hard_error_weight),
        "hard_error_power": float(args.hard_error_power),
        "hard_error_max_multiplier": float(args.hard_error_max_multiplier),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_triggered": early_stop_triggered,
        "best_val": best_val if best_val != float("inf") else None,
        "model_parameter_count": sum(p.numel() for p in (model._orig_mod.parameters() if hasattr(model, "_orig_mod") else model.parameters())),
        "num_workers": num_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": persistent_workers,
        "history": history,
        "train_metrics": metrics_log,
        "val_metrics": val_log,
        "val_tiles_per_sec": val_rate,
        "preview_path": str(preview_path),
        "validation_preview_dir": str(validation_preview_dir),
        "validation_preview_paths": validation_preview_paths,
        "latest_checkpoint_path": str(latest_checkpoint_written_path),
        "best_checkpoint_path": str(best_checkpoint_written_path),
        "compat_checkpoint_path": str(compat_checkpoint_written_path),
        "preferred_latest_checkpoint_path": str(latest_path),
        "preferred_best_checkpoint_path": str(best_path),
        "preferred_compat_checkpoint_path": str(compat_model_path),
        "seed": args.seed,
    }
    metrics_path = args.output_dir / f"{args.run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote height-only training output to {args.output_dir}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main_with_args())
