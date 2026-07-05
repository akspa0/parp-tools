"""Train the Spec 089 V23 height predictor."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v23.channels import IMAGENET_MEAN, IMAGENET_STD, InputMode
from harvester.v23.checkpoint import (
    V23Checkpoint,
    load_checkpoint,
    path_hash,
    resolve_commit_sha,
    save_checkpoint,
)
from harvester.v23.dataset import V23HeightDataset
from harvester.v23.losses import apply_bias_free_masking, compute_v23_loss
from harvester.v23.model import V23HeightPredictor
from transformers import DepthAnythingConfig

try:
    import bitsandbytes as bnb
except Exception:  # pragma: no cover - CPU fallback
    bnb = None


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v22"
_DEFAULT_MODEL_ROOT = _PROJECT_ROOT / "models" / "v23" / "height" / "runs"


def _resolve_amp_dtype(device: torch.device, amp_dtype: str) -> tuple[torch.dtype | None, str]:
    if device.type != "cuda":
        return None, "disabled"
    if amp_dtype == "bf16":
        return torch.bfloat16, "bf16"
    if amp_dtype == "fp16":
        return torch.float16, "fp16"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def _apply_memory_profile(args: argparse.Namespace) -> argparse.Namespace:
    runtime = argparse.Namespace(**vars(args))
    profile = str(runtime.memory_profile)
    if profile == "auto":
        target_vram = float(runtime.target_vram_gb)
        if target_vram <= 12.5:
            profile = "12gb"
        elif target_vram <= 24.5:
            profile = "24gb"
        else:
            profile = "none"
    runtime.effective_memory_profile = profile

    if profile == "12gb":
        runtime.batch_size = min(int(runtime.batch_size), 1)
        if float(runtime.gpct_weight) > 0.0:
            runtime.gpct_K = min(int(runtime.gpct_K), 2)
            runtime.grad_accum_steps = max(int(runtime.grad_accum_steps), 4)
        else:
            runtime.grad_accum_steps = max(int(runtime.grad_accum_steps), 2)
        if str(runtime.amp_dtype) == "auto":
            runtime.amp_dtype = "fp16"
    elif profile == "24gb":
        if float(runtime.gpct_weight) > 0.0:
            runtime.grad_accum_steps = max(int(runtime.grad_accum_steps), 2)
        if str(runtime.amp_dtype) == "auto":
            runtime.amp_dtype = "bf16"

    return runtime


def _resolve_autotune_batch_candidates(base_batch_size: int, requested: list[int] | None) -> list[int]:
    candidates = [int(base_batch_size)]
    if requested:
        candidates.extend(int(value) for value in list(requested))
    else:
        candidates.extend([2, 4, 8, 12, 16, 24, 32])
    return sorted({value for value in candidates if value > 0})


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _effective_seed(args: argparse.Namespace) -> int:
    return 0 if bool(args.deterministic) else int(args.seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_config(path: str | None) -> DepthAnythingConfig | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DepthAnythingConfig.from_dict(payload)


def _subset(dataset: torch.utils.data.Dataset, limit: int | None) -> torch.utils.data.Dataset:
    if limit is None or limit <= 0 or len(dataset) <= limit:
        return dataset
    return Subset(dataset, list(range(int(limit))))


def _curation_score(dataset: torch.utils.data.Dataset, idx: int) -> float:
    if hasattr(dataset, "curation_mismatch_score"):
        return float(dataset.curation_mismatch_score(idx))  # type: ignore[attr-defined]
    return 0.0


def _has_curation(dataset: torch.utils.data.Dataset) -> bool:
    if not hasattr(dataset, "curation_metadata"):
        return False
    return any(dataset.curation_metadata(idx) is not None for idx in range(len(dataset)))  # type: ignore[attr-defined]


def _curated_train_val_indices(
    dataset: torch.utils.data.Dataset,
    *,
    train_max_tiles: int | None,
    val_max_tiles: int | None,
) -> tuple[list[int], list[int]]:
    indices = list(range(len(dataset)))
    if not indices:
        return [], []

    ranked = sorted(indices, key=lambda idx: (_curation_score(dataset, idx), idx), reverse=True)
    val_take = int(val_max_tiles) if val_max_tiles and int(val_max_tiles) > 0 else max(1, math.ceil(len(indices) * 0.10))
    val_take = min(val_take, max(1, len(indices) - 1)) if len(indices) > 1 else 1
    val_set = set(ranked[:val_take])
    train_indices = [idx for idx in ranked if idx not in val_set]
    val_indices = [idx for idx in ranked if idx in val_set]
    if train_max_tiles and int(train_max_tiles) > 0:
        train_indices = train_indices[: int(train_max_tiles)]
    return train_indices, val_indices


def _build_dataset_for_build(
    dataset_dir: Path,
    build: str,
    *,
    maps: list[str] | None,
    input_mode: str,
    tileset_prune_table: str | None,
    curation_manifest: str | None,
    curation_min_terrain_validity: float,
    curation_min_minimap_usefulness: float,
    curation_max_liquid_coverage: float,
    curation_reject_what_plate: bool,
) -> V23HeightDataset:
    return V23HeightDataset(
        dataset_dir / f"{build}.zarr",
        build=build,
        maps=maps,
        input_mode=input_mode,
        tileset_prune_table=tileset_prune_table,
        curation_manifest=curation_manifest,
        curation_min_terrain_validity=curation_min_terrain_validity,
        curation_min_minimap_usefulness=curation_min_minimap_usefulness,
        curation_max_liquid_coverage=curation_max_liquid_coverage,
        curation_reject_what_plate=curation_reject_what_plate,
    )


def _split_datasets(
    dataset_dir: Path,
    builds: list[str],
    *,
    maps: list[str] | None,
    input_mode: str,
    tileset_prune_table: str | None,
    curation_manifest: str | None,
    curation_min_terrain_validity: float,
    curation_min_minimap_usefulness: float,
    curation_max_liquid_coverage: float,
    curation_reject_what_plate: bool,
    train_max_tiles: int | None,
    val_max_tiles: int | None,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_parts = []
    val_parts = []
    for build in builds:
        dataset = _build_dataset_for_build(
            dataset_dir,
            build,
            maps=maps,
            input_mode=input_mode,
            tileset_prune_table=tileset_prune_table,
            curation_manifest=curation_manifest,
            curation_min_terrain_validity=curation_min_terrain_validity,
            curation_min_minimap_usefulness=curation_min_minimap_usefulness,
            curation_max_liquid_coverage=curation_max_liquid_coverage,
            curation_reject_what_plate=curation_reject_what_plate,
        )
        if _has_curation(dataset):
            train_indices, val_indices = _curated_train_val_indices(
                dataset,
                train_max_tiles=train_max_tiles,
                val_max_tiles=val_max_tiles,
            )
            train_parts.append(Subset(dataset, train_indices))
            val_parts.append(Subset(dataset, val_indices))
        else:
            indices = list(range(len(dataset)))
            split = max(1, math.ceil(len(indices) * 0.8))
            train_indices = indices[:split]
            val_indices = indices[split:] or indices[:1]
            train_parts.append(_subset(Subset(dataset, train_indices), train_max_tiles))
            val_parts.append(_subset(Subset(dataset, val_indices), val_max_tiles))
    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_dataset = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    return train_dataset, val_dataset


def _make_optimizer(model: torch.nn.Module, device: torch.device, lr: float):
    if device.type == "cuda" and bnb is not None:
        return bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _denormalize_rgb(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=tensor.device, dtype=tensor.dtype)
    std = IMAGENET_STD.to(device=tensor.device, dtype=tensor.dtype)
    return (tensor * std + mean).clamp(0.0, 1.0)


def _gray_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    if arr.ndim == 2:
        arr = np.repeat(arr[None, :, :], 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0))
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255.0).astype(np.uint8)


def _save_val_preview(batch: dict[str, Any], outputs: Any, preview_dir: Path, epoch: int) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = min(2, int(batch["input"].shape[0]))
    pred = outputs.metric_height.detach().cpu()
    target = batch["target_height"].detach().cpu()
    for idx in range(count):
        minimap = _denormalize_rgb(batch["input"][idx, 0:3]).cpu().permute(1, 2, 0).numpy()
        minimap = np.pad(minimap, ((0, 1), (0, 1), (0, 0)), mode="edge")
        target_img = _gray_rgb(target[idx])
        pred_img = _gray_rgb(pred[idx])
        error_img = _gray_rgb((pred[idx] - target[idx]).abs())
        minimap_img = (np.clip(minimap, 0.0, 1.0) * 255.0).astype(np.uint8)
        strip = np.concatenate([minimap_img, target_img, pred_img, error_img], axis=1)
        image = Image.fromarray(strip, mode="RGB")
        labeled = Image.new("RGB", (image.width, image.height + 36), color=(18, 18, 18))
        labeled.paste(image, (0, 36))
        draw = ImageDraw.Draw(labeled)
        labels = ("minimap", "target_height", "pred_height", "abs_error")
        for panel_idx, label in enumerate(labels):
            draw.text((panel_idx * 257 + 6, 4), label, fill=(235, 235, 235))
        tile_ref = f"{batch.get('map', [''])[idx]} {int(batch.get('tile_x', [-1])[idx])},{int(batch.get('tile_y', [-1])[idx])}"
        if "curation_difficulty_bucket" in batch:
            bucket = batch["curation_difficulty_bucket"][idx]
            score = float(batch.get("curation_mismatch_score", torch.zeros(count))[idx])
            tile_ref = f"{tile_ref} bucket={bucket} mismatch={score:.3f}"
        draw.text((6, 20), tile_ref, fill=(205, 205, 205))
        labeled.save(preview_dir / f"tile_{idx}.png")


def _shift_view(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    shifted = torch.zeros_like(x)
    src_y0 = max(dy, 0)
    src_y1 = x.shape[-2] + min(dy, 0)
    dst_y0 = max(-dy, 0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    src_x0 = max(dx, 0)
    src_x1 = x.shape[-1] + min(dx, 0)
    dst_x0 = max(-dx, 0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return shifted


def _gpct_views(x: torch.Tensor, k: int) -> tuple[list[torch.Tensor], list[tuple[int, ...]]]:
    shift = max(8, x.shape[-1] // 8)
    offsets = [(0, 0)]
    if k >= 2:
        offsets.append((0, shift))
    if k >= 3:
        offsets.append((shift, 0))
    if k >= 4:
        offsets.append((shift, shift))
    while len(offsets) < k:
        offsets.append((0, 0))
    views = [_shift_view(x, dy, dx) for dy, dx in offsets]
    overlaps: list[tuple[int, ...]] = []
    for idx, (dy, dx) in enumerate(offsets[1:], start=1):
        height = 257 - abs(dy)
        width = 257 - abs(dx)
        overlaps.append(
            (
                0,
                idx,
                max(0, dy),
                max(0, dx),
                max(0, -dy),
                max(0, -dx),
                height,
                width,
                0,
                0,
            )
        )
    return views, overlaps


def _format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_cuda_status(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    return f" gpu_alloc_gb={_format_metric(allocated)} gpu_reserved_gb={_format_metric(reserved)}"


def _loss_value(metrics: dict[str, float]) -> float:
    if "loss" in metrics:
        return float(metrics["loss"])
    return float(metrics.get("total", 0.0))


def _format_loss_components(metrics: dict[str, float], *, lr: float | None = None) -> str:
    parts = [f"loss={_format_metric(_loss_value(metrics))}"]
    component_labels = {
        "affine": "affine_loss",
        "gradient": "gradient_loss",
        "sdc": "sdc_loss",
        "gpct": "gpct_loss",
    }
    for key, label in component_labels.items():
        if key in metrics:
            parts.append(f"{label}={_format_metric(metrics.get(key))}")
    if lr is not None:
        parts.append(f"lr={_format_metric(lr)}")
    return " ".join(parts)


def _should_log_batch(batch_index: int, total_batches: int, log_interval: int) -> bool:
    if log_interval <= 0:
        return False
    return batch_index == 1 or batch_index == total_batches or batch_index % log_interval == 0


def _print_status(message: str) -> None:
    print(f"[v23] {message}", flush=True)


def _loss_event_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "loss": _loss_value(metrics),
        "affine_loss": float(metrics.get("affine", 0.0)),
        "gradient_loss": float(metrics.get("gradient", 0.0)),
        "sdc_loss": float(metrics.get("sdc", 0.0)),
        "gpct_loss": float(metrics.get("gpct", 0.0)),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _should_validate_epoch(epoch: int, total_epochs: int, val_interval: int) -> bool:
    if val_interval <= 0:
        return False
    return epoch == total_epochs or epoch % val_interval == 0


def _run_loader(
    *,
    model: V23HeightPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    mask_generator: torch.Generator,
    epoch: int,
    total_epochs: int,
    phase: str,
    loss_history_path: Path | None,
    global_step_start: int,
) -> tuple[float, dict[str, float], tuple[dict[str, Any], Any] | None]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    batches = 0
    preview_payload: tuple[dict[str, Any], Any] | None = None
    amp_dtype, amp_label = _resolve_amp_dtype(device, str(args.amp_dtype))
    use_amp = device.type == "cuda" and amp_dtype is not None
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    total_batches = len(loader)
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else total_batches * int(args.batch_size)
    log_interval = int(getattr(args, "log_interval", 1))
    samples_seen = 0
    phase_start = time.monotonic()

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        should_log = _should_log_batch(batch_index, total_batches, log_interval)
        step_number = global_step_start + batch_index if is_train else batch_index
        if should_log:
            _print_status(
                f"epoch={epoch}/{total_epochs} phase={phase} status=start step={step_number} "
                f"batch={batch_index}/{total_batches}"
            )
        inputs = batch["input"].to(device, non_blocking=True).float()
        target = batch["target_height"].to(device, non_blocking=True).float()
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()
        batch_samples = int(inputs.shape[0])

        if is_train and float(args.bias_free_mask_ratio) > 0.0:
            inputs, _ = apply_bias_free_masking(
                inputs,
                ratio=float(args.bias_free_mask_ratio),
                generator=mask_generator,
            )

        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            outputs = model(inputs)
            sub_tile_preds = None
            overlap_coords = None
            if is_train and float(args.gpct_weight) > 0.0:
                gpct_views, overlap_coords = _gpct_views(inputs, int(args.gpct_K))
                sub_tile_preds = [model(view).metric_height for view in gpct_views]
            total_loss, components = compute_v23_loss(
                outputs,
                target,
                {
                    "affine": 1.0,
                    "gradient": 0.5,
                    "sdc": float(args.sdc_weight),
                    "gpct": float(args.gpct_weight),
                    "gpct_feature": 1.0 if bool(args.gpct_feature_loss) else 0.0,
                },
                valid_mask=valid_mask,
                sub_tile_preds=sub_tile_preds,
                overlap_coords=overlap_coords,
            )

        if optimizer is not None:
            (total_loss / grad_accum_steps).backward()
            should_step = (batch_index % grad_accum_steps == 0) or (batch_index == len(loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batches += 1
        samples_seen += batch_samples
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        if preview_payload is None:
            preview_payload = (batch, outputs)
        batch_metrics = {key: float(value.detach().cpu()) for key, value in components.items()}
        running = {key: value / batches for key, value in totals.items()}
        if loss_history_path is not None:
            event = {
                "type": "batch",
                "epoch": int(epoch),
                "epochs": int(total_epochs),
                "phase": phase,
                "batch": int(batch_index),
                "batches": int(total_batches),
                "global_step": int(global_step_start + batch_index - 1) if is_train else None,
                **_loss_event_metrics(batch_metrics),
                "running_loss": _loss_value(running),
            }
            lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None
            if lr is not None:
                event["lr"] = float(lr)
            _append_jsonl(loss_history_path, event)
        if should_log:
            lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None
            elapsed = time.monotonic() - phase_start
            eta = (elapsed / max(1, batches)) * max(0, total_batches - batch_index)
            pct = (batch_index / max(1, total_batches)) * 100.0
            optimizer_step = (
                "yes"
                if optimizer is not None and ((batch_index % grad_accum_steps == 0) or (batch_index == total_batches))
                else "no"
            )
            _print_status(
                f"epoch={epoch}/{total_epochs} phase={phase} status=done step={step_number} "
                f"batch={batch_index}/{total_batches} samples={samples_seen}/{total_samples} pct={pct:.1f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} optimizer_step={optimizer_step} "
                f"{_format_loss_components(running, lr=lr)}{_format_cuda_status(device)}"
            )

    if batches == 0:
        return 0.0, {}, None
    averaged = {key: value / batches for key, value in totals.items()}
    averaged["loss"] = averaged.get("total", 0.0)
    del amp_label
    return averaged["loss"], averaged, preview_payload


def _build_checkpoint_config(
    args: argparse.Namespace,
    effective_seed: int,
    *,
    in_channels: int,
    model_config: DepthAnythingConfig | None,
) -> dict[str, Any]:
    amp_dtype, amp_label = _resolve_amp_dtype(_resolve_device(args.device), str(args.amp_dtype))
    del amp_dtype
    return {
        "seed": int(args.seed),
        "effective_seed": int(effective_seed),
        "commit_sha": resolve_commit_sha(),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_dir_hash": path_hash(args.dataset_dir),
        "builds": list(args.builds),
        "maps": list(args.maps or []),
        "input_mode": str(args.input_mode),
        "tileset_prune_table": args.tileset_prune_table,
        "tileset_prune_table_hash": path_hash(args.tileset_prune_table),
        "curation_manifest": args.curation_manifest,
        "curation_manifest_hash": path_hash(args.curation_manifest),
        "curation_min_terrain_validity": float(args.curation_min_terrain_validity),
        "curation_min_minimap_usefulness": float(args.curation_min_minimap_usefulness),
        "curation_max_liquid_coverage": float(args.curation_max_liquid_coverage),
        "curation_reject_what_plate": bool(args.curation_reject_what_plate),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accum_steps),
        "autotune_batch_size": bool(args.autotune_batch_size),
        "autotune_batch_candidates": list(args.autotune_batch_candidates or []),
        "autotune_safety_factor": float(args.autotune_safety_factor),
        "gpct_weight": float(args.gpct_weight),
        "gpct_K": int(args.gpct_K),
        "gpct_feature_loss": bool(args.gpct_feature_loss),
        "sdc_weight": float(args.sdc_weight),
        "spectral_weight": float(args.spectral_weight),
        "bias_free_mask_ratio": float(args.bias_free_mask_ratio),
        "target_vram_gb": float(args.target_vram_gb),
        "memory_profile": str(args.memory_profile),
        "effective_memory_profile": str(getattr(args, "effective_memory_profile", args.memory_profile)),
        "amp_dtype": amp_label,
        "deterministic": bool(args.deterministic),
        "pretrained": bool(args.pretrained),
        "model_config_json": args.model_config_json,
        "model_config": model_config.to_dict() if model_config is not None else None,
        "in_channels": int(in_channels),
    }


class _RetryTrainingConfig(RuntimeError):
    pass


def _probe_batch_size(
    *,
    model: V23HeightPredictor,
    dataset: torch.utils.data.Dataset,
    candidate: int,
    device: torch.device,
    args: argparse.Namespace,
    mask_generator: torch.Generator,
) -> dict[str, Any]:
    if device.type != "cuda":
        return {"batch_size": int(candidate), "ok": False, "reason": "non_cuda"}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    probe_args = argparse.Namespace(**vars(args))
    probe_args.batch_size = int(candidate)
    loader = DataLoader(dataset, batch_size=int(candidate), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model.train(True)
    model.zero_grad(set_to_none=True)

    amp_dtype, _ = _resolve_amp_dtype(device, str(probe_args.amp_dtype))
    use_amp = device.type == "cuda" and amp_dtype is not None
    try:
        inputs = batch["input"].to(device, non_blocking=True).float()
        target = batch["target_height"].to(device, non_blocking=True).float()
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()
        if float(probe_args.bias_free_mask_ratio) > 0.0:
            inputs, _ = apply_bias_free_masking(
                inputs,
                ratio=float(probe_args.bias_free_mask_ratio),
                generator=mask_generator,
            )

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(inputs)
            sub_tile_preds = None
            overlap_coords = None
            if float(probe_args.gpct_weight) > 0.0:
                gpct_views, overlap_coords = _gpct_views(inputs, int(probe_args.gpct_K))
                sub_tile_preds = [model(view).metric_height for view in gpct_views]
            total_loss, _ = compute_v23_loss(
                outputs,
                target,
                {
                    "affine": 1.0,
                    "gradient": 0.5,
                    "sdc": float(probe_args.sdc_weight),
                    "gpct": float(probe_args.gpct_weight),
                    "gpct_feature": 1.0 if bool(probe_args.gpct_feature_loss) else 0.0,
                },
                valid_mask=valid_mask,
                sub_tile_preds=sub_tile_preds,
                overlap_coords=overlap_coords,
            )
        total_loss.backward()
        result = {
            "batch_size": int(candidate),
            "ok": True,
            "loss": float(total_loss.detach().cpu()),
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        }
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "batch_size": int(candidate),
            "ok": False,
            "reason": "cuda_oom",
            "error": str(exc).splitlines()[0] if str(exc) else "CUDA out of memory",
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        }
    finally:
        model.zero_grad(set_to_none=True)
        del batch
        if "inputs" in locals():
            del inputs
        if "target" in locals():
            del target
        if "valid_mask" in locals():
            del valid_mask
        if "outputs" in locals():
            del outputs
        if "sub_tile_preds" in locals():
            del sub_tile_preds
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _autotune_batch_size(
    *,
    model: V23HeightPredictor,
    train_dataset: torch.utils.data.Dataset,
    device: torch.device,
    args: argparse.Namespace,
    run_dir: Path,
    mask_generator: torch.Generator,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "autotune_batch_size", False)):
        return None

    evidence_path = run_dir / "batch_autotune.json"
    if device.type != "cuda":
        payload = {
            "enabled": True,
            "skipped": True,
            "reason": "non_cuda_device",
            "batch_size": int(args.batch_size),
        }
        evidence_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _print_status(f"autotune skipped reason=non_cuda_device path={evidence_path}")
        return payload

    candidates = [
        value
        for value in _resolve_autotune_batch_candidates(int(args.batch_size), args.autotune_batch_candidates)
        if value <= len(train_dataset)
    ]
    if not candidates:
        candidates = [int(args.batch_size)]
    effective_target_vram_gb = float(args.target_vram_gb) * float(args.autotune_safety_factor)
    results: list[dict[str, Any]] = []
    selected = int(args.batch_size)
    _print_status(
        f"autotune start candidates={candidates} target_vram_gb={float(args.target_vram_gb):g} "
        f"effective_target_gb={effective_target_vram_gb:g}"
    )

    for candidate in candidates:
        result = _probe_batch_size(
            model=model,
            dataset=train_dataset,
            candidate=int(candidate),
            device=device,
            args=args,
            mask_generator=mask_generator,
        )
        decision_reserved_gb = float(result.get("max_memory_reserved_gb", math.inf))
        fits_target = bool(result.get("ok")) and decision_reserved_gb <= effective_target_vram_gb
        result["fits_target"] = fits_target
        results.append(result)
        _print_status(
            f"autotune candidate={candidate} ok={bool(result.get('ok'))} "
            f"reserved_gb={_format_metric(result.get('max_memory_reserved_gb'))} "
            f"allocated_gb={_format_metric(result.get('max_memory_allocated_gb'))} "
            f"fits={fits_target}"
        )
        if fits_target:
            selected = int(candidate)
            continue
        if not bool(result.get("ok")):
            break

    args.batch_size = int(selected)
    payload = {
        "enabled": True,
        "target_vram_gb": float(args.target_vram_gb),
        "effective_target_vram_gb": effective_target_vram_gb,
        "safety_factor": float(args.autotune_safety_factor),
        "candidates": candidates,
        "selected_batch_size": int(selected),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int(selected) * int(args.grad_accum_steps),
        "gpct_K": int(args.gpct_K),
        "gpct_weight": float(args.gpct_weight),
        "results": results,
    }
    evidence_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_status(
        f"autotune selected_batch_size={selected} effective_batch_size={int(selected) * int(args.grad_accum_steps)} "
        f"path={evidence_path}"
    )
    torch.cuda.reset_peak_memory_stats(device)
    return payload


def _write_peak_vram_json(run_dir: Path, device: torch.device, args: argparse.Namespace) -> None:
    if device.type != "cuda":
        return
    peak_path = run_dir / "peak_vram.json"
    payload = {
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "target_vram_gb": float(args.target_vram_gb),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accum_steps),
        "autotune_batch_size": bool(args.autotune_batch_size),
        "gpct_K": int(args.gpct_K),
        "gpct_weight": float(args.gpct_weight),
    }
    peak_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _train_once(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    effective_seed = _effective_seed(args)
    _seed_everything(effective_seed)
    if bool(args.deterministic):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    dataset_dir = Path(args.dataset_dir)
    train_dataset, val_dataset = _split_datasets(
        dataset_dir,
        list(args.builds),
        maps=list(args.maps or []),
        input_mode=args.input_mode,
        tileset_prune_table=args.tileset_prune_table,
        curation_manifest=args.curation_manifest,
        curation_min_terrain_validity=float(args.curation_min_terrain_validity),
        curation_min_minimap_usefulness=float(args.curation_min_minimap_usefulness),
        curation_max_liquid_coverage=float(args.curation_max_liquid_coverage),
        curation_reject_what_plate=bool(args.curation_reject_what_plate),
        train_max_tiles=args.train_max_tiles,
        val_max_tiles=args.val_max_tiles,
    )

    run_dir = Path(args.output_dir) if args.output_dir else (_DEFAULT_MODEL_ROOT / args.run_name)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _, amp_label = _resolve_amp_dtype(device, str(args.amp_dtype))
    maps_label = ",".join(str(value) for value in (args.maps or [])) or "all"
    curation_label = str(args.curation_manifest) if args.curation_manifest else "none"
    _print_status(f"run={args.run_name} run_dir={run_dir}")
    _print_status(
        f"data dataset_dir={dataset_dir} builds={','.join(str(value) for value in args.builds)} "
        f"maps={maps_label} curation_manifest={curation_label}"
    )

    sample_channels = train_dataset[0]["input"].shape[0]
    model_config = _load_model_config(args.model_config_json)
    _print_status(
        f"model in_channels={int(sample_channels)} pretrained={bool(args.pretrained)} "
        f"model_config_json={args.model_config_json or 'default'}"
    )
    model = V23HeightPredictor(
        in_channels=int(sample_channels),
        config=model_config,
        load_pretrained=bool(args.pretrained),
    ).to(device)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable()

    mask_generator = torch.Generator(device="cpu").manual_seed(effective_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _autotune_batch_size(
        model=model,
        train_dataset=train_dataset,
        device=device,
        args=args,
        run_dir=run_dir,
        mask_generator=mask_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=not bool(args.deterministic),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )
    effective_batch = int(args.batch_size) * int(args.grad_accum_steps)
    _print_status(
        f"loader train_tiles={len(train_dataset)} val_tiles={len(val_dataset)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )
    _print_status(
        f"device={device} memory_profile={getattr(args, 'effective_memory_profile', args.memory_profile)} "
        f"amp={amp_label} target_vram_gb={float(args.target_vram_gb):g} "
        f"batch_size={int(args.batch_size)} grad_accum_steps={int(args.grad_accum_steps)} "
        f"effective_batch_size={effective_batch} gpct_K={int(args.gpct_K)} "
        f"gpct_weight={float(args.gpct_weight):g} autotune_batch_size={bool(args.autotune_batch_size)} "
        f"log_interval={int(args.log_interval)}"
    )

    optimizer = _make_optimizer(model, device, float(args.lr))
    start_epoch = 1
    best_val = float("inf")
    global_step = 0
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        start_epoch = checkpoint.epoch + 1
        global_step = int(checkpoint.global_step)
        if checkpoint.best_val is not None:
            best_val = float(checkpoint.best_val)

    loss_history_path = run_dir / "loss_history.jsonl"
    if start_epoch <= 1:
        loss_history_path.write_text("", encoding="utf-8")
    elif not loss_history_path.exists():
        loss_history_path.write_text("", encoding="utf-8")
    _print_status(f"loss_history path={loss_history_path}")

    history: list[dict[str, Any]] = []
    total_epochs = int(args.epochs)
    for epoch in range(start_epoch, total_epochs + 1):
        _print_status(f"epoch={epoch}/{total_epochs} start")
        try:
            train_loss, train_metrics, _ = _run_loader(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                args=args,
                mask_generator=mask_generator,
                epoch=epoch,
                total_epochs=total_epochs,
                phase="train",
                loss_history_path=loss_history_path,
                global_step_start=global_step,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda" and int(args.batch_size) > 1:
                torch.cuda.empty_cache()
                gc.collect()
                raise _RetryTrainingConfig(str(exc))
            if device.type == "cuda" and float(args.gpct_weight) > 0.0 and int(args.gpct_K) > 1:
                torch.cuda.empty_cache()
                gc.collect()
                raise _RetryTrainingConfig(str(exc))
            raise

        should_validate = _should_validate_epoch(epoch, total_epochs, int(args.val_interval))
        val_loss: float | None = None
        val_metrics: dict[str, float] = {}
        preview: tuple[dict[str, Any], Any] | None = None
        if should_validate:
            with torch.inference_mode():
                val_loss, val_metrics, preview = _run_loader(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    args=args,
                    mask_generator=mask_generator,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    phase="val",
                    loss_history_path=loss_history_path,
                    global_step_start=global_step,
                )
        else:
            _print_status(f"epoch={epoch}/{total_epochs} phase=val skipped val_interval={int(args.val_interval)}")

        if preview is not None and int(args.val_preview_interval) > 0 and epoch % int(args.val_preview_interval) == 0:
            preview_dir = run_dir / f"val_preview_{epoch}"
            _save_val_preview(preview[0], preview[1], preview_dir, epoch)
            _print_status(f"preview epoch={epoch} dir={preview_dir}")

        is_new_best = val_loss is not None and val_loss <= best_val
        if is_new_best:
            best_val = val_loss
        config = _build_checkpoint_config(
            args,
            effective_seed,
            in_channels=int(sample_channels),
            model_config=model_config,
        )
        checkpoint = V23Checkpoint(
            config=config,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            best_val=None if math.isinf(best_val) else best_val,
            global_step=global_step,
        )
        last_path = ckpt_dir / "v23_height_last.pt"
        best_path = ckpt_dir / "v23_height_best.pt"
        save_checkpoint(last_path, checkpoint)
        _print_status(f"checkpoint last={last_path}")
        if is_new_best:
            save_checkpoint(best_path, checkpoint)
            _print_status(f"checkpoint best={best_path}")

        history.append(
            {
                "epoch": epoch,
                "train": {"loss": train_loss, **train_metrics},
                "val": {"loss": val_loss, **val_metrics} if val_loss is not None else None,
            }
        )
        global_step += len(train_loader)
        _write_peak_vram_json(run_dir, device, args)
        _append_jsonl(
            loss_history_path,
            {
                "type": "epoch",
                "epoch": int(epoch),
                "epochs": int(total_epochs),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
                "best_val_loss": float(best_val) if not math.isinf(best_val) else None,
                "train": _loss_event_metrics(train_metrics),
                "val": _loss_event_metrics(val_metrics) if val_loss is not None else None,
                "validation_skipped": val_loss is None,
            },
        )
        if val_loss is None:
            _print_status(
                f"epoch={epoch}/{total_epochs} summary train_loss={_format_metric(train_loss)} "
                f"val_loss=skipped best_val_loss={_format_metric(None if math.isinf(best_val) else best_val)} "
                f"train_components=({_format_loss_components(train_metrics)})"
            )
        else:
            _print_status(
                f"epoch={epoch}/{total_epochs} summary train_loss={_format_metric(train_loss)} "
                f"val_loss={_format_metric(val_loss)} best_val_loss={_format_metric(best_val)} "
                f"train_components=({_format_loss_components(train_metrics)}) "
                f"val_components=({_format_loss_components(val_metrics)})"
            )
        if device.type == "cuda":
            _print_status(
                "peak_vram "
                f"allocated_gb={_format_metric(torch.cuda.max_memory_allocated(device) / 1e9)} "
                f"reserved_gb={_format_metric(torch.cuda.max_memory_reserved(device) / 1e9)}"
            )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "memory_profile": str(getattr(args, "effective_memory_profile", args.memory_profile)),
                "history": history,
                "best_val": None if math.isinf(best_val) else best_val,
                "checkpoint_dir": str(ckpt_dir),
                "loss_history": str(loss_history_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    reported_best_val = None if math.isinf(best_val) else best_val
    _print_status(f"metrics path={metrics_path} best_val={_format_metric(reported_best_val)}")
    return {"run_dir": run_dir, "best_val": reported_best_val, "history": history}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Spec 089 V23 height model.")
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--builds", nargs="+", default=["3_3_5_12340"])
    parser.add_argument("--maps", nargs="+", default=None)
    parser.add_argument("--input-mode", default=InputMode.FULL.value)
    parser.add_argument("--tileset-prune-table", default=None)
    parser.add_argument("--curation-manifest", default=None)
    parser.add_argument("--curation-min-terrain-validity", type=float, default=0.20)
    parser.add_argument("--curation-min-minimap-usefulness", type=float, default=0.10)
    parser.add_argument("--curation-max-liquid-coverage", type=float, default=0.05)
    parser.add_argument("--curation-reject-what-plate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--autotune-batch-size", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--autotune-batch-candidates", nargs="+", type=int, default=None)
    parser.add_argument("--autotune-safety-factor", type=float, default=0.85)
    parser.add_argument("--gpct-K", dest="gpct_K", type=int, default=4)
    parser.add_argument("--gpct-weight", type=float, default=0.0)
    parser.add_argument("--gpct-feature-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sdc-weight", type=float, default=0.1)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--bias-free-mask-ratio", type=float, default=0.0)
    parser.add_argument("--train-max-tiles", type=int, default=None)
    parser.add_argument("--val-max-tiles", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-preview-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--target-vram-gb", type=float, default=12.0)
    parser.add_argument("--memory-profile", choices=["auto", "12gb", "24gb", "none"], default="auto")
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--model-config-json", default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _apply_memory_profile(build_arg_parser().parse_args(argv))
    while True:
        try:
            return _train_once(args)
        except _RetryTrainingConfig:
            if int(args.batch_size) > 1:
                args.batch_size = max(1, int(args.batch_size) // 2)
                continue
            if float(args.gpct_weight) > 0.0 and int(args.gpct_K) > 1:
                args.gpct_K = max(1, int(args.gpct_K) // 2)
                continue
            if str(args.amp_dtype) == "bf16":
                args.amp_dtype = "fp16"
                continue
            if str(args.amp_dtype) == "auto":
                args.amp_dtype = "fp16"
                continue
            raise


if __name__ == "__main__":
    main()
