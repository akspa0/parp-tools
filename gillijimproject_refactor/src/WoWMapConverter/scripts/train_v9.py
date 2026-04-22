from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from v7_losses import build_recovery_mask


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "ml-training" / "v9"
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 8
MAX_ALLOWED_EPOCHS = 500
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 1337
DEFAULT_HEIGHT_SCALE = 1024.0
DEFAULT_RESIDUAL_SCALE = 128.0
DEFAULT_MIN_HEIGHT_RANGE = 4.0
DEFAULT_MIN_MINIMAP_VARIANCE = 1e-5
DEFAULT_MIN_MINIMAP_GRADIENT = 2e-3
DEFAULT_MAX_MEAN_WDL_DELTA = 512.0
DEFAULT_MAX_ABS_WDL_DELTA = 2048.0
DEFAULT_GROUP_BLOCK_SIZE = 4
DEFAULT_HIDDEN_CHANNELS = 32
DEFAULT_BLOCKS_PER_STAGE = 2
DEFAULT_TRAIN_WORKERS = 0
DEFAULT_VAL_WORKERS = 0
DEFAULT_AMP_DTYPE = "auto"
DEFAULT_TARGET_CURATED_SAMPLES = 27
DEFAULT_PREVIEW_COUNT = 4
DEFAULT_TRAIN_SAMPLER = "bucketed"
DEFAULT_HARD_REPLAY_FRACTION = 0.20
DEFAULT_HARD_REPLAY_WARMUP_EPOCHS = 1
DEFAULT_HARD_REPLAY_EMA_DECAY = 0.70
DEFAULT_DETAIL_FOCUS_EVERY_EPOCHS = 12
DEFAULT_DETAIL_FOCUS_MIN_EPOCH = 24
DEFAULT_DETAIL_FOCUS_STALL_THRESHOLD = 24
DEFAULT_DETAIL_FOCUS_TOP_FRACTION = 0.35
DEFAULT_DETAIL_FOCUS_GRADIENT_WEIGHT = 0.40
DEFAULT_DETAIL_FOCUS_DETAIL_RESIDUAL_WEIGHT = 0.12
DEFAULT_PAUSE_EVERY_EPOCHS = 50
DEFAULT_PREVIEW_EVERY_EPOCHS = 10
DEFAULT_PREVIEW_ARCHIVE_EVERY_EPOCHS = 10
DEFAULT_LR_PLATEAU_PATIENCE = 20
DEFAULT_LR_PLATEAU_FACTOR = 0.5
DEFAULT_MIN_LEARNING_RATE = 1e-5
DEFAULT_EARLY_STOP_PATIENCE = 120
DEFAULT_EARLY_STOP_MIN_EPOCHS = 120
DEFAULT_CURATION_MODE = "diverse-quality"
DEFAULT_CURATION_DIVERSITY_BLOCK_SIZE = 8
DEFAULT_CURATION_MAX_PER_GROUP = 2
DEFAULT_SELECTION_METRIC = "auto"
DEFAULT_DEV_EVAL_EVERY = 1
DEFAULT_DEV_EVAL_BLOCK_SIZE = 8
DEFAULT_AUX_LOSS_DECAY_START_EPOCH = 48
DEFAULT_AUX_LOSS_DECAY_EPOCHS = 192
DEFAULT_LATE_MID_L1_WEIGHT = 0.45
DEFAULT_LATE_COARSE_L1_WEIGHT = 0.15
DEFAULT_LATE_GRADIENT_WEIGHT = 0.35
DEFAULT_LATE_MID_RESIDUAL_WEIGHT = 0.08
DEFAULT_LATE_DETAIL_RESIDUAL_WEIGHT = 0.05
MASKED_RGB_ATTENUATION = 0.85
MASKED_NORMAL_ATTENUATION = 0.70
DEFAULT_NORMAL_RGB = (128, 128, 255)

IMAGENET_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

V9_ACTIVE_INPUT_SIGNALS = [
    "terrain_only_or_no_liquid_or_no_object_or_no_mccv_or_image_minimap_rgb",
    "normal_rgb",
    "wdl_17_or_height_17_base_prior",
    "height_min_mask",
    "height_max_mask",
    "liquid_mask",
    "liquid_height_prior",
    "object_footprint_mask",
    "brush_imprint_mask",
    "hole_mask_16x16",
]

V9_NATIVE_TARGET_SIGNALS = [
    "height_17",
    "height_65",
    "height_257",
]

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_tile_coordinates(tile_name: str) -> tuple[str, int, int]:
    parts = tile_name.rsplit("_", 2)
    if len(parts) != 3:
        return tile_name, 0, 0
    map_name, tile_x_text, tile_y_text = parts
    try:
        return map_name, int(tile_x_text), int(tile_y_text)
    except ValueError:
        return tile_name, 0, 0


def resolve_amp_dtype(amp_dtype: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    key = str(amp_dtype).strip().lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def avg_gradient_magnitude(image: np.ndarray) -> float:
    image_float = image.astype(np.float32)
    if image_float.ndim == 3:
        image_float = image_float.mean(axis=2)
    dx = np.diff(image_float, axis=1)
    dy = np.diff(image_float, axis=0)
    dx = dx[:, :-1] if dx.shape[1] > 0 else dx
    dy = dy[:-1, :] if dy.shape[0] > 0 else dy
    if dx.size == 0 or dy.size == 0:
        return 0.0
    magnitude = np.sqrt(dx[: dy.shape[0], :] ** 2 + dy[:, : dx.shape[1]] ** 2)
    return float(np.mean(magnitude))


def _resize_channel_tensor(tensor: torch.Tensor, size: tuple[int, int], mode: str, align_corners: bool | None = None) -> torch.Tensor:
    kwargs: dict[str, object] = {"size": size, "mode": mode}
    if align_corners is not None and mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = align_corners
    return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)


def _rgb_array_to_tensor(rgb: np.ndarray | None, fallback_rgb: tuple[int, int, int]) -> torch.Tensor:
    if rgb is None:
        rgb = np.full((256, 256, 3), fallback_rgb, dtype=np.uint8)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return _resize_channel_tensor(tensor, (257, 257), mode="bilinear", align_corners=False)


def _single_channel_array_to_tensor(array: np.ndarray | None, default_value: float = 0.0, resize_mode: str = "nearest") -> torch.Tensor:
    if array is None:
        return torch.full((1, 257, 257), float(default_value), dtype=torch.float32)

    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    if tensor.shape[-2:] != (257, 257):
        align_corners = False if resize_mode == "bilinear" else None
        tensor = _resize_channel_tensor(tensor, (257, 257), mode=resize_mode, align_corners=align_corners)
    return tensor


def _normalize_rgb_input(rgb_tensor: torch.Tensor, recovery_mask: torch.Tensor, attenuation: float) -> tuple[torch.Tensor, torch.Tensor]:
    attenuated = torch.clamp(rgb_tensor * (1.0 - recovery_mask * attenuation), 0.0, 1.0)
    normalized = (attenuated - IMAGENET_RGB_MEAN) / IMAGENET_RGB_STD
    return attenuated, normalized


def _build_v9_input_channels(
    arrays: dict[str, np.ndarray],
    base_257_scaled: torch.Tensor,
    *,
    include_brush_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    minimap_rgb = _rgb_array_to_tensor(arrays.get("minimap_rgb_256"), fallback_rgb=(0, 0, 0))
    normal_rgb = _rgb_array_to_tensor(arrays.get("normal_rgb_256"), fallback_rgb=DEFAULT_NORMAL_RGB)

    height_hints = arrays.get("height_hints_v7")
    if height_hints is None or len(height_hints) < 2:
        height_min_value = 0.5
        height_max_value = 0.5
    else:
        height_min_value = float(height_hints[0])
        height_max_value = float(height_hints[1])

    height_min_mask = torch.full((1, 257, 257), height_min_value, dtype=torch.float32)
    height_max_mask = torch.full((1, 257, 257), height_max_value, dtype=torch.float32)
    liquid_mask = _single_channel_array_to_tensor(arrays.get("liquid_mask_257"), resize_mode="nearest")
    liquid_height_prior = _single_channel_array_to_tensor(arrays.get("liquid_height_257"), resize_mode="bilinear") * liquid_mask
    object_mask = _single_channel_array_to_tensor(arrays.get("object_mask_257"), resize_mode="nearest")
    brush_mask = _single_channel_array_to_tensor(arrays.get("brush_mask_257"), resize_mode="nearest")
    if not include_brush_mask:
        brush_mask = torch.zeros_like(brush_mask)
    hole_mask_257 = _single_channel_array_to_tensor(arrays.get("hole_mask_16x16"), resize_mode="nearest")

    recovery_mask = build_recovery_mask(object_mask=object_mask, liquid_mask=liquid_mask, brush_mask=brush_mask)
    preview_minimap_rgb, minimap_input = _normalize_rgb_input(minimap_rgb, recovery_mask, MASKED_RGB_ATTENUATION)
    _, normal_input = _normalize_rgb_input(normal_rgb, recovery_mask, MASKED_NORMAL_ATTENUATION)

    inputs = torch.cat(
        [
            minimap_input,
            normal_input,
            base_257_scaled,
            height_min_mask,
            height_max_mask,
            liquid_mask,
            liquid_height_prior,
            object_mask,
            brush_mask,
            hole_mask_257,
        ],
        dim=0,
    )
    return inputs, preview_minimap_rgb


@dataclass(frozen=True)
class V9SampleEntry:
    dataset_root: str
    dataset_key: str
    tile_name: str
    map_name: str
    tile_x: int
    tile_y: int
    shard_path: Path
    source_json: str
    height_min: float
    height_max: float
    liquid_coverage: float
    object_coverage: float
    brush_coverage: float
    hole_coverage: float
    minimap_variance: float
    minimap_gradient: float
    detail_energy: float
    has_wdl_17: bool
    has_minimap_rgb_256: bool

    @property
    def height_range(self) -> float:
        return float(self.height_max - self.height_min)

    @property
    def sample_key(self) -> str:
        return f"{self.dataset_key}:{self.tile_name}"

    @property
    def build_key(self) -> str:
        if "__" in self.dataset_key:
            return self.dataset_key.split("__", 1)[0]
        return self.dataset_key


@dataclass(frozen=True)
class AuditedEntry:
    sample: V9SampleEntry
    minimap_variance: float
    minimap_gradient: float
    mean_wdl_delta: float
    max_abs_wdl_delta: float
    hole_coverage: float
    accepted: bool
    rejection_reason: str | None

    @property
    def quality_score(self) -> float:
        if not self.accepted:
            return -1.0

        height_score = min(self.sample.height_range / 64.0, 4.0)
        minimap_score = min(self.minimap_gradient / 0.02, 3.0) + min(self.minimap_variance / 0.01, 3.0)
        wdl_penalty = min(self.mean_wdl_delta / 128.0, 2.0) + min(self.max_abs_wdl_delta / 512.0, 2.0)
        hole_penalty = min(self.hole_coverage * 2.0, 1.0)
        return float(height_score + minimap_score - wdl_penalty - hole_penalty)


def resolve_manifest_entry_path(manifest_path: Path, raw_value: object) -> Path | None:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return None

    resolved = Path(raw_text)
    if resolved.is_absolute():
        return resolved
    return (manifest_path.parent / resolved).resolve()


def load_cache_manifest(manifest_path: Path) -> list[V9SampleEntry]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    entries: list[V9SampleEntry] = []
    for entry in manifest.get("entries", []):
        tile_name = str(entry.get("tile_name", ""))
        map_name, tile_x, tile_y = parse_tile_coordinates(tile_name)
        shard_path = resolve_manifest_entry_path(manifest_path, entry.get("shard_path", "")) or Path()
        source_json_path = resolve_manifest_entry_path(manifest_path, entry.get("source_json", ""))
        liquid_coverage = entry.get("liquid_coverage")
        object_coverage = entry.get("object_coverage")
        brush_coverage = entry.get("brush_coverage")
        hole_coverage = entry.get("hole_coverage")
        minimap_variance = entry.get("minimap_variance")
        minimap_gradient = entry.get("minimap_gradient")
        detail_energy = entry.get("detail_energy")
        if any(value is None for value in (liquid_coverage, object_coverage, brush_coverage, hole_coverage, minimap_variance, minimap_gradient, detail_energy)) and shard_path.exists():
            arrays = load_npz_arrays(shard_path)
            if liquid_coverage is None:
                liquid_mask = arrays.get("liquid_mask_257")
                liquid_coverage = float(liquid_mask.astype(np.float32).mean()) if liquid_mask is not None else 0.0
            if object_coverage is None:
                object_mask = arrays.get("object_mask_257")
                object_coverage = float(object_mask.astype(np.float32).mean()) if object_mask is not None else 0.0
            if brush_coverage is None:
                brush_mask = arrays.get("brush_mask_257")
                brush_coverage = float(brush_mask.astype(np.float32).mean()) if brush_mask is not None else 0.0
            if hole_coverage is None:
                hole_mask = arrays.get("hole_mask_16x16")
                hole_coverage = float(hole_mask.astype(np.float32).mean()) if hole_mask is not None else 0.0
            if minimap_variance is None or minimap_gradient is None:
                minimap_rgb = arrays.get("minimap_rgb_256")
                if minimap_rgb is not None:
                    minimap_float = minimap_rgb.astype(np.float32) / 255.0
                    if minimap_variance is None:
                        minimap_variance = float(np.var(minimap_float))
                    if minimap_gradient is None:
                        minimap_gradient = avg_gradient_magnitude(minimap_rgb)
                else:
                    if minimap_variance is None:
                        minimap_variance = 0.0
                    if minimap_gradient is None:
                        minimap_gradient = 0.0
            if detail_energy is None:
                height_257 = arrays.get("height_257")
                height_65 = arrays.get("height_65")
                if height_257 is not None and height_65 is not None:
                    height_257_tensor = torch.from_numpy(height_257.astype(np.float32)).unsqueeze(0)
                    height_65_tensor = torch.from_numpy(height_65.astype(np.float32)).unsqueeze(0)
                    upsampled_65 = F.interpolate(height_65_tensor.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=True).squeeze(0)
                    detail_energy = float(torch.mean(torch.abs(height_257_tensor - upsampled_65)).item())
                else:
                    detail_energy = 0.0

        entries.append(
            V9SampleEntry(
                dataset_root=str(entry.get("dataset_root", "")),
                dataset_key=str(entry.get("dataset_key", "")),
                tile_name=tile_name,
                map_name=map_name,
                tile_x=tile_x,
                tile_y=tile_y,
                shard_path=shard_path,
                source_json=str(source_json_path) if source_json_path is not None else "",
                height_min=float(entry.get("height_min", 0.0)),
                height_max=float(entry.get("height_max", 0.0)),
                liquid_coverage=float(liquid_coverage or 0.0),
                object_coverage=float(object_coverage or 0.0),
                brush_coverage=float(brush_coverage or 0.0),
                hole_coverage=float(hole_coverage or 0.0),
                minimap_variance=float(minimap_variance or 0.0),
                minimap_gradient=float(minimap_gradient or 0.0),
                detail_energy=float(detail_energy or 0.0),
                has_wdl_17=bool(entry.get("has_wdl_17", False)),
                has_minimap_rgb_256=bool(entry.get("has_minimap_rgb_256", False)),
            )
        )
    return entries


class OrderedIndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _coverage_bucket(value: float, light_threshold: float, heavy_threshold: float) -> int:
    if value >= heavy_threshold:
        return 2
    if value >= light_threshold:
        return 1
    return 0


def build_sampling_bucket_key(entry: V9SampleEntry) -> tuple[str, int, int, int, int, int]:
    if entry.height_range >= 192.0:
        height_bucket = 3
    elif entry.height_range >= 96.0:
        height_bucket = 2
    elif entry.height_range >= 32.0:
        height_bucket = 1
    else:
        height_bucket = 0

    liquid_bucket = _coverage_bucket(entry.liquid_coverage, 0.01, 0.15)
    object_bucket = _coverage_bucket(entry.object_coverage, 0.01, 0.08)
    hole_bucket = 1 if entry.hole_coverage >= 0.01 else 0
    detail_bucket = _coverage_bucket(entry.detail_energy, 4.0, 12.0)
    return (entry.build_key, height_bucket, liquid_bucket, object_bucket, hole_bucket, detail_bucket)


def is_detail_focus_epoch(epoch: int, args: argparse.Namespace, current_stall: int) -> bool:
    if epoch < max(1, int(args.detail_focus_min_epoch)):
        return False
    periodic_focus = args.detail_focus_every_epochs > 0 and epoch % args.detail_focus_every_epochs == 0
    stall_focus = args.detail_focus_stall_threshold > 0 and current_stall >= args.detail_focus_stall_threshold
    return periodic_focus or stall_focus


def build_epoch_training_order(
    entries: Sequence[V9SampleEntry],
    *,
    epoch: int,
    seed: int,
    sampler_mode: str,
    hard_replay_fraction: float,
    hard_replay_warmup_epochs: int,
    sample_loss_ema: dict[str, float],
    detail_focus_active: bool,
    detail_focus_top_fraction: float,
) -> list[int]:
    indices = list(range(len(entries)))
    rng = random.Random(seed + epoch)
    if sampler_mode == "random":
        rng.shuffle(indices)
        return indices

    if detail_focus_active and indices:
        target_count = min(
            len(indices),
            max(1, int(round(len(indices) * float(np.clip(detail_focus_top_fraction, 0.05, 1.0))))),
        )
        indices = sorted(
            indices,
            key=lambda index: (
                entries[index].detail_energy,
                sample_loss_ema.get(entries[index].sample_key, 0.0),
                entries[index].minimap_gradient,
                entries[index].height_range,
            ),
            reverse=True,
        )[:target_count]
        rng.shuffle(indices)

    grouped_indices: dict[tuple[str, int, int, int, int, int], list[int]] = defaultdict(list)
    for index in indices:
        entry = entries[index]
        grouped_indices[build_sampling_bucket_key(entry)].append(index)

    replay_active = hard_replay_fraction > 0.0 and epoch > hard_replay_warmup_epochs and bool(sample_loss_ema)
    working_groups: list[list[int]] = []
    for group_indices in grouped_indices.values():
        shuffled = list(group_indices)
        rng.shuffle(shuffled)
        if replay_active:
            replay_count = min(len(shuffled), max(1, int(round(len(shuffled) * hard_replay_fraction))))
            replay_head = sorted(
                shuffled,
                key=lambda index: sample_loss_ema.get(entries[index].sample_key, 0.0),
                reverse=True,
            )[:replay_count]
            replay_set = set(replay_head)
            replay_tail = [index for index in shuffled if index not in replay_set]
            shuffled = replay_head + replay_tail
        working_groups.append(shuffled)

    rng.shuffle(working_groups)
    ordered: list[int] = []
    while working_groups:
        next_groups: list[list[int]] = []
        for group in working_groups:
            if not group:
                continue
            ordered.append(group.pop(0))
            if group:
                next_groups.append(group)
        working_groups = next_groups
        rng.shuffle(working_groups)

    return ordered


def update_sample_loss_ema(
    sample_loss_ema: dict[str, float],
    observed_sample_losses: dict[str, float],
    ema_decay: float,
) -> None:
    decay = float(np.clip(ema_decay, 0.0, 0.999))
    retain = 1.0 - decay
    for sample_key, sample_loss in observed_sample_losses.items():
        previous = sample_loss_ema.get(sample_key)
        if previous is None:
            sample_loss_ema[sample_key] = float(sample_loss)
        else:
            sample_loss_ema[sample_key] = (decay * previous) + (retain * float(sample_loss))


def build_train_loader(
    dataset: Dataset,
    entries: Sequence[V9SampleEntry],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    sample_loss_ema: dict[str, float],
) -> tuple[DataLoader, bool]:
    detail_focus_active = is_detail_focus_epoch(epoch, args, getattr(args, "current_stall", 0))
    sampler = OrderedIndexSampler(
        build_epoch_training_order(
            entries,
            epoch=epoch,
            seed=args.seed,
            sampler_mode=args.train_sampler,
            hard_replay_fraction=args.hard_replay_fraction,
            hard_replay_warmup_epochs=args.hard_replay_warmup_epochs,
            sample_loss_ema=sample_loss_ema,
            detail_focus_active=detail_focus_active,
            detail_focus_top_fraction=args.detail_focus_top_fraction,
        )
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.train_workers,
        pin_memory=device.type == "cuda",
    ), detail_focus_active


def reduce_samplewise_mean(value: torch.Tensor) -> torch.Tensor:
    return value.abs().reshape(value.shape[0], -1).mean(dim=1)


def load_npz_arrays(shard_path: Path) -> dict[str, np.ndarray]:
    with np.load(shard_path) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def audit_entry(
    entry: V9SampleEntry,
    require_wdl: bool,
    require_minimap: bool,
    min_height_range: float,
    min_minimap_variance: float,
    min_minimap_gradient: float,
    max_mean_wdl_delta: float,
    max_abs_wdl_delta: float,
) -> AuditedEntry:
    max_mean_wdl_delta_threshold = max_mean_wdl_delta
    max_abs_wdl_delta_threshold = max_abs_wdl_delta
    if not entry.shard_path.exists():
        return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, 0.0, False, "missing_shard")

    arrays = load_npz_arrays(entry.shard_path)
    required_arrays = (
        "height_257",
        "height_17",
        "hole_mask_16x16",
        "normal_rgb_256",
        "height_hints_v7",
        "liquid_mask_257",
        "liquid_height_257",
        "object_mask_257",
        "brush_mask_257",
    )
    for name in required_arrays:
        if name not in arrays:
            return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, 0.0, False, f"missing_{name}")

    if require_wdl and "wdl_17" not in arrays:
        return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, 0.0, False, "missing_wdl_17")
    if require_minimap and "minimap_rgb_256" not in arrays:
        return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, 0.0, False, "missing_minimap_rgb_256")

    height_257 = arrays["height_257"].astype(np.float32)
    height_17 = arrays["height_17"].astype(np.float32)
    hole_mask = arrays["hole_mask_16x16"].astype(np.float32)
    if not np.all(np.isfinite(height_257)):
        return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, 0.0, False, "non_finite_height_257")
    if not np.all(np.isfinite(height_17)):
        return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, 0.0, False, "non_finite_height_17")

    if entry.height_range < min_height_range:
        return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, float(hole_mask.mean()), False, "height_range_too_low")

    minimap_variance = 0.0
    minimap_gradient = 0.0
    if "minimap_rgb_256" in arrays:
        minimap = arrays["minimap_rgb_256"].astype(np.float32) / 255.0
        minimap_variance = float(np.var(minimap))
        minimap_gradient = avg_gradient_magnitude(minimap)
        if minimap_variance < min_minimap_variance:
            return AuditedEntry(entry, minimap_variance, minimap_gradient, math.inf, math.inf, float(hole_mask.mean()), False, "minimap_variance_too_low")
        if minimap_gradient < min_minimap_gradient:
            return AuditedEntry(entry, minimap_variance, minimap_gradient, math.inf, math.inf, float(hole_mask.mean()), False, "minimap_gradient_too_low")

    mean_wdl_delta = 0.0
    max_abs_wdl_delta = 0.0
    if "wdl_17" in arrays:
        wdl_17 = arrays["wdl_17"].astype(np.float32)
        if not np.all(np.isfinite(wdl_17)):
            return AuditedEntry(entry, minimap_variance, minimap_gradient, math.inf, math.inf, float(hole_mask.mean()), False, "non_finite_wdl_17")
        wdl_delta = height_17 - wdl_17
        mean_wdl_delta = float(np.mean(np.abs(wdl_delta)))
        max_abs_wdl_delta = float(np.max(np.abs(wdl_delta)))
        if mean_wdl_delta > max_mean_wdl_delta_threshold:
            return AuditedEntry(entry, minimap_variance, minimap_gradient, mean_wdl_delta, max_abs_wdl_delta, float(hole_mask.mean()), False, "mean_wdl_delta_too_high")
        if max_abs_wdl_delta > max_abs_wdl_delta_threshold:
            return AuditedEntry(entry, minimap_variance, minimap_gradient, mean_wdl_delta, max_abs_wdl_delta, float(hole_mask.mean()), False, "max_wdl_delta_too_high")

    return AuditedEntry(
        sample=entry,
        minimap_variance=minimap_variance,
        minimap_gradient=minimap_gradient,
        mean_wdl_delta=mean_wdl_delta,
        max_abs_wdl_delta=max_abs_wdl_delta,
        hole_coverage=float(hole_mask.mean()),
        accepted=True,
        rejection_reason=None,
    )


def audit_entries(
    entries: Sequence[V9SampleEntry],
    require_wdl: bool,
    require_minimap: bool,
    min_height_range: float,
    min_minimap_variance: float,
    min_minimap_gradient: float,
    max_mean_wdl_delta: float,
    max_abs_wdl_delta: float,
) -> tuple[list[AuditedEntry], list[V9SampleEntry], dict[str, int]]:
    audited: list[AuditedEntry] = []
    accepted: list[V9SampleEntry] = []
    reasons: dict[str, int] = {}
    for entry in entries:
        result = audit_entry(
            entry=entry,
            require_wdl=require_wdl,
            require_minimap=require_minimap,
            min_height_range=min_height_range,
            min_minimap_variance=min_minimap_variance,
            min_minimap_gradient=min_minimap_gradient,
            max_mean_wdl_delta=max_mean_wdl_delta,
            max_abs_wdl_delta=max_abs_wdl_delta,
        )
        audited.append(result)
        if result.accepted:
            accepted.append(entry)
        else:
            reason = str(result.rejection_reason or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
    return audited, accepted, reasons


def select_curated_entries(
    audited_entries: Sequence[AuditedEntry],
    limit: Optional[int],
    curation_mode: str,
    diversity_block_size: int,
    max_per_group: int,
) -> list[V9SampleEntry]:
    accepted = [entry for entry in audited_entries if entry.accepted]
    accepted.sort(
        key=lambda entry: (
            entry.quality_score,
            entry.sample.height_range,
            entry.minimap_gradient,
            entry.minimap_variance,
        ),
        reverse=True,
    )
    if limit is None or curation_mode == "top-quality":
        selected = accepted[:limit] if limit is not None else accepted
        return [entry.sample for entry in selected]

    grouped: dict[str, deque[AuditedEntry]] = defaultdict(deque)
    for entry in accepted:
        block_x = entry.sample.tile_x // max(1, diversity_block_size)
        block_y = entry.sample.tile_y // max(1, diversity_block_size)
        group_key = f"{entry.sample.dataset_key}:{entry.sample.map_name}:{block_x}:{block_y}"
        grouped[group_key].append(entry)

    group_order = sorted(
        grouped.keys(),
        key=lambda key: (
            grouped[key][0].quality_score,
            grouped[key][0].sample.height_range,
            grouped[key][0].minimap_gradient,
        ),
        reverse=True,
    )

    selected_samples: list[V9SampleEntry] = []
    selected_ids: set[str] = set()
    per_group_count: dict[str, int] = defaultdict(int)
    working_order = list(group_order)
    while working_order and len(selected_samples) < limit:
        next_order: list[str] = []
        progressed = False
        for group_key in working_order:
            queue = grouped[group_key]
            if max_per_group > 0 and per_group_count[group_key] >= max_per_group:
                continue
            if not queue:
                continue

            chosen = queue.popleft()
            chosen_id = f"{chosen.sample.dataset_key}:{chosen.sample.tile_name}"
            if chosen_id in selected_ids:
                continue

            selected_samples.append(chosen.sample)
            selected_ids.add(chosen_id)
            per_group_count[group_key] += 1
            progressed = True

            if queue and (max_per_group <= 0 or per_group_count[group_key] < max_per_group):
                next_order.append(group_key)
            if len(selected_samples) >= limit:
                break

        if not progressed:
            break
        working_order = next_order

    if len(selected_samples) < limit:
        for entry in accepted:
            entry_id = f"{entry.sample.dataset_key}:{entry.sample.tile_name}"
            if entry_id in selected_ids:
                continue
            selected_samples.append(entry.sample)
            selected_ids.add(entry_id)
            if len(selected_samples) >= limit:
                break

    return selected_samples


def select_diverse_eval_entries(
    entries: Sequence[V9SampleEntry],
    limit: Optional[int],
    block_size: int,
    seed: int,
) -> list[V9SampleEntry]:
    ordered_entries = sorted(entries, key=lambda entry: (entry.dataset_key, entry.map_name, entry.tile_y, entry.tile_x, entry.tile_name))
    if limit is None or len(ordered_entries) <= limit:
        return ordered_entries

    grouped: dict[str, deque[V9SampleEntry]] = defaultdict(deque)
    for entry in ordered_entries:
        block_x = entry.tile_x // max(1, block_size)
        block_y = entry.tile_y // max(1, block_size)
        group_key = f"{entry.dataset_key}:{entry.map_name}:{block_x}:{block_y}"
        grouped[group_key].append(entry)

    group_order = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_order)
    selected: list[V9SampleEntry] = []
    while group_order and len(selected) < limit:
        next_order: list[str] = []
        for group_key in group_order:
            queue = grouped[group_key]
            if not queue:
                continue
            selected.append(queue.popleft())
            if queue:
                next_order.append(group_key)
            if len(selected) >= limit:
                break
        group_order = next_order

    return selected[:limit]


def build_validation_groups(entries: Sequence[V9SampleEntry], block_size: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        block_x = entry.tile_x // block_size
        block_y = entry.tile_y // block_size
        key = f"{entry.dataset_key}:{entry.map_name}:{block_x}:{block_y}"
        groups.setdefault(key, []).append(index)
    return groups


def split_grouped_indices(entries: Sequence[V9SampleEntry], val_fraction: float, seed: int, block_size: int) -> tuple[list[int], list[int]]:
    if len(entries) <= 1:
        return list(range(len(entries))), []

    groups = build_validation_groups(entries, block_size)
    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)
    target_val_samples = max(1, int(round(len(entries) * val_fraction)))

    val_indices: list[int] = []
    for _, indices in group_items:
        if len(val_indices) >= target_val_samples:
            break
        val_indices.extend(indices)

    val_set = set(val_indices)
    train_indices = [index for index in range(len(entries)) if index not in val_set]
    if not train_indices:
        train_indices = list(range(max(0, len(entries) - 1)))
        val_indices = [len(entries) - 1]
    return train_indices, val_indices


class V9NativeDataset(Dataset):
    def __init__(self, entries: Sequence[V9SampleEntry], height_scale: float, residual_scale: float, include_brush_mask: bool = True):
        self.entries = list(entries)
        self.height_scale = float(height_scale)
        self.residual_scale = float(residual_scale)
        self.include_brush_mask = bool(include_brush_mask)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        arrays = load_npz_arrays(entry.shard_path)

        height_257 = torch.from_numpy(arrays["height_257"].astype(np.float32)).unsqueeze(0)
        height_17 = torch.from_numpy(arrays["height_17"].astype(np.float32)).unsqueeze(0)

        if "wdl_17" in arrays:
            base_17 = torch.from_numpy(arrays["wdl_17"].astype(np.float32)).unsqueeze(0)
        else:
            base_17 = height_17.clone()
        base_65 = F.interpolate(base_17.unsqueeze(0), size=(65, 65), mode="bilinear", align_corners=True).squeeze(0)
        base_257 = F.interpolate(base_17.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=True).squeeze(0)
        height_65 = torch.from_numpy(arrays["height_65"].astype(np.float32)).unsqueeze(0)

        base_65_scaled = base_65 / self.height_scale
        base_257_scaled = base_257 / self.height_scale
        height_257_scaled = height_257 / self.height_scale
        height_65_scaled = height_65 / self.height_scale
        residual_target_257 = (height_257 - base_257) / self.residual_scale
        coarse_target_17 = (height_17 - base_17) / self.residual_scale
        mid_residual_target_65 = (height_65 - base_65) / self.residual_scale
        detail_target_257 = (height_257 - F.interpolate(height_65.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=True).squeeze(0)) / self.residual_scale

        inputs, preview_minimap_rgb = _build_v9_input_channels(
            arrays=arrays,
            base_257_scaled=base_257_scaled,
            include_brush_mask=self.include_brush_mask,
        )

        return {
            "inputs": inputs,
            "preview_minimap_rgb": preview_minimap_rgb,
            "preview_liquid_mask": torch.from_numpy(arrays.get("liquid_mask_257", np.zeros((257, 257), dtype=np.uint8)).astype(np.float32)).unsqueeze(0),
            "preview_liquid_height": torch.from_numpy(arrays.get("liquid_height_257", np.zeros((257, 257), dtype=np.float32)).astype(np.float32)).unsqueeze(0) / self.height_scale,
            "preview_object_mask": torch.from_numpy(arrays.get("object_mask_257", np.zeros((257, 257), dtype=np.uint8)).astype(np.float32)).unsqueeze(0),
            "preview_hole_mask": torch.from_numpy(arrays.get("hole_mask_16x16", np.zeros((16, 16), dtype=np.uint8)).astype(np.float32)).unsqueeze(0),
            "sample_key": entry.sample_key,
            "target_height_257": height_257_scaled,
            "target_height_65": height_65_scaled,
            "target_height_17": height_17 / self.height_scale,
            "base_height_257": base_257_scaled,
            "base_height_65": base_65_scaled,
            "base_height_17": base_17 / self.height_scale,
            "target_residual_257": residual_target_257,
            "target_mid_residual_65": mid_residual_target_65,
            "target_detail_residual_257": detail_target_257,
            "target_coarse_17": coarse_target_17,
        }


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, padding_mode="reflect")
        self.norm1 = nn.GroupNorm(_resolve_group_count(channels, 8), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, padding_mode="reflect")
        self.norm2 = nn.GroupNorm(_resolve_group_count(channels, 8), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.silu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.silu(out + residual, inplace=True)


def _resolve_group_count(channels: int, preferred_groups: int) -> int:
    preferred_groups = max(1, int(preferred_groups))
    for group_count in range(min(preferred_groups, channels), 0, -1):
        if channels % group_count == 0:
            return group_count
    return 1


class V9TerrainModel(nn.Module):
    def __init__(self, in_channels: int = 14, hidden_channels: int = DEFAULT_HIDDEN_CHANNELS, blocks_per_stage: int = DEFAULT_BLOCKS_PER_STAGE):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2, padding_mode="reflect"),
            nn.GroupNorm(_resolve_group_count(hidden_channels, 8), hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.enc1 = nn.Sequential(*[ResidualConvBlock(hidden_channels, dilation=1 + (i % 2)) for i in range(blocks_per_stage)])
        self.down1 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.enc2 = nn.Sequential(*[ResidualConvBlock(hidden_channels * 2, dilation=1 + (i % 3)) for i in range(blocks_per_stage)])
        self.down2 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.enc3 = nn.Sequential(*[ResidualConvBlock(hidden_channels * 4, dilation=1 + (i % 4)) for i in range(blocks_per_stage)])

        self.coarse_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((17, 17)),
            nn.Conv2d(hidden_channels * 2, 1, kernel_size=1),
        )
        self.mid_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, 1, kernel_size=1),
        )

        self.up2 = nn.Conv2d(hidden_channels * 4 + hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, padding_mode="reflect")
        self.dec2 = ResidualConvBlock(hidden_channels * 2, dilation=1)
        self.up1 = nn.Conv2d(hidden_channels * 2 + hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.dec1 = ResidualConvBlock(hidden_channels, dilation=1)
        self.detail_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stem(inputs)
        enc1 = self.enc1(stem)

        down1 = F.silu(self.down1(enc1), inplace=True)
        enc2 = self.enc2(down1)

        down2 = F.silu(self.down2(enc2), inplace=True)
        enc3 = self.enc3(down2)

        coarse_delta_17 = self.coarse_head(enc3)
        mid_delta_65 = self.mid_head(enc3)

        up2 = F.interpolate(enc3, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = F.silu(self.up2(up2), inplace=True)
        up2 = self.dec2(up2)

        up1 = F.interpolate(up2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = F.silu(self.up1(up1), inplace=True)
        up1 = self.dec1(up1)

        detail_delta_257 = self.detail_head(up1)
        return coarse_delta_17, mid_delta_65, detail_delta_257


def build_predictions(
    coarse_delta_17: torch.Tensor,
    mid_delta_65: torch.Tensor,
    detail_delta_257: torch.Tensor,
    base_height_17: torch.Tensor,
    base_height_65: torch.Tensor,
    base_height_257: torch.Tensor,
    residual_scale: float,
    height_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_height_17 = base_height_17 + (coarse_delta_17 * (residual_scale / height_scale))
    coarse_65 = F.interpolate(coarse_height_17, size=base_height_65.shape[-2:], mode="bilinear", align_corners=True)
    mid_height_65 = coarse_65 + (mid_delta_65 * (residual_scale / height_scale))
    mid_257 = F.interpolate(mid_height_65, size=base_height_257.shape[-2:], mode="bilinear", align_corners=True)
    detail_scaled = detail_delta_257 * (residual_scale / height_scale)
    full_height_257 = mid_257 + detail_scaled
    return coarse_height_17, mid_height_65, full_height_257


def resolve_loss_weights(epoch: int, args: argparse.Namespace, current_stall: int = 0) -> dict[str, float]:
    decay_start = max(1, int(args.aux_loss_decay_start_epoch))
    decay_epochs = max(1, int(args.aux_loss_decay_epochs))
    progress = min(max((epoch - decay_start) / decay_epochs, 0.0), 1.0)

    def lerp(start: float, end: float) -> float:
        return float(start + ((end - start) * progress))

    weights = {
        "full_l1": 1.0,
        "mid_l1": lerp(0.70, args.late_mid_l1_weight),
        "coarse_l1": lerp(0.45, args.late_coarse_l1_weight),
        "gradient": lerp(0.25, args.late_gradient_weight),
        "mid_residual": lerp(0.20, args.late_mid_residual_weight),
        "detail_residual": lerp(0.20, args.late_detail_residual_weight),
    }
    if is_detail_focus_epoch(epoch, args, current_stall):
        weights["gradient"] = max(weights["gradient"], float(args.detail_focus_gradient_weight))
        weights["detail_residual"] = max(weights["detail_residual"], float(args.detail_focus_detail_residual_weight))
    return weights


def compute_v9_loss(
    coarse_delta_17: torch.Tensor,
    mid_delta_65: torch.Tensor,
    detail_delta_257: torch.Tensor,
    batch: dict[str, torch.Tensor],
    residual_scale: float,
    height_scale: float,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    coarse_height_17, mid_height_65, full_height_257 = build_predictions(
        coarse_delta_17=coarse_delta_17,
        mid_delta_65=mid_delta_65,
        detail_delta_257=detail_delta_257,
        base_height_17=batch["base_height_17"],
        base_height_65=batch["base_height_65"],
        base_height_257=batch["base_height_257"],
        residual_scale=residual_scale,
        height_scale=height_scale,
    )

    full_l1_per_sample = reduce_samplewise_mean(full_height_257 - batch["target_height_257"])
    mid_l1_per_sample = reduce_samplewise_mean(mid_height_65 - batch["target_height_65"])
    coarse_l1_per_sample = reduce_samplewise_mean(coarse_height_17 - batch["target_height_17"])
    full_l1 = full_l1_per_sample.mean()
    mid_l1 = mid_l1_per_sample.mean()
    coarse_l1 = coarse_l1_per_sample.mean()

    pred_dx = full_height_257[:, :, :, 1:] - full_height_257[:, :, :, :-1]
    pred_dy = full_height_257[:, :, 1:, :] - full_height_257[:, :, :-1, :]
    target_dx = batch["target_height_257"][:, :, :, 1:] - batch["target_height_257"][:, :, :, :-1]
    target_dy = batch["target_height_257"][:, :, 1:, :] - batch["target_height_257"][:, :, :-1, :]
    gradient_loss_x_per_sample = reduce_samplewise_mean(pred_dx - target_dx)
    gradient_loss_y_per_sample = reduce_samplewise_mean(pred_dy - target_dy)
    gradient_loss_per_sample = gradient_loss_x_per_sample + gradient_loss_y_per_sample
    gradient_loss = gradient_loss_per_sample.mean()

    mid_residual_loss_per_sample = reduce_samplewise_mean(mid_delta_65 - batch["target_mid_residual_65"])
    detail_residual_loss_per_sample = reduce_samplewise_mean(detail_delta_257 - batch["target_detail_residual_257"])
    mid_residual_loss = mid_residual_loss_per_sample.mean()
    detail_residual_loss = detail_residual_loss_per_sample.mean()
    total_per_sample = (
        (loss_weights["full_l1"] * full_l1_per_sample)
        + (loss_weights["mid_l1"] * mid_l1_per_sample)
        + (loss_weights["coarse_l1"] * coarse_l1_per_sample)
        + (loss_weights["gradient"] * gradient_loss_per_sample)
        + (loss_weights["mid_residual"] * mid_residual_loss_per_sample)
        + (loss_weights["detail_residual"] * detail_residual_loss_per_sample)
    )
    total = total_per_sample.mean()
    return total, {
        "full_l1": float(full_l1.item()),
        "mid_l1": float(mid_l1.item()),
        "coarse_l1": float(coarse_l1.item()),
        "gradient": float(gradient_loss.item()),
        "mid_residual": float(mid_residual_loss.item()),
        "detail_residual": float(detail_residual_loss.item()),
    }, total_per_sample.detach()


def move_batch_to_device(batch: dict[str, Any], device: torch.device, channels_last: bool) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
            continue
        tensor = value.to(device, non_blocking=device.type == "cuda")
        if channels_last and tensor.ndim == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        moved[key] = tensor
    return moved


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    amp_dtype: torch.dtype,
    residual_scale: float,
    height_scale: float,
    channels_last: bool,
    current_epoch: int,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float], float, dict[str, float]]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    component_sums = {"full_l1": 0.0, "mid_l1": 0.0, "coarse_l1": 0.0, "gradient": 0.0, "mid_residual": 0.0, "detail_residual": 0.0}
    sample_count = 0
    start = time.perf_counter()
    sample_loss_sums: dict[str, float] = {}
    sample_loss_counts: dict[str, int] = {}

    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}
    loss_weights = resolve_loss_weights(current_epoch, args)
    phase_name = "train" if is_training else "val"
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            loader,
            total=len(loader),
            desc=f"epoch {current_epoch:03d}/{args.epochs} {phase_name}",
            dynamic_ncols=True,
            leave=False,
        )
    iterator = progress_bar if progress_bar is not None else loader

    for batch_index, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device, channels_last)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else nullcontext()):
            coarse_delta_17, mid_delta_65, detail_delta_257 = model(batch["inputs"])
            loss, components, per_sample_losses = compute_v9_loss(
                coarse_delta_17=coarse_delta_17,
                mid_delta_65=mid_delta_65,
                detail_delta_257=detail_delta_257,
                batch=batch,
                residual_scale=residual_scale,
                height_scale=height_scale,
                loss_weights=loss_weights,
            )

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = int(batch["inputs"].shape[0])
        sample_keys = batch.get("sample_key", [])
        if sample_keys:
            per_sample_values = per_sample_losses.cpu().tolist()
            for sample_key, sample_loss in zip(sample_keys, per_sample_values):
                sample_loss_sums[sample_key] = sample_loss_sums.get(sample_key, 0.0) + float(sample_loss)
                sample_loss_counts[sample_key] = sample_loss_counts.get(sample_key, 0) + 1
        sample_count += batch_size
        total_loss += float(loss.item()) * batch_size
        for key, value in components.items():
            component_sums[key] += float(value) * batch_size

        if progress_bar is not None:
            elapsed = max(time.perf_counter() - start, 1e-6)
            progress_bar.set_postfix(
                loss=f"{(total_loss / max(sample_count, 1)):.4f}",
                sps=f"{(sample_count / elapsed):.1f}",
                batch=f"{batch_index}/{len(loader)}",
                refresh=False,
            )

    if progress_bar is not None:
        progress_bar.close()

    elapsed = max(time.perf_counter() - start, 1e-6)
    mean_loss = total_loss / max(sample_count, 1)
    mean_components = {key: value / max(sample_count, 1) for key, value in component_sums.items()}
    samples_per_second = sample_count / elapsed
    sample_loss_means = {
        sample_key: sample_loss_sums[sample_key] / max(sample_loss_counts[sample_key], 1)
        for sample_key in sample_loss_sums
    }
    return mean_loss, mean_components, samples_per_second, sample_loss_means


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def build_entry_signature(entries: Sequence[V9SampleEntry]) -> list[str]:
    return [f"{entry.dataset_key}:{entry.tile_name}" for entry in entries]


def build_training_checkpoint(
    *,
    args: argparse.Namespace,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    history: Sequence[dict[str, Any]],
    best_val_loss: float,
    best_val_epoch: int,
    best_selection_metric_name: str,
    best_selection_metric_value: float,
    best_epoch: int,
    epochs_since_best: int,
    selected_entries: Sequence[V9SampleEntry],
    train_entries: Sequence[V9SampleEntry],
    val_entries: Sequence[V9SampleEntry],
) -> dict[str, Any]:
    return {
        "schema_version": "v9-train-checkpoint.v1",
        "created_at_utc": utc_now_iso(),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": list(history),
        "best_val_loss": float(best_val_loss),
        "best_val_epoch": int(best_val_epoch),
        "best_selection_metric_name": best_selection_metric_name,
        "best_selection_metric_value": float(best_selection_metric_value),
        "best_epoch": int(best_epoch),
        "epochs_since_best": int(epochs_since_best),
        "config": vars(args),
        "feature_contract": build_v9_feature_contract(args),
        "selected_signature": build_entry_signature(selected_entries),
        "train_signature": build_entry_signature(train_entries),
        "val_signature": build_entry_signature(val_entries),
    }


def load_resume_state(
    *,
    checkpoint_path: Path,
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    selected_entries: Sequence[V9SampleEntry],
    train_entries: Sequence[V9SampleEntry],
    val_entries: Sequence[V9SampleEntry],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    selected_signature = build_entry_signature(selected_entries)
    train_signature = build_entry_signature(train_entries)
    val_signature = build_entry_signature(val_entries)
    if checkpoint.get("selected_signature") != selected_signature:
        raise SystemExit("Resume checkpoint selected sample signature does not match the current run inputs.")
    if checkpoint.get("train_signature") != train_signature:
        raise SystemExit("Resume checkpoint train split does not match the current run inputs.")
    if checkpoint.get("val_signature") != val_signature:
        raise SystemExit("Resume checkpoint validation split does not match the current run inputs.")

    saved_config = dict(checkpoint.get("config", {}))
    for key in (
        "batch_size",
        "learning_rate",
        "val_fraction",
        "seed",
        "height_scale",
        "residual_scale",
        "group_block_size",
        "hidden_channels",
        "blocks_per_stage",
        "curation_mode",
        "curation_diversity_block_size",
        "curation_max_per_group",
        "selection_metric",
        "dev_eval_cache_manifest",
        "dev_eval_limit",
        "dev_eval_every",
        "dev_eval_block_size",
        "disable_brush_mask",
        "detail_focus_every_epochs",
        "detail_focus_min_epoch",
        "detail_focus_stall_threshold",
        "detail_focus_top_fraction",
        "detail_focus_gradient_weight",
        "detail_focus_detail_residual_weight",
    ):
        if key in saved_config and getattr(args, key) != saved_config[key]:
            raise SystemExit(f"Resume checkpoint config mismatch for '{key}': current={getattr(args, key)!r} saved={saved_config[key]!r}")

    return {
        "history": list(checkpoint.get("history", [])),
        "start_epoch": int(checkpoint.get("epoch", 0)),
        "best_val_loss": float(checkpoint.get("best_val_loss", math.inf)),
        "best_val_epoch": int(checkpoint.get("best_val_epoch", 0)),
        "best_selection_metric_name": str(checkpoint.get("best_selection_metric_name", "val_loss")),
        "best_selection_metric_value": float(checkpoint.get("best_selection_metric_value", math.inf)),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "epochs_since_best": int(checkpoint.get("epochs_since_best", 0)),
    }


def resolve_selection_metric_name(args: argparse.Namespace) -> str:
    if args.selection_metric != "auto":
        return str(args.selection_metric)
    if args.dev_eval_cache_manifest:
        return "dev_wdl_mae_improvement"
    return "val_loss"


def resolve_selection_metric(
    *,
    selection_metric_name: str,
    val_loss: float,
    dev_eval: dict[str, float] | None,
) -> tuple[float, bool, bool]:
    if selection_metric_name == "val_loss":
        return float(val_loss), False, True
    if dev_eval is None:
        return 0.0, True, False
    if selection_metric_name == "dev_global_mae":
        return float(dev_eval["model_global_mae"]), False, True
    if selection_metric_name == "dev_wdl_mae_improvement":
        return float(dev_eval["wdl_global_mae"] - dev_eval["model_global_mae"]), True, True
    raise ValueError(f"Unsupported selection metric: {selection_metric_name}")


def evaluate_model_on_entries(
    *,
    model: nn.Module,
    entries: Sequence[V9SampleEntry],
    device: torch.device,
    amp_dtype: torch.dtype,
    height_scale: float,
    residual_scale: float,
    channels_last: bool,
    include_brush_mask: bool,
) -> dict[str, float]:
    if not entries:
        raise ValueError("Development evaluation requires at least one cache entry.")

    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}
    model_was_training = model.training
    model.eval()

    wdl_abs_sum = 0.0
    wdl_sq_sum = 0.0
    model_abs_sum = 0.0
    model_sq_sum = 0.0
    total_pixels = 0
    tile_wdl_mae_sum = 0.0
    tile_model_mae_sum = 0.0
    tile_count = 0

    with torch.no_grad():
        for entry in entries:
            arrays = load_npz_arrays(entry.shard_path)
            gt = arrays["height_257"].astype(np.float32)

            height_17 = torch.from_numpy(arrays["height_17"].astype(np.float32)).unsqueeze(0)
            if "wdl_17" in arrays:
                base_17 = torch.from_numpy(arrays["wdl_17"].astype(np.float32)).unsqueeze(0)
            else:
                base_17 = height_17.clone()

            base_65 = F.interpolate(base_17.unsqueeze(0), size=(65, 65), mode="bilinear", align_corners=True).squeeze(0)
            base_257 = F.interpolate(base_17.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=True).squeeze(0)
            inputs, _ = _build_v9_input_channels(
                arrays=arrays,
                base_257_scaled=base_257 / height_scale,
                include_brush_mask=include_brush_mask,
            )

            device_inputs = inputs.unsqueeze(0).to(device)
            if channels_last:
                device_inputs = device_inputs.contiguous(memory_format=torch.channels_last)
            device_base_17 = (base_17 / height_scale).unsqueeze(0).to(device)
            device_base_65 = (base_65 / height_scale).unsqueeze(0).to(device)
            device_base_257 = (base_257 / height_scale).unsqueeze(0).to(device)

            with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else nullcontext()):
                coarse_delta_17, mid_delta_65, detail_delta_257 = model(device_inputs)
                _, _, full_height_257 = build_predictions(
                    coarse_delta_17=coarse_delta_17,
                    mid_delta_65=mid_delta_65,
                    detail_delta_257=detail_delta_257,
                    base_height_17=device_base_17,
                    base_height_65=device_base_65,
                    base_height_257=device_base_257,
                    residual_scale=residual_scale,
                    height_scale=height_scale,
                )

            model_pred = (full_height_257.squeeze(0).squeeze(0).detach().cpu().numpy() * height_scale).astype(np.float32)
            wdl_pred = base_257.squeeze(0).cpu().numpy().astype(np.float32)

            wdl_err = wdl_pred - gt
            model_err = model_pred - gt
            wdl_abs = np.abs(wdl_err)
            model_abs = np.abs(model_err)

            wdl_abs_sum += float(wdl_abs.sum())
            wdl_sq_sum += float((wdl_err ** 2).sum())
            model_abs_sum += float(model_abs.sum())
            model_sq_sum += float((model_err ** 2).sum())
            total_pixels += int(gt.size)
            tile_wdl_mae_sum += float(wdl_abs.mean())
            tile_model_mae_sum += float(model_abs.mean())
            tile_count += 1

    model.train(model_was_training)

    return {
        "tile_count": float(tile_count),
        "wdl_global_mae": wdl_abs_sum / max(total_pixels, 1),
        "wdl_global_rmse": math.sqrt(wdl_sq_sum / max(total_pixels, 1)),
        "wdl_mean_tile_mae": tile_wdl_mae_sum / max(tile_count, 1),
        "model_global_mae": model_abs_sum / max(total_pixels, 1),
        "model_global_rmse": math.sqrt(model_sq_sum / max(total_pixels, 1)),
        "model_mean_tile_mae": tile_model_mae_sum / max(tile_count, 1),
        "wdl_mae_improvement": (wdl_abs_sum / max(total_pixels, 1)) - (model_abs_sum / max(total_pixels, 1)),
    }


def build_v9_feature_contract(args: argparse.Namespace | None = None) -> dict[str, list[str] | str]:
    zeroed_input_signals: list[str] = []
    if args is not None and getattr(args, "disable_brush_mask", False):
        zeroed_input_signals.append("brush_imprint_mask")

    return {
        "contract_version": "v9-native-inputs.v2",
        "active_input_signals": list(V9_ACTIVE_INPUT_SIGNALS),
        "native_target_signals": list(V9_NATIVE_TARGET_SIGNALS),
        "zeroed_input_signals": zeroed_input_signals,
        "summary": (
            "V9 now consumes the V7.7 multichannel context stack plus the native hole mask, "
            "while keeping raw native terrain targets instead of PNG heightmaps."
        ),
    }


def describe_v9_input_stack(args: argparse.Namespace) -> str:
    brush_state = "disabled" if getattr(args, "disable_brush_mask", False) else "enabled"
    return (
        "Active v9 inputs | terrain-only/no-liquid/no-object/no-MCCV fallback minimap RGB + normal RGB + WDL/base prior + "
        f"height hints + liquid/object masks + brush mask {brush_state} + liquid height + hole mask"
    )


def parse_cohort_sizes(value: str | None) -> list[int]:
    if not value:
        return []
    sizes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size > 1 and size not in sizes:
            sizes.append(size)
    return sizes


def _height_to_rgb(height: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    scale = max(max_value - min_value, 1e-6)
    normalized = np.clip((height - min_value) / scale, 0.0, 1.0)
    grayscale = (normalized * 255.0).astype(np.uint8)
    return np.repeat(grayscale[:, :, None], 3, axis=2)


def _error_to_rgb(predicted: np.ndarray, target: np.ndarray, height_scale: float) -> np.ndarray:
    error = np.abs(predicted - target) * height_scale
    normalized = np.clip(error / 64.0, 0.0, 1.0)
    red = (normalized * 255.0).astype(np.uint8)
    blue = ((1.0 - normalized) * 48.0).astype(np.uint8)
    return np.stack([red, np.zeros_like(red), blue], axis=2)


def _single_channel_to_rgb(channel: np.ndarray, min_value: float | None = None, max_value: float | None = None) -> np.ndarray:
    channel = channel.astype(np.float32)
    if min_value is None:
        min_value = float(np.min(channel)) if channel.size else 0.0
    if max_value is None:
        max_value = float(np.max(channel)) if channel.size else 1.0
    scale = max(float(max_value) - float(min_value), 1e-6)
    normalized = np.clip((channel - float(min_value)) / scale, 0.0, 1.0)
    grayscale = (normalized * 255.0).astype(np.uint8)
    return np.repeat(grayscale[:, :, None], 3, axis=2)


def _resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8)


def select_preview_entries(entries: Sequence[V9SampleEntry], preview_count: int, preview_seed: int) -> list[V9SampleEntry]:
    if preview_count <= 0 or not entries:
        return []
    if len(entries) <= preview_count:
        return list(entries)

    rng = random.Random(preview_seed)
    selected_indices = sorted(rng.sample(range(len(entries)), preview_count))
    return [entries[index] for index in selected_indices]


def export_preview_images(
    model: nn.Module,
    entries: Sequence[V9SampleEntry],
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    height_scale: float,
    residual_scale: float,
    preview_count: int,
    channels_last: bool,
    preview_seed: int,
    include_brush_mask: bool,
    epoch: int,
    archive_every_epochs: int,
) -> None:
    if not entries or preview_count <= 0:
        return

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for existing_preview in preview_dir.glob("*.png"):
        existing_preview.unlink()

    archive_dir: Path | None = None
    if archive_every_epochs > 0 and epoch % archive_every_epochs == 0:
        archive_dir = preview_dir / "history" / f"epoch_{epoch:04d}"
        archive_dir.mkdir(parents=True, exist_ok=True)

    preview_entries = select_preview_entries(entries, preview_count, preview_seed)
    selected_names = ", ".join(entry.tile_name for entry in preview_entries)
    readme_text = (
        "Preview layout\n"
        "Top row, left to right:\n"
        "1. Minimap RGB input\n"
        "2. Predicted full 257x257 terrain height\n"
        "3. Ground-truth full 257x257 terrain height\n"
        "4. Absolute full-height error heatmap (red = larger error)\n\n"
        "Bottom row, left to right:\n"
        "1. Predicted coarse 17x17 terrain, resized for display\n"
        "2. Ground-truth 17x17 terrain, resized for display\n"
        "3. Predicted mid 65x65 terrain, resized for display\n"
        "4. Ground-truth 65x65 terrain, resized for display\n\n"
        "Third row, left to right:\n"
        "1. Liquid mask input (white = liquid, black = none)\n"
        "2. Liquid height prior input\n"
        "3. Object footprint mask input\n"
        "4. Hole mask input\n\n"
        "Reading guide\n"
        "- Good results have similar large-scale landform shapes in columns 2 and 3 of the top row.\n"
        "- The error panel should get darker and less red over time.\n"
        "- The bottom row shows whether the model is learning the right coarse terrain scaffold before fine detail.\n"
        "- The third row shows whether the cached tile actually carried liquid and other masking priors into training.\n"
        "- Small texture-like minimap details do not have to match height directly; terrain shape matters more than color similarity.\n"
        f"- This export is a rotating validation subset, not a fixed first-N list. Current tiles: {selected_names or 'none'}.\n"
        f"- Source epoch: {epoch}.\n"
    )
    write_text(preview_dir / "README.txt", readme_text)
    if archive_dir is not None:
        write_text(archive_dir / "README.txt", readme_text)
    dataset = V9NativeDataset(
        preview_entries,
        height_scale=height_scale,
        residual_scale=residual_scale,
        include_brush_mask=include_brush_mask,
    )
    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in range(min(preview_count, len(dataset))):
            batch = dataset[index]
            preview_batch: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    preview_batch[key] = value.unsqueeze(0)
                else:
                    preview_batch[key] = [value]
            device_batch = move_batch_to_device(preview_batch, device, channels_last)
            with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else nullcontext()):
                coarse_delta_17, mid_delta_65, detail_delta_257 = model(device_batch["inputs"])
                coarse_height_17, mid_height_65, full_height_257 = build_predictions(
                    coarse_delta_17=coarse_delta_17,
                    mid_delta_65=mid_delta_65,
                    detail_delta_257=detail_delta_257,
                    base_height_17=device_batch["base_height_17"],
                    base_height_65=device_batch["base_height_65"],
                    base_height_257=device_batch["base_height_257"],
                    residual_scale=residual_scale,
                    height_scale=height_scale,
                )

            minimap = (batch["preview_minimap_rgb"].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            target_257 = batch["target_height_257"].squeeze(0).cpu().numpy()
            target_65 = batch["target_height_65"].squeeze(0).cpu().numpy()
            target_17 = batch["target_height_17"].squeeze(0).cpu().numpy()
            pred_257 = full_height_257.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_65 = mid_height_65.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_17 = coarse_height_17.squeeze(0).squeeze(0).detach().cpu().numpy()
            liquid_mask = batch["preview_liquid_mask"].squeeze(0).cpu().numpy()
            liquid_height = batch["preview_liquid_height"].squeeze(0).cpu().numpy()
            object_mask = batch["preview_object_mask"].squeeze(0).cpu().numpy()
            hole_mask = batch["preview_hole_mask"].squeeze(0).cpu().numpy()

            preview_min = float(min(target_257.min(), pred_257.min()))
            preview_max = float(max(target_257.max(), pred_257.max()))
            row_one = np.concatenate(
                [
                    minimap,
                    _height_to_rgb(pred_257, preview_min, preview_max),
                    _height_to_rgb(target_257, preview_min, preview_max),
                    _error_to_rgb(pred_257, target_257, height_scale),
                ],
                axis=1,
            )
            row_two = np.concatenate(
                [
                    _resize_rgb(_height_to_rgb(pred_17, float(min(target_17.min(), pred_17.min())), float(max(target_17.max(), pred_17.max()))), 257),
                    _resize_rgb(_height_to_rgb(target_17, float(min(target_17.min(), pred_17.min())), float(max(target_17.max(), pred_17.max()))), 257),
                    _resize_rgb(_height_to_rgb(pred_65, float(min(target_65.min(), pred_65.min())), float(max(target_65.max(), pred_65.max()))), 257),
                    _resize_rgb(_height_to_rgb(target_65, float(min(target_65.min(), pred_65.min())), float(max(target_65.max(), pred_65.max()))), 257),
                ],
                axis=1,
            )
            row_three = np.concatenate(
                [
                    _single_channel_to_rgb(liquid_mask, 0.0, 1.0),
                    _single_channel_to_rgb(liquid_height, preview_min, preview_max),
                    _single_channel_to_rgb(object_mask, 0.0, 1.0),
                    _resize_rgb(_single_channel_to_rgb(hole_mask, 0.0, 1.0), 257),
                ],
                axis=1,
            )
            preview = np.concatenate([row_one, row_two, row_three], axis=0)
            tile_name = preview_entries[index].tile_name
            preview_name = f"{index:02d}_{tile_name}.png"
            Image.fromarray(preview).save(preview_dir / preview_name)
            if archive_dir is not None:
                Image.fromarray(preview).save(archive_dir / preview_name)

    model.train(model_was_training)


def train_single_run(
    selected_entries: Sequence[V9SampleEntry],
    dev_eval_entries: Sequence[V9SampleEntry],
    args: argparse.Namespace,
    run_output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    if len(selected_entries) < 2:
        raise SystemExit("Need at least 2 accepted samples to train and validate.")

    selected_liquid_coverage = float(sum(entry.liquid_coverage for entry in selected_entries))
    if selected_liquid_coverage <= 0.0:
        print("WARNING: selected run pool has zero liquid coverage; liquid-mask and liquid-height input channels are present in code but effectively empty for this run.")

    train_indices, val_indices = split_grouped_indices(selected_entries, args.val_fraction, args.seed, args.group_block_size)
    train_entries = [selected_entries[index] for index in train_indices]
    val_entries = [selected_entries[index] for index in val_indices]

    train_dataset = V9NativeDataset(
        train_entries,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
    )
    val_dataset = V9NativeDataset(
        val_entries,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.val_workers, pin_memory=device.type == "cuda")
    val_batches = len(val_loader)

    model = V9TerrainModel(hidden_channels=args.hidden_channels, blocks_per_stage=args.blocks_per_stage).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        min_lr=args.min_learning_rate,
    )
    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    selection_metric_name = resolve_selection_metric_name(args)
    best_selection_metric_value = -math.inf if selection_metric_name == "dev_wdl_mae_improvement" else math.inf
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    epochs_since_best = 0
    start_epoch = 0
    resumed_from: str | None = None
    stop_reason = "completed_requested_epochs"
    last_checkpoint_path = run_output_dir / "last_checkpoint.pt"
    sample_loss_ema: dict[str, float] = {}

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint does not exist: {resume_path}")
        resume_state = load_resume_state(
            checkpoint_path=resume_path,
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            selected_entries=selected_entries,
            train_entries=train_entries,
            val_entries=val_entries,
            device=device,
        )
        history = resume_state["history"]
        start_epoch = int(resume_state["start_epoch"])
        best_val_loss = float(resume_state["best_val_loss"])
        selection_metric_name = str(resume_state["best_selection_metric_name"])
        best_selection_metric_value = float(resume_state["best_selection_metric_value"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_since_best = int(resume_state["epochs_since_best"])
        resumed_from = str(resume_path)

    print(
        f"Training V9 | epochs={args.epochs} | batch={args.batch_size} | lr={args.learning_rate:.2e} | "
        f"plateau_patience={args.lr_plateau_patience} | early_stop_patience={args.early_stop_patience} | "
        f"preview_count={args.preview_count} | preview_every={args.preview_every_epochs} | preview_archive_every={args.preview_archive_every_epochs} | "
        f"train_sampler={args.train_sampler} | hard_replay_fraction={args.hard_replay_fraction:.2f} | "
        f"detail_focus_every={args.detail_focus_every_epochs} | detail_focus_top_fraction={args.detail_focus_top_fraction:.2f}"
    )
    args.current_stall = epochs_since_best
    initial_train_loader, _ = build_train_loader(train_dataset, train_entries, args, device, max(start_epoch + 1, 1), sample_loss_ema)
    print(
        f"Dataset split | train_samples={len(train_entries)} ({len(initial_train_loader)} batch(es)) | "
        f"val_samples={len(val_entries)} ({val_batches} batch(es))"
    )
    print(
        "Preview legend: top=minimap | pred257 | target257 | abs-error ; "
        "bottom=pred17 | target17 | pred65 | target65 ; "
        "third=liquid_mask | liquid_height | object_mask | hole_mask"
    )
    print(describe_v9_input_stack(args))
    if resumed_from is not None:
        print(f"Resuming from checkpoint | path={resumed_from} | start_epoch={start_epoch + 1} | best {best_val_loss:.6f}@{best_epoch}")

    if start_epoch >= args.epochs:
        print(f"Resume checkpoint is already at epoch {start_epoch}, which meets or exceeds requested epochs={args.epochs}; skipping training.")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        args.current_stall = epochs_since_best
        train_loader, detail_focus_active = build_train_loader(train_dataset, train_entries, args, device, epoch, sample_loss_ema)
        train_loss, train_components, train_sps, train_sample_losses = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_dtype=amp_dtype,
            residual_scale=args.residual_scale,
            height_scale=args.height_scale,
            channels_last=args.channels_last,
            current_epoch=epoch,
            args=args,
        )
        update_sample_loss_ema(sample_loss_ema, train_sample_losses, args.hard_replay_ema_decay)
        val_loss, val_components, val_sps, _ = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            amp_dtype=amp_dtype,
            residual_scale=args.residual_scale,
            height_scale=args.height_scale,
            channels_last=args.channels_last,
            current_epoch=epoch,
            args=args,
        )

        dev_eval_metrics: dict[str, float] | None = None
        should_run_dev_eval = bool(dev_eval_entries) and args.dev_eval_every > 0 and epoch % args.dev_eval_every == 0
        if should_run_dev_eval:
            dev_eval_metrics = evaluate_model_on_entries(
                model=model,
                entries=dev_eval_entries,
                device=device,
                amp_dtype=amp_dtype,
                height_scale=args.height_scale,
                residual_scale=args.residual_scale,
                channels_last=args.channels_last,
                include_brush_mask=not args.disable_brush_mask,
            )

        loss_weights = resolve_loss_weights(epoch, args, epochs_since_best)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_components": train_components,
            "val_components": val_components,
            "train_samples_per_second": train_sps,
            "val_samples_per_second": val_sps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "loss_weights": loss_weights,
        }
        if dev_eval_metrics is not None:
            epoch_record["dev_eval"] = dev_eval_metrics
        history.append(epoch_record)

        previous_best = best_val_loss
        selection_metric_value, selection_metric_higher_is_better, selection_metric_ready = resolve_selection_metric(
            selection_metric_name=selection_metric_name,
            val_loss=val_loss,
            dev_eval=dev_eval_metrics,
        )
        if not selection_metric_ready:
            is_best = False
        elif selection_metric_higher_is_better:
            is_best = selection_metric_value > best_selection_metric_value
        else:
            is_best = selection_metric_value < best_selection_metric_value
        if is_best:
            best_val_loss = val_loss
            best_selection_metric_value = selection_metric_value
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        val_delta = 0.0 if not math.isfinite(previous_best) else val_loss - previous_best
        train_val_gap = val_loss - train_loss
        current_lr = float(optimizer.param_groups[0]["lr"])
        status = "BEST" if is_best else f"stall={epochs_since_best}"
        print(
            f"epoch {epoch:03d}/{args.epochs} | {status} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"delta {val_delta:+.6f} | gap {train_val_gap:+.6f} | best {best_val_loss:.6f}@{best_epoch} | lr {current_lr:.2e}"
        )
        print(
            f"  full {val_components['full_l1']:.6f} | mid {val_components['mid_l1']:.6f} | coarse {val_components['coarse_l1']:.6f} | "
            f"grad {val_components['gradient']:.6f} | mid_res {val_components['mid_residual']:.6f} | detail_res {val_components['detail_residual']:.6f} | "
            f"train_sps {train_sps:.2f} | val_sps {val_sps:.2f}"
        )
        print(
            f"  loss_w full {loss_weights['full_l1']:.2f} | mid {loss_weights['mid_l1']:.2f} | coarse {loss_weights['coarse_l1']:.2f} | "
            f"grad {loss_weights['gradient']:.2f} | mid_res {loss_weights['mid_residual']:.2f} | detail_res {loss_weights['detail_residual']:.2f}"
        )
        if dev_eval_metrics is not None:
            print(
                f"  dev_eval tiles {dev_eval_metrics['tile_count']:.0f} | model_mae {dev_eval_metrics['model_global_mae']:.6f} | "
                f"wdl_mae {dev_eval_metrics['wdl_global_mae']:.6f} | wdl_gain {dev_eval_metrics['wdl_mae_improvement']:.6f} | "
                f"selection {selection_metric_name}={selection_metric_value:.6f}"
            )
        elif selection_metric_name != "val_loss" and not selection_metric_ready:
            print(f"  dev_eval pending | selection metric {selection_metric_name} not computed this epoch")
        if train_sample_losses:
            hardest_sample_key, hardest_sample_loss = max(train_sample_losses.items(), key=lambda item: item[1])
            focus_label = " detail-focus" if detail_focus_active else ""
            print(f"  sampler {args.train_sampler}{focus_label} | hardest replay sample {hardest_sample_key} loss {hardest_sample_loss:.6f}")

        should_export_previews = is_best or (args.preview_every_epochs > 0 and epoch % args.preview_every_epochs == 0)

        if is_best:
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }

        if should_export_previews:
            export_preview_images(
                model=model,
                entries=val_entries,
                output_dir=run_output_dir,
                device=device,
                amp_dtype=amp_dtype,
                height_scale=args.height_scale,
                residual_scale=args.residual_scale,
                preview_count=args.preview_count,
                channels_last=args.channels_last,
                preview_seed=args.seed + epoch,
                include_brush_mask=not args.disable_brush_mask,
                epoch=epoch,
                archive_every_epochs=args.preview_archive_every_epochs,
            )
            preview_message = f"  refreshed previews in {run_output_dir / 'previews'}"
            if args.preview_archive_every_epochs > 0 and epoch % args.preview_archive_every_epochs == 0:
                preview_message += f" and archived epoch {epoch}"
            print(preview_message)

        if is_best:
            print(f"  saved best checkpoint in {run_output_dir / 'best_model.pt'}")

        previous_lr = current_lr
        scheduler.step(val_loss)
        new_lr = float(optimizer.param_groups[0]["lr"])
        if new_lr < previous_lr:
            print(f"  learning rate reduced: {previous_lr:.2e} -> {new_lr:.2e}")

        torch.save(
            build_training_checkpoint(
                args=args,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_val_loss=best_val_loss,
                best_val_epoch=best_epoch,
                best_selection_metric_name=selection_metric_name,
                best_selection_metric_value=best_selection_metric_value,
                best_epoch=best_epoch,
                epochs_since_best=epochs_since_best,
                selected_entries=selected_entries,
                train_entries=train_entries,
                val_entries=val_entries,
            ),
            last_checkpoint_path,
        )

        if epoch >= args.early_stop_min_epochs and epochs_since_best >= args.early_stop_patience:
            stop_reason = "early_stop_patience"
            print(
                f"  early stop: no new best for {epochs_since_best} epoch(s) after epoch {best_epoch}; "
                f"best val remained {best_val_loss:.6f}"
            )
            break

        if args.pause_every_epochs > 0 and epoch < args.epochs and epoch % args.pause_every_epochs == 0:
            stop_reason = "paused_for_inspection"
            print(
                f"  pause checkpoint: reached epoch {epoch}; review previews and metrics, then resume with "
                f"--resume-from {last_checkpoint_path} --epochs {args.epochs}"
            )
            break

    run_summary = {
        "schema_version": "v9-train-run.v1",
        "created_at_utc": utc_now_iso(),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "selected_samples": len(selected_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "history": history,
        "start_epoch": start_epoch,
        "final_epoch": history[-1]["epoch"] if history else start_epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_selection_metric_name": selection_metric_name,
        "best_selection_metric_value": best_selection_metric_value,
        "stop_reason": stop_reason,
        "resumed_from": resumed_from,
        "config": vars(args),
        "preview_count": args.preview_count,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "last_checkpoint_path": str(last_checkpoint_path),
        "feature_contract": build_v9_feature_contract(args),
    }
    write_json(run_output_dir / "run_summary.json", run_summary)
    if best_state is not None:
        torch.save(best_state, run_output_dir / "best_model.pt")
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the first native-signal V9 terrain model from cached tensor shards.")
    parser.add_argument("cache_manifest", help="Path to v9_tensor_cache_manifest.json produced by build_v9_native_tensor_cache.py.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for reports, logs, and checkpoints.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on accepted audited samples before splitting.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--subset", type=int, default=None,
                        help="Optional limit on the number of sane samples to randomly subsample before training. "
                             "Use for fast iteration on a small representative set. Applied before curation.")
    parser.add_argument("--subset-seed", type=int, default=None,
                        help="Random seed for the subset selection. Defaults to --seed if not specified.")
    parser.add_argument("--height-scale", type=float, default=DEFAULT_HEIGHT_SCALE)
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--min-height-range", type=float, default=DEFAULT_MIN_HEIGHT_RANGE)
    parser.add_argument("--min-minimap-variance", type=float, default=DEFAULT_MIN_MINIMAP_VARIANCE)
    parser.add_argument("--min-minimap-gradient", type=float, default=DEFAULT_MIN_MINIMAP_GRADIENT)
    parser.add_argument("--max-mean-wdl-delta", type=float, default=DEFAULT_MAX_MEAN_WDL_DELTA)
    parser.add_argument("--max-abs-wdl-delta", type=float, default=DEFAULT_MAX_ABS_WDL_DELTA)
    parser.add_argument("--require-wdl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-minimap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-block-size", type=int, default=DEFAULT_GROUP_BLOCK_SIZE)
    parser.add_argument("--hidden-channels", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument("--blocks-per-stage", type=int, default=DEFAULT_BLOCKS_PER_STAGE)
    parser.add_argument("--train-workers", type=int, default=DEFAULT_TRAIN_WORKERS)
    parser.add_argument("--val-workers", type=int, default=DEFAULT_VAL_WORKERS)
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default=DEFAULT_AMP_DTYPE)
    parser.add_argument("--target-curated-samples", type=int, default=DEFAULT_TARGET_CURATED_SAMPLES)
    parser.add_argument("--cohort-sizes", default=None, help="Comma-separated cohort sizes to train side by side from the ranked sane pool, for example '27,48,64'.")
    parser.add_argument("--selection-metric", choices=["auto", "val_loss", "dev_global_mae", "dev_wdl_mae_improvement"], default=DEFAULT_SELECTION_METRIC,
                        help="Metric used to decide the best checkpoint and drive early-stop stall counting.")
    parser.add_argument("--dev-eval-cache-manifest", default=None,
                        help="Optional separate cache manifest used as a stable development holdout for checkpoint selection.")
    parser.add_argument("--dev-eval-limit", type=int, default=None,
                        help="Optional limit on the number of dev-eval holdout entries after diversity selection.")
    parser.add_argument("--dev-eval-every", type=int, default=DEFAULT_DEV_EVAL_EVERY,
                        help="Run development holdout evaluation every N epochs. Set 0 to disable dev-eval passes.")
    parser.add_argument("--dev-eval-block-size", type=int, default=DEFAULT_DEV_EVAL_BLOCK_SIZE,
                        help="Diversity block size used when selecting the dev-eval holdout subset.")
    parser.add_argument("--preview-count", type=int, default=DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--train-sampler", choices=["random", "bucketed"], default=DEFAULT_TRAIN_SAMPLER,
                        help="Training sample order. 'bucketed' interleaves terrain complexity and coverage buckets instead of using naive random shuffle.")
    parser.add_argument("--hard-replay-fraction", type=float, default=DEFAULT_HARD_REPLAY_FRACTION,
                        help="Fraction of each sampling bucket reserved near the front for prior hard samples once replay is active.")
    parser.add_argument("--hard-replay-warmup-epochs", type=int, default=DEFAULT_HARD_REPLAY_WARMUP_EPOCHS,
                        help="Number of initial epochs to run before epoch-to-epoch hard-example replay becomes eligible.")
    parser.add_argument("--hard-replay-ema-decay", type=float, default=DEFAULT_HARD_REPLAY_EMA_DECAY,
                        help="EMA decay used when tracking sample difficulty across epochs for replay ordering.")
    parser.add_argument("--detail-focus-every-epochs", type=int, default=DEFAULT_DETAIL_FOCUS_EVERY_EPOCHS,
                        help="Run a detail-focused epoch every N epochs by training only on the highest-detail tiles. Set 0 to disable periodic detail focus.")
    parser.add_argument("--detail-focus-min-epoch", type=int, default=DEFAULT_DETAIL_FOCUS_MIN_EPOCH,
                        help="Minimum epoch before periodic or stall-triggered detail focus can activate.")
    parser.add_argument("--detail-focus-stall-threshold", type=int, default=DEFAULT_DETAIL_FOCUS_STALL_THRESHOLD,
                        help="Automatically switch to detail-focused epochs after this many epochs without a new best. Set 0 to disable stall-triggered focus.")
    parser.add_argument("--detail-focus-top-fraction", type=float, default=DEFAULT_DETAIL_FOCUS_TOP_FRACTION,
                        help="Fraction of the train split kept during a detail-focused epoch, ranked by detail energy and replay loss.")
    parser.add_argument("--detail-focus-gradient-weight", type=float, default=DEFAULT_DETAIL_FOCUS_GRADIENT_WEIGHT,
                        help="Minimum gradient-loss weight enforced during detail-focused epochs.")
    parser.add_argument("--detail-focus-detail-residual-weight", type=float, default=DEFAULT_DETAIL_FOCUS_DETAIL_RESIDUAL_WEIGHT,
                        help="Minimum detail-residual weight enforced during detail-focused epochs.")
    parser.add_argument("--pause-every-epochs", type=int, default=DEFAULT_PAUSE_EVERY_EPOCHS,
                        help="Write the normal checkpoint and stop cleanly every N epochs so the run can be inspected and resumed. Set 0 to disable periodic pauses.")
    parser.add_argument("--preview-every-epochs", type=int, default=DEFAULT_PREVIEW_EVERY_EPOCHS, help="Refresh the live validation preview set every N epochs. Set 0 to disable periodic refreshes and only update on best epochs.")
    parser.add_argument("--preview-archive-every-epochs", type=int, default=DEFAULT_PREVIEW_ARCHIVE_EVERY_EPOCHS, help="Keep archived preview snapshots every N epochs under previews/history. Set 0 to disable archiving.")
    parser.add_argument("--lr-plateau-patience", type=int, default=DEFAULT_LR_PLATEAU_PATIENCE)
    parser.add_argument("--lr-plateau-factor", type=float, default=DEFAULT_LR_PLATEAU_FACTOR)
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_MIN_LEARNING_RATE)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-epochs", type=int, default=DEFAULT_EARLY_STOP_MIN_EPOCHS)
    parser.add_argument("--aux-loss-decay-start-epoch", type=int, default=DEFAULT_AUX_LOSS_DECAY_START_EPOCH)
    parser.add_argument("--aux-loss-decay-epochs", type=int, default=DEFAULT_AUX_LOSS_DECAY_EPOCHS)
    parser.add_argument("--late-mid-l1-weight", type=float, default=DEFAULT_LATE_MID_L1_WEIGHT)
    parser.add_argument("--late-coarse-l1-weight", type=float, default=DEFAULT_LATE_COARSE_L1_WEIGHT)
    parser.add_argument("--late-gradient-weight", type=float, default=DEFAULT_LATE_GRADIENT_WEIGHT)
    parser.add_argument("--late-mid-residual-weight", type=float, default=DEFAULT_LATE_MID_RESIDUAL_WEIGHT)
    parser.add_argument("--late-detail-residual-weight", type=float, default=DEFAULT_LATE_DETAIL_RESIDUAL_WEIGHT)
    parser.add_argument("--curation-mode", choices=["diverse-quality", "top-quality"], default=DEFAULT_CURATION_MODE)
    parser.add_argument("--curation-diversity-block-size", type=int, default=DEFAULT_CURATION_DIVERSITY_BLOCK_SIZE)
    parser.add_argument("--curation-max-per-group", type=int, default=DEFAULT_CURATION_MAX_PER_GROUP)
    parser.add_argument("--disable-brush-mask", action=argparse.BooleanOptionalAction, default=False, help="Zero the brush imprint mask channel and remove brush-driven RGB attenuation.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", dest="use_compile", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--write-curated-manifest", action="store_true")
    parser.add_argument("--resume-from", default=None, help="Optional path to a last_checkpoint.pt file to resume training from.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.pause_every_epochs < 0:
        raise SystemExit("--pause-every-epochs must be 0 or greater.")
    if args.detail_focus_every_epochs < 0:
        raise SystemExit("--detail-focus-every-epochs must be 0 or greater.")
    if args.detail_focus_min_epoch < 1:
        raise SystemExit("--detail-focus-min-epoch must be at least 1.")
    if args.detail_focus_stall_threshold < 0:
        raise SystemExit("--detail-focus-stall-threshold must be 0 or greater.")
    if not 0.0 < args.detail_focus_top_fraction <= 1.0:
        raise SystemExit("--detail-focus-top-fraction must be greater than 0 and at most 1.")
    if args.dev_eval_every < 0:
        raise SystemExit("--dev-eval-every must be 0 or greater.")
    if args.dev_eval_block_size < 1:
        raise SystemExit("--dev-eval-block-size must be at least 1.")
    if args.epochs > MAX_ALLOWED_EPOCHS:
        print(f"Requested epochs {args.epochs} exceed cap {MAX_ALLOWED_EPOCHS}; clamping to {MAX_ALLOWED_EPOCHS}.")
        args.epochs = MAX_ALLOWED_EPOCHS
    seed_everything(args.seed)

    cache_manifest = Path(args.cache_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_cache_manifest(cache_manifest)
    audited, sane_entries, rejection_counts = audit_entries(
        entries=entries,
        require_wdl=args.require_wdl,
        require_minimap=args.require_minimap,
        min_height_range=args.min_height_range,
        min_minimap_variance=args.min_minimap_variance,
        min_minimap_gradient=args.min_minimap_gradient,
        max_mean_wdl_delta=args.max_mean_wdl_delta,
        max_abs_wdl_delta=args.max_abs_wdl_delta,
    )

    # Subset: randomly sample a smaller subset of the sane pool for fast iteration.
    # This is applied before curation so you can test training behavior on a small
    # representative set without processing the full sane pool.
    subset = getattr(args, "subset", None)
    subset_seed = getattr(args, "subset_seed", None)
    if subset and subset > 0 and subset < len(sane_entries):
        rng_subset = random.Random(subset_seed if subset_seed is not None else args.seed)
        sane_entries = rng_subset.sample(list(sane_entries), subset)
        print(f"  Subset: using {len(sane_entries)} samples (seed={subset_seed or args.seed})")

    curated_limit = args.limit if args.limit is not None else args.target_curated_samples
    accepted_entries = select_curated_entries(
        audited,
        curated_limit,
        args.curation_mode,
        args.curation_diversity_block_size,
        args.curation_max_per_group,
    )
    ranked_sane_entries = select_curated_entries(
        audited,
        None,
        args.curation_mode,
        args.curation_diversity_block_size,
        args.curation_max_per_group,
    )

    audit_payload = {
        "schema_version": "v9-native-audit.v1",
        "created_at_utc": utc_now_iso(),
        "cache_manifest": str(cache_manifest),
        "sane_pool": len(sane_entries),
        "curated_subset": len(accepted_entries),
        "rejected": len(entries) - len(sane_entries),
        "rejection_counts": rejection_counts,
        "entries": [
            {
                "tile_name": item.sample.tile_name,
                "dataset_key": item.sample.dataset_key,
                "shard_path": str(item.sample.shard_path),
                "accepted": item.accepted,
                "rejection_reason": item.rejection_reason,
                "height_range": item.sample.height_range,
                "minimap_variance": item.minimap_variance,
                "minimap_gradient": item.minimap_gradient,
                "mean_wdl_delta": item.mean_wdl_delta,
                "max_abs_wdl_delta": item.max_abs_wdl_delta,
                "hole_coverage": item.hole_coverage,
                "quality_score": item.quality_score,
            }
            for item in audited
        ],
    }
    audit_path = output_dir / "v9_audit_report.json"
    write_json(audit_path, audit_payload)

    if args.write_curated_manifest:
        curated_payload = {
            "schema_version": "v9-native-curated-manifest.v1",
            "created_at_utc": utc_now_iso(),
            "source_cache_manifest": str(cache_manifest),
            "accepted": len(accepted_entries),
            "entries": [
                {
                    "dataset_root": entry.dataset_root,
                    "dataset_key": entry.dataset_key,
                    "tile_name": entry.tile_name,
                    "shard_path": str(entry.shard_path),
                    "source_json": entry.source_json,
                }
                for entry in accepted_entries
            ],
        }
        write_json(output_dir / "v9_curated_manifest.json", curated_payload)

    print(f"Audited {len(entries)} shard(s): sane pool {len(sane_entries)}, curated subset {len(accepted_entries)}, rejected {len(entries) - len(sane_entries)}")
    if rejection_counts:
        print(f"Rejection counts: {rejection_counts}")

    if args.audit_only:
        return

    dev_eval_entries: list[V9SampleEntry] = []
    if args.dev_eval_cache_manifest:
        dev_eval_manifest_path = Path(args.dev_eval_cache_manifest)
        dev_eval_all_entries = load_cache_manifest(dev_eval_manifest_path)
        _, dev_eval_sane_entries, dev_eval_rejection_counts = audit_entries(
            entries=dev_eval_all_entries,
            require_wdl=args.require_wdl,
            require_minimap=args.require_minimap,
            min_height_range=args.min_height_range,
            min_minimap_variance=args.min_minimap_variance,
            min_minimap_gradient=args.min_minimap_gradient,
            max_mean_wdl_delta=args.max_mean_wdl_delta,
            max_abs_wdl_delta=args.max_abs_wdl_delta,
        )
        dev_eval_entries = select_diverse_eval_entries(
            dev_eval_sane_entries,
            args.dev_eval_limit,
            args.dev_eval_block_size,
            args.seed + 1009,
        )
        print(
            f"Dev-eval holdout | source={dev_eval_manifest_path} | sane_pool={len(dev_eval_sane_entries)} | "
            f"selected={len(dev_eval_entries)} | selection_metric={resolve_selection_metric_name(args)}"
        )
        if dev_eval_rejection_counts:
            print(f"Dev-eval rejection counts: {dev_eval_rejection_counts}")
        if not dev_eval_entries and args.selection_metric != "val_loss":
            raise SystemExit("Dev-eval selection metric requested, but the dev-eval holdout resolved to zero accepted entries.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    cohort_sizes = parse_cohort_sizes(args.cohort_sizes)
    if cohort_sizes:
        cohort_results: list[dict[str, Any]] = []
        for cohort_size in cohort_sizes:
            if len(ranked_sane_entries) < cohort_size:
                cohort_results.append({"cohort_size": cohort_size, "status": "skipped_insufficient_samples", "available": len(ranked_sane_entries)})
                continue
            cohort_entries = ranked_sane_entries[:cohort_size]
            cohort_dir = output_dir / f"cohort_{cohort_size:03d}"
            cohort_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                cohort_dir / "v9_curated_manifest.json",
                {
                    "schema_version": "v9-native-curated-manifest.v1",
                    "created_at_utc": utc_now_iso(),
                    "source_cache_manifest": str(cache_manifest),
                    "accepted": len(cohort_entries),
                    "entries": [
                        {
                            "dataset_root": entry.dataset_root,
                            "dataset_key": entry.dataset_key,
                            "tile_name": entry.tile_name,
                            "shard_path": str(entry.shard_path),
                            "source_json": entry.source_json,
                        }
                        for entry in cohort_entries
                    ],
                },
            )
            print(f"Starting cohort {cohort_size} with {len(cohort_entries)} ranked sane samples")
            summary = train_single_run(cohort_entries, args, cohort_dir, device, amp_dtype)
            cohort_results.append({"cohort_size": cohort_size, "status": "trained", "best_val_loss": summary["best_val_loss"]})
        write_json(output_dir / "cohort_summary.json", {"schema_version": "v9-cohort-summary.v1", "created_at_utc": utc_now_iso(), "results": cohort_results, "config": vars(args)})
        return

    train_single_run(accepted_entries, dev_eval_entries, args, output_dir, device, amp_dtype)


if __name__ == "__main__":
    main()
