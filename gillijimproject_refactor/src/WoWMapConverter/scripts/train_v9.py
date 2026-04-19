from __future__ import annotations

import argparse
import json
import math
import random
import time
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
from torch.utils.data import DataLoader, Dataset, Subset


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "ml-training" / "v9"
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 8
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
    has_wdl_17: bool
    has_minimap_rgb_256: bool

    @property
    def height_range(self) -> float:
        return float(self.height_max - self.height_min)


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


def load_cache_manifest(manifest_path: Path) -> list[V9SampleEntry]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    entries: list[V9SampleEntry] = []
    for entry in manifest.get("entries", []):
        tile_name = str(entry.get("tile_name", ""))
        map_name, tile_x, tile_y = parse_tile_coordinates(tile_name)
        entries.append(
            V9SampleEntry(
                dataset_root=str(entry.get("dataset_root", "")),
                dataset_key=str(entry.get("dataset_key", "")),
                tile_name=tile_name,
                map_name=map_name,
                tile_x=tile_x,
                tile_y=tile_y,
                shard_path=Path(str(entry.get("shard_path", ""))),
                source_json=str(entry.get("source_json", "")),
                height_min=float(entry.get("height_min", 0.0)),
                height_max=float(entry.get("height_max", 0.0)),
                has_wdl_17=bool(entry.get("has_wdl_17", False)),
                has_minimap_rgb_256=bool(entry.get("has_minimap_rgb_256", False)),
            )
        )
    return entries


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
    if not entry.shard_path.exists():
        return AuditedEntry(entry, 0.0, 0.0, math.inf, math.inf, 0.0, False, "missing_shard")

    arrays = load_npz_arrays(entry.shard_path)
    required_arrays = ("height_257", "height_17", "hole_mask_16x16")
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
        if mean_wdl_delta > max_mean_wdl_delta:
            return AuditedEntry(entry, minimap_variance, minimap_gradient, mean_wdl_delta, max_abs_wdl_delta, float(hole_mask.mean()), False, "mean_wdl_delta_too_high")
        if max_abs_wdl_delta > max_abs_wdl_delta:
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
    selected = accepted[:limit] if limit is not None else accepted
    return [entry.sample for entry in selected]


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
    def __init__(self, entries: Sequence[V9SampleEntry], height_scale: float, residual_scale: float):
        self.entries = list(entries)
        self.height_scale = float(height_scale)
        self.residual_scale = float(residual_scale)

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

        minimap = arrays.get("minimap_rgb_256")
        if minimap is None:
            minimap_tensor = torch.zeros((3, 257, 257), dtype=torch.float32)
        else:
            minimap_tensor = torch.from_numpy(minimap.astype(np.float32) / 255.0).permute(2, 0, 1)
            minimap_tensor = F.interpolate(minimap_tensor.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=False).squeeze(0)

        hole_mask = torch.from_numpy(arrays["hole_mask_16x16"].astype(np.float32)).unsqueeze(0)
        hole_mask_257 = F.interpolate(hole_mask.unsqueeze(0), size=(257, 257), mode="nearest").squeeze(0)

        base_65_scaled = base_65 / self.height_scale
        base_257_scaled = base_257 / self.height_scale
        height_257_scaled = height_257 / self.height_scale
        height_65_scaled = height_65 / self.height_scale
        residual_target_257 = (height_257 - base_257) / self.residual_scale
        coarse_target_17 = (height_17 - base_17) / self.residual_scale
        mid_residual_target_65 = (height_65 - base_65) / self.residual_scale
        detail_target_257 = (height_257 - F.interpolate(height_65.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=True).squeeze(0)) / self.residual_scale

        inputs = torch.cat([minimap_tensor, base_257_scaled, hole_mask_257], dim=0)

        return {
            "inputs": inputs,
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
    def __init__(self, in_channels: int = 5, hidden_channels: int = DEFAULT_HIDDEN_CHANNELS, blocks_per_stage: int = DEFAULT_BLOCKS_PER_STAGE):
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


def compute_v9_loss(
    coarse_delta_17: torch.Tensor,
    mid_delta_65: torch.Tensor,
    detail_delta_257: torch.Tensor,
    batch: dict[str, torch.Tensor],
    residual_scale: float,
    height_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
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

    full_l1 = F.l1_loss(full_height_257, batch["target_height_257"])
    mid_l1 = F.l1_loss(mid_height_65, batch["target_height_65"])
    coarse_l1 = F.l1_loss(coarse_height_17, batch["target_height_17"])

    pred_dx = full_height_257[:, :, :, 1:] - full_height_257[:, :, :, :-1]
    pred_dy = full_height_257[:, :, 1:, :] - full_height_257[:, :, :-1, :]
    target_dx = batch["target_height_257"][:, :, :, 1:] - batch["target_height_257"][:, :, :, :-1]
    target_dy = batch["target_height_257"][:, :, 1:, :] - batch["target_height_257"][:, :, :-1, :]
    gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    mid_residual_loss = F.l1_loss(mid_delta_65, batch["target_mid_residual_65"])
    detail_residual_loss = F.l1_loss(detail_delta_257, batch["target_detail_residual_257"])
    total = (
        full_l1
        + (0.7 * mid_l1)
        + (0.45 * coarse_l1)
        + (0.25 * gradient_loss)
        + (0.20 * mid_residual_loss)
        + (0.20 * detail_residual_loss)
    )
    return total, {
        "full_l1": float(full_l1.item()),
        "mid_l1": float(mid_l1.item()),
        "coarse_l1": float(coarse_l1.item()),
        "gradient": float(gradient_loss.item()),
        "mid_residual": float(mid_residual_loss.item()),
        "detail_residual": float(detail_residual_loss.item()),
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device, channels_last: bool) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
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
) -> tuple[float, dict[str, float], float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    component_sums = {"full_l1": 0.0, "mid_l1": 0.0, "coarse_l1": 0.0, "gradient": 0.0, "mid_residual": 0.0, "detail_residual": 0.0}
    sample_count = 0
    start = time.perf_counter()

    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    for batch in loader:
        batch = move_batch_to_device(batch, device, channels_last)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else nullcontext()):
            coarse_delta_17, mid_delta_65, detail_delta_257 = model(batch["inputs"])
            loss, components = compute_v9_loss(
                coarse_delta_17=coarse_delta_17,
                mid_delta_65=mid_delta_65,
                detail_delta_257=detail_delta_257,
                batch=batch,
                residual_scale=residual_scale,
                height_scale=height_scale,
            )

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = int(batch["inputs"].shape[0])
        sample_count += batch_size
        total_loss += float(loss.item()) * batch_size
        for key, value in components.items():
            component_sums[key] += float(value) * batch_size

    elapsed = max(time.perf_counter() - start, 1e-6)
    mean_loss = total_loss / max(sample_count, 1)
    mean_components = {key: value / max(sample_count, 1) for key, value in component_sums.items()}
    samples_per_second = sample_count / elapsed
    return mean_loss, mean_components, samples_per_second


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def _resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8)


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
) -> None:
    if not entries or preview_count <= 0:
        return

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    dataset = V9NativeDataset(entries[:preview_count], height_scale=height_scale, residual_scale=residual_scale)
    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in range(min(preview_count, len(dataset))):
            batch = dataset[index]
            device_batch = move_batch_to_device({key: value.unsqueeze(0) for key, value in batch.items()}, device, channels_last)
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

            minimap = (batch["inputs"][:3].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            target_257 = batch["target_height_257"].squeeze(0).cpu().numpy()
            target_65 = batch["target_height_65"].squeeze(0).cpu().numpy()
            target_17 = batch["target_height_17"].squeeze(0).cpu().numpy()
            pred_257 = full_height_257.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_65 = mid_height_65.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_17 = coarse_height_17.squeeze(0).squeeze(0).detach().cpu().numpy()

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
            preview = np.concatenate([row_one, row_two], axis=0)
            tile_name = entries[index].tile_name
            Image.fromarray(preview).save(preview_dir / f"{index:02d}_{tile_name}.png")

    model.train(model_was_training)


def train_single_run(
    selected_entries: Sequence[V9SampleEntry],
    args: argparse.Namespace,
    run_output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    if len(selected_entries) < 2:
        raise SystemExit("Need at least 2 accepted samples to train and validate.")

    train_indices, val_indices = split_grouped_indices(selected_entries, args.val_fraction, args.seed, args.group_block_size)
    train_entries = [selected_entries[index] for index in train_indices]
    val_entries = [selected_entries[index] for index in val_indices]

    train_dataset = V9NativeDataset(train_entries, height_scale=args.height_scale, residual_scale=args.residual_scale)
    val_dataset = V9NativeDataset(val_entries, height_scale=args.height_scale, residual_scale=args.residual_scale)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.train_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.val_workers, pin_memory=device.type == "cuda")

    model = V9TerrainModel(hidden_channels=args.hidden_channels, blocks_per_stage=args.blocks_per_stage).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    best_state: Optional[dict[str, Any]] = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_components, train_sps = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_dtype=amp_dtype,
            residual_scale=args.residual_scale,
            height_scale=args.height_scale,
            channels_last=args.channels_last,
        )
        val_loss, val_components, val_sps = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            amp_dtype=amp_dtype,
            residual_scale=args.residual_scale,
            height_scale=args.height_scale,
            channels_last=args.channels_last,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_components": train_components,
            "val_components": val_components,
            "train_samples_per_second": train_sps,
            "val_samples_per_second": val_sps,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"train_sps {train_sps:.2f} | val_sps {val_sps:.2f} | "
            f"mid {val_components['mid_l1']:.6f} | coarse {val_components['coarse_l1']:.6f} | grad {val_components['gradient']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }
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
            )

    run_summary = {
        "schema_version": "v9-train-run.v1",
        "created_at_utc": utc_now_iso(),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "selected_samples": len(selected_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "history": history,
        "best_val_loss": best_val_loss,
        "config": vars(args),
        "preview_count": args.preview_count,
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
    parser.add_argument("--preview-count", type=int, default=DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", dest="use_compile", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--write-curated-manifest", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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

    curated_limit = args.limit if args.limit is not None else args.target_curated_samples
    accepted_entries = select_curated_entries(audited, curated_limit)
    ranked_sane_entries = select_curated_entries(audited, None)

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

    train_single_run(accepted_entries, args, output_dir, device, amp_dtype)


if __name__ == "__main__":
    main()