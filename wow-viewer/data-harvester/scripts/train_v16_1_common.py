"""Shared V16.1 training entrypoint used by task-specific wrappers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402

from harvester.v16_1_dataset import V161Dataset  # noqa: E402
from harvester.v16_1_models import (  # noqa: E402
    V161HeightModel,
    V161HolesModel,
    V161LiquidModel,
    V161NormalModel,
    V161TexcompModel,
    recompose_from_mcly_alpha,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_MODELS_ROOT = _PROJECT_ROOT / "models" / "v16_1"
_PANEL_SIZE = 256
_PANEL_LABEL_HEIGHT = 18
_ROW_LABEL_HEIGHT = 18
_DIFFICULTY_BUCKETS = ("easy", "medium", "hard", "pathological")
_BUCKET_SAMPLING_PROFILES: dict[str, dict[str, float]] = {
    "uniform": {bucket: 1.0 for bucket in _DIFFICULTY_BUCKETS},
    "v16_1_1_normal": {
        "easy": 1.0,
        "medium": 1.75,
        "hard": 3.5,
        "pathological": 1.25,
    },
}


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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _normalize_bucket_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in _DIFFICULTY_BUCKETS:
        return label
    return "unbucketed"


def _bucket_sampling_weights(profile: str | None) -> dict[str, float] | None:
    if profile is None:
        return None
    return _BUCKET_SAMPLING_PROFILES.get(str(profile))


class _DeterministicEpochSampler(Sampler[int]):
    """Deterministic sampler with optional per-epoch subset rotation."""

    def __init__(
        self,
        n: int,
        seed: int,
        order_log_path: Path | None = None,
        bucket_log_path: Path | None = None,
        epoch_size: int | None = None,
        build_labels: list[str] | None = None,
        build_balanced: bool = False,
        bucket_labels: list[str] | None = None,
        bucket_sampling_profile: str | None = None,
        sample_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._n = int(n)
        self._seed = int(seed)
        self._epoch = 0
        self._order_log_path = order_log_path
        self._bucket_log_path = bucket_log_path
        self._epoch_size = None if epoch_size is None or int(epoch_size) <= 0 else min(int(epoch_size), self._n)
        self._build_labels = list(build_labels) if build_labels is not None else None
        self._build_balanced = bool(build_balanced)
        self._bucket_labels = [_normalize_bucket_label(label) for label in bucket_labels] if bucket_labels is not None else None
        self._bucket_sampling_profile = bucket_sampling_profile
        self._bucket_sampling_weights = _bucket_sampling_weights(bucket_sampling_profile)
        self._sample_rows = [dict(row) for row in sample_rows] if sample_rows is not None else None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _sample_subset(self, rng: np.random.RandomState) -> list[int]:
        if self._n <= 0:
            return []
        if self._epoch_size is None or self._epoch_size >= self._n:
            return list(range(self._n))
        return _sample_positions(
            self._n,
            seed=self._seed + self._epoch,
            take=self._epoch_size,
            build_labels=self._build_labels,
            build_balanced=self._build_balanced,
            bucket_labels=self._bucket_labels,
            bucket_sampling_weights=self._bucket_sampling_weights,
        )

    def __iter__(self):
        rng = np.random.RandomState(self._seed + self._epoch)
        selected = self._sample_subset(rng)
        order = list(selected)
        rng.shuffle(order)
        available_bucket_counts = _bucket_counts(self._bucket_labels or [])
        selected_bucket_labels = [self._bucket_labels[pos] for pos in selected] if self._bucket_labels is not None else []
        ordered_bucket_labels = [self._bucket_labels[pos] for pos in order] if self._bucket_labels is not None else []
        selected_rows = [self._sample_rows[pos] for pos in selected] if self._sample_rows is not None else None
        order_rows = [self._sample_rows[pos] for pos in order] if self._sample_rows is not None else None
        if self._order_log_path is not None:
            payload = {
                "epoch": self._epoch,
                "num_samples": self._n,
                "epoch_size": len(order),
                "bucket_sampling_profile": self._bucket_sampling_profile,
                "bucket_sampling_weights": self._bucket_sampling_weights,
                "available_bucket_counts": available_bucket_counts,
                "selected_bucket_counts": _bucket_counts(selected_bucket_labels),
                "ordered_bucket_counts": _bucket_counts(ordered_bucket_labels),
                "selected_positions_sha256": hashlib.sha256(np.asarray(selected, dtype=np.int32).tobytes()).hexdigest(),
                "selected_positions": selected,
                "selected_rows": selected_rows,
                "order_sha256": hashlib.sha256(np.asarray(order, dtype=np.int32).tobytes()).hexdigest(),
                "order": order,
                "order_rows": order_rows,
            }
            with self._order_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        if self._bucket_log_path is not None:
            payload = {
                "epoch": self._epoch,
                "num_samples": self._n,
                "epoch_size": len(order),
                "bucket_sampling_profile": self._bucket_sampling_profile,
                "bucket_sampling_weights": self._bucket_sampling_weights,
                "available_bucket_counts": available_bucket_counts,
                "selected_bucket_counts": _bucket_counts(selected_bucket_labels),
                "selected_build_counts": _count_string_values([row.get("build", "unknown") for row in selected_rows] if selected_rows is not None else []),
            }
            with self._bucket_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        return iter(order)

    def __len__(self) -> int:
        return self._epoch_size if self._epoch_size is not None else self._n


def _sample_positions(
    n: int,
    seed: int,
    take: int,
    build_labels: list[str] | None = None,
    build_balanced: bool = True,
    bucket_labels: list[str] | None = None,
    bucket_sampling_weights: dict[str, float] | None = None,
) -> list[int]:
    if n <= 0 or take <= 0:
        return []
    if take >= n:
        return list(range(n))

    rng = np.random.RandomState(int(seed))
    if not build_balanced or not build_labels or len(build_labels) != n:
        return _sample_weighted_positions(
            list(range(n)),
            take=take,
            rng=rng,
            bucket_labels=bucket_labels,
            bucket_sampling_weights=bucket_sampling_weights,
        )

    by_build: dict[str, list[int]] = {}
    for pos, build in enumerate(build_labels):
        by_build.setdefault(str(build), []).append(pos)

    build_order = sorted(by_build.keys())
    rng.shuffle(build_order)
    out: list[int] = []
    while len(out) < take:
        progressed = False
        for build in build_order:
            items = by_build[build]
            if not items:
                continue
            chosen = _sample_weighted_positions(
                items,
                take=1,
                rng=rng,
                bucket_labels=bucket_labels,
                bucket_sampling_weights=bucket_sampling_weights,
            )
            if not chosen:
                continue
            selected = chosen[0]
            items.remove(selected)
            out.append(selected)
            progressed = True
            if len(out) >= take:
                break
        if not progressed:
            break

    if len(out) < take:
        remaining: list[int] = []
        for build in build_order:
            remaining.extend(by_build[build])
        if remaining:
            extra = _sample_weighted_positions(
                remaining,
                take=take - len(out),
                rng=rng,
                bucket_labels=bucket_labels,
                bucket_sampling_weights=bucket_sampling_weights,
            )
            out.extend(extra)

    rng.shuffle(out)
    return out[:take]


def _count_string_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _bucket_counts(bucket_labels: list[str]) -> dict[str, int]:
    counts = _count_string_values([_normalize_bucket_label(label) for label in bucket_labels if str(label or "").strip()])
    return {bucket: counts.get(bucket, 0) for bucket in _DIFFICULTY_BUCKETS if counts.get(bucket, 0) > 0}


def _sample_weighted_positions(
    positions: list[int],
    *,
    take: int,
    rng: np.random.RandomState,
    bucket_labels: list[str] | None = None,
    bucket_sampling_weights: dict[str, float] | None = None,
) -> list[int]:
    if take <= 0 or not positions:
        return []
    if take >= len(positions):
        return list(positions)

    weights = None
    if bucket_labels is not None and bucket_sampling_weights:
        raw = np.asarray(
            [
                max(float(bucket_sampling_weights.get(_normalize_bucket_label(bucket_labels[pos]), 1.0)), 0.0)
                for pos in positions
            ],
            dtype=np.float64,
        )
        if float(raw.sum()) > 0.0 and not np.allclose(raw, raw[0]):
            weights = raw / raw.sum()
    chosen_idx = rng.choice(len(positions), size=take, replace=False, p=weights)
    return [positions[int(idx)] for idx in chosen_idx.tolist()]


def _pool_row(entry: dict[str, Any], subset_pos: int, split_pos: int) -> dict[str, Any]:
    return {
        "subset_pos": int(subset_pos),
        "split_pos": int(split_pos),
        "build": str(entry.get("_build", "unknown")),
        "map": entry.get("map"),
        "tile_id": int(entry.get("tile_id", -1)),
        "tile_x": int(entry.get("tile_x", -1) if entry.get("tile_x") is not None else -1),
        "tile_y": int(entry.get("tile_y", -1) if entry.get("tile_y") is not None else -1),
        "height_mean": float(entry.get("height_mean", 0.0) or 0.0),
        "height_std": float(entry.get("height_std", 0.0) or 0.0),
        "has_normal_xyz": bool(entry.get("has_normal_xyz", False)),
        "has_alpha_256": bool(entry.get("has_alpha_256", False)),
        "has_liquid_mask": bool(entry.get("has_liquid_mask", False)),
        "has_mcly_texture_ids": bool(entry.get("has_mcly_texture_ids", False)),
        "n_mddf": int(entry.get("n_mddf", 0) or 0),
        "n_modf": int(entry.get("n_modf", 0) or 0),
        "difficulty_bucket": str(entry.get("_curation_difficulty_bucket", "")),
        "difficulty_rank": int(entry.get("_curation_difficulty_rank", -1)),
        "quality_score": float(entry.get("_curation_quality_score", 0.0) or 0.0),
        "usefulness_score": float(entry.get("_curation_usefulness_score", 0.0) or 0.0),
        "difficulty_score": float(entry.get("_curation_difficulty_score", 0.0) or 0.0),
    }


def _apply_dataset_pool(
    ds: V161Dataset,
    split: str,
    max_tiles: int,
    seed: int,
    evidence_dir: Path,
    build_balanced: bool = True,
) -> dict[str, Any]:
    available = len(ds._indices)
    build_labels = [
        str(ds._index_entries[global_idx].get("_build", "unknown"))
        for global_idx in ds._indices
    ]
    bucket_labels = [
        _normalize_bucket_label(ds._index_entries[global_idx].get("_curation_difficulty_bucket", ""))
        for global_idx in ds._indices
    ]
    selected_positions = _sample_positions(
        available,
        seed=seed,
        take=max_tiles if max_tiles > 0 else available,
        build_labels=build_labels,
        build_balanced=build_balanced,
    )
    selected_global_indices = [ds._indices[pos] for pos in selected_positions]
    ds._indices = selected_global_indices

    rows: list[dict[str, Any]] = []
    build_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for subset_pos, split_pos in enumerate(selected_positions):
        entry = ds._index_entries[selected_global_indices[subset_pos]]
        row = _pool_row(entry, subset_pos=subset_pos, split_pos=split_pos)
        rows.append(row)
        build = str(row["build"])
        build_counts[build] = build_counts.get(build, 0) + 1
        bucket = _normalize_bucket_label(row.get("difficulty_bucket", ""))
        if bucket != "unbucketed":
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    selection_jsonl = evidence_dir / f"{split}_pool_selection.jsonl"
    with selection_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "split": split,
        "available_tiles": int(available),
        "selected_tiles": int(len(ds._indices)),
        "requested_max_tiles": int(max_tiles),
        "build_balanced": bool(build_balanced),
        "seed": int(seed),
        "selected_positions_sha256": hashlib.sha256(
            np.asarray(selected_positions, dtype=np.int32).tobytes()
        ).hexdigest(),
        "build_tile_counts": build_counts,
        "available_bucket_counts": _bucket_counts(bucket_labels),
        "selected_bucket_counts": dict(sorted(bucket_counts.items())),
        "selection_jsonl": str(selection_jsonl),
    }
    (evidence_dir / f"{split}_pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _apply_dataset_limit(ds: V161Dataset, max_samples: int) -> None:
    if max_samples > 0:
        ds._indices = ds._indices[: min(int(max_samples), len(ds._indices))]


def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_map * mask).sum() / mask.sum().clamp_min(1e-8)


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target).abs(), weight)


def _weighted_l2(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target) ** 2, weight)


def _resize_weight(weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(weight.shape[-2:]) == tuple(size):
        return weight
    return F.interpolate(weight, size=size, mode="bilinear", align_corners=False)


def _normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    return ((normals.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0).astype(np.uint8)


def _draw_text_strip(img: Image.Image, text: str, height: int) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), color=(0, 0, 0))
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (canvas.width, height - 1)], fill=(14, 14, 14))
    draw.text((4, 3), str(text), fill=(240, 240, 240))
    return canvas


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
    canvas = _compose_horizontal_panel(panels)
    canvas.save(out_path)


def _save_preview_grid(rows: list[list[tuple[str, torch.Tensor]]], out_path: Path, row_titles: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError("Cannot save preview grid with no rows.")
    row_images = []
    for idx, row in enumerate(rows):
        row_img = _compose_horizontal_panel(row)
        if row_titles is not None:
            row_img = _draw_text_strip(row_img, row_titles[idx], _ROW_LABEL_HEIGHT)
        row_images.append(row_img)
    width = max(img.width for img in row_images)
    height = sum(img.height for img in row_images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for img in row_images:
        canvas.paste(img, (0, y))
        y += img.height
    canvas.save(out_path)


def _meta_value(x: Any, idx: int) -> Any:
    if isinstance(x, torch.Tensor):
        return x[idx].item()
    if isinstance(x, np.ndarray):
        return x[idx]
    if isinstance(x, (list, tuple)):
        return x[idx]
    return x


def _preview_row_title(batch: dict[str, Any], idx: int) -> str:
    build = _meta_value(batch.get("meta_build", "unknown"), idx)
    map_name = _meta_value(batch.get("meta_map", ""), idx)
    tile_id = _meta_value(batch.get("meta_tile_id", -1), idx)
    tile_x = _meta_value(batch.get("meta_tile_x", -1), idx)
    tile_y = _meta_value(batch.get("meta_tile_y", -1), idx)
    return f"{build} | {map_name} | tile={tile_id} | ({tile_x},{tile_y})"


def _coarse_type_to_rgb(type_grid: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.45, 0.9],
            [0.1, 0.7, 0.95],
            [0.9, 0.35, 0.1],
            [0.25, 0.85, 0.2],
        ],
        dtype=torch.float32,
        device=type_grid.device,
    )
    rgb = palette[type_grid.long().clamp(0, 4)]
    return rgb.permute(2, 0, 1)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    model_factory: Callable[[], torch.nn.Module]
    loss_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device, argparse.Namespace], tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]]
    save_preview: Callable[[dict[str, Any], dict[str, torch.Tensor], Path], None]


def _height_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["height_norm"].to(device, non_blocking=True)
    weight = batch["weight_257"].to(device, non_blocking=True)
    pred = model(inp)
    loss = _weighted_l1(pred, target, weight)
    return loss, {"height": float(loss.item())}, {"pred": pred, "target": target, "weight": weight}


def _gradient_magnitude_257(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt((dx * dx) + (dy * dy) + 1e-8)


def _hard_region_weight_from_targets(
    height_raw: torch.Tensor,
    target_normals: torch.Tensor,
    alpha_painted_256: torch.Tensor,
    mcly_any_16: torch.Tensor,
    terrain_valid_mask: torch.Tensor,
    base_mask: torch.Tensor,
    detail_boost: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    height_grad = _gradient_magnitude_257(height_raw)
    normal_grad = _gradient_magnitude_257(target_normals)
    normal_grad = normal_grad.mean(dim=1, keepdim=True)
    alpha_painted_257 = _resize_weight(alpha_painted_256, target_normals.shape[-2:])
    alpha_grad = _gradient_magnitude_257(alpha_painted_257)
    mcly_any_257 = _resize_weight(mcly_any_16, target_normals.shape[-2:])
    mcly_grad = _gradient_magnitude_257(mcly_any_257)

    valid_mean_height = _masked_mean(height_grad, base_mask)
    valid_mean_normal = _masked_mean(normal_grad, base_mask)
    valid_mean_alpha = _masked_mean(alpha_grad, base_mask)
    valid_mean_mcly = _masked_mean(mcly_grad, base_mask)
    height_grad_n = (height_grad / valid_mean_height.clamp_min(1e-6)).clamp(0.0, 4.0)
    normal_grad_n = (normal_grad / valid_mean_normal.clamp_min(1e-6)).clamp(0.0, 4.0)
    alpha_grad_n = (alpha_grad / valid_mean_alpha.clamp_min(1e-6)).clamp(0.0, 4.0)
    mcly_grad_n = (mcly_grad / valid_mean_mcly.clamp_min(1e-6)).clamp(0.0, 4.0)

    transition_signal = torch.maximum(alpha_grad_n, mcly_grad_n)
    hard_region_signal = ((0.50 * height_grad_n) + (0.25 * normal_grad_n) + (0.25 * transition_signal)).clamp(0.0, 4.0)
    hard_region_signal = hard_region_signal * terrain_valid_mask
    hard_region_weight = 1.0 + (float(detail_boost) * hard_region_signal)
    return hard_region_weight, {
        "hard_region_signal": hard_region_signal,
        "height_grad_signal": height_grad_n,
        "normal_grad_signal": normal_grad_n,
        "alpha_grad_signal": alpha_grad_n,
        "mcly_grad_signal": mcly_grad_n,
        "transition_signal": transition_signal,
    }


def _normal_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["normals"].to(device, non_blocking=True)
    height_raw = batch["height_raw"].to(device, non_blocking=True)
    normal_mask = batch["normal_mask"].to(device, non_blocking=True)
    terrain_valid_mask = batch["terrain_valid_mask_257"].to(device, non_blocking=True)
    object_weight = batch["weight_257"].to(device, non_blocking=True)
    mddf_mask = batch["mddf_mask"].to(device, non_blocking=True)
    modf_mask = batch["modf_mask"].to(device, non_blocking=True)
    liquid_mask = batch["liquid_mask"].to(device, non_blocking=True)
    what_plate_flag = batch["what_plate_flag"].to(device, non_blocking=True).view(-1, 1, 1, 1)
    alpha_painted_cov = batch["alpha_painted_cov"].to(device, non_blocking=True)
    mcly_cov = batch["mcly_cov"].to(device, non_blocking=True)
    alpha_painted_256 = batch["alpha_painted_256"].to(device, non_blocking=True)
    mcly_any_16 = batch["mcly_any_16"].to(device, non_blocking=True)
    pred = model(inp)
    pred_n = F.normalize(pred, dim=1, eps=1e-6)
    target_n = F.normalize(target, dim=1, eps=1e-6)
    cosine = 1.0 - (pred_n * target_n).sum(dim=1, keepdim=True)
    liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
    object_presence = torch.maximum(mddf_mask, modf_mask)
    liquid_weight = 1.0 - (0.85 * liquid_mask_257)
    instance_weight = 1.0 - (0.75 * object_presence)
    base_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight
    base_mask = base_mask * (1.0 - what_plate_flag)
    hard_region_weight, hard_region_debug = _hard_region_weight_from_targets(
        height_raw=height_raw,
        target_normals=target_n,
        alpha_painted_256=alpha_painted_256,
        mcly_any_16=mcly_any_16,
        terrain_valid_mask=terrain_valid_mask,
        base_mask=base_mask,
        detail_boost=float(args.normal_detail_boost),
    )
    train_mask = base_mask * hard_region_weight
    vec_l1 = (pred_n - target_n).abs().mean(dim=1, keepdim=True)
    nz_l2 = (pred_n[:, 2:3] - target_n[:, 2:3]) ** 2
    loss_cos = _masked_mean(cosine, train_mask)
    loss_vec = _masked_mean(vec_l1, train_mask)
    loss_nz = _masked_mean(nz_l2, train_mask)
    loss = loss_cos + (0.35 * loss_vec) + (0.15 * loss_nz)
    return loss, {
        "normal": float(loss.item()),
        "normal_cos": float(loss_cos.item()),
        "normal_vec": float(loss_vec.item()),
        "normal_nz": float(loss_nz.item()),
        "normal_mask_cov": float(base_mask.mean().item()),
        "normal_detail_mean": float(_masked_mean(hard_region_weight, base_mask).item()),
        "normal_hard_region_mean": float(_masked_mean(hard_region_debug["hard_region_signal"], base_mask).item()),
        "normal_transition_mean": float(_masked_mean(hard_region_debug["transition_signal"], base_mask).item()),
        "what_plate_rate": float(what_plate_flag.mean().item()),
        "alpha_painted_cov": float(alpha_painted_cov.mean().item()),
        "mcly_cov": float(mcly_cov.mean().item()),
    }, {
        "pred": pred_n,
        "target": target_n,
        "train_mask": train_mask,
        "base_mask": base_mask,
        "detail_weight": hard_region_weight,
        "hard_region_signal": hard_region_debug["hard_region_signal"],
        "transition_signal": hard_region_debug["transition_signal"],
        "terrain_valid_mask": terrain_valid_mask,
        "object_weight": object_weight,
        "liquid_mask": liquid_mask_257,
        "instance_weight": instance_weight,
    }


def _holes_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["holes"].to(device, non_blocking=True)
    weight = batch["weight_16"].to(device, non_blocking=True)
    pred = model(inp)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    loss = _masked_mean(bce, weight)
    return loss, {"holes": float(loss.item())}, {"pred": pred, "target": target}


def _liquid_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target_mask = batch["liquid_mask"].to(device, non_blocking=True)
    target_type = batch["liquid_type_16"].to(device, non_blocking=True)
    type_valid = batch["liquid_type_valid_16"].to(device, non_blocking=True)
    weight_256 = batch["weight_256"].to(device, non_blocking=True)
    weight_16 = batch["weight_16"].to(device, non_blocking=True)
    pred_mask, pred_type = model(inp)
    mask_loss = _weighted_l1(pred_mask, target_mask, weight_256)
    type_ce = F.cross_entropy(pred_type, target_type, reduction="none").unsqueeze(1)
    type_loss = _masked_mean(type_ce, type_valid * weight_16)
    loss = mask_loss + (0.5 * type_loss)
    type_pred = pred_type.argmax(dim=1)
    return loss, {"liquid_mask": float(mask_loss.item()), "liquid_type": float(type_loss.item())}, {"pred_mask": pred_mask, "target_mask": target_mask, "pred_type": type_pred, "target_type": target_type}


def _texcomp_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    alpha_target = batch["alpha"].to(device, non_blocking=True)
    mcly_ids = batch["mcly_ids"].to(device, non_blocking=True)
    mcly_mask = batch["mcly_mask"].to(device, non_blocking=True)
    weight_256 = batch["weight_256"].to(device, non_blocking=True)
    weight_16 = batch["weight_16"].to(device, non_blocking=True)
    pred_alpha, pred_mask, pred_ids = model(inp)
    alpha_loss = _weighted_l1(pred_alpha, alpha_target, weight_256)
    mask_bce = F.binary_cross_entropy(pred_mask, mcly_mask.permute(0, 3, 1, 2), reduction="none")
    mask_loss = _masked_mean(mask_bce, weight_16.repeat(1, 4, 1, 1))
    pred_ids_r = pred_ids.permute(0, 1, 3, 4, 2)
    id_ce = F.cross_entropy(pred_ids_r.reshape(-1, 16), mcly_ids.reshape(-1), reduction="none")
    id_ce = id_ce.view_as(mcly_ids).permute(0, 3, 1, 2)
    id_loss = _masked_mean(id_ce, mcly_mask.permute(0, 3, 1, 2) * weight_16.repeat(1, 4, 1, 1))
    recomposed = recompose_from_mcly_alpha(pred_alpha, pred_ids, pred_mask)
    recompose_loss = _weighted_l1(recomposed, inp, weight_256)
    loss = alpha_loss + (0.35 * mask_loss) + (0.25 * id_loss) + (0.5 * recompose_loss)
    pred_ids_cls = pred_ids.argmax(dim=2)
    return loss, {
        "alpha": float(alpha_loss.item()),
        "mcly_mask": float(mask_loss.item()),
        "mcly_id": float(id_loss.item()),
        "recompose": float(recompose_loss.item()),
    }, {
        "pred_alpha": pred_alpha,
        "target_alpha": alpha_target,
        "pred_mask": pred_mask,
        "target_mask": mcly_mask.permute(0, 3, 1, 2),
        "pred_ids": pred_ids_cls,
        "target_ids": mcly_ids,
        "recomposed": recomposed,
    }


def _preview_height(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("height_gt", batch["height_norm"][idx]),
                ("height_pred", outputs["pred"][idx]),
                ("weight", batch["weight_257"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_normal(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("normal_gt", _normals_to_rgb(outputs["target"][idx])),
                ("normal_pred", _normals_to_rgb(outputs["pred"][idx])),
                ("terrain_valid", outputs["terrain_valid_mask"][idx]),
                ("base_mask", outputs["base_mask"][idx]),
                ("hard_region", outputs["hard_region_signal"][idx] / outputs["hard_region_signal"][idx].max().clamp_min(1e-6)),
                ("transition", outputs["transition_signal"][idx] / outputs["transition_signal"][idx].max().clamp_min(1e-6)),
                ("detail_weight", outputs["detail_weight"][idx] / outputs["detail_weight"][idx].max().clamp_min(1e-6)),
                ("train_mask", outputs["train_mask"][idx] / outputs["train_mask"][idx].max().clamp_min(1e-6)),
                ("liquid_mask", outputs["liquid_mask"][idx]),
                ("object_weight", outputs["object_weight"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_holes(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("holes_gt", outputs["target"][idx]),
                ("holes_pred", outputs["pred"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_liquid(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        pred_type_rgb = _coarse_type_to_rgb(outputs["pred_type"][idx])
        target_type_rgb = _coarse_type_to_rgb(outputs["target_type"][idx])
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("liq_gt", outputs["target_mask"][idx]),
                ("liq_pred", outputs["pred_mask"][idx]),
                ("type_gt", target_type_rgb),
                ("type_pred", pred_type_rgb),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_texcomp(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        gt_alpha_painted = batch["alpha"][idx, 1:].max(dim=0).values.unsqueeze(0)
        pred_alpha_painted = outputs["pred_alpha"][idx, 1:].max(dim=0).values.unsqueeze(0)
        gt_mask = outputs["target_mask"][idx].max(dim=0).values.unsqueeze(0)
        pred_mask = outputs["pred_mask"][idx].max(dim=0).values.unsqueeze(0)
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("alpha_gt", gt_alpha_painted),
                ("alpha_pred", pred_alpha_painted),
                ("mask_gt", gt_mask),
                ("mask_pred", pred_mask),
                ("recomposed", outputs["recomposed"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


TASKS: dict[str, TaskSpec] = {
    "height": TaskSpec("height", V161HeightModel, _height_loss, _preview_height),
    "normal": TaskSpec("normal", V161NormalModel, _normal_loss, _preview_normal),
    "holes": TaskSpec("holes", V161HolesModel, _holes_loss, _preview_holes),
    "liquid": TaskSpec("liquid", V161LiquidModel, _liquid_loss, _preview_liquid),
    "texcomp": TaskSpec("texcomp", V161TexcompModel, _texcomp_loss, _preview_texcomp),
}


def _parse_args(task_name: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train V16.1 {task_name} model")
    p.add_argument("--dataset-dir", type=Path, default=_DATASET_ROOT)
    p.add_argument(
        "--curation-manifest",
        type=Path,
        default=None,
        help="Optional curation manifest directory/file produced by build_v16_curation_manifest.py",
    )
    p.add_argument("--builds", nargs="+", default=["3_3_5_12340"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader worker count. Use -1 to auto-resolve a CUDA-friendly default.",
    )
    p.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader worker processes alive between epochs (defaults to true when workers > 0)",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched per worker (effective when --num-workers > 0)",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--train-max-tiles",
        type=int,
        default=0,
        help="If >0, curate this many tiles into the persistent train pool before epoch rotation.",
    )
    p.add_argument(
        "--train-epoch-tiles",
        type=int,
        default=0,
        help="If >0, sample this many train-pool tiles per epoch.",
    )
    p.add_argument(
        "--train-epoch-build-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance per-epoch train sampling across builds when possible.",
    )
    p.add_argument(
        "--bucket-sampling-profile",
        choices=sorted(_BUCKET_SAMPLING_PROFILES.keys()),
        default=("v16_1_1_normal" if task_name == "normal" else "uniform"),
        help="Per-epoch difficulty-bucket sampling profile used when curated manifests carry bucket metadata.",
    )
    p.add_argument(
        "--val-max-tiles",
        type=int,
        default=0,
        help="If >0, curate this many tiles into the fixed validation pool.",
    )
    p.add_argument(
        "--curation-seed",
        type=int,
        default=None,
        help="Seed for train/val pool selection (defaults to --seed).",
    )
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument(
        "--val-preview-interval",
        type=int,
        default=1,
        help="If >0, write a validation preview only when a new best checkpoint is found. 0 disables preview writes.",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--normal-detail-boost",
        type=float,
        default=1.0,
        help="Extra weight placed on high-deformation normal targets relative to broad flat terrain.",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (useful for CPU-only or limited toolchains)",
    )
    p.add_argument("--target-vram-gb", type=float, default=0.0)
    return p.parse_args()


def run_task(task_name: str) -> None:
    if task_name not in TASKS:
        raise RuntimeError(f"Unknown V16.1 task: {task_name}")
    args = _parse_args(task_name)
    task = TASKS[task_name]
    if args.grad_accum_steps < 1:
        raise RuntimeError("--grad-accum-steps must be >= 1")
    if args.prefetch_factor < 1:
        raise RuntimeError("--prefetch-factor must be >= 1")
    if args.train_epoch_tiles < 0:
        raise RuntimeError("--train-epoch-tiles must be >= 0")
    _seed_all(args.seed)
    device = _resolve_device(args.device)
    resolved_num_workers = _resolve_num_workers(int(args.num_workers), device)
    resolved_persistent_workers = _resolve_persistent_workers(args.persistent_workers, resolved_num_workers)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _MODELS_ROOT / task_name / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    val_dir = run_dir / "validation"
    evidence_dir = run_dir / "evidence"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    train_ds = V161Dataset(
        args.dataset_dir,
        builds=args.builds,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=not args.no_augment,
        curation_manifest=args.curation_manifest,
    )
    val_ds = V161Dataset(
        args.dataset_dir,
        builds=args.builds,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=False,
        curation_manifest=args.curation_manifest,
    )
    curation_seed = int(args.curation_seed) if args.curation_seed is not None else int(args.seed)
    train_pool = _apply_dataset_pool(
        train_ds,
        split="train",
        max_tiles=int(args.train_max_tiles),
        seed=curation_seed + 101,
        evidence_dir=evidence_dir,
        build_balanced=True,
    )
    val_pool = _apply_dataset_pool(
        val_ds,
        split="val",
        max_tiles=int(args.val_max_tiles),
        seed=curation_seed + 202,
        evidence_dir=evidence_dir,
        build_balanced=True,
    )
    _apply_dataset_limit(train_ds, int(args.max_train_samples))
    _apply_dataset_limit(val_ds, int(args.max_val_samples))
    train_build_labels = [
        str(train_ds._index_entries[global_idx].get("_build", "unknown"))
        for global_idx in train_ds._indices
    ]
    train_bucket_labels = [
        _normalize_bucket_label(train_ds._index_entries[global_idx].get("_curation_difficulty_bucket", ""))
        for global_idx in train_ds._indices
    ]
    train_sample_rows = [
        _pool_row(train_ds._index_entries[global_idx], subset_pos=idx, split_pos=idx)
        for idx, global_idx in enumerate(train_ds._indices)
    ]
    train_order_log = evidence_dir / "train_epoch_orders.jsonl"
    train_bucket_log = evidence_dir / "train_epoch_bucket_usage.jsonl"
    train_sampler = _DeterministicEpochSampler(
        len(train_ds),
        seed=int(args.seed),
        order_log_path=train_order_log,
        bucket_log_path=train_bucket_log,
        epoch_size=int(args.train_epoch_tiles),
        build_labels=train_build_labels,
        build_balanced=bool(args.train_epoch_build_balanced),
        bucket_labels=train_bucket_labels,
        bucket_sampling_profile=args.bucket_sampling_profile,
        sample_rows=train_sample_rows,
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": resolved_num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(resolved_persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = task.model_factory().to(device)
    can_compile = hasattr(torch, "compile") and not args.no_compile and device.type == "cuda"
    if can_compile:
        try:
            model = torch.compile(model)
            compile_status = "enabled"
        except Exception as ex:
            compile_status = f"disabled (compile failed: {ex})"
    else:
        compile_status = "disabled"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    best_val = float("inf")
    best_epoch: int | None = None
    start_epoch = 1
    log_entries: list[dict[str, Any]] = []

    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        best_epoch = int(ckpt["best_epoch"]) if ckpt.get("best_epoch") is not None else None

    config = {
        "task": task_name,
        "run_name": run_name,
        "dataset_dir": str(Path(args.dataset_dir)),
        "curation_manifest": str(args.curation_manifest) if args.curation_manifest else None,
        "builds": list(args.builds),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "num_workers": args.num_workers,
        "resolved_num_workers": resolved_num_workers,
        "persistent_workers": args.persistent_workers,
        "resolved_persistent_workers": resolved_persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "resolved_curation_seed": curation_seed,
        "val_fraction": args.val_fraction,
        "train_max_tiles": args.train_max_tiles,
        "train_epoch_tiles": args.train_epoch_tiles,
        "train_epoch_build_balanced": args.train_epoch_build_balanced,
        "bucket_sampling_profile": args.bucket_sampling_profile,
        "bucket_sampling_weights": _bucket_sampling_weights(args.bucket_sampling_profile),
        "val_max_tiles": args.val_max_tiles,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "normal_detail_boost": args.normal_detail_boost,
        "train_pool": train_pool,
        "val_pool": val_pool,
        "no_compile": args.no_compile,
        "compile_status": compile_status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "best_val": best_val if np.isfinite(best_val) else None,
        "best_epoch": best_epoch,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Task: {task_name}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Dataset: train={len(train_ds)} val={len(val_ds)}", flush=True)
    if args.curation_manifest is not None:
        print(f"Curation manifest: {args.curation_manifest}", flush=True)
    print(
        "Curated pools: "
        f"train={train_pool['selected_tiles']}/{train_pool['available_tiles']} "
        f"val={val_pool['selected_tiles']}/{val_pool['available_tiles']}",
        flush=True,
    )
    print(f"Curated build mix (train): {train_pool.get('build_tile_counts', {})}", flush=True)
    print(f"Curated build mix (val): {val_pool.get('build_tile_counts', {})}", flush=True)
    if train_pool.get("selected_bucket_counts"):
        print(f"Curated difficulty mix (train): {train_pool.get('selected_bucket_counts', {})}", flush=True)
        print(f"Available difficulty mix (train): {train_pool.get('available_bucket_counts', {})}", flush=True)
    if val_pool.get("selected_bucket_counts"):
        print(f"Curated difficulty mix (val): {val_pool.get('selected_bucket_counts', {})}", flush=True)
    if args.train_epoch_tiles > 0:
        print(
            "Epoch sampling: "
            f"train_epoch_tiles={min(int(args.train_epoch_tiles), len(train_ds))}/{len(train_ds)} "
            f"build_balanced={bool(args.train_epoch_build_balanced)} "
            f"bucket_profile={args.bucket_sampling_profile}",
            flush=True,
        )
    print(
        f"DataLoader: workers={resolved_num_workers} "
        f"persistent_workers={resolved_persistent_workers} "
        f"prefetch_factor={(args.prefetch_factor if resolved_num_workers > 0 else 'n/a')}",
        flush=True,
    )
    print(
        f"Batching: micro={args.batch_size} accum={args.grad_accum_steps} "
        f"effective={args.batch_size * args.grad_accum_steps}",
        flush=True,
    )
    if task_name == "normal":
        print(f"Normal detail steering: boost={args.normal_detail_boost:.2f}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    print(f"torch.compile: {compile_status}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        metric_sums: dict[str, float] = {}
        train_loss_sum = 0.0
        optimizer_steps = 0
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                loss, metrics, _outputs = task.loss_fn(model, batch, device, args)
            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = (batch_idx % args.grad_accum_steps == 0) or (batch_idx == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            train_loss_sum += float(loss.item())
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        scheduler.step()

        n_train = max(len(train_loader), 1)
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss_sum / n_train,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.perf_counter() - t0,
            "optimizer_steps": optimizer_steps,
        }
        for key, value in metric_sums.items():
            entry[f"train_{key}"] = value / n_train

        print(
            f"Epoch {epoch:3d}/{args.epochs} | loss={entry['train_loss']:.4f} "
            f"lr={entry['lr']:.2e} opt_steps={optimizer_steps} {entry['elapsed_s']:.1f}s",
            flush=True,
        )

        if epoch % args.val_interval == 0 and len(val_loader) > 0:
            model.eval()
            val_loss_sum = 0.0
            val_metric_sums: dict[str, float] = {}
            preview_batch = None
            preview_outputs = None
            with torch.no_grad():
                for batch in val_loader:
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                        loss, metrics, outputs = task.loss_fn(model, batch, device, args)
                    val_loss_sum += float(loss.item())
                    for key, value in metrics.items():
                        val_metric_sums[key] = val_metric_sums.get(key, 0.0) + float(value)
                    if preview_batch is None:
                        preview_batch = batch
                        preview_outputs = outputs
            n_val = max(len(val_loader), 1)
            entry["val_loss"] = val_loss_sum / n_val
            for key, value in val_metric_sums.items():
                entry[f"val_{key}"] = value / n_val
            print(f"        val | loss={entry['val_loss']:.4f}", flush=True)
            is_new_best = False
            if entry["val_loss"] < best_val:
                best_val = float(entry["val_loss"])
                best_epoch = int(epoch)
                is_new_best = True
                entry["is_new_best"] = True
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "task": task_name,
                    },
                    ckpt_dir / f"v16_1_{task_name}_best.pt",
                )
                print(f"        *** new best val_loss={best_val:.4f}", flush=True)
            if (
                is_new_best
                and preview_batch is not None
                and preview_outputs is not None
                and args.val_preview_interval > 0
            ):
                task.save_preview(preview_batch, preview_outputs, val_dir / f"best_epoch_{epoch:04d}.png")

        log_entries.append(entry)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "task": task_name,
            },
            ckpt_dir / f"v16_1_{task_name}_last.pt",
        )
        (run_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2), encoding="utf-8")

    config["best_val"] = best_val if np.isfinite(best_val) else None
    config["best_epoch"] = best_epoch
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "task": task_name,
        },
        ckpt_dir / f"v16_1_{task_name}_final.pt",
    )
    print(f"Done. Run dir: {run_dir}", flush=True)
