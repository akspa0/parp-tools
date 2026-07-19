"""Optimized, observable trainer for the Spec 114 universal relief student.

Training remains user-owned. Calling this module without ``--confirm-run`` validates and prints the
complete plan without downloading Hub weights, creating output, or allocating CUDA training state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from harvester.height_to_normal import analytic_normals_from_height
from harvester.v50.height_relative_model import encode_relative_height
from harvester.v50.relief_teacher_labels import relief_array_sha256
from harvester.v50.universal_relief_curriculum import (
    UNIVERSAL_CURRICULUM_SCHEMA,
    source_store_identity,
)
from harvester.v50.universal_relief_model import (
    INPUT_TILE_SIZE,
    UniversalReliefNet,
    download_pinned_student_backbone,
    student_identity_dict,
)

TRAINING_RECIPE_ID = "v114.3-universal-guided-onecycle-ema"
ADT_TILE_SIZE = 1600.0 / 3.0


class UniversalTrainingError(ValueError):
    """Raised when training identity, data, or output violates the universal contract."""


@dataclass(frozen=True)
class TileSample:
    row: dict[str, Any]
    x: int
    y: int
    padded_width: int
    padded_height: int


def _axis_origins(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    padded = max(length, tile_size)
    if padded == tile_size:
        return (0,)
    stride = tile_size - overlap
    origins = list(range(0, padded - tile_size + 1, stride))
    last = padded - tile_size
    if origins[-1] != last:
        origins.append(last)
    return tuple(origins)


def _resize_2d(values: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    mode = "nearest" if nearest else "bilinear"
    kwargs = {} if nearest else {"align_corners": False}
    return (
        nn.functional.interpolate(tensor, size=size, mode=mode, **kwargs)
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )


def _resize_normals(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).permute(2, 0, 1)[None]
    resized = nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)[0]
    resized = nn.functional.normalize(resized, dim=0, eps=1e-8)
    return resized.permute(1, 2, 0).numpy()


def _pad_hwc(values: np.ndarray, padded_height: int, padded_width: int) -> np.ndarray:
    height, width = values.shape[:2]
    pad_left = (padded_width - width) // 2
    pad_right = padded_width - width - pad_left
    pad_top = (padded_height - height) // 2
    pad_bottom = padded_height - height - pad_top
    pad_spec = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if values.ndim == 3:
        pad_spec.append((0, 0))
    return np.pad(values, tuple(pad_spec), mode="edge")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_image_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class UniversalReliefDataset(Dataset):
    """Reference-index dataset expanded into deterministic 224px overlapping source tiles."""

    def __init__(
        self,
        curriculum: str | Path,
        splits: set[str],
        *,
        overlap: int = 28,
        augment: bool = False,
    ) -> None:
        self.curriculum = Path(curriculum).resolve()
        self.overlap = overlap
        self.augment = augment
        table = pq.read_table(self.curriculum / "index.parquet")
        self.rows = [row for row in table.to_pylist() if str(row["split"]) in splits]
        self.samples: list[TileSample] = []
        for row in self.rows:
            width = int(row["width"])
            height = int(row["height"])
            padded_width = max(width, INPUT_TILE_SIZE)
            padded_height = max(height, INPUT_TILE_SIZE)
            for y in _axis_origins(height, INPUT_TILE_SIZE, overlap):
                for x in _axis_origins(width, INPUT_TILE_SIZE, overlap):
                    self.samples.append(
                        TileSample(
                            row=row,
                            x=x,
                            y=y,
                            padded_width=padded_width,
                            padded_height=padded_height,
                        )
                    )
        if not self.samples:
            raise UniversalTrainingError(f"no samples selected for splits {sorted(splits)}")
        self._stores: dict[str, zarr.Group] = {}
        self._verified_inputs: set[str] = set()
        self._verified_targets: set[str] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def _store(self, path: str) -> zarr.Group:
        if path not in self._stores:
            self._stores[path] = zarr.open_group(path, mode="r")
        return self._stores[path]

    def _load_source(
        self, sample: TileSample
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        row = sample.row
        if row["target_authority"] == "exact_numeric":
            store = self._store(str(row["source_store"]))
            index = int(row["source_index"])
            rgb = np.asarray(store["minimap_rgb"][index], dtype=np.uint8)
            height_world = np.asarray(store["height_257"][index], dtype=np.float32)
            relative, tile_min, tile_max = encode_relative_height(height_world)
            target = _resize_2d(relative, rgb.shape[:2])
            height_range = max(tile_max - tile_min, 1.0)
            if all(name in store for name in ("normal_xyz", "normal_mask", "mcnr_mask_257")):
                normals = _resize_normals(np.asarray(store["normal_xyz"][index]), rgb.shape[:2])
                normal_mask = _resize_2d(
                    np.asarray(store["normal_mask"][index], dtype=np.float32),
                    rgb.shape[:2],
                    nearest=True,
                )
                mcnr_mask = _resize_2d(
                    np.asarray(store["mcnr_mask_257"][index], dtype=np.float32),
                    rgb.shape[:2],
                    nearest=True,
                )
                guidance_mask = normal_mask * mcnr_mask
            else:
                normals = np.zeros((*rgb.shape[:2], 3), dtype=np.float32)
                guidance_mask = np.zeros(rgb.shape[:2], dtype=np.float32)
            if "liquid_mask" in store:
                liquid = _resize_2d(
                    np.asarray(store["liquid_mask"][index], dtype=np.float32),
                    rgb.shape[:2],
                    nearest=True,
                )
            else:
                liquid = np.zeros(rgb.shape[:2], dtype=np.float32)
            guidance_mask *= 1.0 - np.clip(liquid, 0.0, 1.0)
            point_weight = 1.0 - 0.75 * np.clip(liquid, 0.0, 1.0)
            return rgb, target, point_weight, normals, guidance_mask, height_range

        store = self._store(str(row["source_store"]))
        input_path = str(row["input_path"])
        if input_path not in self._verified_inputs:
            observed = _sha256_file(Path(input_path))
            if observed != row["input_sha256"]:
                raise UniversalTrainingError(
                    f"teacher input image drift for {input_path}: expected {row['input_sha256']}, "
                    f"observed {observed}"
                )
            self._verified_inputs.add(input_path)
        rgb = _load_image_rgb(input_path)
        source_row_key = str(row["source_row_key"])
        target = np.asarray(store["rows"][source_row_key]["relative_relief"][:], dtype=np.float32)
        target_identity = f"{row['source_store']}#{source_row_key}"
        if target_identity not in self._verified_targets:
            observed_target_sha256 = relief_array_sha256(target)
            if observed_target_sha256 != row["target_sha256"]:
                raise UniversalTrainingError(
                    f"teacher relief target drift for {target_identity}: "
                    f"expected {row['target_sha256']}, observed {observed_target_sha256}"
                )
            self._verified_targets.add(target_identity)
        if target.shape != rgb.shape[:2]:
            raise UniversalTrainingError(
                f"teacher image/target shape mismatch for {row['row_id']}: "
                f"image={rgb.shape[:2]} target={target.shape}"
            )
        point_weight = np.ones(target.shape, dtype=np.float32)
        normals = np.zeros((*target.shape, 3), dtype=np.float32)
        guidance_mask = np.zeros(target.shape, dtype=np.float32)
        return rgb, target, point_weight, normals, guidance_mask, 1.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rgb, target, point_weight, normals, normal_mask, height_range = self._load_source(sample)
        rgb = _pad_hwc(rgb, sample.padded_height, sample.padded_width)
        target = _pad_hwc(target, sample.padded_height, sample.padded_width)
        point_weight = _pad_hwc(point_weight, sample.padded_height, sample.padded_width)
        normals = _pad_hwc(normals, sample.padded_height, sample.padded_width)
        normal_mask = _pad_hwc(normal_mask, sample.padded_height, sample.padded_width)
        y_slice = slice(sample.y, sample.y + INPUT_TILE_SIZE)
        x_slice = slice(sample.x, sample.x + INPUT_TILE_SIZE)
        batch = {
            "rgb": torch.from_numpy(
                np.transpose(rgb[y_slice, x_slice].astype(np.float32) / 255.0, (2, 0, 1)).copy()
            ),
            "target": torch.from_numpy(target[y_slice, x_slice].astype(np.float32).copy()),
            "point_weight": torch.from_numpy(
                point_weight[y_slice, x_slice].astype(np.float32).copy()
            ),
            "normals": torch.from_numpy(
                np.transpose(normals[y_slice, x_slice].astype(np.float32), (2, 0, 1)).copy()
            ),
            "normal_mask": torch.from_numpy(
                normal_mask[y_slice, x_slice].astype(np.float32).copy()
            ),
            "height_range": torch.tensor(height_range, dtype=torch.float32),
            "authority_weight": torch.tensor(
                1.0 if sample.row["target_authority"] == "exact_numeric" else 0.5,
                dtype=torch.float32,
            ),
            "row_id": str(sample.row["row_id"]),
            "visual_family": str(sample.row["visual_family"]),
            "split": str(sample.row["split"]),
            "tile_x": sample.x,
            "tile_y": sample.y,
        }
        return augment_relief_sample(batch) if self.augment else batch


def _transform_normal_components(normals: torch.Tensor, rotation: int) -> torch.Tensor:
    transformed = normals.clone()
    x = normals[0].clone()
    y = normals[1].clone()
    if rotation == 1:
        transformed[0], transformed[1] = -y, x
    elif rotation == 2:
        transformed[0], transformed[1] = -x, -y
    elif rotation == 3:
        transformed[0], transformed[1] = y, -x
    return transformed


def augment_relief_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Paired D4 geometry plus broad photometric/style augmentation."""
    result = dict(sample)
    spatial_keys = ("rgb", "target", "point_weight", "normals", "normal_mask")
    rotation = int(torch.randint(0, 4, ()).item())
    for key in spatial_keys:
        result[key] = torch.rot90(result[key], rotation, dims=(-2, -1))
    result["normals"] = _transform_normal_components(result["normals"], rotation)
    if bool(torch.rand(()) < 0.5):
        for key in spatial_keys:
            result[key] = torch.flip(result[key], dims=(-1,))
        result["normals"][0] *= -1.0
    if bool(torch.rand(()) < 0.5):
        for key in spatial_keys:
            result[key] = torch.flip(result[key], dims=(-2,))
        result["normals"][1] *= -1.0

    rgb = result["rgb"]
    brightness = 0.6 + 0.8 * torch.rand(())
    contrast = 0.6 + 0.8 * torch.rand(())
    saturation = 0.2 + 1.4 * torch.rand(())
    gamma = 0.7 + 0.8 * torch.rand(())
    rgb = rgb * brightness
    mean = rgb.mean(dim=(-2, -1), keepdim=True)
    rgb = (rgb - mean) * contrast + mean
    luma = (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])[:, None, None]).sum(
        dim=0, keepdim=True
    )
    rgb = luma + (rgb - luma) * saturation
    rgb = rgb.clamp(0.0, 1.0).pow(gamma)
    if bool(torch.rand(()) < 0.2):
        rgb = luma.expand_as(rgb).clamp(0.0, 1.0)
    if bool(torch.rand(()) < 0.05):
        rgb = 1.0 - rgb
    result["rgb"] = rgb.clamp(0.0, 1.0)
    return result


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_universal_relief_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    point_weight: torch.Tensor,
    authority_weight: torch.Tensor,
    target_normals: torch.Tensor,
    normal_mask: torch.Tensor,
    height_range: torch.Tensor,
    multiscale_weight: float = 0.2,
    gradient_weight: float = 0.05,
    normal_weight: float = 0.10,
    hard_error_weight: float = 0.05,
    hard_error_max_multiplier: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sample_weights = authority_weight[:, None, None]
    pixel_weights = point_weight * sample_weights
    point_terms = []
    for size in (224, 112, 56, 28, 14):
        pred_scaled = nn.functional.interpolate(
            predicted[:, None], size=(size, size), mode="bilinear", align_corners=False
        )[:, 0]
        target_scaled = nn.functional.interpolate(
            target[:, None], size=(size, size), mode="bilinear", align_corners=False
        )[:, 0]
        weights_scaled = nn.functional.interpolate(
            pixel_weights[:, None], size=(size, size), mode="area"
        )[:, 0]
        point_terms.append(_masked_mean((pred_scaled - target_scaled).abs(), weights_scaled))
    point = multiscale_weight * torch.stack(point_terms).sum()

    dx_error = ((predicted[:, :, 1:] - predicted[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])).abs()
    dy_error = ((predicted[:, 1:, :] - predicted[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])).abs()
    dx_weight = 0.5 * (pixel_weights[:, :, 1:] + pixel_weights[:, :, :-1])
    dy_weight = 0.5 * (pixel_weights[:, 1:, :] + pixel_weights[:, :-1, :])
    gradient = 0.5 * (_masked_mean(dx_error, dx_weight) + _masked_mean(dy_error, dy_weight))

    world_height = predicted * height_range[:, None, None]
    predicted_normals = analytic_normals_from_height(
        world_height,
        spacing=ADT_TILE_SIZE / (predicted.shape[-1] - 1),
    )
    cosine_error = 1.0 - (predicted_normals * target_normals).sum(dim=1).clamp(-1.0, 1.0)
    normal_weights = normal_mask * sample_weights
    normal = _masked_mean(cosine_error, normal_weights)

    absolute_error = (predicted - target).abs()
    mean_error = _masked_mean(absolute_error.detach(), pixel_weights).clamp_min(1e-6)
    hard_multiplier = (absolute_error.detach() / mean_error).clamp(1.0, hard_error_max_multiplier)
    hard = _masked_mean(absolute_error * hard_multiplier, pixel_weights)

    total = point + gradient_weight * gradient + normal_weight * normal + hard_error_weight * hard
    return total, {
        "point": point.detach(),
        "gradient": gradient.detach(),
        "normal": normal.detach(),
        "hard": hard.detach(),
        "total": total.detach(),
    }


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def _gradient_mae(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx = ((predicted[:, :, 1:] - predicted[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])).abs().mean(dim=(1, 2))
    dy = ((predicted[:, 1:, :] - predicted[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])).abs().mean(dim=(1, 2))
    return 0.5 * (dx + dy)


def _border_mae(predicted: torch.Tensor, target: torch.Tensor, width: int = 8) -> torch.Tensor:
    error = (predicted - target).abs()
    mask = torch.zeros_like(error, dtype=torch.bool)
    mask[:, :width, :] = True
    mask[:, -width:, :] = True
    mask[:, :, :width] = True
    mask[:, :, -width:] = True
    return (error * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)


def _luminance_baseline(rgb: torch.Tensor) -> torch.Tensor:
    luma = (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(dim=1)
    low = luma.amin(dim=(1, 2), keepdim=True)
    high = luma.amax(dim=(1, 2), keepdim=True)
    return torch.where(high - low > 1e-8, (luma - low) / (high - low).clamp_min(1e-8), torch.zeros_like(luma))


@torch.no_grad()
def evaluate_universal_relief(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    family_previews: dict[str, dict[str, Any]] = {}
    worst_previews: list[dict[str, Any]] = []
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            predicted = model(rgb)
        predicted = predicted.float()
        constant = target.mean(dim=(1, 2), keepdim=True).expand_as(target)
        luminance = _luminance_baseline(rgb)
        model_mae = (predicted - target).abs().mean(dim=(1, 2))
        constant_mae = (constant - target).abs().mean(dim=(1, 2))
        luminance_mae = (luminance - target).abs().mean(dim=(1, 2))
        gradient = _gradient_mae(predicted, target)
        constant_gradient = _gradient_mae(constant, target)
        luminance_gradient = _gradient_mae(luminance, target)
        border = _border_mae(predicted, target)
        for index in range(rgb.shape[0]):
            records.append(
                {
                    "row_id": batch["row_id"][index],
                    "visual_family": batch["visual_family"][index],
                    "split": batch["split"][index],
                    "tile_x": int(batch["tile_x"][index]),
                    "tile_y": int(batch["tile_y"][index]),
                    "mae": float(model_mae[index]),
                    "gradient_mae": float(gradient[index]),
                    "border_mae": float(border[index]),
                    "constant_mae": float(constant_mae[index]),
                    "constant_gradient_mae": float(constant_gradient[index]),
                    "luminance_mae": float(luminance_mae[index]),
                    "luminance_gradient_mae": float(luminance_gradient[index]),
                }
            )
            preview = {
                "rgb": rgb[index].permute(1, 2, 0).float().cpu().numpy(),
                "truth": target[index].float().cpu().numpy(),
                "prediction": predicted[index].float().cpu().numpy(),
                "luminance": luminance[index].float().cpu().numpy(),
                "row_id": str(batch["row_id"][index]),
                "visual_family": str(batch["visual_family"][index]),
                "split": str(batch["split"][index]),
                "mae": float(model_mae[index]),
            }
            family_previews.setdefault(preview["visual_family"], preview)
            worst_previews.append(preview)
            worst_previews.sort(key=lambda item: float(item["mae"]), reverse=True)
            del worst_previews[24:]

    metric_keys = (
        "mae",
        "gradient_mae",
        "border_mae",
        "constant_mae",
        "constant_gradient_mae",
        "luminance_mae",
        "luminance_gradient_mae",
    )

    def aggregate(values: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
        family_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            family_values[value["visual_family"]].append(value)
        family_metrics: dict[str, Any] = {}
        for family, grouped in sorted(family_values.items()):
            family_metrics[family] = {
                key: float(np.mean([value[key] for value in grouped])) for key in metric_keys
            }
            family_metrics[family]["tiles"] = len(grouped)
        macro = {
            key: float(np.mean([metrics[key] for metrics in family_metrics.values()]))
            for key in metric_keys
        }
        return family_metrics, macro

    family_metrics, all_macro = aggregate(records)
    selection_records = [record for record in records if record["split"] == "validation"]
    if not selection_records:
        raise UniversalTrainingError("validation loader has no validation-only selection tiles")
    selection_family_metrics, selection_macro = aggregate(selection_records)
    compatibility_records = [record for record in records if record["split"] == "compatibility"]
    if not compatibility_records:
        raise UniversalTrainingError("validation loader has no whole-family compatibility tiles")
    compatibility_family_metrics, compatibility_macro = aggregate(compatibility_records)
    gains = {
        "constant_mae": (compatibility_macro["constant_mae"] - compatibility_macro["mae"])
        / max(compatibility_macro["constant_mae"], 1e-8),
        "luminance_mae": (compatibility_macro["luminance_mae"] - compatibility_macro["mae"])
        / max(compatibility_macro["luminance_mae"], 1e-8),
        "constant_gradient": (
            compatibility_macro["constant_gradient_mae"] - compatibility_macro["gradient_mae"]
        )
        / max(compatibility_macro["constant_gradient_mae"], 1e-8),
        "luminance_gradient": (
            compatibility_macro["luminance_gradient_mae"] - compatibility_macro["gradient_mae"]
        )
        / max(compatibility_macro["luminance_gradient_mae"], 1e-8),
    }
    summary = {
        "tiles": len(records),
        "family_metrics": family_metrics,
        "all_macro": all_macro,
        "selection_scope": "validation_only",
        "selection_family_metrics": selection_family_metrics,
        "macro": selection_macro,
        "promotion_scope": "whole_family_compatibility_only",
        "compatibility_family_metrics": compatibility_family_metrics,
        "compatibility_macro": compatibility_macro,
        "relative_gains": gains,
        "beats_constant_mae": compatibility_macro["mae"]
        < compatibility_macro["constant_mae"],
        "beats_luminance_mae": compatibility_macro["mae"]
        < compatibility_macro["luminance_mae"],
        "beats_constant_gradient": compatibility_macro["gradient_mae"]
        < compatibility_macro["constant_gradient_mae"],
        "beats_luminance_gradient": compatibility_macro["gradient_mae"]
        < compatibility_macro["luminance_gradient_mae"],
        "passes_five_percent_gate": all(value >= 0.05 for value in gains.values()),
    }
    previews = list(family_previews.values()) + worst_previews
    return summary, records, previews


def save_validation_sheet(previews: list[dict[str, Any]], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not previews:
        raise UniversalTrainingError("cannot write an empty validation sheet")
    selected = previews[: min(8, len(previews))]
    figure, axes = plt.subplots(len(selected), 5, figsize=(15, 3 * len(selected)), squeeze=False)
    for row, preview in enumerate(selected):
        error = np.abs(preview["prediction"] - preview["truth"])
        panels = (
            (preview["rgb"], "input RGB", None),
            (preview["truth"], "truth", "viridis"),
            (preview["prediction"], "prediction", "viridis"),
            (preview["luminance"], "luminance baseline", "viridis"),
            (error, "absolute error", "magma"),
        )
        for column, (image, label, cmap) in enumerate(panels):
            axes[row, column].imshow(np.clip(image, 0.0, 1.0), cmap=cmap, vmin=0.0, vmax=1.0)
            if column == 0:
                identity = (
                    f"{preview.get('visual_family', 'unknown')}/"
                    f"{preview.get('split', 'unknown')}\n"
                    f"{str(preview.get('row_id', ''))[:32]}"
                )
                axes[row, column].set_title(f"{label}\n{identity}")
            else:
                axes[row, column].set_title(label)
            axes[row, column].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _validate_curriculum(curriculum: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not curriculum.is_dir():
        raise UniversalTrainingError(f"curriculum does not exist: {curriculum}")
    summary_path = curriculum / "summary.json"
    index_path = curriculum / "index.parquet"
    if not summary_path.is_file() or not index_path.is_file():
        raise UniversalTrainingError("curriculum requires summary.json and index.parquet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != UNIVERSAL_CURRICULUM_SCHEMA:
        raise UniversalTrainingError(f"unexpected curriculum schema {summary.get('schema')!r}")
    authorities = summary.get("target_authorities", {})
    has_teacher_rows = authorities.get("teacher_pseudo", 0) > 0
    minimum_families = 5 if has_teacher_rows else 2
    if len(summary.get("visual_families", {})) < minimum_families:
        raise UniversalTrainingError(
            f"curriculum requires at least {minimum_families} visual families for this authority mix"
        )
    if not summary.get("held_out_families") or summary.get("split_counts", {}).get("compatibility", 0) < 1:
        raise UniversalTrainingError("universal training requires a non-empty whole-family holdout")
    if summary.get("group_leak_count") != 0 or summary.get("family_leak_count") != 0:
        raise UniversalTrainingError("curriculum leakage counters must both be zero")
    if authorities.get("exact_numeric", 0) < 1:
        raise UniversalTrainingError("curriculum requires exact numeric targets")
    for source in summary.get("source_inputs", []):
        source_path = Path(source["path"])
        observed = source_store_identity(source_path)
        if observed != source["sha256"]:
            raise UniversalTrainingError(
                f"source store identity drift for {source_path}: expected {source['sha256']}, "
                f"observed {observed}"
            )
    return summary, pq.read_table(index_path).to_pylist()


def build_training_plan(
    *,
    curriculum: str | Path,
    output: str | Path,
    batch_size: int,
    epochs: int,
    workers: int,
    seed: int,
    overlap: int,
    learning_rate: float,
    weight_decay: float,
    pseudo_weight: float,
    freeze_backbone: bool,
    gradient_weight: float = 0.05,
    normal_weight: float = 0.10,
    hard_error_weight: float = 0.05,
    hard_error_max_multiplier: float = 4.0,
    ema_decay: float = 0.999,
    warmup_fraction: float = 0.05,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> dict[str, Any]:
    curriculum_path = Path(curriculum).resolve()
    summary, rows = _validate_curriculum(curriculum_path)
    if workers != 0:
        raise UniversalTrainingError("Windows universal training currently requires --workers 0")
    if batch_size < 1 or epochs < 1:
        raise UniversalTrainingError("batch_size and epochs must be positive")
    if not 0 <= overlap < INPUT_TILE_SIZE:
        raise UniversalTrainingError(f"overlap must be in [0,{INPUT_TILE_SIZE})")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise UniversalTrainingError("learning rate must be positive and weight decay non-negative")
    if not 0.0 < pseudo_weight <= 1.0:
        raise UniversalTrainingError("pseudo authority weight must be in (0,1]")
    if min(gradient_weight, normal_weight, hard_error_weight) < 0.0:
        raise UniversalTrainingError("loss weights must be non-negative")
    if hard_error_max_multiplier < 1.0:
        raise UniversalTrainingError("hard-error maximum multiplier must be at least 1")
    if not 0.0 < ema_decay < 1.0 or not 0.0 < warmup_fraction < 1.0 or grad_clip <= 0.0:
        raise UniversalTrainingError("EMA, warmup, or gradient-clip setting is invalid")
    output_path = Path(output).resolve()
    if output_path.is_file() or (output_path.is_dir() and any(output_path.iterdir())):
        raise UniversalTrainingError(f"refusing to use occupied output {output_path}")
    tile_counts = Counter()
    for row in rows:
        x_count = len(_axis_origins(int(row["width"]), INPUT_TILE_SIZE, overlap))
        y_count = len(_axis_origins(int(row["height"]), INPUT_TILE_SIZE, overlap))
        tile_counts[str(row["split"])] += x_count * y_count
    if min(tile_counts["train"], tile_counts["validation"], tile_counts["compatibility"]) < 1:
        raise UniversalTrainingError(
            "training, validation, and compatibility tile counts must all be nonzero"
        )
    return {
        "schema": "v114-universal-training-plan-v1",
        "training_recipe": TRAINING_RECIPE_ID,
        "curriculum": str(curriculum_path),
        "curriculum_id": summary["curriculum_id"],
        "output": str(output_path),
        "student": student_identity_dict(),
        "deployment_inputs": ["rgb"],
        "output_signal": "relative_relief",
        "visual_families": summary["visual_families"],
        "held_out_families": summary["held_out_families"],
        "source_rows": summary["row_count"],
        "tile_counts": dict(sorted(tile_counts.items())),
        "batch_size": batch_size,
        "steps_per_epoch": math.ceil(tile_counts["train"] / batch_size),
        "epochs": epochs,
        "workers": workers,
        "seed": seed,
        "overlap": overlap,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "pseudo_authority_weight": pseudo_weight,
        "freeze_backbone": freeze_backbone,
        "loss": {
            "multiscale_l1": {"scales": [224, 112, 56, 28, 14], "weight_each": 0.2},
            "gradient_weight": gradient_weight,
            "normal_weight": normal_weight,
            "hard_error_weight": hard_error_weight,
            "hard_error_max_multiplier": hard_error_max_multiplier,
            "liquid_point_weight": 0.25,
        },
        "optimization": {
            "optimizer": "AdamW",
            "amp": use_amp,
            "ema_decay": ema_decay,
            "schedule": "OneCycleLR warmup+cosine",
            "warmup_fraction": warmup_fraction,
            "gradient_clip": grad_clip,
            "sampler": "family_balanced_weighted_replacement",
        },
        "validation": {
            "selection_metric": "macro_family_mae",
            "baselines": ["constant_relief", "direct_luminance_relief"],
            "artifacts": ["best_epoch_sheets", "per_row_metrics", "final_sheet", "worst_cases"],
        },
    }


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_model_stage_summary(
    *,
    plan: dict[str, Any],
    final_validation: dict[str, Any],
    checkpoint_path: Path,
    best_epoch: int,
    epochs_completed: int,
    parameter_count: int,
    peak_vram_gb: float,
    wall_seconds: float,
) -> dict[str, Any]:
    """Build the exact Spec 114 ``model_stage_run`` contract."""
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    curriculum_sha256 = str(plan["curriculum_id"]).removeprefix("sha256:")
    if len(curriculum_sha256) != 64:
        raise UniversalTrainingError("curriculum identity is not a SHA-256 value")
    architecture_payload = {
        "student": plan["student"],
        "freeze_backbone": plan["freeze_backbone"],
        "loss": plan["loss"],
        "optimization": plan["optimization"],
    }
    config_sha256 = hashlib.sha256(
        json.dumps(architecture_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    created_utc = datetime.now(UTC).isoformat()
    run_id = "sha256:" + hashlib.sha256(
        f"{checkpoint_sha256}:{curriculum_sha256}:{created_utc}".encode()
    ).hexdigest()
    compatibility = final_validation["compatibility_macro"]
    numeric_gate = bool(final_validation["passes_five_percent_gate"])
    return {
        "schema": "v50-model-stage-run-v1",
        "run_id": run_id,
        "created_utc": created_utc,
        "stage": "universal_relief",
        "output_signal": "relative_relief",
        "architecture": {
            "id": plan["student"]["architecture_id"],
            "config_sha256": config_sha256,
            "parameter_count": parameter_count,
        },
        "pretrained_source": {
            "hub_id": plan["student"]["hub_id"],
            "revision": plan["student"]["revision"],
            "sha256": plan["student"]["weights_sha256"],
            "license": plan["student"]["license"],
            "role": "student_initialization",
        },
        "curriculum": {
            "path": plan["curriculum"],
            "sha256": curriculum_sha256,
        },
        "upstream_models": [],
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "best_epoch": best_epoch,
        },
        "baselines": {
            "scope": "whole_family_compatibility_only",
            "constant_relief": {
                "mae": compatibility["constant_mae"],
                "gradient_mae": compatibility["constant_gradient_mae"],
            },
            "direct_luminance_relief": {
                "mae": compatibility["luminance_mae"],
                "gradient_mae": compatibility["luminance_gradient_mae"],
            },
        },
        "metrics": {
            "training_recipe": TRAINING_RECIPE_ID,
            "epochs_completed": epochs_completed,
            "peak_vram_gb": peak_vram_gb,
            "wall_seconds": wall_seconds,
            "numeric_gate_passed": numeric_gate,
            "validation": final_validation,
        },
        "visual_evidence": {
            "best_epoch_directory": "validation/",
            "final_validation": "final_validation.png",
            "worst_cases": "worst_cases.png",
            "per_row_metrics": "per_row_metrics.json",
            "user_verdict": "pending",
        },
        "promotion_verdict": "pending" if numeric_gate else "rejected",
    }


def run_training(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    output = Path(plan["output"])
    generator = _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise UniversalTrainingError("confirmed universal training requires an available CUDA device")
    if output.is_file() or (output.is_dir() and any(output.iterdir())):
        raise UniversalTrainingError(f"refusing to overwrite occupied output {output}")

    train_dataset = UniversalReliefDataset(
        args.curriculum, {"train"}, overlap=args.overlap, augment=True
    )
    validation_dataset = UniversalReliefDataset(
        args.curriculum, {"validation", "compatibility"}, overlap=args.overlap, augment=False
    )
    family_counts = Counter(sample.row["visual_family"] for sample in train_dataset.samples)
    sample_weights = [1.0 / family_counts[sample.row["visual_family"]] for sample in train_dataset.samples]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    backbone = download_pinned_student_backbone(cache_dir=args.hf_cache)
    model = UniversalReliefNet(backbone, freeze_backbone=not args.unfreeze_backbone).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    ema_model.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "training_plan.json", plan)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_fraction,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    torch.cuda.reset_peak_memory_stats(device)
    history = []
    best_score = math.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Counter[str] = Counter()
        train_items = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            point_weight = batch["point_weight"].to(device, non_blocking=True)
            authority = batch["authority_weight"].to(device, non_blocking=True)
            authority = torch.where(
                authority < 1.0,
                torch.full_like(authority, args.pseudo_weight),
                authority,
            )
            normals = batch["normals"].to(device, non_blocking=True)
            normal_mask = batch["normal_mask"].to(device, non_blocking=True)
            height_range = batch["height_range"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                predicted = model(rgb)
                loss, components = compute_universal_relief_loss(
                    predicted,
                    target,
                    point_weight=point_weight,
                    authority_weight=authority,
                    target_normals=normals,
                    normal_mask=normal_mask,
                    height_range=height_range,
                    gradient_weight=args.gradient_weight,
                    normal_weight=args.normal_weight,
                    hard_error_weight=args.hard_error_weight,
                    hard_error_max_multiplier=args.hard_error_max_multiplier,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema(ema_model, model, args.ema_decay)
            count = rgb.shape[0]
            train_items += count
            for name, value in components.items():
                totals[name] += float(value) * count
            totals["gradient_norm"] += float(gradient_norm) * count

        validation, records, previews = evaluate_universal_relief(
            ema_model, validation_loader, device=device, use_amp=use_amp
        )
        score = float(validation["macro"]["mae"])
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema": "v114-universal-relief-checkpoint-v1",
                    "epoch": epoch,
                    "model": ema_model.state_dict(),
                    "raw_model": model.state_dict(),
                    "student": student_identity_dict(),
                    "training_recipe": TRAINING_RECIPE_ID,
                    "curriculum_id": plan["curriculum_id"],
                    "validation": validation,
                    "freeze_backbone": not args.unfreeze_backbone,
                },
                output / "checkpoint_best.pt",
            )
            save_validation_sheet(
                previews,
                output / "validation" / f"best_epoch_{epoch:04d}.png",
                f"Universal relief best epoch {epoch}",
            )
        else:
            stale += 1
        epoch_record = {
            "epoch": epoch,
            "train": {name: total / train_items for name, total in totals.items()},
            "validation": validation,
            "selection_score": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - epoch_started,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
        history.append(epoch_record)
        _write_json(output / "history.json", history)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
            },
            output / "checkpoint_last.pt",
        )
        print(
            f"[epoch {epoch}] train={epoch_record['train']['total']:.6f} "
            f"macro_mae={score:.6f} constant={validation['macro']['constant_mae']:.6f} "
            f"luminance={validation['macro']['luminance_mae']:.6f} "
            f"best={best_score:.6f} stale={stale}/{args.patience}"
        )
        if stale >= args.patience:
            break

    best_checkpoint = torch.load(output / "checkpoint_best.pt", map_location=device, weights_only=False)
    ema_model.load_state_dict(best_checkpoint["model"])
    final_validation, records, previews = evaluate_universal_relief(
        ema_model, validation_loader, device=device, use_amp=use_amp
    )
    _write_json(output / "per_row_metrics.json", records)
    save_validation_sheet(previews, output / "final_validation.png", "Universal relief final best")
    worst_previews = sorted(previews, key=lambda item: float(item["mae"]), reverse=True)[:8]
    if worst_previews:
        save_validation_sheet(worst_previews, output / "worst_cases.png", "Universal relief worst cases")
    summary = build_model_stage_summary(
        plan=plan,
        final_validation=final_validation,
        checkpoint_path=output / "checkpoint_best.pt",
        best_epoch=best_epoch,
        epochs_completed=len(history),
        parameter_count=ema_model.total_parameter_count(),
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / (1024**3),
        wall_seconds=time.perf_counter() - started,
    )
    _write_json(output / "training_summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train universal image-to-relief student; dry-run unless --confirm-run."
    )
    parser.add_argument("--curriculum", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--overlap", type=int, default=28)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pseudo-weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--normal-weight", type=float, default=0.10)
    parser.add_argument("--hard-error-weight", type=float, default=0.05)
    parser.add_argument("--hard-error-max-multiplier", type=float, default=4.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_training_plan(
        curriculum=args.curriculum,
        output=args.output,
        batch_size=args.batch,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
        overlap=args.overlap,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pseudo_weight=args.pseudo_weight,
        freeze_backbone=not args.unfreeze_backbone,
        gradient_weight=args.gradient_weight,
        normal_weight=args.normal_weight,
        hard_error_weight=args.hard_error_weight,
        hard_error_max_multiplier=args.hard_error_max_multiplier,
        ema_decay=args.ema_decay,
        warmup_fraction=args.warmup_fraction,
        grad_clip=args.grad_clip,
        use_amp=not args.no_amp,
    )
    print(json.dumps(plan, indent=2))
    if not args.confirm_run:
        print("DRY RUN: add --confirm-run to download the pinned student and launch CUDA training.")
        return 0
    summary = run_training(args, plan)
    print(json.dumps(summary, indent=2))
    return 0
