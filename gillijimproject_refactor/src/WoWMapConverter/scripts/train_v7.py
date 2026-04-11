#!/usr/bin/env python3
"""
WoW Height Regressor V7.1 - multichannel terrain model.

V7.1 is not a pure minimap-to-height regressor. It works because it combines
the minimap with auxiliary terrain context that resolves ambiguity the minimap
cannot carry on its own.

The key long-term reconstruction seam is WDL:
- for many maps we have minimaps and WDL even when we do not have full ADTs
- WDL gives us a low-resolution terrain prior that the viewer already uses as a
    practical performance fallback
- training against both minimap->WDL and minimap->heightmap correlations gives
    us a path to reconstruct terrain even when only the lower-resolution teacher is
    available

The remaining auxiliary channels mark known losses in the minimap surface:
- liquids flatten or overwrite visible terrain cues
- placed objects obscure terrain and often imply locally flat support surfaces

Inputs:
- minimap RGB
- normal map RGB
- low-resolution WDL height prior
- per-tile height min/max hint masks
- liquid mask
- liquid height prior
- object footprint mask

Outputs:
- global heightmap
- local heightmap
- height bounds head

The current CLI keeps the object-aware V7.1 model design intact while exposing
dataset-root selection and training controls needed by the viewer GUI.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


INPUT_SIZE = 512
OUTPUT_SIZE = 512
MODEL_INPUT_CHANNELS = 12
MODEL_OUTPUT_CHANNELS = 2

HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
HEIGHT_GLOBAL_RANGE = HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN

DEFAULT_OUTPUT_DIR = Path("./vlm_output")
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 500
DEFAULT_EARLY_STOP_PATIENCE = 25
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_SPATIAL_GROUP_SIZE = 4
DEFAULT_SEED = 1337
DEFAULT_BLUR_SIGMA = 0.5
DEFAULT_PREVIEW_COUNT = 4
PREVIEW_MIN_VISUAL_VARIANCE = 0.008
DEFAULT_MIN_HEIGHT_RANGE = 0.5

DEFAULT_DATASET_SEARCH_ROOTS = [
    Path(r"i:\parp\parp-tools\gillijimproject_refactor\test_data\vlm-datasets"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets"),
]

PROFILE_PRESETS = {
    "manual": {
        "description": "Use only explicit --dataset-root values.",
        "include_maps": [],
        "discover": [],
    },
    "development-map": {
        "description": "Prioritize 3.0.1 Northrend and optionally supplement with 4.0.0.11927 LostIsles.",
        "include_maps": ["Northrend", "LostIsles"],
        "discover": [
            {
                "label": "wrath-northrend",
                "map_tokens": ["northrend"],
                "build_tokens": ["301", "3.0.1", "8303", "wrath", "wotlk"],
            },
            {
                "label": "cata-lostisles",
                "map_tokens": ["lostisles", "lost_isles"],
                "build_tokens": ["400", "4.0.0", "11927", "cata", "cataclysm"],
            },
        ],
    },
}

LOSS_WEIGHTS = {
    "heightmap_global": 0.15,
    "heightmap_local": 0.35,
    "bounds": 0.05,
    "ssim": 0.05,
    "gradient": 0.05,
    "edge": 0.25,
}


@dataclass(frozen=True)
class TileSample:
    dataset_root: Path
    dataset_name: str
    json_path: Path
    tile_name: str
    map_name: str
    tile_x: int
    tile_y: int
    minimap_path: Path
    normalmap_path: Path
    heightmap_global_path: Path
    heightmap_local_path: Path
    liquid_mask_path: Optional[Path]
    liquid_height_path: Optional[Path]


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_tile_identity(tile_name: str) -> Tuple[str, int, int]:
    parts = tile_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Tile name '{tile_name}' is not in <map>_<x>_<y> form.")
    map_name = "_".join(parts[:-2])
    tile_x = int(parts[-2])
    tile_y = int(parts[-1])
    return map_name, tile_x, tile_y


def collect_explicit_roots(dataset_roots: Sequence[str]) -> List[Path]:
    roots: List[Path] = []
    for root_str in dataset_roots:
        root = Path(root_str)
        if root.exists() and root.is_dir():
            roots.append(root)
        else:
            print(f"Warning: dataset root not found, skipping: {root}")
    return roots


def discover_profile_roots(profile_name: str, search_roots: Sequence[str]) -> List[Path]:
    profile = PROFILE_PRESETS[profile_name]
    discovered: List[Path] = []
    seen: set[Path] = set()

    for search_root_str in search_roots:
        search_root = Path(search_root_str)
        if not search_root.exists() or not search_root.is_dir():
            continue

        for child in sorted(search_root.iterdir()):
            if not child.is_dir():
                continue
            normalized_name = normalize_token(child.name)

            for rule in profile["discover"]:
                map_hit = any(normalize_token(token) in normalized_name for token in rule["map_tokens"])
                build_hit = any(normalize_token(token) in normalized_name for token in rule["build_tokens"])
                if map_hit and build_hit and child not in seen:
                    seen.add(child)
                    discovered.append(child)
                    break

    return discovered


def resolve_dataset_roots(args: argparse.Namespace) -> List[Path]:
    explicit = collect_explicit_roots(args.dataset_root)
    if explicit:
        return explicit

    if args.profile == "manual":
        raise SystemExit("No dataset roots provided. Use --dataset-root or choose a non-manual --profile.")

    discovered = discover_profile_roots(args.profile, args.search_root)
    if discovered:
        return discovered

    profile = PROFILE_PRESETS[args.profile]
    raise SystemExit(
        "No dataset roots resolved for profile "
        f"'{args.profile}'. Expected roots containing the era/map markers for: {profile['description']}. "
        "Pass explicit --dataset-root paths instead."
    )


class MultiChannelUNetV7(nn.Module):
    """5-level U-Net for 512x512 multichannel terrain inputs."""

    def __init__(self, in_channels: int = MODEL_INPUT_CHANNELS, out_channels: int = MODEL_OUTPUT_CHANNELS):
        super().__init__()

        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 1024)
        self.bottleneck = self._conv_block(1024, 2048)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.height_bounds_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

        self.up5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec5 = self._conv_block(2048, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        bottleneck = self.bottleneck(self.pool(enc5))

        pooled = self.global_pool(bottleneck).view(bottleneck.size(0), -1)
        bounds = self.height_bounds_fc(pooled)

        dec5 = self.up5(bottleneck)
        dec5 = torch.cat([dec5, enc5], dim=1)
        dec5 = self.dec5(dec5)

        dec4 = self.up4(dec5)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        outputs = torch.sigmoid(self.out_conv(dec1))
        if outputs.shape[-2:] != (OUTPUT_SIZE, OUTPUT_SIZE):
            outputs = F.interpolate(outputs, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False)

        return outputs, bounds


def load_heightmap_16bit(path: Path, target_size: int = OUTPUT_SIZE) -> torch.Tensor:
    image = Image.open(path)
    if image.mode == "I;16":
        array = np.asarray(image, dtype=np.float32) / 65535.0
    elif image.mode == "I":
        array = np.asarray(image, dtype=np.float32)
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    else:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    if array.shape[0] != target_size or array.shape[1] != target_size:
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return tensor.squeeze(0)

    return torch.from_numpy(array).unsqueeze(0)


class WoWTileDatasetV7(Dataset):
    def __init__(
        self,
        dataset_roots: Sequence[Path],
        include_maps: Sequence[str],
        exclude_maps: Sequence[str],
        input_size: int = INPUT_SIZE,
        augment: bool = True,
        limit: Optional[int] = None,
        min_height_range: float = DEFAULT_MIN_HEIGHT_RANGE,
    ) -> None:
        self.input_size = input_size
        self.augment = augment
        self.min_height_range = min_height_range
        self.include_maps = {value.lower() for value in include_maps if value}
        self.exclude_maps = {value.lower() for value in exclude_maps if value}
        self.samples: List[TileSample] = []

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=DEFAULT_BLUR_SIGMA)
        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        blank_skipped = 0
        print("Loading V7 dataset roots...")
        for dataset_root in dataset_roots:
            samples, skipped = self._collect_root_samples(dataset_root, limit)
            self.samples.extend(samples)
            blank_skipped += skipped

        print(f"Loaded {len(self.samples)} valid samples (V7.1 strict mode, {blank_skipped} blank tiles skipped)")

    def _collect_root_samples(self, dataset_root: Path, limit: Optional[int]) -> Tuple[List[TileSample], int]:
        dataset_dir = dataset_root / "dataset"
        if not dataset_dir.exists():
            print(f"Warning: dataset folder missing in {dataset_root}")
            return [], 0

        collected: List[TileSample] = []
        blank_skipped = 0
        for json_path in sorted(dataset_dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception as exc:
                print(f"Warning: failed to read {json_path}: {exc}")
                continue

            terrain = payload.get("terrain_data", {})
            tile_name = terrain.get("adt_tile") or json_path.stem

            try:
                map_name, tile_x, tile_y = parse_tile_identity(tile_name)
            except ValueError:
                continue

            map_key = map_name.lower()
            if self.include_maps and map_key not in self.include_maps:
                continue
            if map_key in self.exclude_maps:
                continue

            # Skip blank/flat tiles (ocean, void) that have no useful height variation
            height_min = float(terrain.get("height_min", 0.0))
            height_max = float(terrain.get("height_max", 0.0))
            if (height_max - height_min) < self.min_height_range:
                blank_skipped += 1
                continue

            heightmap_global_rel = terrain.get("heightmap_global") or terrain.get("heightmap")
            heightmap_local_rel = terrain.get("heightmap_local") or terrain.get("heightmap")
            normalmap_rel = terrain.get("normalmap")
            if not heightmap_global_rel or not heightmap_local_rel or not normalmap_rel:
                continue

            minimap_path = dataset_root / "images" / f"{tile_name}.png"
            normalmap_path = dataset_root / normalmap_rel
            heightmap_global_path = dataset_root / heightmap_global_rel
            heightmap_local_path = dataset_root / heightmap_local_rel
            if not minimap_path.exists() or not normalmap_path.exists() or not heightmap_global_path.exists() or not heightmap_local_path.exists():
                continue

            liquid_mask_path = dataset_root / terrain["liquid_mask"] if terrain.get("liquid_mask") else None
            liquid_height_path = dataset_root / terrain["liquid_height"] if terrain.get("liquid_height") else None

            collected.append(
                TileSample(
                    dataset_root=dataset_root,
                    dataset_name=dataset_root.name,
                    json_path=json_path,
                    tile_name=tile_name,
                    map_name=map_name,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    minimap_path=minimap_path,
                    normalmap_path=normalmap_path,
                    heightmap_global_path=heightmap_global_path,
                    heightmap_local_path=heightmap_local_path,
                    liquid_mask_path=liquid_mask_path,
                    liquid_height_path=liquid_height_path,
                )
            )

            if limit is not None and len(collected) >= limit:
                break

        print(f"  {dataset_root.name}: {len(collected)} usable samples ({blank_skipped} blank skipped)")
        return collected, blank_skipped

    def __len__(self) -> int:
        return len(self.samples)

    def _render_wdl(self, wdl_data: Optional[Dict[str, object]]) -> torch.Tensor:
        if not wdl_data:
            return torch.full((1, self.input_size, self.input_size), 0.5)

        outer = np.asarray(wdl_data.get("outer_17", []), dtype=np.float32)
        if len(outer) != 289:
            return torch.full((1, self.input_size, self.input_size), 0.5)

        grid = outer.reshape(17, 17)
        minimum = float(grid.min())
        maximum = float(grid.max())
        if maximum - minimum > 1e-6:
            grid = (grid - minimum) / (maximum - minimum)
        else:
            grid[:] = 0.5

        image = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        return self.to_tensor(image)

    def _build_object_mask(self, objects: Optional[Sequence[Dict[str, object]]]) -> torch.Tensor:
        object_mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        if not objects:
            return object_mask

        image = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        tile_size = 533.33333
        for obj in objects:
            pos_x = float(obj.get("x", obj.get("pos_x", 0.0)))
            pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
            scale = float(obj.get("scale", 1.0))

            bounds_min = obj.get("bounds_min")
            bounds_max = obj.get("bounds_max")
            if bounds_min and bounds_max and len(bounds_min) >= 2 and len(bounds_max) >= 2:
                half_width = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
                half_depth = abs(float(bounds_max[1]) - float(bounds_min[1])) * 0.5 * scale
                pixels_per_unit = self.input_size / tile_size
                radius_x = max(1, int(half_width * pixels_per_unit))
                radius_y = max(1, int(half_depth * pixels_per_unit))
            else:
                radius_x = max(1, int(5 * scale))
                radius_y = radius_x

            if abs(pos_x) < 2 and abs(pos_y) < 2:
                normalized_x = int((pos_x + 1) * 0.5 * self.input_size)
                normalized_y = int((pos_y + 1) * 0.5 * self.input_size)
            else:
                normalized_x = int((pos_x / tile_size) * self.input_size) % self.input_size
                normalized_y = int((pos_y / tile_size) * self.input_size) % self.input_size

            x1 = max(0, normalized_x - radius_x)
            y1 = max(0, normalized_y - radius_y)
            x2 = min(self.input_size, normalized_x + radius_x)
            y2 = min(self.input_size, normalized_y + radius_y)
            image[y1:y2, x1:x2] = 1.0

        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        with open(sample.json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        terrain = payload.get("terrain_data", {})

        minimap = Image.open(sample.minimap_path).convert("RGB")
        normalmap = Image.open(sample.normalmap_path).convert("RGB")
        minimap = minimap.resize((self.input_size, self.input_size), Image.BILINEAR)
        normalmap = normalmap.resize((self.input_size, self.input_size), Image.BILINEAR)

        minimap = self.blur(minimap)
        if self.augment:
            minimap = self.color_jitter(minimap)

        minimap_tensor = self.normalize(self.to_tensor(minimap))
        normalmap_tensor = self.normalize(self.to_tensor(normalmap))
        wdl_tensor = self._render_wdl(terrain.get("wdl_heights"))

        height_min = float(terrain.get("height_min", 0.0))
        height_max = float(terrain.get("height_max", 100.0))
        global_min = float(terrain.get("height_global_min", HEIGHT_GLOBAL_MIN))
        global_max = float(terrain.get("height_global_max", HEIGHT_GLOBAL_MAX))
        global_range = max(global_max - global_min, 1e-6)

        height_min_normalized = np.clip((height_min - global_min) / global_range, 0.0, 1.0)
        height_max_normalized = np.clip((height_max - global_min) / global_range, 0.0, 1.0)
        global_min_normalized = np.clip((global_min - global_min) / global_range, 0.0, 1.0)
        global_max_normalized = np.clip((global_max - global_min) / global_range, 0.0, 1.0)

        height_min_mask = torch.full((1, self.input_size, self.input_size), float(height_min_normalized), dtype=torch.float32)
        height_max_mask = torch.full((1, self.input_size, self.input_size), float(height_max_normalized), dtype=torch.float32)

        liquid_mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        if sample.liquid_mask_path and sample.liquid_mask_path.exists():
            liquid_image = Image.open(sample.liquid_mask_path).convert("L").resize((self.input_size, self.input_size), Image.NEAREST)
            liquid_tensor = self.to_tensor(liquid_image)
            liquid_mask = (liquid_tensor > 0.1).float()

        liquid_height_prior = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        if sample.liquid_height_path and sample.liquid_height_path.exists():
            liquid_height_prior = load_heightmap_16bit(sample.liquid_height_path, self.input_size) * liquid_mask

        object_mask = self._build_object_mask(terrain.get("objects"))

        input_tensor = torch.cat(
            [
                minimap_tensor,
                normalmap_tensor,
                wdl_tensor,
                height_min_mask,
                height_max_mask,
                liquid_mask,
                liquid_height_prior,
                object_mask,
            ],
            dim=0,
        )

        heightmap_global_tensor = load_heightmap_16bit(sample.heightmap_global_path, OUTPUT_SIZE)
        heightmap_local_tensor = load_heightmap_16bit(sample.heightmap_local_path, OUTPUT_SIZE)

        bounds_tensor = torch.tensor(
            [height_min_normalized, height_max_normalized, global_min_normalized, global_max_normalized],
            dtype=torch.float32,
        )

        if self.augment and torch.rand(1).item() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            heightmap_global_tensor = torch.flip(heightmap_global_tensor, dims=[2])
            heightmap_local_tensor = torch.flip(heightmap_local_tensor, dims=[2])

        return {
            "input": input_tensor,
            "target": torch.cat([heightmap_global_tensor, heightmap_local_tensor], dim=0),
            "height_bounds": bounds_tensor,
        }


def ssim_loss(predicted: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gaussian / gaussian.sum()

    gaussian = gaussian_window(window_size).to(predicted.device)
    window = (gaussian[:, None] @ gaussian[None, :]).unsqueeze(0).unsqueeze(0)
    window = window.expand(predicted.shape[1], 1, window_size, window_size)

    mu_pred = F.conv2d(predicted, window, padding=window_size // 2, groups=predicted.shape[1])
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(predicted * predicted, window, padding=window_size // 2, groups=predicted.shape[1]) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=target.shape[1]) - mu_target_sq
    sigma_pred_target = F.conv2d(predicted * target, window, padding=window_size // 2, groups=predicted.shape[1]) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return 1 - ssim_map.mean()


def edge_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=predicted.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    def compute(tensor: torch.Tensor) -> torch.Tensor:
        edges = torch.zeros_like(tensor)
        for channel in range(tensor.shape[1]):
            current = tensor[:, channel:channel + 1]
            edges[:, channel:channel + 1] = F.conv2d(current, sobel_x, padding=1).abs() + F.conv2d(current, sobel_y, padding=1).abs()
        return edges

    return F.l1_loss(compute(predicted[:, :2]), compute(target[:, :2]))


def combined_loss(
    predicted_heightmap: torch.Tensor,
    predicted_bounds: torch.Tensor,
    target_heightmap: torch.Tensor,
    target_bounds: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    global_loss = F.l1_loss(predicted_heightmap[:, 0:1], target_heightmap[:, 0:1])
    local_loss = F.l1_loss(predicted_heightmap[:, 1:2], target_heightmap[:, 1:2])
    bounds_loss = F.mse_loss(predicted_bounds, target_bounds)

    def get_gradient(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor[:, :, :, 1:] - tensor[:, :, :, :-1], tensor[:, :, 1:, :] - tensor[:, :, :-1, :]

    predicted_dx, predicted_dy = get_gradient(predicted_heightmap[:, :2])
    target_dx, target_dy = get_gradient(target_heightmap[:, :2])
    gradient_component = F.l1_loss(predicted_dx, target_dx) + F.l1_loss(predicted_dy, target_dy)

    ssim_component = ssim_loss(predicted_heightmap[:, :2], target_heightmap[:, :2])
    edge_component = edge_loss(predicted_heightmap, target_heightmap)

    total = (
        LOSS_WEIGHTS["heightmap_global"] * global_loss
        + LOSS_WEIGHTS["heightmap_local"] * local_loss
        + LOSS_WEIGHTS["bounds"] * bounds_loss
        + LOSS_WEIGHTS["gradient"] * gradient_component
        + LOSS_WEIGHTS["ssim"] * ssim_component
        + LOSS_WEIGHTS["edge"] * edge_component
    )

    return total, {
        "heightmap_global": float(global_loss.item()),
        "heightmap_local": float(local_loss.item()),
        "bounds": float(bounds_loss.item()),
        "gradient": float(gradient_component.item()),
        "ssim": float(ssim_component.item()),
        "edge": float(edge_component.item()),
    }


def build_validation_groups(samples: Sequence[TileSample], block_size: int) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for index, sample in enumerate(samples):
        block_x = sample.tile_x // block_size
        block_y = sample.tile_y // block_size
        group_key = f"{sample.dataset_name}:{sample.map_name}:{block_x}:{block_y}"
        groups.setdefault(group_key, []).append(index)
    return groups


def split_grouped_indices(samples: Sequence[TileSample], val_fraction: float, seed: int, block_size: int) -> Tuple[List[int], List[int], int, int]:
    groups = build_validation_groups(samples, block_size)
    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    target_val_samples = max(1, int(round(len(samples) * val_fraction)))
    val_indices: List[int] = []
    val_group_count = 0
    for group_key in group_keys:
        if len(val_indices) >= target_val_samples and val_group_count > 0:
            break
        val_indices.extend(groups[group_key])
        val_group_count += 1

    val_index_set = set(val_indices)
    train_indices = [index for index in range(len(samples)) if index not in val_index_set]
    if not train_indices:
        split_point = max(1, len(val_indices) // 5)
        train_indices = val_indices[:split_point]
        val_indices = val_indices[split_point:]
        val_index_set = set(val_indices)

    if not val_indices:
        val_indices = train_indices[-max(1, len(train_indices) // 10):]
        val_index_set = set(val_indices)
        train_indices = [index for index in train_indices if index not in val_index_set]

    train_groups = len({key for key, indices in groups.items() if any(index in train_indices for index in indices)})
    val_groups = len({key for key, indices in groups.items() if any(index in val_index_set for index in indices)})
    return train_indices, val_indices, train_groups, val_groups


def _image_luma_variance(path: Path, size: int = 64) -> float:
    try:
        with Image.open(path).convert("L") as image:
            reduced = image.resize((size, size), Image.BILINEAR)
            pixels = np.asarray(reduced, dtype=np.float32) / 255.0
        return float(np.var(pixels))
    except Exception:
        return 0.0


def _liquid_coverage(path: Optional[Path], size: int = 64) -> float:
    if not path or not path.exists():
        return 0.0

    try:
        with Image.open(path).convert("L") as image:
            reduced = image.resize((size, size), Image.NEAREST)
            pixels = np.asarray(reduced, dtype=np.float32)
        return float((pixels > 8.0).mean())
    except Exception:
        return 0.0


def compute_preview_interest_metrics(sample: TileSample) -> Dict[str, float]:
    height_span = 0.0
    object_count = 0

    try:
        with open(sample.json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        terrain = payload.get("terrain_data", {})

        height_min = float(terrain.get("height_min", 0.0))
        height_max = float(terrain.get("height_max", height_min))
        height_span = max(0.0, height_max - height_min)

        objects = terrain.get("objects")
        if isinstance(objects, list):
            object_count = len(objects)
    except Exception:
        pass

    minimap_variance = _image_luma_variance(sample.minimap_path)
    normal_variance = _image_luma_variance(sample.normalmap_path)
    liquid_coverage = _liquid_coverage(sample.liquid_mask_path)

    visual_component = min((minimap_variance + normal_variance) / 0.06, 1.0)
    height_component = min(height_span / 300.0, 1.0)
    object_component = min(object_count / 25.0, 1.0)
    liquid_component = min(liquid_coverage * 2.0, 1.0)

    score = float(
        0.60 * visual_component
        + 0.20 * height_component
        + 0.12 * object_component
        + 0.08 * liquid_component
    )

    return {
        "score": score,
        "visual_variance": float(minimap_variance + normal_variance),
        "minimap_variance": float(minimap_variance),
        "normal_variance": float(normal_variance),
    }


def compute_preview_interest_score(sample: TileSample) -> float:
    return compute_preview_interest_metrics(sample)["score"]


def select_preview_candidates(samples: Sequence[TileSample], val_indices: Sequence[int], preview_count: int) -> List[Tuple[int, float, float]]:
    scored: List[Tuple[int, float, float]] = []
    for index in val_indices:
        if index < 0 or index >= len(samples):
            continue
        metrics = compute_preview_interest_metrics(samples[index])
        scored.append((index, metrics["score"], metrics["visual_variance"]))

    if not scored:
        return []

    target_count = max(1, preview_count)
    scored.sort(
        key=lambda item: (
            -item[1],
            -item[2],
            samples[item[0]].dataset_name,
            samples[item[0]].map_name,
            samples[item[0]].tile_x,
            samples[item[0]].tile_y,
        )
    )

    non_blank = [item for item in scored if item[2] >= PREVIEW_MIN_VISUAL_VARIANCE]
    if non_blank:
        return non_blank[:target_count]

    return scored[:target_count]


def build_preview_batch(dataset: WoWTileDatasetV7, dataset_indices: Sequence[int]) -> Tuple[Dict[str, torch.Tensor], List[int], List[Tuple[int, str]]]:
    if not dataset_indices:
        raise ValueError("No preview indices were provided.")

    items: List[Dict[str, torch.Tensor]] = []
    loaded_indices: List[int] = []
    skipped_indices: List[Tuple[int, str]] = []

    for index in dataset_indices:
        try:
            item = dataset[index]
        except Exception as exc:
            skipped_indices.append((index, str(exc)))
            continue

        items.append(item)
        loaded_indices.append(index)

    if not items:
        raise ValueError("No preview tiles could be loaded from the selected candidates.")

    batch = {
        "input": torch.stack([item["input"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
        "height_bounds": torch.stack([item["height_bounds"] for item in items], dim=0),
    }
    return batch, loaded_indices, skipped_indices


def save_training_preview(model: nn.Module, batch: Dict[str, torch.Tensor], epoch: int, output_dir: Path, device: torch.device) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions, _ = model(inputs)

    rows: List[torch.Tensor] = []
    sample_count = min(DEFAULT_PREVIEW_COUNT, inputs.shape[0])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for index in range(sample_count):
        minimap = torch.clamp(inputs[index, 0:3].cpu() * std + mean, 0.0, 1.0)
        normal = torch.clamp(inputs[index, 3:6].cpu() * std + mean, 0.0, 1.0)
        water = inputs[index, 9:10].cpu().repeat(3, 1, 1) * torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
        prediction = predictions[index, 1:2].cpu()
        target = targets[index, 1:2].cpu()

        def normalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor - tensor.min()
            return tensor / (tensor.max() + 1e-6)

        prediction_viz = normalize_for_display(prediction).repeat(3, 1, 1)
        target_viz = normalize_for_display(target).repeat(3, 1, 1)
        rows.append(torch.cat([minimap, normal, water, prediction_viz, target_viz], dim=2))

    grid = torch.cat(rows, dim=1)
    grid = torch.clamp(grid, 0.0, 1.0)
    transforms.ToPILImage()(grid).save(output_dir / f"val_epoch_{epoch:04d}.png")


def checkpoint_metadata_from_args(
    args: argparse.Namespace,
    dataset_roots: Sequence[Path],
    sample_count: int,
    train_count: int,
    val_count: int,
    train_groups: int,
    val_groups: int,
) -> Dict[str, object]:
    return {
        "profile": args.profile,
        "dataset_roots": [str(root) for root in dataset_roots],
        "include_maps": list(args.include_map),
        "exclude_maps": list(args.exclude_map),
        "input_channels": MODEL_INPUT_CHANNELS,
        "output_channels": MODEL_OUTPUT_CHANNELS,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "sample_count": sample_count,
        "train_count": train_count,
        "val_count": val_count,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "spatial_group_size": args.spatial_group_size,
        "seed": args.seed,
        "height_global_min": HEIGHT_GLOBAL_MIN,
        "height_global_max": HEIGHT_GLOBAL_MAX,
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_roots = resolve_dataset_roots(args)
    include_maps = list(args.include_map)
    if not include_maps and args.profile != "manual":
        include_maps = list(PROFILE_PRESETS[args.profile]["include_maps"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("WoW V7.1 Training - multichannel terrain model")
    print("=" * 72)
    print("Dataset roots:")
    for root in dataset_roots:
        print(f"  - {root}")
    if include_maps:
        print(f"Included maps: {', '.join(include_maps)}")
    if args.exclude_map:
        print(f"Excluded maps: {', '.join(args.exclude_map)}")

    dataset = WoWTileDatasetV7(
        dataset_roots=dataset_roots,
        include_maps=include_maps,
        exclude_maps=args.exclude_map,
        input_size=INPUT_SIZE,
        augment=not args.no_augment,
        limit=args.limit,
        min_height_range=args.min_height_range,
    )
    if len(dataset) == 0:
        raise SystemExit("No samples found. Regenerate the dataset with V7 exporter outputs intact.")

    train_indices, val_indices, train_groups, val_groups = split_grouped_indices(
        dataset.samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        block_size=args.spatial_group_size,
    )
    print(f"Train/val samples: {len(train_indices)} / {len(val_indices)}")
    print(f"Train/val spatial groups: {train_groups} / {val_groups}")

    train_dataset = Subset(dataset, train_indices)
    val_base_dataset = WoWTileDatasetV7(
        dataset_roots=dataset_roots,
        include_maps=include_maps,
        exclude_maps=args.exclude_map,
        input_size=INPUT_SIZE,
        augment=False,
        limit=args.limit,
        min_height_range=args.min_height_range,
    )
    val_dataset = Subset(val_base_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    preview_batch: Optional[Dict[str, torch.Tensor]] = None
    try:
        preview_candidates = select_preview_candidates(val_base_dataset.samples, val_indices, DEFAULT_PREVIEW_COUNT)
        preview_indices = [index for index, _, _ in preview_candidates]
        preview_batch, loaded_preview_indices, skipped_preview_indices = build_preview_batch(val_base_dataset, preview_indices)
        loaded_preview_index_set = set(loaded_preview_indices)

        print("Preview tiles (ranked for visual signal):")
        for index, score, visual_variance in preview_candidates:
            sample = val_base_dataset.samples[index]
            skipped_suffix = " [skipped]" if index not in loaded_preview_index_set else ""
            print(
                f"  - {sample.dataset_name}:{sample.tile_name} "
                f"(score={score:.3f}, visual_var={visual_variance:.5f}){skipped_suffix}"
            )

        if preview_candidates and preview_candidates[0][2] < PREVIEW_MIN_VISUAL_VARIANCE:
            print(
                "Warning: no preview tiles met the visual-variance floor "
                f"({PREVIEW_MIN_VISUAL_VARIANCE:.5f}); using low-signal fallback candidates."
            )

        if skipped_preview_indices:
            print(f"Warning: skipped {len(skipped_preview_indices)} preview tile(s) that failed to load.")
    except Exception as exc:
        print(f"Warning: failed to precompute preview batch, falling back to first val batch: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = MultiChannelUNetV7().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "components": [],
        "metadata": checkpoint_metadata_from_args(args, dataset_roots, len(dataset), len(train_indices), len(val_indices), train_groups, val_groups),
    }

    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("val_loss", best_loss))
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses: List[float] = []
        epoch_parts: Dict[str, float] = {}

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            bounds = batch["height_bounds"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs, output_bounds = model(inputs)
            loss, parts = combined_loss(outputs, output_bounds, targets, bounds)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            for key, value in parts.items():
                epoch_parts[key] = epoch_parts.get(key, 0.0) + value
            progress.set_postfix(loss=f"{loss.item():.4f}")

        for key in epoch_parts:
            epoch_parts[key] /= max(len(train_loader), 1)

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                bounds = batch["height_bounds"].to(device)
                outputs, output_bounds = model(inputs)
                loss, _ = combined_loss(outputs, output_bounds, targets, bounds)
                val_losses.append(float(loss.item()))

        average_train_loss = float(np.mean(train_losses))
        average_val_loss = float(np.mean(val_losses))
        current_lr = optimizer.param_groups[0]["lr"]

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(average_train_loss)
        history["val_loss"].append(average_val_loss)
        history["components"].append(epoch_parts)
        with open(output_dir / "training_log.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": average_val_loss,
            "metadata": checkpoint_metadata_from_args(args, dataset_roots, len(dataset), len(train_indices), len(val_indices), train_groups, val_groups),
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {average_train_loss:.4f} | Val Loss: {average_val_loss:.4f} | Best: {best_loss:.4f}")
        print(
            "  HM_G: {global_loss:.4f} | HM_L: {local_loss:.4f} | Bounds: {bounds:.4f}".format(
                global_loss=epoch_parts.get("heightmap_global", 0.0),
                local_loss=epoch_parts.get("heightmap_local", 0.0),
                bounds=epoch_parts.get("bounds", 0.0),
            )
        )
        print(
            "  Grad: {gradient:.4f} | SSIM: {ssim:.4f} | Edge: {edge:.4f} | LR: {lr:.2e} | Patience: {patience}/{limit}".format(
                gradient=epoch_parts.get("gradient", 0.0),
                ssim=epoch_parts.get("ssim", 0.0),
                edge=epoch_parts.get("edge", 0.0),
                lr=current_lr,
                patience=patience_counter,
                limit=args.patience,
            )
        )

        scheduler.step(average_val_loss)

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            patience_counter = 0
            torch.save(checkpoint, output_dir / "best.pt")
            print("  Saved best model")
            try:
                if preview_batch is not None:
                    save_training_preview(model, preview_batch, epoch + 1, output_dir / "previews", device)
                else:
                    fallback_batch = next(iter(val_loader))
                    save_training_preview(model, fallback_batch, epoch + 1, output_dir / "previews", device)
            except Exception as exc:
                print(f"  Failed to save preview: {exc}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs")
                break

    print(f"\nTraining complete. Best validation loss: {best_loss:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the V7.1 multichannel terrain regressor.")
    parser.add_argument("--dataset-root", action="append", default=[], help="Explicit dataset root. Repeat for multiple roots.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=[str(path) for path in DEFAULT_DATASET_SEARCH_ROOTS],
        help="Root folder to scan when using an auto-discovery profile.",
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS.keys()), default="development-map")
    parser.add_argument("--include-map", action="append", default=[], help="Restrict training to these map names.")
    parser.add_argument("--exclude-map", action="append", default=[], help="Exclude these map names.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--spatial-group-size", type=int, default=DEFAULT_SPATIAL_GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, help="Optional per-root sample cap for quick debugging.")
    parser.add_argument("--resume", type=str, help="Resume from an existing checkpoint.")
    parser.add_argument("--no-augment", action="store_true", help="Disable RGB jitter and random flips.")
    parser.add_argument("--min-height-range", type=float, default=DEFAULT_MIN_HEIGHT_RANGE,
                        help=f"Skip tiles with less than this height variation in game units (default: {DEFAULT_MIN_HEIGHT_RANGE}).")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
