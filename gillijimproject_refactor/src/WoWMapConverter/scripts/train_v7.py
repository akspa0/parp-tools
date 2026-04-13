#!/usr/bin/env python3
"""
WoW Height Regressor V7.3 - multichannel terrain model with adversarial sharpening.

V7.3 is not a pure minimap-to-height regressor. It works because it combines
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
- brush imprint mask

Outputs:
- global heightmap
- local heightmap
- height bounds head

V7.3 changes over V7.2:
- Replaced ConvTranspose2d upsampling with bilinear + Conv2d to eliminate
  checkerboard artifacts that soften terrain detail.
- Replaced sigmoid output activation with hard clamp — sigmoid squashes
  gradients near 0/1 and kills sharp height extremes.
- Added residual connections in conv blocks for better gradient flow.
- Added lightweight PatchGAN discriminator that enforces local realism in
  heightmap patches, the proven fix for L1 regression blur.
- Same enlarged multi-version corpus and FFT frequency loss from V7.2.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
MODEL_INPUT_CHANNELS = 13
MODEL_OUTPUT_CHANNELS = 2

HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
HEIGHT_GLOBAL_RANGE = HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN

DEFAULT_OUTPUT_DIR = Path("./vlm_output")
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 100
DEFAULT_EARLY_STOP_PATIENCE = 5
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_SPATIAL_GROUP_SIZE = 4
DEFAULT_SEED = 1337
DEFAULT_BLUR_SIGMA = 0.0
DEFAULT_PREVIEW_COUNT = 4
DEFAULT_STATIC_PREVIEW_COUNT = 2
DEFAULT_RANDOM_PREVIEW_COUNT = 2
PREVIEW_MIN_VISUAL_VARIANCE = 0.008
DEFAULT_MIN_HEIGHT_RANGE = 0.5
TILE_SIZE = 533.33333
MAP_ORIGIN = 32.0 * TILE_SIZE
MASK_CONTEXT_MARGIN_TILES = 0.20
MASK_MAX_ABOVE_TERRAIN = 8.0
MASK_MIN_BELOW_TERRAIN = -3.0
DATASET_INDEX_CACHE_VERSION = 1
DATASET_INDEX_CACHE_FILE = ".v7_dataset_index_cache.json"
BRUSH_MANIFEST_FILE = "brush_imprint_manifest.json"
DEFAULT_TRAIN_WORKERS = 4
DEFAULT_VAL_WORKERS = 2
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_LIVE_LOG_EVERY = 20
DEFAULT_AMP_DTYPE = "auto"
DEFAULT_ADVERSARIAL_SCALE = 0.20
DEFAULT_START_GAN_EPOCH = 101
DEFAULT_DISC_EVERY = 2
DEFAULT_DISC_REAL_TARGET = 0.90
DEFAULT_DISC_FAKE_TARGET = 0.10
DEFAULT_DISC_LABEL_NOISE = 0.02
DEFAULT_DISC_INPUT_NOISE_STD = 0.01
DEFAULT_DISC_GRAD_CLIP = 1.0
DEFAULT_GAN_CYCLE_LENGTH = 0
DEFAULT_GAN_CYCLE_ON_EPOCHS = 0
DEFAULT_GAN_COOLDOWN_AFTER_BEST = 0
DEFAULT_GAN_BURST_AFTER_BEST = 0
DEFAULT_EARLY_STOP_START_EPOCH = 1
DEFAULT_LR_PLATEAU_PATIENCE = 8
DEFAULT_LR_PLATEAU_FACTOR = 0.5
DEFAULT_USE_WDL_GLOBAL_TRESTLE = True
DEFAULT_GLOBAL_RESIDUAL_SCALE = 0.20
MODEL_VARIANT_WDL_TRESTLE_REFLECT = "wdl-trestle-reflect-v1"
MODEL_VARIANT_LEGACY = "legacy-absolute-v1"

DEFAULT_DATASET_SEARCH_ROOTS = [
    Path(r"i:\parp\parp-tools\gillijimproject_refactor\test_data\vlm-datasets"),
    Path(r"i:\parp\parp-tools\output\ml-corpus"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets"),
    Path(r"J:\wowDev\parp-tools\output\ml-corpus"),
]

PROFILE_PRESETS = {
    "manual": {
        "description": "Use only explicit --dataset-root values.",
        "include_maps": [],
        "discover": [],
    },
    "development-map": {
        "description": "Prioritize 3.0.1 Northrend, add EmeraldDream cross-version signals, and optionally supplement with 4.0.0.11927 LostIsles.",
        "include_maps": ["Northrend", "LostIsles", "EmeraldDream"],
        "discover": [
            {
                "label": "prototype-emerald-dream",
                "map_tokens": ["emeralddream", "emerald_dream", "emerald dream"],
                "build_tokens": ["070", "0.7.0", "3694", "301", "3.0.1", "8303", "335", "3.3.5", "12340", "400", "4.0.0", "11927"],
            },
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
    "heightmap_global": 0.08,
    "heightmap_local": 0.14,
    "bounds": 0.04,
    "ssim": 0.05,
    "gradient": 0.10,
    "edge": 0.12,
    "frequency": 0.08,
    "adversarial": 0.12,
    "laplacian": 0.12,
    "transition": 0.10,
    "tile_edge": 0.12,
}

EDGE_FOCUS_WIDTH = 12
TRANSITION_FOCUS_GAIN = 3.0

DISCRIMINATOR_LR = 2e-4


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
    brush_mask_path: Optional[Path]
    height_range: float
    object_count: int
    has_liquid: bool


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


def tile_uv_candidates(world_a: float, world_b: float, tile_x: int, tile_y: int) -> List[Tuple[float, float]]:
    """Return plausible tile-local UV candidates for map-space and centered-world conventions."""
    return [
        (world_a / TILE_SIZE - float(tile_x), world_b / TILE_SIZE - float(tile_y)),
        ((MAP_ORIGIN - world_b) / TILE_SIZE - float(tile_x), (MAP_ORIGIN - world_a) / TILE_SIZE - float(tile_y)),
    ]


def is_quarantined_root(root: Path) -> bool:
    marker = "__untrusted_do_not_use"
    return any(marker in part.lower() for part in root.parts)


def collect_explicit_roots(dataset_roots: Sequence[str]) -> List[Path]:
    roots: List[Path] = []
    for root_str in dataset_roots:
        root = Path(root_str)
        if is_quarantined_root(root):
            raise SystemExit(
                f"Refusing quarantined dataset root: {root}. "
                "Use trusted lineage roots only."
            )
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
            if is_quarantined_root(child):
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


class ResConvBlock(nn.Module):
    """Two-conv block with a residual skip when channel counts match."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + identity
        return F.relu(out, inplace=True)


class BilinearUp(nn.Module):
    """Bilinear upsample + 1x1 conv to replace ConvTranspose2d (no checkerboard)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class MultiChannelUNetV7(nn.Module):
    """5-level U-Net for 512x512 multichannel terrain inputs.

    V7.3: residual conv blocks, bilinear upsampling, hard-clamp output.
    """

    def __init__(
        self,
        in_channels: int = MODEL_INPUT_CHANNELS,
        out_channels: int = MODEL_OUTPUT_CHANNELS,
        use_wdl_global_trestle: bool = False,
        global_residual_scale: float = DEFAULT_GLOBAL_RESIDUAL_SCALE,
    ):
        super().__init__()
        self.use_wdl_global_trestle = use_wdl_global_trestle
        self.global_residual_scale = float(global_residual_scale)

        self.enc1 = ResConvBlock(in_channels, 64)
        self.enc2 = ResConvBlock(64, 128)
        self.enc3 = ResConvBlock(128, 256)
        self.enc4 = ResConvBlock(256, 512)
        self.enc5 = ResConvBlock(512, 1024)
        self.bottleneck = ResConvBlock(1024, 2048)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.height_bounds_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

        self.up5 = BilinearUp(2048, 1024)
        self.dec5 = ResConvBlock(2048, 1024)
        self.up4 = BilinearUp(1024, 512)
        self.dec4 = ResConvBlock(1024, 512)
        self.up3 = BilinearUp(512, 256)
        self.dec3 = ResConvBlock(512, 256)
        self.up2 = BilinearUp(256, 128)
        self.dec2 = ResConvBlock(256, 128)
        self.up1 = BilinearUp(128, 64)
        self.dec1 = ResConvBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

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

        # Hard clamp instead of sigmoid — preserves gradients at extremes
        raw_outputs = self.out_conv(dec1)
        global_output = raw_outputs[:, 0:1]
        if self.use_wdl_global_trestle and inputs.shape[1] > 6:
            wdl_base = inputs[:, 6:7]
            global_delta = torch.tanh(global_output) * self.global_residual_scale
            global_output = torch.clamp(wdl_base + global_delta, 0.0, 1.0)
        else:
            global_output = torch.clamp(global_output, 0.0, 1.0)

        local_output = torch.clamp(raw_outputs[:, 1:2], 0.0, 1.0)
        outputs = torch.cat([global_output, local_output], dim=1)
        if outputs.shape[-2:] != (OUTPUT_SIZE, OUTPUT_SIZE):
            outputs = F.interpolate(outputs, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False)

        return outputs, bounds


class PatchDiscriminator(nn.Module):
    """Lightweight 70x70 PatchGAN discriminator.

    Takes the 2-channel heightmap (global + local) and classifies each
    overlapping 70x70 patch as real or generated.  Adds ~2M params and
    ~10-15 % training overhead.
    """

    def __init__(self, in_channels: int = MODEL_OUTPUT_CHANNELS):
        super().__init__()
        self.model = nn.Sequential(
            # 512 -> 256
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 -> 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 63 (stride=1)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 63 -> 62 patch map
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
        preloaded_samples: Optional[Sequence[TileSample]] = None,
    ) -> None:
        self.input_size = input_size
        self.augment = augment
        self.min_height_range = min_height_range
        self.include_maps = {value.lower() for value in include_maps if value}
        self.exclude_maps = {value.lower() for value in exclude_maps if value}
        self.samples: List[TileSample] = list(preloaded_samples) if preloaded_samples is not None else []

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=DEFAULT_BLUR_SIGMA) if DEFAULT_BLUR_SIGMA > 0 else None
        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self._brush_manifest_cache: Dict[Path, Optional[Dict[str, Any]]] = {}

        if preloaded_samples is not None:
            print(f"Reusing preloaded V7 sample index ({len(self.samples)} samples)")
            return

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

        json_paths = sorted(dataset_dir.glob("*.json"))
        signature = self._build_index_signature(json_paths)

        cached_entries, cached_stats = self._load_index_cache(dataset_root, signature)
        if cached_entries is not None:
            entries = cached_entries
            print(f"  {dataset_root.name}: index cache hit ({len(entries)} entries)")
        else:
            entries, cached_stats = self._build_index_entries(json_paths)
            self._save_index_cache(dataset_root, signature, entries, cached_stats)
            print(f"  {dataset_root.name}: index cache rebuilt ({len(entries)} entries)")

        collected: List[TileSample] = []
        blank_skipped = 0
        rejected = {
            "json_read_error": int(cached_stats.get("json_read_error", 0)),
            "tile_name_invalid": int(cached_stats.get("tile_name_invalid", 0)),
            "map_filtered": 0,
            "height_range_too_low": 0,
            "missing_path_refs": 0,
            "missing_input_files": 0,
        }
        for entry in entries:
            tile_name = str(entry["tile_name"])
            map_name = str(entry["map_name"])
            tile_x = int(entry["tile_x"])
            tile_y = int(entry["tile_y"])

            map_key = map_name.lower()
            if self.include_maps and map_key not in self.include_maps:
                rejected["map_filtered"] += 1
                continue
            if map_key in self.exclude_maps:
                rejected["map_filtered"] += 1
                continue

            # Skip blank/flat tiles (ocean, void) that have no useful height variation
            height_min = float(entry.get("height_min", 0.0))
            height_max = float(entry.get("height_max", 0.0))
            if (height_max - height_min) < self.min_height_range:
                blank_skipped += 1
                rejected["height_range_too_low"] += 1
                continue

            heightmap_global_rel = entry.get("heightmap_global")
            heightmap_local_rel = entry.get("heightmap_local")
            normalmap_rel = entry.get("normalmap")
            if not heightmap_global_rel or not heightmap_local_rel or not normalmap_rel:
                rejected["missing_path_refs"] += 1
                continue

            minimap_rel = entry.get("image") or entry.get("no_object_minimap")
            if minimap_rel:
                minimap_path = dataset_root / str(minimap_rel)
            else:
                minimap_path = dataset_root / "images" / f"{tile_name}.png"

            if not minimap_path.exists() and entry.get("no_object_minimap"):
                fallback_minimap = entry.get("no_object_minimap")
                if fallback_minimap:
                    minimap_path = dataset_root / str(fallback_minimap)

            normalmap_path = dataset_root / normalmap_rel
            heightmap_global_path = dataset_root / heightmap_global_rel
            heightmap_local_path = dataset_root / heightmap_local_rel
            if not minimap_path.exists() or not normalmap_path.exists() or not heightmap_global_path.exists() or not heightmap_local_path.exists():
                rejected["missing_input_files"] += 1
                continue

            liquid_mask_rel = entry.get("liquid_mask")
            liquid_height_rel = entry.get("liquid_height")
            liquid_mask_path = dataset_root / str(liquid_mask_rel) if liquid_mask_rel else None
            liquid_height_path = dataset_root / str(liquid_height_rel) if liquid_height_rel else None
            brush_mask_path = self._resolve_brush_mask_path(dataset_root, tile_name)

            tile_object_count = int(entry.get("object_count", 0))
            tile_has_liquid = bool(liquid_mask_path and liquid_mask_path.exists())

            json_name = str(entry.get("json_name", f"{tile_name}.json"))
            json_path = dataset_dir / json_name

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
                    brush_mask_path=brush_mask_path,
                    height_range=height_max - height_min,
                    object_count=tile_object_count,
                    has_liquid=tile_has_liquid,
                )
            )

            if limit is not None and len(collected) >= limit:
                break

        print(
            f"  {dataset_root.name}: {len(collected)} usable samples ({blank_skipped} blank skipped; "
            f"missing refs {rejected['missing_path_refs']}, missing files {rejected['missing_input_files']}, "
            f"filtered {rejected['map_filtered']}, invalid tile ids {rejected['tile_name_invalid']})"
        )
        return collected, blank_skipped

    def _build_index_signature(self, json_paths: Sequence[Path]) -> Dict[str, int]:
        latest_mtime_ns = 0
        total_size = 0
        for path in json_paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
            total_size += int(stat.st_size)

        return {
            "json_count": len(json_paths),
            "latest_mtime_ns": latest_mtime_ns,
            "total_size": total_size,
        }

    def _cache_file_path(self, dataset_root: Path) -> Path:
        return dataset_root / DATASET_INDEX_CACHE_FILE

    def _load_index_cache(
        self,
        dataset_root: Path,
        signature: Dict[str, int],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, int]]:
        cache_path = self._cache_file_path(dataset_root)
        if not cache_path.exists():
            return None, {"json_read_error": 0, "tile_name_invalid": 0}

        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None, {"json_read_error": 0, "tile_name_invalid": 0}

        if int(payload.get("version", -1)) != DATASET_INDEX_CACHE_VERSION:
            return None, {"json_read_error": 0, "tile_name_invalid": 0}

        cached_signature = payload.get("signature") or {}
        if (
            int(cached_signature.get("json_count", -1)) != int(signature["json_count"])
            or int(cached_signature.get("latest_mtime_ns", -1)) != int(signature["latest_mtime_ns"])
            or int(cached_signature.get("total_size", -1)) != int(signature["total_size"])
        ):
            return None, {"json_read_error": 0, "tile_name_invalid": 0}

        entries = payload.get("entries")
        if not isinstance(entries, list):
            return None, {"json_read_error": 0, "tile_name_invalid": 0}

        stats = payload.get("build_stats") or {}
        return entries, {
            "json_read_error": int(stats.get("json_read_error", 0)),
            "tile_name_invalid": int(stats.get("tile_name_invalid", 0)),
        }

    def _save_index_cache(
        self,
        dataset_root: Path,
        signature: Dict[str, int],
        entries: List[Dict[str, Any]],
        build_stats: Dict[str, int],
    ) -> None:
        cache_path = self._cache_file_path(dataset_root)
        payload = {
            "version": DATASET_INDEX_CACHE_VERSION,
            "signature": signature,
            "entries": entries,
            "build_stats": build_stats,
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:
            # Cache write failure should never block training.
            return

    def _load_brush_manifest(self, dataset_root: Path) -> Optional[Dict[str, Any]]:
        if dataset_root in self._brush_manifest_cache:
            return self._brush_manifest_cache[dataset_root]

        manifest_path = dataset_root / "brush_imprints" / BRUSH_MANIFEST_FILE
        if not manifest_path.exists():
            self._brush_manifest_cache[dataset_root] = None
            return None

        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = None

        self._brush_manifest_cache[dataset_root] = manifest
        return manifest

    def _resolve_brush_mask_path(self, dataset_root: Path, tile_name: str) -> Optional[Path]:
        manifest = self._load_brush_manifest(dataset_root)
        if not manifest:
            return None

        for tile in manifest.get("tiles", []):
            if str(tile.get("tile_name", "")) != tile_name:
                continue

            rel = tile.get("brush_mask_path")
            if not rel:
                return None

            candidate = dataset_root / "brush_imprints" / str(rel)
            return candidate if candidate.exists() else None

        return None

    def _build_index_entries(self, json_paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        entries: List[Dict[str, Any]] = []
        stats = {"json_read_error": 0, "tile_name_invalid": 0}

        for json_path in json_paths:
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                stats["json_read_error"] += 1
                continue

            terrain = payload.get("terrain_data", {})
            tile_name = terrain.get("adt_tile") or json_path.stem

            try:
                map_name, tile_x, tile_y = parse_tile_identity(str(tile_name))
            except ValueError:
                stats["tile_name_invalid"] += 1
                continue

            heightmap_global_rel = terrain.get("heightmap_global") or terrain.get("heightmap")
            heightmap_local_rel = terrain.get("heightmap_local") or terrain.get("heightmap")
            normalmap_rel = terrain.get("normalmap")
            objects = terrain.get("objects")

            entries.append(
                {
                    "json_name": json_path.name,
                    "tile_name": str(tile_name),
                    "map_name": map_name,
                    "tile_x": int(tile_x),
                    "tile_y": int(tile_y),
                    "height_min": float(terrain.get("height_min", 0.0)),
                    "height_max": float(terrain.get("height_max", 0.0)),
                    "image": payload.get("image"),
                    "no_object_minimap": terrain.get("no_object_minimap"),
                    "normalmap": normalmap_rel,
                    "heightmap_global": heightmap_global_rel,
                    "heightmap_local": heightmap_local_rel,
                    "liquid_mask": terrain.get("liquid_mask"),
                    "liquid_height": terrain.get("liquid_height"),
                    "object_count": len(objects) if isinstance(objects, list) else 0,
                }
            )

        return entries, stats

    def __len__(self) -> int:
        return len(self.samples)

    def _render_wdl(self, wdl_data: Optional[Dict[str, object]], global_min: float, global_max: float) -> torch.Tensor:
        if not wdl_data:
            return torch.full((1, self.input_size, self.input_size), 0.5)

        outer = np.asarray(wdl_data.get("outer_17", []), dtype=np.float32)
        if len(outer) != 289:
            return torch.full((1, self.input_size, self.input_size), 0.5)

        grid = outer.reshape(17, 17)
        global_range = max(global_max - global_min, 1e-6)
        grid = np.clip((grid - global_min) / global_range, 0.0, 1.0)

        image = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        return self.to_tensor(image)

    def _load_optional_binary_mask(self, dataset_root: Path, terrain: Dict[str, object], keys: Sequence[str]) -> torch.Tensor:
        for key in keys:
            rel = terrain.get(key)
            if not rel:
                continue
            candidate = dataset_root / str(rel)
            if not candidate.exists():
                continue
            mask_image = Image.open(candidate).convert("L").resize((self.input_size, self.input_size), Image.NEAREST)
            return (self.to_tensor(mask_image) > 0.1).float()

        return torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)

    def _build_wdl_height_sampler(
        self,
        wdl_data: Optional[Dict[str, object]],
    ) -> Optional[Callable[[float, float, Optional[float]], Optional[float]]]:
        if not wdl_data:
            return None

        outer = np.asarray(wdl_data.get("outer_17", []), dtype=np.float32)
        if len(outer) != 289 or not np.all(np.isfinite(outer)):
            return None

        grid = outer.reshape(17, 17)

        def bilinear(sample_x: float, sample_y: float) -> float:
            x = float(np.clip(sample_x, 0.0, 16.0))
            y = float(np.clip(sample_y, 0.0, 16.0))
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(16, x0 + 1)
            y1 = min(16, y0 + 1)
            tx = x - x0
            ty = y - y0

            v00 = float(grid[y0, x0])
            v01 = float(grid[y1, x0])
            v10 = float(grid[y0, x1])
            v11 = float(grid[y1, x1])
            return (
                v00 * (1.0 - tx) * (1.0 - ty)
                + v10 * tx * (1.0 - ty)
                + v01 * (1.0 - tx) * ty
                + v11 * tx * ty
            )

        def sample(local_x: float, local_y: float, reference_height: Optional[float] = None) -> Optional[float]:
            gx = float(np.clip(local_x, 0.0, 1.0)) * 16.0
            gy = float(np.clip(local_y, 0.0, 1.0)) * 16.0

            height_xy = bilinear(gx, gy)
            height_yx = bilinear(gy, gx)

            if reference_height is None or not np.isfinite(reference_height):
                return height_xy

            if abs(reference_height - height_yx) < abs(reference_height - height_xy):
                return height_yx

            return height_xy

        return sample

    def _build_object_mask(
        self,
        objects: Optional[Sequence[Dict[str, object]]],
        tile_x: int,
        tile_y: int,
        wdl_heights: Optional[Dict[str, object]],
    ) -> torch.Tensor:
        object_mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        if not objects:
            return object_mask

        image = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        pixels_per_world = self.input_size / TILE_SIZE
        wdl_sampler = self._build_wdl_height_sampler(wdl_heights)
        for obj in objects:
            pos_x = float(obj.get("x", obj.get("pos_x", 0.0)))
            pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
            pos_z = float(obj.get("z", obj.get("pos_z", pos_y)))
            scale = float(obj.get("scale", 1.0))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0

            candidate_uvs: List[Tuple[float, float]] = []
            if abs(pos_x) < 2 and abs(pos_y) < 2:
                # Legacy fallback for normalized tile-local coordinates.
                candidate_uvs.append(((pos_y + 1.0) * 0.5, (pos_x + 1.0) * 0.5))

            candidate_uvs.extend(tile_uv_candidates(pos_x, pos_z, tile_x, tile_y))
            if np.isfinite(pos_y):
                candidate_uvs.extend(tile_uv_candidates(pos_x, pos_y, tile_x, tile_y))

            local_x = 0.0
            local_y = 0.0
            best_overflow = float("inf")
            for cand_x, cand_y in candidate_uvs:
                overflow = (
                    max(0.0, -cand_x)
                    + max(0.0, cand_x - 1.0)
                    + max(0.0, -cand_y)
                    + max(0.0, cand_y - 1.0)
                )
                if overflow < best_overflow:
                    best_overflow = overflow
                    local_x = cand_x
                    local_y = cand_y
                if overflow <= 1e-6:
                    break

            if (
                local_x < -MASK_CONTEXT_MARGIN_TILES
                or local_x > 1.0 + MASK_CONTEXT_MARGIN_TILES
                or local_y < -MASK_CONTEXT_MARGIN_TILES
                or local_y > 1.0 + MASK_CONTEXT_MARGIN_TILES
            ):
                continue

            center_x = int(round(local_x * (self.input_size - 1)))
            center_y = int(round(local_y * (self.input_size - 1)))

            category = str(obj.get("category", "")).lower()
            bounds_min = obj.get("bounds_min")
            bounds_max = obj.get("bounds_max")
            if bounds_min and bounds_max and len(bounds_min) >= 3 and len(bounds_max) >= 3:
                half_width_world = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
                half_depth_world = abs(float(bounds_max[2]) - float(bounds_min[2])) * 0.5 * scale
                radius_x = max(1, int(round(half_width_world * pixels_per_world)))
                radius_y = max(1, int(round(half_depth_world * pixels_per_world)))
            elif bounds_min and bounds_max and len(bounds_min) >= 2 and len(bounds_max) >= 2:
                half_width_world = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
                half_depth_world = abs(float(bounds_max[1]) - float(bounds_min[1])) * 0.5 * scale
                radius_x = max(1, int(round(half_width_world * pixels_per_world)))
                radius_y = max(1, int(round(half_depth_world * pixels_per_world)))
            else:
                base_radius_world = 3.0 * scale
                if "wmo" in category:
                    base_radius_world *= 2.0
                radius_x = max(1, int(round(base_radius_world * pixels_per_world)))
                radius_y = radius_x

            is_wmo = "wmo" in category
            if not is_wmo:
                # Minimap captures are WMO-focused; skip M2/doodad mask contribution.
                continue

            if np.isfinite(pos_y) and wdl_sampler is not None:
                terrain_height = wdl_sampler(local_x, local_y, pos_y)
                if terrain_height is not None and np.isfinite(terrain_height):
                    delta = float(pos_y - terrain_height)
                    if delta < MASK_MIN_BELOW_TERRAIN or delta > MASK_MAX_ABOVE_TERRAIN:
                        continue

            x1 = max(0, center_x - radius_x)
            y1 = max(0, center_y - radius_y)
            x2 = min(self.input_size, center_x + radius_x + 1)
            y2 = min(self.input_size, center_y + radius_y + 1)
            if x1 >= x2 or y1 >= y2:
                continue
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

        if self.blur is not None:
            minimap = self.blur(minimap)
        if self.augment:
            minimap = self.color_jitter(minimap)

        height_min = float(terrain.get("height_min", 0.0))
        height_max = float(terrain.get("height_max", 100.0))
        global_min = float(terrain.get("height_global_min", HEIGHT_GLOBAL_MIN))
        global_max = float(terrain.get("height_global_max", HEIGHT_GLOBAL_MAX))
        global_range = max(global_max - global_min, 1e-6)

        minimap_tensor = self.normalize(self.to_tensor(minimap))
        normalmap_tensor = self.normalize(self.to_tensor(normalmap))
        wdl_tensor = self._render_wdl(terrain.get("wdl_heights"), global_min, global_max)

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

        brush_mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        if sample.brush_mask_path and sample.brush_mask_path.exists():
            brush_image = Image.open(sample.brush_mask_path).convert("L").resize((self.input_size, self.input_size), Image.NEAREST)
            brush_tensor = self.to_tensor(brush_image)
            brush_mask = (brush_tensor > 0.1).float()

        object_mask = self._build_object_mask(
            terrain.get("objects"),
            sample.tile_x,
            sample.tile_y,
            terrain.get("wdl_heights"),
        )
        pm4_mask = self._load_optional_binary_mask(
            sample.dataset_root,
            terrain,
            keys=["object_visibility_mask_cv2", "object_visibility_mask", "pm4_mask", "pm4_object_mask", "collision_mask"],
        )
        object_mask = torch.maximum(object_mask, pm4_mask)
        # NOTE: object_mask is passed as ch11 input context only.
        # We do NOT blank minimap/normal map here because _build_object_mask produces
        # inaccurate footprints (WMOs have no bounds in the JSON, fallback radius is ~6px
        # while real buildings are hundreds of yards wide, and coordinate mapping is unreliable).
        # Masking with bad silhouettes is worse than not masking at all.
        # Fix required upstream: exporter must render real WMO footprint images.

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
                brush_mask,
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
    predicted = predicted.float()
    target = target.float()
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

    sigma_pred_sq = torch.clamp(
        F.conv2d(predicted * predicted, window, padding=window_size // 2, groups=predicted.shape[1]) - mu_pred_sq,
        min=0.0,
    )
    sigma_target_sq = torch.clamp(
        F.conv2d(target * target, window, padding=window_size // 2, groups=target.shape[1]) - mu_target_sq,
        min=0.0,
    )
    sigma_pred_target = F.conv2d(predicted * target, window, padding=window_size // 2, groups=predicted.shape[1]) - mu_pred_target

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = torch.clamp(
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2),
        min=1e-8,
    )
    ssim_map = torch.clamp(numerator / denominator, min=-1.0, max=1.0)
    return torch.clamp(1 - ssim_map.mean(), min=0.0)


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


def frequency_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize differences in frequency-domain magnitude to preserve high-frequency detail.

    Uses log1p on FFT magnitudes so that high-frequency content (ridges, cliffs,
    terrain edges) receives fair weight relative to the dominant low-frequency bulk.
    """
    # Keep FFT in float32 for numerical stability under AMP.
    pred_float = predicted[:, :2].float()
    target_float = target[:, :2].float()
    pred_fft = torch.fft.rfft2(pred_float)
    target_fft = torch.fft.rfft2(target_float)
    pred_mag = torch.log1p(pred_fft.abs())
    target_mag = torch.log1p(target_fft.abs())
    return F.l1_loss(pred_mag, target_mag)


def laplacian_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize differences in second-order derivatives (curvature) to preserve sharp terrain features.

    The Laplacian captures ridgelines, cliff edges, and terrain curvature that
    first-order gradient losses miss.  Proven to improve sharpness in image
    super-resolution and depth estimation tasks.
    """
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=predicted.device
    ).view(1, 1, 3, 3)

    def apply_laplacian(tensor: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(tensor)
        for ch in range(tensor.shape[1]):
            result[:, ch : ch + 1] = F.conv2d(tensor[:, ch : ch + 1], kernel, padding=1)
        return result

    return F.l1_loss(apply_laplacian(predicted[:, :2]), apply_laplacian(target[:, :2]))


def weighted_l1_loss(predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted_error = (predicted - target).abs() * weights
    return weighted_error.sum() / torch.clamp(weights.sum(), min=1e-6)


def transition_focus_loss(predicted: torch.Tensor, target: torch.Tensor, gain: float = TRANSITION_FOCUS_GAIN) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    weight_maps = []
    for channel in range(target.shape[1]):
        current = target[:, channel:channel + 1]
        grad_x = F.conv2d(current, sobel_x, padding=1)
        grad_y = F.conv2d(current, sobel_y, padding=1)
        magnitude = torch.sqrt(torch.clamp(grad_x * grad_x + grad_y * grad_y, min=0.0))
        normalized = magnitude / (magnitude.mean(dim=(2, 3), keepdim=True) + 1e-6)
        weight_maps.append(1.0 + gain * torch.clamp(normalized, min=0.0, max=1.0))

    weights = torch.cat(weight_maps, dim=1)
    return weighted_l1_loss(predicted[:, :2], target[:, :2], weights)


def tile_edge_loss(predicted: torch.Tensor, target: torch.Tensor, edge_width: int = EDGE_FOCUS_WIDTH) -> torch.Tensor:
    if edge_width <= 0:
        return torch.zeros((), dtype=predicted.dtype, device=predicted.device)

    _, channels, height, width = predicted[:, :2].shape
    border_mask = torch.zeros((1, 1, height, width), dtype=predicted.dtype, device=predicted.device)
    border_mask[:, :, :edge_width, :] = 1.0
    border_mask[:, :, -edge_width:, :] = 1.0
    border_mask[:, :, :, :edge_width] = 1.0
    border_mask[:, :, :, -edge_width:] = 1.0
    border_mask = border_mask.expand(predicted.shape[0], channels, height, width)
    return weighted_l1_loss(predicted[:, :2], target[:, :2], border_mask)


def combined_loss(
    predicted_heightmap: torch.Tensor,
    predicted_bounds: torch.Tensor,
    target_heightmap: torch.Tensor,
    target_bounds: torch.Tensor,
    adv_loss: Optional[torch.Tensor] = None,
    adversarial_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    predicted_heightmap = predicted_heightmap.float()
    predicted_bounds = predicted_bounds.float()
    target_heightmap = target_heightmap.float()
    target_bounds = target_bounds.float()

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
    frequency_component = frequency_loss(predicted_heightmap, target_heightmap)
    laplacian_component = laplacian_loss(predicted_heightmap, target_heightmap)
    transition_component = transition_focus_loss(predicted_heightmap, target_heightmap)
    tile_edge_component = tile_edge_loss(predicted_heightmap, target_heightmap)

    total = (
        LOSS_WEIGHTS["heightmap_global"] * global_loss
        + LOSS_WEIGHTS["heightmap_local"] * local_loss
        + LOSS_WEIGHTS["bounds"] * bounds_loss
        + LOSS_WEIGHTS["gradient"] * gradient_component
        + LOSS_WEIGHTS["ssim"] * ssim_component
        + LOSS_WEIGHTS["edge"] * edge_component
        + LOSS_WEIGHTS["frequency"] * frequency_component
        + LOSS_WEIGHTS["laplacian"] * laplacian_component
        + LOSS_WEIGHTS["transition"] * transition_component
        + LOSS_WEIGHTS["tile_edge"] * tile_edge_component
    )

    adv_value = 0.0
    if adv_loss is not None:
        adv_loss = adv_loss.float()
        total = total + (LOSS_WEIGHTS["adversarial"] * adversarial_scale) * adv_loss
        adv_value = float(adv_loss.item())

    return total, {
        "heightmap_global": float(global_loss.item()),
        "heightmap_local": float(local_loss.item()),
        "bounds": float(bounds_loss.item()),
        "gradient": float(gradient_component.item()),
        "ssim": float(ssim_component.item()),
        "edge": float(edge_component.item()),
        "frequency": float(frequency_component.item()),
        "laplacian": float(laplacian_component.item()),
        "transition": float(transition_component.item()),
        "tile_edge": float(tile_edge_component.item()),
        "adversarial": adv_value,
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


def curate_training_set(
    samples: Sequence[TileSample],
    train_indices: List[int],
    seed: int,
) -> List[int]:
    """Build a complexity-balanced training subset.

    Scores each sample by terrain complexity (height variation, objects,
    water presence) and selects a balanced subset that over-represents
    interesting terrain while keeping some simple tiles for grounding.
    """
    if len(train_indices) < 100:
        return train_indices

    rng = random.Random(seed + 7)
    max_range = max((samples[i].height_range for i in train_indices), default=1.0)
    max_range = max(max_range, 1.0)

    scored: List[Tuple[int, float]] = []
    for idx in train_indices:
        s = samples[idx]
        score = (
            (s.height_range / max_range) * 0.55
            + min(s.object_count / 30.0, 1.0) * 0.15
            + (0.30 if s.has_liquid else 0.0)
        )
        scored.append((idx, score))

    scored.sort(key=lambda x: -x[1])
    n = len(scored)

    high_cutoff = int(n * 0.30)
    mid_cutoff = int(n * 0.75)

    tier_high = [idx for idx, _ in scored[:high_cutoff]]
    tier_mid = [idx for idx, _ in scored[high_cutoff:mid_cutoff]]
    tier_low = [idx for idx, _ in scored[mid_cutoff:]]
    water_tiles = {idx for idx in train_indices if samples[idx].has_liquid}

    rng.shuffle(tier_mid)
    rng.shuffle(tier_low)

    selected: set[int] = set(tier_high)
    selected.update(water_tiles)
    selected.update(tier_mid[: int(len(tier_mid) * 0.50)])
    selected.update(tier_low[: int(len(tier_low) * 0.20)])

    curated = sorted(selected)
    print(f"  Curation: {n} -> {len(curated)} training tiles ({len(curated) / n * 100:.0f}%)")
    print(
        f"    High complexity: {len(tier_high)} | Water: {len(water_tiles)}"
        f" | Mid 50%: {int(len(tier_mid) * 0.50)} | Low 20%: {int(len(tier_low) * 0.20)}"
    )
    return curated


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


def select_epoch_preview_indices(
    samples: Sequence[TileSample],
    val_indices: Sequence[int],
    static_indices: Sequence[int],
    epoch_number: int,
    static_preview_count: int,
    random_preview_count: int,
    seed: int,
) -> List[int]:
    selected: List[int] = []

    for index in static_indices[: max(static_preview_count, 0)]:
        if index not in selected:
            selected.append(index)

    if random_preview_count > 0:
        pool = [index for index in val_indices if index not in selected]
        rng = random.Random(seed + 100_003 + epoch_number)
        rng.shuffle(pool)
        for index in pool[:random_preview_count]:
            if index not in selected:
                selected.append(index)

    if not selected:
        fallback_count = max(1, static_preview_count + random_preview_count)
        for index in list(val_indices)[:fallback_count]:
            if index not in selected:
                selected.append(index)

    return selected


def save_training_preview(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    epoch: int,
    output_dir: Path,
    device: torch.device,
    preview_labels: Optional[Sequence[str]] = None,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions, _ = model(inputs)

    rows_global: List[torch.Tensor] = []
    rows_local: List[torch.Tensor] = []
    rows_context: List[torch.Tensor] = []
    sample_count = min(DEFAULT_PREVIEW_COUNT, inputs.shape[0])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    blue = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
    amber = torch.tensor([1.0, 0.65, 0.0]).view(3, 1, 1)

    def normalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor - tensor.min()
        return tensor / (tensor.max() + 1e-6)

    for index in range(sample_count):
        minimap = torch.clamp(inputs[index, 0:3].cpu() * std + mean, 0.0, 1.0)
        normal = torch.clamp(inputs[index, 3:6].cpu() * std + mean, 0.0, 1.0)
        water_mask = inputs[index, 9:10].cpu()
        water = water_mask.repeat(3, 1, 1) * blue
        object_mask = inputs[index, 11:12].cpu()
        brush_mask = inputs[index, 12:13].cpu()
        object_overlay = torch.clamp(minimap * (1.0 - 0.65 * object_mask) + red * object_mask * 0.85, 0.0, 1.0)
        masked_minimap = torch.clamp(minimap * (1.0 - object_mask), 0.0, 1.0)
        object_mask_rgb = object_mask.repeat(3, 1, 1) * red
        brush_mask_rgb = brush_mask.repeat(3, 1, 1) * amber

        # Global channel (ch 0): absolute height — most readable in early training
        pred_global = normalize_for_display(predictions[index, 0:1].cpu()).repeat(3, 1, 1)
        gt_global = normalize_for_display(targets[index, 0:1].cpu()).repeat(3, 1, 1)
        rows_global.append(torch.cat([minimap, normal, water, pred_global, gt_global], dim=2))

        # Local channel (ch 1): within-tile normalized detail
        pred_local = normalize_for_display(predictions[index, 1:2].cpu()).repeat(3, 1, 1)
        gt_local = normalize_for_display(targets[index, 1:2].cpu()).repeat(3, 1, 1)
        rows_local.append(torch.cat([minimap, normal, water, pred_local, gt_local], dim=2))

        rows_context.append(
            torch.cat([minimap, object_overlay, masked_minimap, object_mask_rgb, water, brush_mask_rgb], dim=2)
        )

    global_grid = torch.clamp(torch.cat(rows_global, dim=1), 0.0, 1.0)
    local_grid = torch.clamp(torch.cat(rows_local, dim=1), 0.0, 1.0)
    context_grid = torch.clamp(torch.cat(rows_context, dim=1), 0.0, 1.0)
    transforms.ToPILImage()(global_grid).save(output_dir / f"val_epoch_{epoch:04d}.png")
    transforms.ToPILImage()(local_grid).save(output_dir / f"val_epoch_{epoch:04d}_local.png")
    transforms.ToPILImage()(context_grid).save(output_dir / f"val_epoch_{epoch:04d}_context.png")
    if preview_labels is not None:
        metadata = {
            "epoch": epoch,
            "labels": list(preview_labels)[:sample_count],
            "context_columns": [
                "minimap",
                "object_overlay",
                "masked_minimap",
                "object_mask",
                "liquid_mask",
                "brush_mask",
            ],
        }
        with open(output_dir / f"val_epoch_{epoch:04d}.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


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
        "model_variant": MODEL_VARIANT_WDL_TRESTLE_REFLECT if DEFAULT_USE_WDL_GLOBAL_TRESTLE else MODEL_VARIANT_LEGACY,
        "use_wdl_global_trestle": DEFAULT_USE_WDL_GLOBAL_TRESTLE,
        "global_residual_scale": DEFAULT_GLOBAL_RESIDUAL_SCALE,
        "conv_padding_mode": "reflect",
        "blur_sigma": DEFAULT_BLUR_SIGMA,
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
        "adversarial_scale": args.adversarial_scale,
        "start_gan_epoch": args.start_gan_epoch,
        "gan_cycle_length": args.gan_cycle_length,
        "gan_cycle_on_epochs": args.gan_cycle_on_epochs,
        "gan_cooldown_after_best": args.gan_cooldown_after_best,
        "gan_burst_after_best": args.gan_burst_after_best,
    }


def resolve_model_architecture_from_metadata(metadata: Optional[Dict[str, object]]) -> Tuple[bool, float]:
    if not metadata:
        return False, DEFAULT_GLOBAL_RESIDUAL_SCALE

    use_wdl_global_trestle = bool(metadata.get("use_wdl_global_trestle", False))
    global_residual_scale = float(metadata.get("global_residual_scale", DEFAULT_GLOBAL_RESIDUAL_SCALE))
    return use_wdl_global_trestle, global_residual_scale


def resolve_gan_schedule(epoch_number: int, args: argparse.Namespace, gan_schedule_remaining: int) -> Tuple[bool, str]:
    if args.adversarial_scale <= 0.0:
        return False, "disabled"
    if args.gan_burst_after_best > 0:
        if gan_schedule_remaining > 0:
            return True, f"best-burst({gan_schedule_remaining})"
        return False, "waiting-for-best"
    if epoch_number < args.start_gan_epoch:
        return False, f"warmup-until-{args.start_gan_epoch}"
    if gan_schedule_remaining > 0:
        return False, f"cooldown({gan_schedule_remaining})"
    if args.gan_cycle_length > 0 and args.gan_cycle_on_epochs > 0:
        cycle_on_epochs = min(args.gan_cycle_on_epochs, args.gan_cycle_length)
        cycle_offset = (epoch_number - args.start_gan_epoch) % args.gan_cycle_length
        gan_enabled = cycle_offset < cycle_on_epochs
        return gan_enabled, f"cycle({cycle_offset + 1}/{args.gan_cycle_length})"
    return True, "steady"


def build_discriminator_targets(prediction: torch.Tensor, target_value: float, label_noise: float) -> torch.Tensor:
    targets = torch.full_like(prediction, target_value)
    if label_noise > 0.0:
        jitter = (torch.rand_like(prediction) * 2.0 - 1.0) * label_noise
        targets = torch.clamp(targets + jitter, 0.0, 1.0)
    return targets


def apply_discriminator_input_noise(tensor: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0.0:
        return tensor
    return torch.clamp(tensor + torch.randn_like(tensor) * noise_std, 0.0, 1.0)


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
    print("WoW V7.3 Training - GAN + curated data + laplacian sharpness")
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

    if not args.no_curate:
        train_indices = curate_training_set(dataset.samples, train_indices, args.seed)
        print(f"Curated train samples: {len(train_indices)}")

    train_dataset = Subset(dataset, train_indices)
    val_base_dataset = WoWTileDatasetV7(
        dataset_roots=dataset_roots,
        include_maps=include_maps,
        exclude_maps=args.exclude_map,
        input_size=INPUT_SIZE,
        augment=False,
        limit=args.limit,
        min_height_range=args.min_height_range,
        preloaded_samples=dataset.samples,
    )
    val_dataset = Subset(val_base_dataset, val_indices)

    use_cuda = torch.cuda.is_available()

    train_loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.train_workers,
        "pin_memory": use_cuda,
    }
    if args.train_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    val_loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.val_workers,
        "pin_memory": use_cuda,
    }
    if args.val_workers > 0:
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    static_preview_indices: List[int] = []
    try:
        preview_candidates = select_preview_candidates(
            val_base_dataset.samples,
            val_indices,
            max(args.static_preview_count, 1),
        )
        static_preview_indices = [index for index, _, _ in preview_candidates]
        _, loaded_preview_indices, skipped_preview_indices = build_preview_batch(val_base_dataset, static_preview_indices)
        loaded_preview_index_set = set(loaded_preview_indices)

        print("Static preview tiles (ranked for visual signal):")
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
        print(f"Warning: failed to precompute static preview candidates, falling back to validation order: {exc}")

    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        if args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    amp_dtype: Optional[torch.dtype] = None
    if use_cuda and not args.no_amp:
        if args.amp_dtype == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    use_amp = amp_dtype is not None
    print(f"Training on {device}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        amp_name = str(amp_dtype).split(".")[-1] if amp_dtype is not None else "off"
        print(
            f"GPU: {gpu_name} | AMP: {amp_name} | "
            f"TF32(matmul/cudnn): {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}/"
            f"{'on' if torch.backends.cudnn.allow_tf32 else 'off'} | "
            f"cuDNN benchmark: {'on' if torch.backends.cudnn.benchmark else 'off'}"
        )
    print(
        "DataLoader config: "
        f"train_workers={args.train_workers}, val_workers={args.val_workers}, "
        f"prefetch_factor={args.prefetch_factor}, pin_memory={'on' if use_cuda else 'off'}"
    )

    resume_checkpoint: Optional[Dict[str, Any]] = None
    use_wdl_global_trestle = DEFAULT_USE_WDL_GLOBAL_TRESTLE
    global_residual_scale = DEFAULT_GLOBAL_RESIDUAL_SCALE
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        use_wdl_global_trestle, global_residual_scale = resolve_model_architecture_from_metadata(
            dict(resume_checkpoint.get("metadata", {}))
        )

    model = MultiChannelUNetV7(
        use_wdl_global_trestle=use_wdl_global_trestle,
        global_residual_scale=global_residual_scale,
    ).to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.lr_plateau_patience,
        factor=args.lr_plateau_factor,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_cuda else None

    print(
        "Fine-tune config: "
        f"adv_scale={args.adversarial_scale:.3f}, "
        f"start_gan_epoch={args.start_gan_epoch}, "
        f"gan_cycle={args.gan_cycle_on_epochs}/{args.gan_cycle_length if args.gan_cycle_length > 0 else 0}, "
        f"gan_cooldown_after_best={args.gan_cooldown_after_best}, "
        f"gan_burst_after_best={args.gan_burst_after_best}, "
        f"early_stop_start_epoch={args.early_stop_start_epoch}, "
        f"disc_every={args.disc_every}, "
        f"disc_lr={args.disc_learning_rate:.2e}, "
        f"disc_targets={args.disc_real_target:.2f}/{args.disc_fake_target:.2f}, "
        f"disc_label_noise={args.disc_label_noise:.3f}, "
        f"disc_input_noise_std={args.disc_input_noise_std:.3f}, "
        f"lr_plateau_patience={args.lr_plateau_patience}"
    )

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
    gan_schedule_remaining = 0

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = resume_checkpoint if resume_checkpoint is not None else torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("val_loss", best_loss))
        patience_counter = int(checkpoint.get("patience_counter", patience_counter))
        gan_schedule_remaining = int(
            checkpoint.get(
                "gan_schedule_remaining",
                checkpoint.get(
                    "gan_refinement_remaining",
                    checkpoint.get("gan_cooldown_remaining", gan_schedule_remaining),
                ),
            )
        )
        if not args.no_resume_optimizer:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "disc_optimizer_state_dict" in checkpoint:
                disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
        print("Resume optimizer state: enabled" if not args.no_resume_optimizer else "Resume optimizer state: disabled (--no-resume-optimizer)")
        print(f"Resume GAN schedule remaining: {gan_schedule_remaining}")

    for epoch in range(start_epoch, args.epochs):
        epoch_number = epoch + 1
        gan_enabled_epoch, gan_phase = resolve_gan_schedule(epoch_number, args, gan_schedule_remaining)
        gan_schedule_active_this_epoch = gan_schedule_remaining > 0
        model.train()
        discriminator.train()
        train_losses: List[float] = []
        disc_losses: List[float] = []
        disc_real_scores: List[float] = []
        disc_fake_scores: List[float] = []
        epoch_parts: Dict[str, float] = {}

        epoch_train_start = time.perf_counter()
        progress = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step_index, batch in progress:
            inputs = batch["input"].to(device, non_blocking=use_cuda)
            targets = batch["target"].to(device, non_blocking=use_cuda)
            bounds = batch["height_bounds"].to(device, non_blocking=use_cuda)

            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

            # --- Generator forward ---
            with amp_context:
                outputs, output_bounds = model(inputs)

            # --- Discriminator step ---
            do_disc_step = args.disc_every <= 1 or (step_index % args.disc_every == 0)
            if gan_enabled_epoch and do_disc_step:
                disc_optimizer.zero_grad(set_to_none=True)
                with amp_context:
                    real_disc_input = apply_discriminator_input_noise(targets, args.disc_input_noise_std)
                    fake_disc_input = apply_discriminator_input_noise(outputs.detach(), args.disc_input_noise_std)
                    real_pred = discriminator(real_disc_input)
                    fake_pred = discriminator(fake_disc_input)
                    real_targets = build_discriminator_targets(real_pred, args.disc_real_target, args.disc_label_noise)
                    fake_targets = build_discriminator_targets(fake_pred, args.disc_fake_target, args.disc_label_noise)
                    disc_real_loss = F.mse_loss(real_pred, real_targets)
                    disc_fake_loss = F.mse_loss(fake_pred, fake_targets)
                    disc_loss = (disc_real_loss + disc_fake_loss) * 0.5
                if scaler is not None:
                    scaler.scale(disc_loss).backward()
                    scaler.unscale_(disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.disc_grad_clip)
                    scaler.step(disc_optimizer)
                else:
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.disc_grad_clip)
                    disc_optimizer.step()
                disc_losses.append(float(disc_loss.item()))
                disc_real_scores.append(float(real_pred.mean().item()))
                disc_fake_scores.append(float(fake_pred.mean().item()))

            # --- Generator step ---
            optimizer.zero_grad(set_to_none=True)
            use_gan_objective = gan_enabled_epoch
            with amp_context:
                adv_loss = None
                if use_gan_objective:
                    fake_pred_for_gen = discriminator(outputs)
                    gen_targets = torch.full_like(fake_pred_for_gen, args.disc_real_target)
                    adv_loss = F.mse_loss(fake_pred_for_gen, gen_targets)
                loss, parts = combined_loss(
                    outputs,
                    output_bounds,
                    targets,
                    bounds,
                    adv_loss=adv_loss,
                    adversarial_scale=args.adversarial_scale,
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(float(loss.item()))
            for key, value in parts.items():
                epoch_parts[key] = epoch_parts.get(key, 0.0) + value

            if step_index % max(args.log_every, 1) == 0 or step_index == len(train_loader):
                current_lr = optimizer.param_groups[0]["lr"]
                avg_disc_window = float(np.mean(disc_losses[-args.log_every:])) if disc_losses else 0.0
                postfix: Dict[str, str] = {
                    "g": f"{float(np.mean(train_losses[-args.log_every:])):.4f}",
                    "d": f"{avg_disc_window:.4f}",
                    "lr": f"{current_lr:.1e}",
                }
                postfix["gan"] = "on" if use_gan_objective else "off"
                if use_cuda:
                    vram_gb = torch.cuda.memory_allocated() / (1024.0 ** 3)
                    postfix["vram"] = f"{vram_gb:.2f}G"
                progress.set_postfix(postfix)

        train_phase_seconds = max(time.perf_counter() - epoch_train_start, 1e-9)
        train_steps_per_second = len(train_loader) / train_phase_seconds
        train_samples_per_second = len(train_dataset) / train_phase_seconds

        for key in epoch_parts:
            epoch_parts[key] /= max(len(train_loader), 1)

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device, non_blocking=use_cuda)
                targets = batch["target"].to(device, non_blocking=use_cuda)
                bounds = batch["height_bounds"].to(device, non_blocking=use_cuda)
                amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp_context:
                    outputs, output_bounds = model(inputs)
                    loss, _ = combined_loss(outputs, output_bounds, targets, bounds)
                val_losses.append(float(loss.item()))

        average_train_loss = float(np.mean(train_losses))
        average_val_loss = float(np.mean(val_losses))
        val_loss_valid = bool(np.isfinite(average_val_loss) and average_val_loss >= 0.0)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epochs"].append(epoch_number)
        history["train_loss"].append(average_train_loss)
        history["val_loss"].append(average_val_loss)
        history.setdefault("val_loss_valid", []).append(val_loss_valid)
        history.setdefault("gan_enabled", []).append(gan_enabled_epoch)
        history.setdefault("gan_phase", []).append(gan_phase)
        history.setdefault("gan_schedule_remaining", []).append(gan_schedule_remaining)
        history["components"].append(epoch_parts)
        with open(output_dir / "training_log.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {average_train_loss:.4f} | Val Loss: {average_val_loss:.4f} | Best: {best_loss:.4f}")
        print(
            "  HM_G: {global_loss:.4f} | HM_L: {local_loss:.4f} | Bounds: {bounds:.4f}".format(
                global_loss=epoch_parts.get("heightmap_global", 0.0),
                local_loss=epoch_parts.get("heightmap_local", 0.0),
                bounds=epoch_parts.get("bounds", 0.0),
            )
        )
        avg_disc = float(np.mean(disc_losses)) if disc_losses else 0.0
        print(
            "  Grad: {gradient:.4f} | SSIM: {ssim:.4f} | Edge: {edge:.4f} | Freq: {freq:.4f} | Lap: {lap:.4f}".format(
                gradient=epoch_parts.get("gradient", 0.0),
                ssim=epoch_parts.get("ssim", 0.0),
                edge=epoch_parts.get("edge", 0.0),
                freq=epoch_parts.get("frequency", 0.0),
                lap=epoch_parts.get("laplacian", 0.0),
            )
        )
        print(
            "  Adv: {adv:.4f} | Disc: {disc:.4f} | LR: {lr:.2e} | Patience: {patience}/{limit}".format(
                adv=epoch_parts.get("adversarial", 0.0),
                disc=avg_disc,
                lr=current_lr,
                patience=patience_counter,
                limit=args.patience,
            )
        )
        print(
            "  Disc Real/Fake Mean: {real:.4f}/{fake:.4f}".format(
                real=float(np.mean(disc_real_scores)) if disc_real_scores else 0.0,
                fake=float(np.mean(disc_fake_scores)) if disc_fake_scores else 0.0,
            )
        )
        print(
            "  Throughput: {steps:.2f} steps/s | {samples:.1f} samples/s".format(
                steps=train_steps_per_second,
                samples=train_samples_per_second,
            )
        )
        print(
            f"  GAN: {'on' if gan_enabled_epoch else 'off'} | Phase: {gan_phase} | Schedule Remaining: {gan_schedule_remaining}"
        )

        if val_loss_valid:
            scheduler.step(average_val_loss)
        else:
            print(
                "  Warning: validation loss was invalid; skipping LR scheduler and best-checkpoint update for this epoch"
            )

        # Save a preview every epoch unconditionally so there is always something to inspect.
        try:
            preview_indices = select_epoch_preview_indices(
                val_base_dataset.samples,
                val_indices,
                static_preview_indices,
                epoch_number,
                args.static_preview_count,
                args.random_preview_count,
                args.seed,
            )
            preview_batch, loaded_preview_indices, skipped_preview_indices = build_preview_batch(
                val_base_dataset,
                preview_indices,
            )
            preview_labels = [
                f"{val_base_dataset.samples[index].dataset_name}:{val_base_dataset.samples[index].tile_name}"
                for index in loaded_preview_indices
            ]
            save_training_preview(
                model,
                preview_batch,
                epoch + 1,
                output_dir / "previews",
                device,
                preview_labels=preview_labels,
            )
            if skipped_preview_indices:
                print(f"  Skipped {len(skipped_preview_indices)} preview tile(s) while building epoch preview batch")
        except Exception as exc:
            print(f"  Failed to save mixed preview batch: {exc}")

        saved_new_best = False
        schedule_rearmed_this_epoch = False
        if val_loss_valid and average_val_loss < best_loss:
            best_loss = average_val_loss
            patience_counter = 0
            saved_new_best = True
            if args.gan_burst_after_best > 0:
                gan_schedule_remaining = args.gan_burst_after_best
                schedule_rearmed_this_epoch = True
            elif gan_enabled_epoch and args.gan_cooldown_after_best > 0:
                gan_schedule_remaining = args.gan_cooldown_after_best
                schedule_rearmed_this_epoch = True
            print("  Saved best model")
            if args.gan_burst_after_best > 0:
                print(
                    f"  GAN refinement burst armed for next {args.gan_burst_after_best} epoch(s) after best checkpoint"
                )
            elif gan_enabled_epoch and args.gan_cooldown_after_best > 0:
                print(
                    f"  GAN cooldown armed for next {args.gan_cooldown_after_best} epoch(s) after GAN-assisted best"
                )
        else:
            if not val_loss_valid:
                continue
            if epoch_number >= args.early_stop_start_epoch:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping: no improvement for {args.patience} epochs")
                    break
            else:
                print(
                    f"  Early-stop warmup active until epoch {args.early_stop_start_epoch}; "
                    "patience not counting yet"
                )

        if gan_schedule_active_this_epoch and not schedule_rearmed_this_epoch:
            gan_schedule_remaining = max(gan_schedule_remaining - 1, 0)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "disc_optimizer_state_dict": disc_optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "val_loss": average_val_loss,
            "val_loss_valid": val_loss_valid,
            "patience_counter": patience_counter,
            "gan_schedule_remaining": gan_schedule_remaining,
            "metadata": checkpoint_metadata_from_args(args, dataset_roots, len(dataset), len(train_indices), len(val_indices), train_groups, val_groups),
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")
        if saved_new_best:
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"\nTraining complete. Best validation loss: {best_loss:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the V7.3 multichannel terrain regressor with GAN and curated data.")
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
    parser.add_argument("--disc-learning-rate", type=float, default=DISCRIMINATOR_LR,
                        help=f"Discriminator learning rate (default: {DISCRIMINATOR_LR}).")
    parser.add_argument("--disc-real-target", type=float, default=DEFAULT_DISC_REAL_TARGET,
                        help=f"Least-squares GAN target for real patches (default: {DEFAULT_DISC_REAL_TARGET}).")
    parser.add_argument("--disc-fake-target", type=float, default=DEFAULT_DISC_FAKE_TARGET,
                        help=f"Least-squares GAN target for fake patches (default: {DEFAULT_DISC_FAKE_TARGET}).")
    parser.add_argument("--disc-label-noise", type=float, default=DEFAULT_DISC_LABEL_NOISE,
                        help=f"Uniform label jitter applied around discriminator targets (default: {DEFAULT_DISC_LABEL_NOISE}).")
    parser.add_argument("--disc-input-noise-std", type=float, default=DEFAULT_DISC_INPUT_NOISE_STD,
                        help=f"Gaussian noise std added to discriminator inputs while GAN is active (default: {DEFAULT_DISC_INPUT_NOISE_STD}).")
    parser.add_argument("--disc-grad-clip", type=float, default=DEFAULT_DISC_GRAD_CLIP,
                        help=f"Gradient clipping max-norm for discriminator updates (default: {DEFAULT_DISC_GRAD_CLIP}).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-start-epoch", type=int, default=DEFAULT_EARLY_STOP_START_EPOCH,
                        help=f"Epoch number (1-based) when early-stop patience begins counting (default: {DEFAULT_EARLY_STOP_START_EPOCH}).")
    parser.add_argument("--adversarial-scale", type=float, default=DEFAULT_ADVERSARIAL_SCALE,
                        help=f"Scale applied to adversarial loss weight (default: {DEFAULT_ADVERSARIAL_SCALE}, geometry-first safer default).")
    parser.add_argument("--start-gan-epoch", type=int, default=DEFAULT_START_GAN_EPOCH,
                        help=f"Epoch number (1-based) when adversarial loss turns on (default: {DEFAULT_START_GAN_EPOCH}).")
    parser.add_argument("--gan-cycle-length", type=int, default=DEFAULT_GAN_CYCLE_LENGTH,
                        help=f"Optional GAN cadence length in epochs after start-gan-epoch (default: {DEFAULT_GAN_CYCLE_LENGTH}, disabled).")
    parser.add_argument("--gan-cycle-on-epochs", type=int, default=DEFAULT_GAN_CYCLE_ON_EPOCHS,
                        help=f"How many epochs per GAN cadence run with adversarial loss enabled (default: {DEFAULT_GAN_CYCLE_ON_EPOCHS}, disabled).")
    parser.add_argument("--gan-cooldown-after-best", type=int, default=DEFAULT_GAN_COOLDOWN_AFTER_BEST,
                        help=f"Force GAN off for this many subsequent epochs after a new GAN-assisted best checkpoint (default: {DEFAULT_GAN_COOLDOWN_AFTER_BEST}).")
    parser.add_argument("--gan-burst-after-best", type=int, default=DEFAULT_GAN_BURST_AFTER_BEST,
                        help=f"Automatically arm GAN for this many subsequent epochs after any new best checkpoint (default: {DEFAULT_GAN_BURST_AFTER_BEST}, disabled). Overrides epoch-based GAN cadence when > 0.")
    parser.add_argument("--disc-every", type=int, default=DEFAULT_DISC_EVERY,
                        help=f"Update discriminator every N train steps (default: {DEFAULT_DISC_EVERY}).")
    parser.add_argument("--lr-plateau-patience", type=int, default=DEFAULT_LR_PLATEAU_PATIENCE,
                        help=f"ReduceLROnPlateau patience in epochs (default: {DEFAULT_LR_PLATEAU_PATIENCE}).")
    parser.add_argument("--lr-plateau-factor", type=float, default=DEFAULT_LR_PLATEAU_FACTOR,
                        help=f"ReduceLROnPlateau factor (default: {DEFAULT_LR_PLATEAU_FACTOR}).")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--spatial-group-size", type=int, default=DEFAULT_SPATIAL_GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--static-preview-count", type=int, default=DEFAULT_STATIC_PREVIEW_COUNT,
                        help=f"How many fixed high-signal validation tiles to keep in every preview grid (default: {DEFAULT_STATIC_PREVIEW_COUNT}).")
    parser.add_argument("--random-preview-count", type=int, default=DEFAULT_RANDOM_PREVIEW_COUNT,
                        help=f"How many random validation tiles to add to each epoch preview grid (default: {DEFAULT_RANDOM_PREVIEW_COUNT}).")
    parser.add_argument("--limit", type=int, help="Optional per-root sample cap for quick debugging.")
    parser.add_argument("--resume", type=str, help="Resume from an existing checkpoint.")
    parser.add_argument("--no-resume-optimizer", action="store_true",
                        help="Do not restore optimizer/discriminator/scheduler/scaler state from checkpoint.")
    parser.add_argument("--train-workers", type=int, default=DEFAULT_TRAIN_WORKERS,
                        help=f"DataLoader worker count for training (default: {DEFAULT_TRAIN_WORKERS}).")
    parser.add_argument("--val-workers", type=int, default=DEFAULT_VAL_WORKERS,
                        help=f"DataLoader worker count for validation (default: {DEFAULT_VAL_WORKERS}).")
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR,
                        help=f"Batches prefetched per worker when workers > 0 (default: {DEFAULT_PREFETCH_FACTOR}).")
    parser.add_argument("--log-every", type=int, default=DEFAULT_LIVE_LOG_EVERY,
                        help=f"Update tqdm live metrics every N training steps (default: {DEFAULT_LIVE_LOG_EVERY}).")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--amp-dtype", choices=["auto", "float16", "bfloat16"], default=DEFAULT_AMP_DTYPE,
                        help=f"Autocast dtype when AMP is enabled (default: {DEFAULT_AMP_DTYPE}).")
    parser.add_argument("--no-tf32", action="store_true",
                        help="Disable TF32 Tensor Core paths for matmul/cuDNN.")
    parser.add_argument("--no-cudnn-benchmark", action="store_true",
                        help="Disable cuDNN benchmark autotuning for fixed-size training tensors.")
    parser.add_argument("--no-augment", action="store_true", help="Disable RGB jitter and random flips.")
    parser.add_argument("--no-curate", action="store_true", help="Disable complexity-based dataset curation.")
    parser.add_argument("--min-height-range", type=float, default=DEFAULT_MIN_HEIGHT_RANGE,
                        help=f"Skip tiles with less than this height variation in game units (default: {DEFAULT_MIN_HEIGHT_RANGE}).")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
