#!/usr/bin/env python3
"""
WoW Height Regressor V7.1 - inference engine.

Inference restores the active multichannel V7.x contract:
- minimap RGB
- normal map RGB
- WDL prior
- per-tile bounds hints
- liquid mask
- liquid height prior
- object footprint mask
- brush imprint mask when the checkpoint expects it

The primary mesh export still uses the predicted global height channel. The
terrain checkpoint no longer owns alpha-mask prediction.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from train_v7 import (
        BRUSH_MANIFEST_FILE,
        DEFAULT_BLUR_SIGMA,
        DEFAULT_GLOBAL_RESIDUAL_SCALE,
        HEIGHT_GLOBAL_MAX,
        HEIGHT_GLOBAL_MIN,
        MultiChannelUNetV7,
        OUTPUT_SIZE,
        resolve_model_architecture_from_metadata,
    )
except ImportError:
    OUTPUT_SIZE = 512
    HEIGHT_GLOBAL_MIN = -1000.0
    HEIGHT_GLOBAL_MAX = 3000.0
    BRUSH_MANIFEST_FILE = "brush_imprint_manifest.json"
    DEFAULT_BLUR_SIGMA = 0.0
    DEFAULT_GLOBAL_RESIDUAL_SCALE = 0.20

    def resolve_model_architecture_from_metadata(metadata: Optional[Dict[str, object]]) -> Tuple[bool, float]:
        return False, DEFAULT_GLOBAL_RESIDUAL_SCALE

    class MultiChannelUNetV7(nn.Module):
        def __init__(
            self,
            in_channels: int = 13,
            out_channels: int = 2,
            use_wdl_global_trestle: bool = False,
            global_residual_scale: float = DEFAULT_GLOBAL_RESIDUAL_SCALE,
        ):
            super().__init__()
            raise ImportError("train_v7.py is required so the V7.1 architecture matches the checkpoint.")


TILE_SIZE = 533.33333
MAP_ORIGIN = 32.0 * TILE_SIZE
MASK_CONTEXT_MARGIN_TILES = 0.20
MASK_MAX_ABOVE_TERRAIN = 8.0
MASK_MIN_BELOW_TERRAIN = -3.0
PRECISE_OBJECT_MASK_KEYS = (
    "object_visibility_mask_cv2",
    "pm4_mask",
    "pm4_object_mask",
    "collision_mask",
)
SEEDED_OBJECT_MASK_KEYS = (
    "object_visibility_mask",
)
_SCIPY_GAUSSIAN_FILTER = None
_SCIPY_IMPORT_ATTEMPTED = False
DEFAULT_EDGE_ANCHOR_WIDTH = 12


def tile_uv_candidates(world_a: float, world_b: float, tile_x: int, tile_y: int) -> List[Tuple[float, float]]:
    return [
        (world_a / TILE_SIZE - float(tile_x), world_b / TILE_SIZE - float(tile_y)),
        ((MAP_ORIGIN - world_b) / TILE_SIZE - float(tile_x), (MAP_ORIGIN - world_a) / TILE_SIZE - float(tile_y)),
    ]


def apply_gaussian_smoothing(heightmap: np.ndarray, sigma: float) -> np.ndarray:
    global _SCIPY_GAUSSIAN_FILTER
    global _SCIPY_IMPORT_ATTEMPTED

    if sigma <= 0:
        return heightmap

    if not _SCIPY_IMPORT_ATTEMPTED:
        _SCIPY_IMPORT_ATTEMPTED = True
        try:
            from scipy.ndimage import gaussian_filter as _imported_gaussian_filter

            _SCIPY_GAUSSIAN_FILTER = _imported_gaussian_filter
        except Exception:
            _SCIPY_GAUSSIAN_FILTER = None

    if _SCIPY_GAUSSIAN_FILTER is not None:
        return _SCIPY_GAUSSIAN_FILTER(heightmap, sigma=sigma)

    # Fallback for environments where SciPy is unavailable or ABI-incompatible.
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()

    tensor = torch.from_numpy(heightmap.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    tensor = torch.nn.functional.conv2d(tensor, kernel_x, padding=(0, radius))
    tensor = torch.nn.functional.conv2d(tensor, kernel_y, padding=(radius, 0))
    return tensor.squeeze(0).squeeze(0).numpy()


def render_wdl_height_prior_world(wdl_data: Optional[Dict[str, object]], target_size: int) -> Optional[np.ndarray]:
    if not wdl_data:
        return None

    outer = np.asarray(wdl_data.get("outer_17", []), dtype=np.float32)
    if len(outer) != 289 or not np.all(np.isfinite(outer)):
        return None

    grid = outer.reshape(17, 17)
    tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=True)
    return resized.squeeze(0).squeeze(0).numpy()


def anchor_heightmap_edges(heightmap_world: np.ndarray, edge_prior_world: Optional[np.ndarray], edge_width: int) -> np.ndarray:
    if edge_prior_world is None or edge_width <= 0:
        return heightmap_world

    result = np.array(heightmap_world, copy=True)
    height, width = result.shape
    border_distance = np.minimum.reduce(
        np.meshgrid(
            np.minimum(np.arange(width), np.arange(width)[::-1]),
            np.minimum(np.arange(height), np.arange(height)[::-1]),
            indexing="xy",
        )
    ).astype(np.float32)
    border_weight = np.clip((float(edge_width) - border_distance) / max(float(edge_width), 1.0), 0.0, 1.0)
    if not np.any(border_weight > 0):
        return result

    result = result * (1.0 - border_weight) + edge_prior_world * border_weight
    return result.astype(np.float32, copy=False)


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


class V7InferenceEngine:
    def __init__(self, model_path: Path, device: str = "auto") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        print(f"Loading V7.1 model from {model_path} on {self.device}...")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.metadata: Dict[str, object] = dict(checkpoint.get("metadata", {}))
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        in_channels = self._infer_input_channels(state_dict)
        out_channels = self._infer_output_channels(state_dict)
        self.expected_in_channels = int(in_channels)
        use_wdl_global_trestle, global_residual_scale = resolve_model_architecture_from_metadata(self.metadata)
        print(f"Detected input channels: {in_channels}")

        self.model = MultiChannelUNetV7(
            in_channels=in_channels,
            out_channels=out_channels,
            use_wdl_global_trestle=use_wdl_global_trestle,
            global_residual_scale=global_residual_scale,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        blur_sigma = float(self.metadata.get("blur_sigma", 0.5))
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=blur_sigma) if blur_sigma > 0 else None
        self._brush_manifest_cache: Dict[Path, Optional[Dict[str, object]]] = {}

    @staticmethod
    def _infer_input_channels(state_dict: Dict[str, torch.Tensor]) -> int:
        for key, value in state_dict.items():
            if key.endswith("weight") and value.ndim == 4:
                return int(value.shape[1])
        raise KeyError("Could not infer input channel count from checkpoint state_dict.")

    @staticmethod
    def _infer_output_channels(state_dict: Dict[str, torch.Tensor]) -> int:
        for key, value in reversed(list(state_dict.items())):
            if key.endswith("weight") and value.ndim == 4:
                return int(value.shape[0])
        raise KeyError("Could not infer output channel count from checkpoint state_dict.")

    def _load_brush_manifest(self, dataset_root: Path) -> Optional[Dict[str, object]]:
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

    def prepare_input(self, dataset_root: Path, tile_name: str) -> Tuple[torch.Tensor, Dict[str, float], Path]:
        json_path = dataset_root / "dataset" / f"{tile_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        terrain = payload.get("terrain_data", {})

        minimap_rel = terrain.get("no_object_minimap") or terrain.get("no_mccv_minimap") or payload.get("image")
        if minimap_rel:
            minimap_path = dataset_root / str(minimap_rel)
        else:
            minimap_path = dataset_root / "images" / f"{tile_name}.png"

        if not minimap_path.exists() and payload.get("image"):
            fallback_minimap = payload.get("image")
            if fallback_minimap:
                minimap_path = dataset_root / str(fallback_minimap)

        normalmap_rel = terrain.get("normalmap")
        if not normalmap_rel:
            raise FileNotFoundError(f"Missing normalmap reference for {tile_name}")
        normalmap_path = dataset_root / normalmap_rel

        if not minimap_path.exists() or not normalmap_path.exists():
            raise FileNotFoundError(f"Missing input images for {tile_name}")

        minimap = Image.open(minimap_path).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        if self.blur is not None:
            minimap = self.blur(minimap)
        normalmap = Image.open(normalmap_path).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)

        height_min = float(terrain.get("height_min", 0.0))
        height_max = float(terrain.get("height_max", 100.0))
        global_min = float(terrain.get("height_global_min", HEIGHT_GLOBAL_MIN))
        global_max = float(terrain.get("height_global_max", HEIGHT_GLOBAL_MAX))
        global_range = max(global_max - global_min, 1e-6)

        minimap_tensor = self.normalize(self.to_tensor(minimap))
        normalmap_tensor = self.normalize(self.to_tensor(normalmap))
        wdl_tensor = self._render_wdl(terrain.get("wdl_heights"), global_min, global_max)

        height_min_mask = torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), np.clip((height_min - global_min) / global_range, 0.0, 1.0), dtype=torch.float32)
        height_max_mask = torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), np.clip((height_max - global_min) / global_range, 0.0, 1.0), dtype=torch.float32)

        liquid_mask = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        liquid_mask_rel = terrain.get("liquid_mask")
        if liquid_mask_rel:
            liquid_mask_path = dataset_root / liquid_mask_rel
            if liquid_mask_path.exists():
                liquid_image = Image.open(liquid_mask_path).convert("L").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
                liquid_mask = (self.to_tensor(liquid_image) > 0.1).float()

        liquid_height_prior = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        liquid_height_rel = terrain.get("liquid_height")
        if liquid_height_rel:
            liquid_height_path = dataset_root / liquid_height_rel
            if liquid_height_path.exists():
                liquid_height_prior = load_heightmap_16bit(liquid_height_path, OUTPUT_SIZE) * liquid_mask

        brush_mask = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        if self.expected_in_channels >= 13:
            brush_mask_path = self._resolve_brush_mask_path(dataset_root, tile_name)
            if brush_mask_path and brush_mask_path.exists():
                brush_image = Image.open(brush_mask_path).convert("L").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
                brush_tensor = self.to_tensor(brush_image)
                brush_mask = (brush_tensor > 0.1).float()

        tile_x, tile_y = parse_tile_coords(tile_name)
        object_mask = self._build_object_context_mask(dataset_root, terrain, tile_x, tile_y)

        channels = [
            minimap_tensor,
            normalmap_tensor,
            wdl_tensor,
            height_min_mask,
            height_max_mask,
            liquid_mask,
            liquid_height_prior,
            object_mask,
        ]
        if self.expected_in_channels >= 13:
            channels.append(brush_mask)

        input_tensor = torch.cat(channels, dim=0)
        if input_tensor.shape[0] != self.expected_in_channels:
            raise RuntimeError(
                f"Prepared {input_tensor.shape[0]} input channels for {tile_name}, but checkpoint expects {self.expected_in_channels}."
            )

        input_tensor = input_tensor.unsqueeze(0)

        bounds_info = {
            "h_min": height_min,
            "h_max": height_max,
            "g_min": global_min,
            "g_max": global_max,
        }
        return input_tensor.to(self.device), bounds_info, minimap_path

    def _render_wdl(self, wdl_data: Optional[Dict[str, object]], global_min: float, global_max: float) -> torch.Tensor:
        if not wdl_data:
            return torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), 0.5)

        outer = np.asarray(wdl_data.get("outer_17", []), dtype=np.float32)
        if len(outer) != 289:
            return torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), 0.5)

        grid = outer.reshape(17, 17)
        global_range = max(global_max - global_min, 1e-6)
        grid = np.clip((grid - global_min) / global_range, 0.0, 1.0)

        image = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        return self.to_tensor(image)

    def _load_optional_binary_mask(self, dataset_root: Path, terrain: Dict[str, object], keys: Sequence[str]) -> torch.Tensor:
        for key in keys:
            rel = terrain.get(key)
            if not rel:
                continue
            candidate = dataset_root / str(rel)
            if not candidate.exists():
                continue
            mask_image = Image.open(candidate).convert("L").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
            return (self.to_tensor(mask_image) > 0.1).float()

        return torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)

    def _build_object_context_mask(
        self,
        dataset_root: Path,
        terrain: Dict[str, object],
        tile_x: int,
        tile_y: int,
    ) -> torch.Tensor:
        precise_mask = self._load_optional_binary_mask(dataset_root, terrain, keys=PRECISE_OBJECT_MASK_KEYS)
        if bool(torch.any(precise_mask > 0)):
            return precise_mask

        seeded_mask = self._load_optional_binary_mask(dataset_root, terrain, keys=SEEDED_OBJECT_MASK_KEYS)
        if bool(torch.any(seeded_mask > 0)):
            return seeded_mask

        return self._build_object_mask(terrain.get("objects"), tile_x, tile_y, terrain.get("wdl_heights"))

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
        object_mask = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        if not objects:
            return object_mask

        image = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        pixels_per_world = OUTPUT_SIZE / TILE_SIZE
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

            center_x = int(round(local_x * (OUTPUT_SIZE - 1)))
            center_y = int(round(local_y * (OUTPUT_SIZE - 1)))

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
                continue

            if np.isfinite(pos_y) and wdl_sampler is not None:
                terrain_height = wdl_sampler(local_x, local_y, pos_y)
                if terrain_height is not None and np.isfinite(terrain_height):
                    delta = float(pos_y - terrain_height)
                    if delta < MASK_MIN_BELOW_TERRAIN or delta > MASK_MAX_ABOVE_TERRAIN:
                        continue

            x1 = max(0, center_x - radius_x)
            y1 = max(0, center_y - radius_y)
            x2 = min(OUTPUT_SIZE, center_x + radius_x + 1)
            y2 = min(OUTPUT_SIZE, center_y + radius_y + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            image[y1:y2, x1:x2] = 1.0

        return torch.from_numpy(image).unsqueeze(0)

    def predict(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            predicted_heightmap, predicted_bounds = self.model(input_tensor)
        return predicted_heightmap.cpu().numpy(), predicted_bounds.cpu().numpy()

    @staticmethod
    def denormalize_heightmap(normalized_heightmap: np.ndarray, global_min: float = HEIGHT_GLOBAL_MIN, global_max: float = HEIGHT_GLOBAL_MAX) -> np.ndarray:
        return normalized_heightmap * (global_max - global_min) + global_min

    @staticmethod
    def save_obj(heightmap: np.ndarray, output_path: Path, tile_x: int, tile_y: int, texture_path: Optional[Path] = None) -> None:
        resolution = heightmap.shape[0]
        step = TILE_SIZE / max(resolution - 1, 1)

        vertices = []
        uvs = []
        faces = []

        for grid_y in range(resolution):
            for grid_x in range(resolution):
                world_x = tile_x * TILE_SIZE + grid_x * step
                world_y = float(heightmap[grid_y, grid_x])
                world_z = tile_y * TILE_SIZE + grid_y * step
                vertices.append((world_x, world_y, world_z))
                uvs.append((grid_x / max(resolution - 1, 1), 1.0 - (grid_y / max(resolution - 1, 1))))

        for grid_y in range(resolution - 1):
            for grid_x in range(resolution - 1):
                v0 = grid_y * resolution + grid_x + 1
                v1 = grid_y * resolution + (grid_x + 1) + 1
                v2 = (grid_y + 1) * resolution + (grid_x + 1) + 1
                v3 = (grid_y + 1) * resolution + grid_x + 1
                faces.append((v0, v3, v2))
                faces.append((v0, v2, v1))

        mtl_path = output_path.with_suffix(".mtl")
        texture_name = texture_path.name if texture_path else "default.png"
        if texture_path and texture_path.exists():
            copied_texture = output_path.parent / texture_name
            if not copied_texture.exists():
                shutil.copy(texture_path, copied_texture)

        with open(mtl_path, "w", encoding="utf-8") as handle:
            handle.write("newmtl TerrainMat\n")
            handle.write("Ka 1.0 1.0 1.0\n")
            handle.write("Kd 1.0 1.0 1.0\n")
            handle.write(f"map_Kd {texture_name}\n")

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"mtllib {mtl_path.name}\n")
            handle.write("usemtl TerrainMat\n")
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.2f} {vertex[1]:.2f} {vertex[2]:.2f}\n")
            for uv in uvs:
                handle.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
            for face in faces:
                handle.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")

    def save_debug_image(self, input_tensor: torch.Tensor, predicted_heightmap: torch.Tensor, output_path: Path) -> None:
        minimap = input_tensor[0, 0:3].cpu()
        normalmap = input_tensor[0, 3:6].cpu()
        wdl = input_tensor[0, 6:7].cpu().repeat(3, 1, 1)
        water = input_tensor[0, 9:10].cpu().repeat(3, 1, 1) * torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
        objects = input_tensor[0, 11:12].cpu().repeat(3, 1, 1) * torch.tensor([1.0, 0.5, 0.0]).view(3, 1, 1)
        brush = None
        if input_tensor.shape[1] > 12:
            brush = input_tensor[0, 12:13].cpu().repeat(3, 1, 1) * torch.tensor([1.0, 0.75, 0.0]).view(3, 1, 1)
        prediction = predicted_heightmap[0, 0:1].cpu()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        minimap = torch.clamp(minimap * std + mean, 0.0, 1.0)
        normalmap = torch.clamp(normalmap * std + mean, 0.0, 1.0)

        prediction = prediction - prediction.min()
        prediction = prediction / (prediction.max() + 1e-6)
        prediction = prediction.repeat(3, 1, 1)

        columns = [minimap, normalmap, wdl, water, objects]
        if brush is not None:
            columns.append(brush)
        columns.append(prediction)

        grid = torch.cat(columns, dim=2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transforms.ToPILImage()(grid).save(output_path)


def save_height_outputs(heightmap_world: np.ndarray, output_dir: Path, tile_name: str) -> None:
    normalized = np.clip((heightmap_world - HEIGHT_GLOBAL_MIN) / max(HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN, 1e-6), 0.0, 1.0)
    height_u16 = (normalized * 65535.0).astype(np.uint16)
    Image.fromarray(height_u16, mode="I;16").save(output_dir / f"{tile_name}_height.png")

    metadata = {
        "tile_name": tile_name,
        "height_encoding": "v7_1_global_range",
        "height_global_min": HEIGHT_GLOBAL_MIN,
        "height_global_max": HEIGHT_GLOBAL_MAX,
        "predicted_height_min": float(heightmap_world.min()),
        "predicted_height_max": float(heightmap_world.max()),
    }
    with open(output_dir / f"{tile_name}_height_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_tile_coords(tile_name: str) -> Tuple[int, int]:
    parts = tile_name.split("_")
    if len(parts) < 3:
        return 0, 0
    return int(parts[-2]), int(parts[-1])


def run_batch_inference(
    model_path: Path,
    dataset_root: Path,
    output_dir: Path,
    tile_filter: Optional[str] = None,
    debug: bool = False,
    z_scale: float = 1.0,
    smooth_sigma: float = 0.0,
    out_res: int = OUTPUT_SIZE,
    edge_anchor_width: int = DEFAULT_EDGE_ANCHOR_WIDTH,
) -> None:
    engine = V7InferenceEngine(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = dataset_root / "dataset"
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    tiles = sorted(dataset_dir.glob("*.json"))
    print(f"Found {len(tiles)} tiles in {dataset_root}")
    success_count = 0

    for json_file in tiles:
        tile_name = json_file.stem
        if tile_filter and tile_filter not in tile_name:
            continue

        try:
            input_tensor, bounds_info, texture_path = engine.prepare_input(dataset_root, tile_name)
            with open(json_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            terrain = payload.get("terrain_data", {})
            predicted_heightmap, predicted_bounds = engine.predict(input_tensor)

            heightmap_normalized = predicted_heightmap[0, 0]
            heightmap_world = engine.denormalize_heightmap(heightmap_normalized, bounds_info["g_min"], bounds_info["g_max"])

            if abs(z_scale - 1.0) > 1e-6:
                heightmap_world *= z_scale
            if smooth_sigma > 0:
                heightmap_world = apply_gaussian_smoothing(heightmap_world, sigma=smooth_sigma)
            edge_prior_world = render_wdl_height_prior_world(terrain.get("wdl_heights"), heightmap_world.shape[0])
            heightmap_world = anchor_heightmap_edges(heightmap_world, edge_prior_world, edge_anchor_width)
            if out_res != OUTPUT_SIZE:
                resized = Image.fromarray(heightmap_world, mode="F")
                resized = resized.resize((out_res, out_res), Image.BILINEAR)
                heightmap_world = np.asarray(resized, dtype=np.float32)

            tile_x, tile_y = parse_tile_coords(tile_name)
            engine.save_obj(heightmap_world, output_dir / f"{tile_name}.obj", tile_x, tile_y, texture_path)
            save_height_outputs(heightmap_world, output_dir, tile_name)

            if debug:
                engine.save_debug_image(input_tensor, torch.from_numpy(predicted_heightmap), output_dir / f"{tile_name}_debug.png")

            if predicted_bounds.size > 0:
                with open(output_dir / f"{tile_name}_bounds.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "predicted_bounds": predicted_bounds[0].tolist(),
                            "input_bounds": bounds_info,
                        },
                        handle,
                        indent=2,
                    )

            success_count += 1
        except Exception as exc:
            print(f"Failed {tile_name}: {exc}")

    print(f"Processed {success_count} tiles")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V7.1 multichannel inference.")
    parser.add_argument("--model", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to dataset root containing dataset/ and images/")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--filter", help="Optional string filter for tile names")
    parser.add_argument("--debug", action="store_true", help="Save debug composite images")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Optional scale factor for output world heights")
    parser.add_argument("--smooth-output", type=float, default=0.0, help="Optional Gaussian smoothing sigma")
    parser.add_argument("--res", type=int, default=OUTPUT_SIZE, help="Output mesh and heightmap resolution")
    parser.add_argument("--edge-anchor-width", type=int, default=DEFAULT_EDGE_ANCHOR_WIDTH,
                        help=f"Feather width in pixels for anchoring tile borders to the WDL prior (default: {DEFAULT_EDGE_ANCHOR_WIDTH}).")
    return parser


if __name__ == "__main__":
    arguments = build_arg_parser().parse_args()
    run_batch_inference(
        model_path=Path(arguments.model),
        dataset_root=Path(arguments.dataset),
        output_dir=Path(arguments.out),
        tile_filter=arguments.filter,
        debug=arguments.debug,
        z_scale=arguments.z_scale,
        smooth_sigma=arguments.smooth_output,
        out_res=arguments.res,
        edge_anchor_width=arguments.edge_anchor_width,
    )
