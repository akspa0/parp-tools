from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

TILE_SIZE = 533.33333
MAP_ORIGIN = 32.0 * TILE_SIZE
MASK_CONTEXT_MARGIN_TILES = 0.20
MASK_MAX_ABOVE_TERRAIN = 8.0
MASK_MIN_BELOW_TERRAIN = -3.0
MAX_MASK_OBJECT_HEIGHT_WORLD = 80.0
MAX_MASK_OBJECT_TOP_ABOVE_TERRAIN = 48.0
MAX_MASK_M2_SCALE_WITHOUT_BOUNDS = 4.0
EXPORTED_MASK_WINDOW_SCALE = 3.0
EXPORTED_MASK_MIN_RADIUS = 4
PRECISE_OBJECT_MASK_KEYS = (
    "object_visibility_mask_cv2",
    "pm4_mask",
    "pm4_object_mask",
    "collision_mask",
)
SEEDED_OBJECT_MASK_KEYS = (
    "object_visibility_mask",
)
MAX_PRECISE_OBJECT_MASK_COVERAGE = 0.30
MAX_SEEDED_OBJECT_MASK_COVERAGE = 0.25
MAX_FALLBACK_OBJECT_MASK_COVERAGE = 0.20

_TREE_LIKE_PATTERN = re.compile(
    r"(^|[/_ .-])(tree|trees|treelog|treelogs|treestump|treestumps|canopy|foliage|bush|bushes|shrub|shrubs)([/_ .-]|$)",
    re.IGNORECASE,
)


@dataclass
class ProjectedObject:
    center_x: int
    center_y: int
    radius_x: int
    radius_y: int
    category: str
    name: str
    model_path: str
    unique_id: int
    has_bounds: bool
    allow_fallback: bool


def tile_uv_candidates(world_a: float, world_b: float, tile_x: int, tile_y: int) -> List[Tuple[float, float]]:
    return [
        (world_a / TILE_SIZE - float(tile_x), world_b / TILE_SIZE - float(tile_y)),
        ((MAP_ORIGIN - world_b) / TILE_SIZE - float(tile_x), (MAP_ORIGIN - world_a) / TILE_SIZE - float(tile_y)),
    ]


def load_binary_mask(
    dataset_root: Path,
    terrain: Dict[str, object],
    keys: Sequence[str],
    output_size: int,
) -> Optional[np.ndarray]:
    for key in keys:
        rel = terrain.get(key)
        if not rel:
            continue
        candidate = dataset_root / str(rel)
        if not candidate.exists():
            continue
        with Image.open(candidate).convert("L") as image:
            if image.size != (output_size, output_size):
                image = image.resize((output_size, output_size), Image.NEAREST)
            return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)

    return None


def mask_coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def is_mask_usable(mask: np.ndarray, max_coverage: float) -> bool:
    if mask.size == 0 or not np.any(mask > 0):
        return False
    return mask_coverage(mask) <= max_coverage


def build_wdl_height_sampler(
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
        x1 = min(x0 + 1, 16)
        y1 = min(y0 + 1, 16)
        tx = x - x0
        ty = y - y0

        v00 = float(grid[y0, x0])
        v10 = float(grid[y0, x1])
        v01 = float(grid[y1, x0])
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


def _normalized_object_text(obj: Dict[str, object]) -> str:
    name = str(obj.get("name", "") or "")
    model_path = str(obj.get("model_path", "") or "")
    return f"{name} {model_path}".replace("\\", "/").lower()


def is_tree_like_object(obj: Dict[str, object]) -> bool:
    category = str(obj.get("category", "") or "").lower()
    if category != "m2":
        return False
    return bool(_TREE_LIKE_PATTERN.search(_normalized_object_text(obj)))


def _safe_scale(obj: Dict[str, object]) -> float:
    scale = float(obj.get("scale", 1.0) or 1.0)
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _choose_local_uv(obj: Dict[str, object], tile_x: int, tile_y: int) -> Optional[Tuple[float, float]]:
    pos_x = float(obj.get("x", obj.get("pos_x", 0.0)))
    pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
    pos_z = float(obj.get("z", obj.get("pos_z", pos_y)))

    candidate_uvs: List[Tuple[float, float]] = []
    if abs(pos_x) < 2 and abs(pos_y) < 2:
        candidate_uvs.append(((pos_y + 1.0) * 0.5, (pos_x + 1.0) * 0.5))

    candidate_uvs.extend(tile_uv_candidates(pos_x, pos_z, tile_x, tile_y))
    if np.isfinite(pos_y):
        candidate_uvs.extend(tile_uv_candidates(pos_x, pos_y, tile_x, tile_y))

    best: Optional[Tuple[float, float]] = None
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
            best = (cand_x, cand_y)
            if overflow <= 1e-6:
                break

    if best is None:
        return None

    local_x, local_y = best
    if (
        local_x < -MASK_CONTEXT_MARGIN_TILES
        or local_x > 1.0 + MASK_CONTEXT_MARGIN_TILES
        or local_y < -MASK_CONTEXT_MARGIN_TILES
        or local_y > 1.0 + MASK_CONTEXT_MARGIN_TILES
    ):
        return None

    return best


def _object_height_world(obj: Dict[str, object], scale: float) -> Optional[float]:
    bounds_min = obj.get("bounds_min")
    bounds_max = obj.get("bounds_max")
    if not isinstance(bounds_min, list) or not isinstance(bounds_max, list):
        return None
    if len(bounds_min) < 3 or len(bounds_max) < 3:
        return None
    return abs(float(bounds_max[1]) - float(bounds_min[1])) * scale


def _object_top_above_terrain(obj: Dict[str, object], scale: float, terrain_height: float) -> Optional[float]:
    bounds_min = obj.get("bounds_min")
    bounds_max = obj.get("bounds_max")
    if not isinstance(bounds_min, list) or not isinstance(bounds_max, list):
        return None
    if len(bounds_min) < 3 or len(bounds_max) < 3:
        return None

    pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
    top = pos_y + max(float(bounds_min[1]), float(bounds_max[1])) * scale
    return top - terrain_height


def _estimate_radii(obj: Dict[str, object], output_size: int, scale: float) -> Tuple[int, int, bool]:
    pixels_per_world = output_size / TILE_SIZE
    bounds_min = obj.get("bounds_min")
    bounds_max = obj.get("bounds_max")
    category = str(obj.get("category", "") or "").lower()

    if isinstance(bounds_min, list) and isinstance(bounds_max, list) and len(bounds_min) >= 3 and len(bounds_max) >= 3:
        half_width_world = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
        half_depth_world = abs(float(bounds_max[2]) - float(bounds_min[2])) * 0.5 * scale
        radius_x = max(1, int(round(half_width_world * pixels_per_world)))
        radius_y = max(1, int(round(half_depth_world * pixels_per_world)))
        return radius_x, radius_y, True

    if isinstance(bounds_min, list) and isinstance(bounds_max, list) and len(bounds_min) >= 2 and len(bounds_max) >= 2:
        half_width_world = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
        half_depth_world = abs(float(bounds_max[1]) - float(bounds_min[1])) * 0.5 * scale
        radius_x = max(1, int(round(half_width_world * pixels_per_world)))
        radius_y = max(1, int(round(half_depth_world * pixels_per_world)))
        return radius_x, radius_y, False

    base_radius_world = 3.0 * scale
    if category == "wmo":
        base_radius_world *= 2.0
    radius = max(1, int(round(base_radius_world * pixels_per_world)))
    return radius, radius, False


def _project_objects(
    objects: Optional[Sequence[Dict[str, object]]],
    tile_x: int,
    tile_y: int,
    output_size: int,
    wdl_heights: Optional[Dict[str, object]],
) -> Tuple[List[ProjectedObject], Dict[str, Any]]:
    projected: List[ProjectedObject] = []
    excluded_counts: Dict[str, int] = {}
    excluded_examples: Dict[str, List[str]] = {}
    included_examples: List[str] = []
    wdl_sampler = build_wdl_height_sampler(wdl_heights)

    def record_exclusion(reason: str, obj: Dict[str, object]) -> None:
        excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
        examples = excluded_examples.setdefault(reason, [])
        if len(examples) < 6:
            label = str(obj.get("model_path") or obj.get("name") or "unknown")
            examples.append(label)

    for obj in objects or []:
        if not isinstance(obj, dict):
            record_exclusion("invalid_object", {})
            continue

        category = str(obj.get("category", "") or "").lower()
        if category not in {"m2", "wmo"}:
            record_exclusion("unsupported_category", obj)
            continue

        if is_tree_like_object(obj):
            record_exclusion("tree_m2", obj)
            continue

        scale = _safe_scale(obj)
        local_uv = _choose_local_uv(obj, tile_x, tile_y)
        if local_uv is None:
            record_exclusion("outside_tile", obj)
            continue

        local_x, local_y = local_uv
        terrain_height = None
        pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
        if np.isfinite(pos_y) and wdl_sampler is not None:
            terrain_height = wdl_sampler(local_x, local_y, pos_y)
            if terrain_height is not None and np.isfinite(terrain_height):
                delta = float(pos_y - terrain_height)
                if delta < MASK_MIN_BELOW_TERRAIN or delta > MASK_MAX_ABOVE_TERRAIN:
                    record_exclusion("origin_height_delta", obj)
                    continue

        object_height_world = _object_height_world(obj, scale)
        if object_height_world is not None and object_height_world > MAX_MASK_OBJECT_HEIGHT_WORLD:
            record_exclusion("height_too_large", obj)
            continue

        if object_height_world is None and category == "m2" and scale > MAX_MASK_M2_SCALE_WITHOUT_BOUNDS:
            record_exclusion("scale_too_large_no_bounds", obj)
            continue

        if terrain_height is not None and np.isfinite(terrain_height):
            top_above_terrain = _object_top_above_terrain(obj, scale, float(terrain_height))
            if top_above_terrain is not None and top_above_terrain > MAX_MASK_OBJECT_TOP_ABOVE_TERRAIN:
                record_exclusion("top_above_terrain", obj)
                continue

        radius_x, radius_y, has_planar_bounds = _estimate_radii(obj, output_size, scale)
        center_x = int(round(np.clip(local_x, 0.0, 1.0) * (output_size - 1)))
        center_y = int(round(np.clip(local_y, 0.0, 1.0) * (output_size - 1)))
        allow_fallback = not (category == "wmo" and not has_planar_bounds)
        projected_obj = ProjectedObject(
            center_x=center_x,
            center_y=center_y,
            radius_x=radius_x,
            radius_y=radius_y,
            category=category,
            name=str(obj.get("name", "") or ""),
            model_path=str(obj.get("model_path", "") or ""),
            unique_id=int(obj.get("unique_id", 0) or 0),
            has_bounds=has_planar_bounds,
            allow_fallback=allow_fallback,
        )
        projected.append(projected_obj)
        if len(included_examples) < 8:
            included_examples.append(projected_obj.model_path or projected_obj.name or "unknown")

    debug = {
        "object_count": len(list(objects or [])),
        "included_count": len(projected),
        "excluded_counts": excluded_counts,
        "excluded_examples": excluded_examples,
        "included_examples": included_examples,
    }
    return projected, debug


def _ellipse_window(shape: Tuple[int, int], center_x: int, center_y: int, radius_x: int, radius_y: int) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    norm_x = ((xx - center_x) / max(radius_x, 1)) ** 2
    norm_y = ((yy - center_y) / max(radius_y, 1)) ** 2
    return (norm_x + norm_y) <= 1.0


def _filter_exported_mask(exported_mask: np.ndarray, projected: Sequence[ProjectedObject]) -> np.ndarray:
    result = np.zeros_like(exported_mask, dtype=np.uint8)
    active = exported_mask > 0
    if not np.any(active):
        return result

    for obj in projected:
        window_radius_x = max(EXPORTED_MASK_MIN_RADIUS, int(round(obj.radius_x * EXPORTED_MASK_WINDOW_SCALE)))
        window_radius_y = max(EXPORTED_MASK_MIN_RADIUS, int(round(obj.radius_y * EXPORTED_MASK_WINDOW_SCALE)))
        window = _ellipse_window(exported_mask.shape, obj.center_x, obj.center_y, window_radius_x, window_radius_y)
        result[np.logical_and(active, window)] = 1

    return result


def _build_fallback_mask(output_size: int, projected: Sequence[ProjectedObject]) -> np.ndarray:
    image = np.zeros((output_size, output_size), dtype=np.uint8)
    for obj in projected:
        if not obj.allow_fallback:
            continue
        ellipse = _ellipse_window(image.shape, obj.center_x, obj.center_y, obj.radius_x, obj.radius_y)
        image[ellipse] = 1
    return image


def build_object_context_mask(
    dataset_root: Path,
    terrain: Dict[str, object],
    tile_x: int,
    tile_y: int,
    output_size: int,
    precise_keys: Sequence[str],
    seeded_keys: Sequence[str],
    max_precise_coverage: float,
    max_seeded_coverage: float,
    max_fallback_coverage: float,
    return_debug: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    projected, debug = _project_objects(
        objects=terrain.get("objects"),
        tile_x=tile_x,
        tile_y=tile_y,
        output_size=output_size,
        wdl_heights=terrain.get("wdl_heights"),
    )

    zero = torch.zeros((1, output_size, output_size), dtype=torch.float32)
    if not projected:
        debug.update({"selected_source": "none", "coverage": 0.0})
        return (zero, debug) if return_debug else zero

    precise_mask = load_binary_mask(dataset_root, terrain, precise_keys, output_size)
    if precise_mask is not None:
        filtered_precise = _filter_exported_mask(precise_mask, projected)
        if is_mask_usable(filtered_precise, max_precise_coverage):
            tensor = torch.from_numpy(filtered_precise.astype(np.float32)).unsqueeze(0)
            debug.update({"selected_source": "precise_filtered", "coverage": mask_coverage(filtered_precise)})
            return (tensor, debug) if return_debug else tensor

    seeded_mask = load_binary_mask(dataset_root, terrain, seeded_keys, output_size)
    if seeded_mask is not None:
        filtered_seeded = _filter_exported_mask(seeded_mask, projected)
        if is_mask_usable(filtered_seeded, max_seeded_coverage):
            tensor = torch.from_numpy(filtered_seeded.astype(np.float32)).unsqueeze(0)
            debug.update({"selected_source": "seeded_filtered", "coverage": mask_coverage(filtered_seeded)})
            return (tensor, debug) if return_debug else tensor

    fallback_mask = _build_fallback_mask(output_size, projected)
    if is_mask_usable(fallback_mask, max_fallback_coverage):
        tensor = torch.from_numpy(fallback_mask.astype(np.float32)).unsqueeze(0)
        debug.update({"selected_source": "fallback_filtered", "coverage": mask_coverage(fallback_mask)})
        return (tensor, debug) if return_debug else tensor

    debug.update({"selected_source": "none", "coverage": 0.0})
    return (zero, debug) if return_debug else zero