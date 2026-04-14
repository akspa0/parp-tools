#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TILE_SIZE = 533.33333
MAP_ORIGIN = 32.0 * TILE_SIZE
MASK_CONTEXT_MARGIN_TILES = 0.20
PATCHES_PER_TILE = 256
PATCHES_PER_CHUNK = 16
OBJECT_GROUP_MARGIN_PATCHES = 8.0
OBJECT_CLUSTER_RADIUS_PATCHES = 10.0
OBJECT_SAME_MODEL_BONUS_RADIUS_PATCHES = 16.0
DEFAULT_MIN_FRACTAL_SCORE = 0.035
DEFAULT_SIMILARITY_THRESHOLD = 0.80


@dataclass(frozen=True)
class ProjectedObject:
    patch_x: float
    patch_y: float
    world_x: float
    world_y: float
    world_z: float
    category: str
    model_key: str
    name: str
    model_path: str
    scale: float
    unique_id: int
    rot_x: float
    rot_y: float
    rot_z: float
    bounds_min: Optional[Tuple[float, float, float]]
    bounds_max: Optional[Tuple[float, float, float]]


@dataclass
class TileContext:
    tile_name: str
    map_name: str
    tile_x: int
    tile_y: int
    terrain: Dict[str, Any]
    chunk_layers_by_idx: Dict[int, Dict[str, Any]]
    projected_objects: List[ProjectedObject]


def sha1_bytes(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def parse_bounds_triplet(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None

    try:
        parsed = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None

    return parsed if all(np.isfinite(component) for component in parsed) else None


def normalize_model_key(obj: Dict[str, Any]) -> str:
    model_path = str(obj.get("model_path") or "").strip().replace("\\", "/").lower()
    if model_path:
        return model_path
    category = str(obj.get("category") or "unknown").strip().lower()
    name = str(obj.get("name") or "unnamed").strip().lower()
    return f"{category}:{name}"


def normalize_texture_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/").lower()
    return text or "none"


def parse_tile_coords(tile_name: str) -> Tuple[int, int]:
    parts = tile_name.split("_")
    if len(parts) < 3:
        return 0, 0
    return int(parts[-2]), int(parts[-1])


def tile_uv_candidates(world_a: float, world_b: float, tile_x: int, tile_y: int) -> List[Tuple[float, float]]:
    return [
        (world_a / TILE_SIZE - float(tile_x), world_b / TILE_SIZE - float(tile_y)),
        ((MAP_ORIGIN - world_b) / TILE_SIZE - float(tile_x), (MAP_ORIGIN - world_a) / TILE_SIZE - float(tile_y)),
    ]


def choose_local_uv(obj: Dict[str, Any], tile_x: int, tile_y: int) -> Optional[Tuple[float, float]]:
    pos_x = float(obj.get("x", obj.get("pos_x", 0.0)) or 0.0)
    pos_y = float(obj.get("y", obj.get("pos_y", 0.0)) or 0.0)
    pos_z = float(obj.get("z", obj.get("pos_z", pos_y)) or pos_y)

    candidates: List[Tuple[float, float]] = []
    if abs(pos_x) < 2 and abs(pos_y) < 2:
        candidates.append(((pos_y + 1.0) * 0.5, (pos_x + 1.0) * 0.5))

    candidates.extend(tile_uv_candidates(pos_x, pos_z, tile_x, tile_y))
    if np.isfinite(pos_y):
        candidates.extend(tile_uv_candidates(pos_x, pos_y, tile_x, tile_y))

    best: Optional[Tuple[float, float]] = None
    best_overflow = float("inf")
    for local_x, local_y in candidates:
        overflow = (
            max(0.0, -local_x)
            + max(0.0, local_x - 1.0)
            + max(0.0, -local_y)
            + max(0.0, local_y - 1.0)
        )
        if overflow < best_overflow:
            best_overflow = overflow
            best = (local_x, local_y)
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


def project_objects(objects: Sequence[Dict[str, Any]], tile_x: int, tile_y: int) -> List[ProjectedObject]:
    projected: List[ProjectedObject] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue

        pos_x = float(obj.get("x", obj.get("pos_x", 0.0)) or 0.0)
        pos_y = float(obj.get("y", obj.get("pos_y", 0.0)) or 0.0)
        pos_z = float(obj.get("z", obj.get("pos_z", pos_y)) or pos_y)
        local_uv = choose_local_uv(obj, tile_x, tile_y)
        if local_uv is None:
            continue

        local_x, local_y = local_uv
        patch_x = float(np.clip(local_x, 0.0, 1.0) * (PATCHES_PER_TILE - 1))
        patch_y = float(np.clip(local_y, 0.0, 1.0) * (PATCHES_PER_TILE - 1))
        scale = float(obj.get("scale", 1.0) or 1.0)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        bounds_min = parse_bounds_triplet(obj.get("bounds_min"))
        bounds_max = parse_bounds_triplet(obj.get("bounds_max"))

        projected.append(
            ProjectedObject(
                patch_x=patch_x,
                patch_y=patch_y,
                world_x=pos_x,
                world_y=pos_y,
                world_z=pos_z,
                category=str(obj.get("category") or "").strip().lower(),
                model_key=normalize_model_key(obj),
                name=str(obj.get("name") or ""),
                model_path=str(obj.get("model_path") or ""),
                scale=scale,
                unique_id=int(obj.get("unique_id", 0) or 0),
                rot_x=float(obj.get("rot_x", 0.0) or 0.0),
                rot_y=float(obj.get("rot_y", 0.0) or 0.0),
                rot_z=float(obj.get("rot_z", 0.0) or 0.0),
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )
        )

    return projected


def hash_alpha_bits(alpha_bits: Any) -> str:
    if alpha_bits is None:
        return "none"

    encoded = str(alpha_bits).strip()
    if not encoded:
        return "none"

    try:
        payload = base64.b64decode(encoded, validate=False)
    except Exception:
        payload = encoded.encode("utf-8")

    return sha1_bytes(payload)


def quantize_grid(values: Sequence[float], width: int, height: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size != width * height:
        raise ValueError(f"Expected {width * height} values, got {array.size}.")
    grid = array.reshape(height, width)
    return np.clip(np.round(grid * 255.0), 0, 255).astype(np.uint8)


def resize_grid_nearest(grid: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if grid.shape == (target_height, target_width):
        return grid
    y_idx = np.clip(np.round(np.linspace(0, max(grid.shape[0] - 1, 0), target_height)).astype(np.int32), 0, max(grid.shape[0] - 1, 0))
    x_idx = np.clip(np.round(np.linspace(0, max(grid.shape[1] - 1, 0), target_width)).astype(np.int32), 0, max(grid.shape[1] - 1, 0))
    return grid[np.ix_(y_idx, x_idx)]


def coarsen_grid(grid: np.ndarray, max_size: int, levels: int) -> np.ndarray:
    current = grid.astype(np.float32) / 255.0
    while current.shape[0] > max_size or current.shape[1] > max_size:
        downsampled = block_average(current)
        if downsampled is None:
            break
        current = downsampled
    quantized = np.clip(np.round(current * float(levels - 1)), 0, levels - 1).astype(np.uint8)
    return resize_grid_nearest(quantized, max_size, max_size)


def block_average(grid: np.ndarray) -> Optional[np.ndarray]:
    height, width = grid.shape
    pooled_height = height // 2
    pooled_width = width // 2
    if pooled_height < 2 or pooled_width < 2:
        return None

    trimmed = grid[: pooled_height * 2, : pooled_width * 2]
    return trimmed.reshape(pooled_height, 2, pooled_width, 2).mean(axis=(1, 3))


def upsample_nearest(grid: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    upsampled = np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
    return upsampled[:target_height, :target_width]


def compute_fractal_detail_score(values: Sequence[float], width: int, height: int) -> float:
    current = np.asarray(values, dtype=np.float32).reshape(height, width)
    residuals: List[float] = []
    while True:
        downsampled = block_average(current)
        if downsampled is None:
            break
        approximated = upsample_nearest(downsampled, current.shape[0], current.shape[1])
        sample_height = min(current.shape[0], approximated.shape[0])
        sample_width = min(current.shape[1], approximated.shape[1])
        residuals.append(float(np.mean(np.abs(current[:sample_height, :sample_width] - approximated[:sample_height, :sample_width]))))
        current = downsampled
        if len(residuals) >= 4:
            break

    if not residuals:
        return 0.0

    weighted = sum(score * float(index + 1) for index, score in enumerate(residuals))
    normalizer = float(sum(range(1, len(residuals) + 1)))
    return round(weighted / max(normalizer, 1.0), 6)


def build_patch_mask(group: Dict[str, Any]) -> np.ndarray:
    width = int(group.get("patch_width", 0) or 0)
    height = int(group.get("patch_height", 0) or 0)
    mask = np.zeros((height, width), dtype=np.uint8)
    for patch in group.get("patches", []):
        local_x = int(patch.get("x", -1))
        local_y = int(patch.get("y", -1))
        if 0 <= local_x < width and 0 <= local_y < height:
            mask[local_y, local_x] = 255
    return mask


def build_chunk_signature(
    chunk_layers_by_idx: Dict[int, Dict[str, Any]],
    chunk_indices: Sequence[int],
) -> Tuple[str, List[str], int]:
    if not chunk_indices:
        return "none", [], 0

    min_chunk_x = min(int(index) % PATCHES_PER_CHUNK for index in chunk_indices)
    min_chunk_y = min(int(index) // PATCHES_PER_CHUNK for index in chunk_indices)
    seen_textures: set[str] = set()
    non_empty_alpha_count = 0
    tokens: List[str] = []

    for chunk_index in sorted({int(index) for index in chunk_indices}):
        chunk = chunk_layers_by_idx.get(chunk_index, {})
        rel_chunk_x = (chunk_index % PATCHES_PER_CHUNK) - min_chunk_x
        rel_chunk_y = (chunk_index // PATCHES_PER_CHUNK) - min_chunk_y

        layers = chunk.get("layers") if isinstance(chunk, dict) else None
        layer_tokens: List[str] = []
        for layer_index, layer in enumerate(layers or []):
            texture_path = normalize_texture_path(layer.get("texture_path"))
            flags = int(layer.get("flags", 0) or 0)
            effect_id = int(layer.get("effect_id", -1) or -1)
            alpha_hash = hash_alpha_bits(layer.get("alpha_bits"))
            if alpha_hash != "none":
                non_empty_alpha_count += 1
            if texture_path != "none":
                seen_textures.add(texture_path)
            layer_tokens.append(f"{layer_index}:{texture_path}:{flags}:{effect_id}:{alpha_hash}")

        if not layer_tokens:
            layer_tokens.append("empty")

        tokens.append(f"{rel_chunk_x},{rel_chunk_y}=>{'|'.join(layer_tokens)}")

    return sha1_bytes("||".join(tokens).encode("utf-8")), sorted(seen_textures), non_empty_alpha_count


def build_coarse_chunk_signature(
    chunk_texture_signatures: Sequence[str],
    non_empty_alpha_count: int,
    chunk_count: int,
) -> str:
    texture_token = "|".join(sorted({normalize_texture_path(value) for value in chunk_texture_signatures}))
    alpha_bucket = min(int(non_empty_alpha_count), 8)
    chunk_bucket = min(int(chunk_count), 16)
    return sha1_bytes(f"{texture_token}|alpha:{alpha_bucket}|chunks:{chunk_bucket}".encode("utf-8"))


def cluster_objects(objects: Sequence[ProjectedObject]) -> List[List[ProjectedObject]]:
    clusters: List[List[ProjectedObject]] = []
    visited: set[int] = set()

    for index, seed in enumerate(objects):
        if index in visited:
            continue

        queue = [index]
        visited.add(index)
        cluster: List[ProjectedObject] = []

        while queue:
            current_index = queue.pop()
            current = objects[current_index]
            cluster.append(current)

            for other_index, other in enumerate(objects):
                if other_index in visited:
                    continue
                delta_x = other.patch_x - current.patch_x
                delta_y = other.patch_y - current.patch_y
                distance = math.hypot(delta_x, delta_y)
                if distance <= OBJECT_CLUSTER_RADIUS_PATCHES or (
                    other.model_key == current.model_key and distance <= OBJECT_SAME_MODEL_BONUS_RADIUS_PATCHES
                ):
                    visited.add(other_index)
                    queue.append(other_index)

        clusters.append(sorted(cluster, key=lambda obj: (obj.model_key, obj.patch_y, obj.patch_x)))

    return clusters


def select_nearby_projected_objects(
    projected_objects: Sequence[ProjectedObject],
    group: Dict[str, Any],
) -> List[ProjectedObject]:
    min_patch_x = float(group.get("patch_min_x", 0) or 0) - OBJECT_GROUP_MARGIN_PATCHES
    min_patch_y = float(group.get("patch_min_y", 0) or 0) - OBJECT_GROUP_MARGIN_PATCHES
    max_patch_x = float(group.get("patch_max_x", 0) or 0) + OBJECT_GROUP_MARGIN_PATCHES
    max_patch_y = float(group.get("patch_max_y", 0) or 0) + OBJECT_GROUP_MARGIN_PATCHES

    return [
        obj
        for obj in projected_objects
        if min_patch_x <= obj.patch_x <= max_patch_x and min_patch_y <= obj.patch_y <= max_patch_y
    ]


def cluster_distance_to_group(cluster: Sequence[ProjectedObject], group: Dict[str, Any]) -> float:
    if not cluster:
        return float("inf")

    group_center_x = (float(group.get("patch_min_x", 0) or 0) + float(group.get("patch_max_x", 0) or 0)) * 0.5
    group_center_y = (float(group.get("patch_min_y", 0) or 0) + float(group.get("patch_max_y", 0) or 0)) * 0.5
    cluster_center_x = float(sum(obj.patch_x for obj in cluster)) / float(len(cluster))
    cluster_center_y = float(sum(obj.patch_y for obj in cluster)) / float(len(cluster))
    return math.hypot(cluster_center_x - group_center_x, cluster_center_y - group_center_y)


def select_prefab_object_cluster(
    projected_objects: Sequence[ProjectedObject],
    group: Dict[str, Any],
) -> Tuple[List[ProjectedObject], int]:
    nearby = select_nearby_projected_objects(projected_objects, group)
    if not nearby:
        return [], 0

    clusters = cluster_objects(nearby)

    def cluster_key(cluster: Sequence[ProjectedObject]) -> Tuple[int, int, int, float]:
        unique_models = len({obj.model_key for obj in cluster})
        return (
            1 if len(cluster) >= 2 else 0,
            len(cluster),
            unique_models,
            -cluster_distance_to_group(cluster, group),
        )

    selected = max(clusters, key=cluster_key)
    ordered = sorted(selected, key=lambda obj: (obj.model_key, obj.patch_y, obj.patch_x, obj.unique_id))
    return ordered, len(clusters)


def object_world_extents(obj: ProjectedObject) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if obj.bounds_min is not None and obj.bounds_max is not None:
        scale = max(float(obj.scale), 0.05)
        min_corner = (
            obj.world_x + obj.bounds_min[0] * scale,
            obj.world_y + obj.bounds_min[1] * scale,
            obj.world_z + obj.bounds_min[2] * scale,
        )
        max_corner = (
            obj.world_x + obj.bounds_max[0] * scale,
            obj.world_y + obj.bounds_max[1] * scale,
            obj.world_z + obj.bounds_max[2] * scale,
        )
        return min_corner, max_corner

    footprint = PATCHES_PER_TILE and (TILE_SIZE / float(PATCHES_PER_TILE)) * max(0.75, min(float(obj.scale) * 1.5, 6.0))
    height = (TILE_SIZE / float(PATCHES_PER_TILE)) * max(2.0, min(float(obj.scale) * 6.0, 12.0))
    min_corner = (obj.world_x - footprint, obj.world_y, obj.world_z - footprint)
    max_corner = (obj.world_x + footprint, obj.world_y + height, obj.world_z + footprint)
    return min_corner, max_corner


def build_prefab_object_records(selected_objects: Sequence[ProjectedObject]) -> List[Dict[str, Any]]:
    if not selected_objects:
        return []

    patch_origin_x = min(obj.patch_x for obj in selected_objects)
    patch_origin_y = min(obj.patch_y for obj in selected_objects)

    extents = [object_world_extents(obj) for obj in selected_objects]
    world_origin_x = min(min_corner[0] for min_corner, _ in extents)
    world_origin_y = min(min_corner[1] for min_corner, _ in extents)
    world_origin_z = min(min_corner[2] for min_corner, _ in extents)

    records: List[Dict[str, Any]] = []
    for obj in selected_objects:
        records.append(
            {
                "name": obj.name,
                "category": obj.category,
                "model_key": obj.model_key,
                "model_path": obj.model_path,
                "unique_id": obj.unique_id,
                "scale": round(float(obj.scale), 6),
                "rot_x": round(float(obj.rot_x), 6),
                "rot_y": round(float(obj.rot_y), 6),
                "rot_z": round(float(obj.rot_z), 6),
                "patch_x": round(float(obj.patch_x), 6),
                "patch_y": round(float(obj.patch_y), 6),
                "rel_patch_x": round(float(obj.patch_x - patch_origin_x), 6),
                "rel_patch_y": round(float(obj.patch_y - patch_origin_y), 6),
                "world_x": round(float(obj.world_x), 6),
                "world_y": round(float(obj.world_y), 6),
                "world_z": round(float(obj.world_z), 6),
                "local_x": round(float(obj.world_x - world_origin_x), 6),
                "local_y": round(float(obj.world_y - world_origin_y), 6),
                "local_z": round(float(obj.world_z - world_origin_z), 6),
                "bounds_min": list(obj.bounds_min) if obj.bounds_min is not None else None,
                "bounds_max": list(obj.bounds_max) if obj.bounds_max is not None else None,
            }
        )

    return records


def build_object_signature(
    projected_objects: Sequence[ProjectedObject],
    group: Dict[str, Any],
) -> Tuple[str, str, List[str], int, int, List[Dict[str, Any]]]:
    selected_cluster, cluster_count = select_prefab_object_cluster(projected_objects, group)
    prefab_objects = build_prefab_object_records(selected_cluster)
    if not prefab_objects:
        return "none", "none", [], 0, cluster_count, []

    descriptors: List[str] = []
    histogram: Dict[str, int] = defaultdict(int)
    for obj in prefab_objects:
        rel_x = int(round(float(obj.get("rel_patch_x", 0.0) or 0.0) * 2.0))
        rel_y = int(round(float(obj.get("rel_patch_y", 0.0) or 0.0) * 2.0))
        scale_bucket = round(float(obj.get("scale", 1.0) or 1.0), 2)
        model_key = str(obj.get("model_key") or "unknown")
        histogram[model_key] += 1
        descriptors.append(f"{model_key}@{rel_x}:{rel_y}:{scale_bucket}")

    object_models = sorted(histogram.keys())
    coarse_tokens = [f"{model}:{min(histogram[model], 4)}" for model in object_models]
    coarse_tokens.append(f"objects:{min(len(prefab_objects), 8)}")
    overall_signature = sha1_bytes("|".join(descriptors).encode("utf-8"))
    coarse_signature = sha1_bytes("|".join(coarse_tokens).encode("utf-8"))
    return overall_signature, coarse_signature, object_models, len(prefab_objects), cluster_count, prefab_objects


def build_occurrence(
    group: Dict[str, Any],
    tile_context: TileContext,
    group_rel_path: str,
    min_fractal_score: float,
) -> Dict[str, Any]:
    patch_mask = build_patch_mask(group)
    coarse_patch_mask_grid = coarsen_grid(patch_mask, max_size=8, levels=8)
    patch_mask_signature = sha1_bytes(patch_mask.tobytes())
    coarse_patch_mask_signature = sha1_bytes(coarse_patch_mask_grid.tobytes())

    height_grid_width = int(group.get("height_grid_width", 0) or 0)
    height_grid_height = int(group.get("height_grid_height", 0) or 0)
    height_values = group.get("normalized_height_grid", [])
    height_grid = quantize_grid(height_values, height_grid_width, height_grid_height)
    coarse_height_grid = coarsen_grid(height_grid, max_size=8, levels=16)
    height_signature = sha1_bytes(height_grid.tobytes())
    coarse_height_signature = sha1_bytes(coarse_height_grid.tobytes())
    fractal_detail_score = compute_fractal_detail_score(height_values, height_grid_width, height_grid_height)
    fractal_candidate = fractal_detail_score >= min_fractal_score

    chunk_indices = sorted({int(patch.get("chunk_index", -1)) for patch in group.get("patches", []) if int(patch.get("chunk_index", -1)) >= 0})
    alpha_signature, chunk_texture_signatures, non_empty_alpha_count = build_chunk_signature(
        tile_context.chunk_layers_by_idx,
        chunk_indices,
    )
    coarse_alpha_signature = build_coarse_chunk_signature(chunk_texture_signatures, non_empty_alpha_count, len(chunk_indices))
    object_signature, coarse_object_signature, object_models, prefab_object_count, object_cluster_count, prefab_objects = build_object_signature(
        tile_context.projected_objects,
        group,
    )

    exact_terrain_signature = sha1_bytes(
        "|".join(
            [
                patch_mask_signature,
                height_signature,
                alpha_signature,
                ",".join(chunk_texture_signatures),
            ]
        ).encode("utf-8")
    )
    terrain_signature = sha1_bytes(
        "|".join(
            [
                coarse_patch_mask_signature,
                coarse_height_signature,
                coarse_alpha_signature,
            ]
        ).encode("utf-8")
    )
    prefab_signature = sha1_bytes(f"{terrain_signature}|{coarse_object_signature}".encode("utf-8"))
    prefab_signature_exact = sha1_bytes(f"{exact_terrain_signature}|{object_signature}".encode("utf-8"))

    return {
        "group_id": str(group.get("group_id") or Path(group_rel_path).stem),
        "group_file": group_rel_path,
        "tile_name": tile_context.tile_name,
        "map_name": tile_context.map_name,
        "prefab_signature": prefab_signature,
        "prefab_signature_exact": prefab_signature_exact,
        "terrain_signature": terrain_signature,
        "terrain_signature_exact": exact_terrain_signature,
        "height_signature": height_signature,
        "height_signature_coarse": coarse_height_signature,
        "patch_mask_signature": patch_mask_signature,
        "patch_mask_signature_coarse": coarse_patch_mask_signature,
        "alpha_signature": alpha_signature,
        "alpha_signature_coarse": coarse_alpha_signature,
        "object_signature": object_signature,
        "object_signature_coarse": coarse_object_signature,
        "patch_min_x": int(group.get("patch_min_x", 0) or 0),
        "patch_min_y": int(group.get("patch_min_y", 0) or 0),
        "patch_max_x": int(group.get("patch_max_x", 0) or 0),
        "patch_max_y": int(group.get("patch_max_y", 0) or 0),
        "patch_width": int(group.get("patch_width", 0) or 0),
        "patch_height": int(group.get("patch_height", 0) or 0),
        "patch_count": int(group.get("patch_count", 0) or 0),
        "chunk_indices": chunk_indices,
        "chunk_texture_signatures": chunk_texture_signatures,
        "group_texture_signatures": sorted({normalize_texture_path(value) for value in group.get("texture_signatures", [])}),
        "non_empty_alpha_count": non_empty_alpha_count,
        "fractal_detail_score": fractal_detail_score,
        "fractal_candidate": fractal_candidate,
        "brush_mean_score": float(group.get("mean_score", 0.0) or 0.0),
        "brush_max_score": float(group.get("max_score", 0.0) or 0.0),
        "nearby_object_count": len(select_nearby_projected_objects(tile_context.projected_objects, group)),
        "prefab_object_count": prefab_object_count,
        "object_cluster_count": object_cluster_count,
        "object_models": object_models,
        "prefab_objects": prefab_objects,
        "source_image_path": group.get("source_image_path"),
        "heightmap_global_path": group.get("heightmap_global_path"),
        "_coarse_height_grid": coarse_height_grid.reshape(-1).tolist(),
        "_coarse_patch_mask_grid": coarse_patch_mask_grid.reshape(-1).tolist(),
        "_texture_tokens": sorted({*chunk_texture_signatures, *({normalize_texture_path(value) for value in group.get("texture_signatures", [])})}),
        "_object_tokens": [str(item.get("model_key") or "") for item in prefab_objects],
    }


def jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(value) for value in left if str(value)}
    right_set = {str(value) for value in right if str(value)}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set)) / float(len(union))


def ratio_similarity(left: int, right: int) -> float:
    max_value = max(int(left), int(right), 1)
    min_value = min(int(left), int(right), 1)
    return float(min_value) / float(max_value)


def occurrence_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_height = np.asarray(left.get("_coarse_height_grid", []), dtype=np.float32)
    right_height = np.asarray(right.get("_coarse_height_grid", []), dtype=np.float32)
    left_mask = np.asarray(left.get("_coarse_patch_mask_grid", []), dtype=np.float32)
    right_mask = np.asarray(right.get("_coarse_patch_mask_grid", []), dtype=np.float32)

    if left_height.size == 0 or right_height.size == 0 or left_mask.size == 0 or right_mask.size == 0:
        return 0.0

    height_similarity = 1.0 - float(np.mean(np.abs(left_height - right_height)) / 15.0)
    mask_similarity = 1.0 - float(np.mean(np.abs(left_mask - right_mask)) / 7.0)
    texture_similarity = jaccard_similarity(left.get("_texture_tokens", []), right.get("_texture_tokens", []))
    object_similarity = jaccard_similarity(left.get("_object_tokens", []), right.get("_object_tokens", []))
    patch_similarity = ratio_similarity(int(left.get("patch_count", 0) or 0), int(right.get("patch_count", 0) or 0))

    has_prefab_objects = int(left.get("prefab_object_count", 0) or 0) > 0 or int(right.get("prefab_object_count", 0) or 0) > 0
    if has_prefab_objects:
        score = (
            0.55 * object_similarity
            + 0.20 * max(0.0, min(height_similarity, 1.0))
            + 0.10 * max(0.0, min(mask_similarity, 1.0))
            + 0.10 * texture_similarity
            + 0.05 * patch_similarity
        )
    else:
        score = (
            0.45 * max(0.0, min(height_similarity, 1.0))
            + 0.20 * max(0.0, min(mask_similarity, 1.0))
            + 0.15 * texture_similarity
            + 0.10 * object_similarity
            + 0.10 * patch_similarity
        )
    return round(max(0.0, min(score, 1.0)), 6)


def primary_texture_token(occurrence: Dict[str, Any]) -> str:
    tokens = [str(value) for value in occurrence.get("_texture_tokens", []) if str(value) and str(value) != "none"]
    return tokens[0] if tokens else "none"


def primary_object_token(occurrence: Dict[str, Any]) -> str:
    tokens = [str(value) for value in occurrence.get("_object_tokens", []) if str(value) and str(value) != "none"]
    return tokens[0] if tokens else "none"


def sanitize_occurrence(occurrence: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in occurrence.items() if not key.startswith("_")}


def cluster_occurrences_by_similarity(
    occurrences: Sequence[Dict[str, Any]],
    similarity_threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    sorted_occurrences = sorted(
        occurrences,
        key=lambda entry: (
            primary_texture_token(entry),
            str(entry.get("tile_name") or ""),
            str(entry.get("group_id") or ""),
        ),
    )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    exemplars_by_texture: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    exemplars_by_object: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

    for occurrence in sorted_occurrences:
        texture_key = primary_texture_token(occurrence)
        object_key = primary_object_token(occurrence)
        candidate_map: Dict[str, Dict[str, Any]] = {}
        for candidate_signature, exemplar in exemplars_by_texture.get(texture_key, []):
            candidate_map[candidate_signature] = exemplar
        if texture_key != "none":
            for candidate_signature, exemplar in exemplars_by_texture.get("none", []):
                candidate_map[candidate_signature] = exemplar
        for candidate_signature, exemplar in exemplars_by_object.get(object_key, []):
            candidate_map[candidate_signature] = exemplar
        if object_key != "none":
            for candidate_signature, exemplar in exemplars_by_object.get("none", []):
                candidate_map[candidate_signature] = exemplar

        best_signature: Optional[str] = None
        best_score = -1.0
        for candidate_signature, exemplar in candidate_map.items():
            score = occurrence_similarity(occurrence, exemplar)
            if score >= similarity_threshold and score > best_score:
                best_signature = candidate_signature
                best_score = score

        if best_signature is None:
            best_signature = sha1_bytes(f"cluster|{occurrence['prefab_signature_exact']}".encode("utf-8"))
            exemplars_by_texture[texture_key].append((best_signature, occurrence))
            exemplars_by_object[object_key].append((best_signature, occurrence))

        occurrence["prefab_signature"] = best_signature
        occurrence["similarity_score_to_exemplar"] = None if best_score < 0.0 else best_score
        grouped[best_signature].append(occurrence)

    return grouped


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tile_context(dataset_root: Path, tile_name: str, cache: Dict[str, TileContext]) -> TileContext:
    cached = cache.get(tile_name)
    if cached is not None:
        return cached

    tile_path = dataset_root / "dataset" / f"{tile_name}.json"
    payload = load_json(tile_path)
    terrain = payload.get("terrain_data", {})
    tile_x, tile_y = parse_tile_coords(tile_name)
    chunk_layers_by_idx = {
        int(entry.get("idx", -1)): entry
        for entry in terrain.get("chunk_layers", [])
        if int(entry.get("idx", -1)) >= 0
    }
    projected_objects = project_objects(terrain.get("objects", []), tile_x, tile_y)

    context = TileContext(
        tile_name=tile_name,
        map_name=str(terrain.get("adt_tile") or tile_name).split("_")[0],
        tile_x=tile_x,
        tile_y=tile_y,
        terrain=terrain,
        chunk_layers_by_idx=chunk_layers_by_idx,
        projected_objects=projected_objects,
    )
    cache[tile_name] = context
    return context


def collect_group_files(
    dataset_root: Path,
    tile_filter: Optional[str],
    limit_groups: Optional[int],
) -> Tuple[Path, List[str]]:
    manifest_path = dataset_root / "brush_imprints" / "brush_imprint_manifest.json"
    manifest = load_json(manifest_path)
    group_files = [str(rel) for rel in manifest.get("group_files", [])]
    if tile_filter:
        lowered = tile_filter.lower()
        group_files = [rel for rel in group_files if lowered in rel.lower()]
    if limit_groups is not None and limit_groups > 0:
        group_files = group_files[:limit_groups]
    return manifest_path, group_files


def build_prefab_summary(
    prefab_id: str,
    occurrences: Sequence[Dict[str, Any]],
    prefab_file_rel: str,
    min_fractal_score: float,
) -> Dict[str, Any]:
    first = occurrences[0]
    fractal_scores = [float(entry.get("fractal_detail_score", 0.0) or 0.0) for entry in occurrences]
    texture_signatures = sorted(
        {
            texture
            for entry in occurrences
            for texture in entry.get("chunk_texture_signatures", []) + entry.get("group_texture_signatures", [])
        }
    )
    object_models = sorted({model for entry in occurrences for model in entry.get("object_models", [])})
    maps = sorted({str(entry.get("map_name") or "") for entry in occurrences})
    tiles = sorted({str(entry.get("tile_name") or "") for entry in occurrences})

    return {
        "prefab_id": prefab_id,
        "hash": first["prefab_signature"],
        "hash_exact_representative": first["prefab_signature_exact"],
        "terrain_signature": first["terrain_signature"],
        "terrain_signature_exact": first["terrain_signature_exact"],
        "alpha_signature": first["alpha_signature"],
        "object_signature": first["object_signature"],
        "count": len(occurrences),
        "maps": maps,
        "tiles": tiles,
        "texture_signatures": texture_signatures,
        "object_models": object_models,
        "mean_fractal_detail_score": round(float(np.mean(fractal_scores)), 6),
        "max_fractal_detail_score": round(float(np.max(fractal_scores)), 6),
        "fractal_candidate_count": int(sum(1 for score in fractal_scores if score >= min_fractal_score)),
        "representative_group_id": first["group_id"],
        "file": prefab_file_rel,
    }


def write_prefab_library(
    dataset_root: Path,
    output_dir: Path,
    manifest_path: Path,
    grouped_occurrences: Dict[str, List[Dict[str, Any]]],
    groups_seen: int,
    groups_processed: int,
    groups_skipped: int,
    min_fractal_score: float,
    error_examples: Sequence[Dict[str, str]],
    similarity_threshold: float,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefabs_dir = output_dir / "prefabs"
    prefabs_dir.mkdir(parents=True, exist_ok=True)

    sorted_entries = sorted(
        grouped_occurrences.items(),
        key=lambda item: (-len(item[1]), -max(float(entry.get("fractal_detail_score", 0.0) or 0.0) for entry in item[1]), item[0]),
    )

    summaries: List[Dict[str, Any]] = []
    instances_map: Dict[str, List[str]] = {}
    prefab_files: List[str] = []
    fractal_candidate_occurrences = 0

    for index, (signature, occurrences) in enumerate(sorted_entries, start=1):
        prefab_id = f"prefab_{index:05d}"
        prefab_path = prefabs_dir / f"{prefab_id}.json"
        prefab_file_rel = Path("prefabs") / prefab_path.name
        prefab_files.append(prefab_file_rel.as_posix())
        instances_map[signature] = [str(entry.get("group_id") or "") for entry in occurrences]
        fractal_candidate_occurrences += sum(1 for entry in occurrences if bool(entry.get("fractal_candidate")))
        output_occurrences = [sanitize_occurrence(entry) for entry in occurrences]

        prefab_payload = {
            "schema_version": "wowviewer-ml-prefab-object.v2",
            "prefab_id": prefab_id,
            "dataset_root": str(dataset_root),
            "prefab_signature": signature,
            "prefab_kind": "object-assembly",
            "grouping_strategy": "similarity-cluster-v2-object-weighted",
            "similarity_threshold": similarity_threshold,
            "terrain_signature": output_occurrences[0]["terrain_signature"],
            "terrain_signature_exact": output_occurrences[0]["terrain_signature_exact"],
            "alpha_signature": output_occurrences[0]["alpha_signature"],
            "object_signature": output_occurrences[0]["object_signature"],
            "occurrence_count": len(output_occurrences),
            "occurrences": output_occurrences,
        }
        with open(prefab_path, "w", encoding="utf-8") as handle:
            json.dump(prefab_payload, handle, indent=2)

        summaries.append(build_prefab_summary(prefab_id, output_occurrences, prefab_file_rel.as_posix(), min_fractal_score))

    library_payload = {
        "schema_version": "wowviewer-ml-prefab-library.v2",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_root": str(dataset_root),
        "source_brush_manifest": str(manifest_path.relative_to(dataset_root)).replace("\\", "/"),
        "groups_seen": groups_seen,
        "groups_processed": groups_processed,
        "groups_skipped": groups_skipped,
        "unique_prefabs": len(summaries),
        "fractal_candidate_occurrences": fractal_candidate_occurrences,
        "min_fractal_score": min_fractal_score,
        "prefab_kind": "object-assembly",
        "grouping_strategy": "similarity-cluster-v2-object-weighted",
        "similarity_threshold": similarity_threshold,
        "error_examples": list(error_examples),
        "prefabs": summaries,
    }

    with open(output_dir / "prefab_library.json", "w", encoding="utf-8") as handle:
        json.dump(library_payload, handle, indent=2)

    with open(output_dir / "prefab_instances.json", "w", encoding="utf-8") as handle:
        json.dump(instances_map, handle, indent=2)

    manifest_payload = {
        "schema_version": "wowviewer-ml-prefab-library-manifest.v2",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_root": str(dataset_root),
        "source_brush_manifest": str(manifest_path.relative_to(dataset_root)).replace("\\", "/"),
        "output_directory": str(output_dir),
        "groups_seen": groups_seen,
        "groups_processed": groups_processed,
        "groups_skipped": groups_skipped,
        "prefabs_written": len(summaries),
        "occurrences_written": sum(len(entries) for entries in grouped_occurrences.values()),
        "fractal_candidate_occurrences": fractal_candidate_occurrences,
        "min_fractal_score": min_fractal_score,
        "prefab_kind": "object-assembly",
        "grouping_strategy": "similarity-cluster-v2-object-weighted",
        "similarity_threshold": similarity_threshold,
        "error_examples": list(error_examples),
        "prefab_files": prefab_files,
    }

    with open(output_dir / "prefab_library_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    return manifest_payload


def build_library(
    dataset_root: Path,
    output_dir: Path,
    tile_filter: Optional[str],
    limit_groups: Optional[int],
    min_fractal_score: float,
    similarity_threshold: float,
) -> Dict[str, Any]:
    manifest_path, group_files = collect_group_files(dataset_root, tile_filter, limit_groups)
    tile_cache: Dict[str, TileContext] = {}
    all_occurrences: List[Dict[str, Any]] = []
    groups_processed = 0
    groups_skipped = 0
    error_examples: List[Dict[str, str]] = []

    for group_rel in group_files:
        group_path = dataset_root / "brush_imprints" / group_rel
        try:
            group = load_json(group_path)
            tile_name = str(group.get("tile_name") or "")
            if not tile_name:
                groups_skipped += 1
                continue
            tile_context = load_tile_context(dataset_root, tile_name, tile_cache)
            occurrence = build_occurrence(group, tile_context, group_rel, min_fractal_score)
            all_occurrences.append(occurrence)
            groups_processed += 1
        except Exception as exc:
            groups_skipped += 1
            if len(error_examples) < 20:
                error_examples.append(
                    {
                        "group_file": group_rel,
                        "error": str(exc),
                    }
                )

    grouped_occurrences = cluster_occurrences_by_similarity(all_occurrences, similarity_threshold)

    return write_prefab_library(
        dataset_root=dataset_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        grouped_occurrences=grouped_occurrences,
        groups_seen=len(group_files),
        groups_processed=groups_processed,
        groups_skipped=groups_skipped,
        min_fractal_score=min_fractal_score,
        error_examples=error_examples,
        similarity_threshold=similarity_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an object-assembly prefab library from brush groups and tile dataset payloads.")
    parser.add_argument("dataset_root", help="Dataset root that already contains dataset/ and brush_imprints/.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <dataset_root>/prefab_library.")
    parser.add_argument("--tile-filter", help="Optional substring filter applied to brush group paths.")
    parser.add_argument("--limit-groups", type=int, help="Optional hard cap on processed brush groups.")
    parser.add_argument("--min-fractal-score", type=float, default=DEFAULT_MIN_FRACTAL_SCORE, help="Score threshold used to flag a prefab occurrence as fractal-heavy.")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Similarity score threshold used to merge occurrences into the same prefab cluster.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_root / "prefab_library"

    manifest = build_library(
        dataset_root=dataset_root,
        output_dir=output_dir,
        tile_filter=args.tile_filter,
        limit_groups=args.limit_groups,
        min_fractal_score=float(args.min_fractal_score),
        similarity_threshold=float(args.similarity_threshold),
    )

    print("Prefab library build complete")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_dir: {output_dir}")
    print(f"  groups_seen: {manifest['groups_seen']}")
    print(f"  groups_processed: {manifest['groups_processed']}")
    print(f"  groups_skipped: {manifest['groups_skipped']}")
    print(f"  prefabs_written: {manifest['prefabs_written']}")
    print(f"  occurrences_written: {manifest['occurrences_written']}")
    print(f"  fractal_candidate_occurrences: {manifest['fractal_candidate_occurrences']}")


if __name__ == "__main__":
    main()