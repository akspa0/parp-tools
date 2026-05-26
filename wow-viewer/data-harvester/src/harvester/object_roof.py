"""Shared helpers for object-roof exemplar curation and mask projection.

This module centralizes the schema and deterministic ID rules used by the
object-roof library scripts so the curation and mask stages stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Iterable

import numpy as np

TILE_WORLD_SIZE = 533.3333333333334
WORLD_TILE_ORIGIN = 32.0


@dataclass(frozen=True)
class RoofFamilySummary:
    family_id: str
    canonical_asset_path: str
    exemplar_count: int
    canonical_exemplar_id: str
    review_state: str
    review_required: bool


@dataclass(frozen=True)
class RoofExemplarRecord:
    exemplar_id: str
    family_id: str
    variant_rank: int
    is_canonical: bool
    asset_path: str
    instance_type: str
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    instance_idx: int
    unique_id: int
    pose_rot_x: float
    pose_rot_y: float
    pose_rot_z: float
    pose_scale: float
    bbox_xyxy: tuple[int, int, int, int]
    bbox_wh: tuple[int, int]
    crop_size: int
    mask_coverage: float
    minimap_mean: float
    minimap_std: float
    provenance_key: str
    review_state: str
    review_required: bool


def normalize_asset_path(path: str) -> str:
    """Normalize asset paths for stable family IDs and catalog joins."""
    text = (path or "").replace("\\", "/").strip().lower()
    while "//" in text:
        text = text.replace("//", "/")
    return text


def family_id_from_asset_path(path: str) -> str:
    normalized = normalize_asset_path(path)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"rooffam_{digest[:14]}"


def exemplar_id_from_parts(parts: Iterable[str | int]) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"roof_{digest[:16]}"


def variant_fingerprint_from_rgb(rgb: np.ndarray, hash_size: int = 16) -> str:
    """Compact perceptual hash used for variant dedupe within an asset family."""
    if rgb.ndim != 3:
        raise ValueError("Expected RGB array for fingerprint")
    luminance = rgb.astype(np.float32).mean(axis=2)
    # Use nearest-neighbor downsample to keep deterministic edge profile.
    target_w = hash_size + 1
    target_h = hash_size
    ys = np.linspace(0, luminance.shape[0] - 1, target_h).astype(np.int32)
    xs = np.linspace(0, luminance.shape[1] - 1, target_w).astype(np.int32)
    sampled = luminance[np.ix_(ys, xs)]
    diff = sampled[:, 1:] > sampled[:, :-1]
    bits = "".join("1" if flag else "0" for flag in diff.reshape(-1).tolist())
    width = (len(bits) + 3) // 4
    return f"{int(bits, 2):0{width}x}"


def build_map_tile_key(build: str, map_name: str, tile_x: int, tile_y: int) -> str:
    return f"{build}|{map_name}|{int(tile_x)}|{int(tile_y)}"


def tile_world_bounds(tile_x: int, tile_y: int) -> tuple[float, float, float, float]:
    """Return world-space bounds (x_min, x_max, y_min, y_max) for a tile."""
    x_max = (WORLD_TILE_ORIGIN - float(tile_x)) * TILE_WORLD_SIZE
    x_min = x_max - TILE_WORLD_SIZE
    y_max = (WORLD_TILE_ORIGIN - float(tile_y)) * TILE_WORLD_SIZE
    y_min = y_max - TILE_WORLD_SIZE
    return x_min, x_max, y_min, y_max


def world_xy_to_tile_pixel(x: float, y: float, tile_x: int, tile_y: int, tile_size: int = 256) -> tuple[float, float]:
    """Project world coordinates into tile-local pixel space.

    The transform follows the same 32-origin tile convention used by ADT tile
    naming. Output is not clipped so callers can apply bbox clipping policy.
    """
    x_min, x_max, y_min, y_max = tile_world_bounds(tile_x, tile_y)
    px = ((x_max - float(x)) / TILE_WORLD_SIZE) * float(tile_size - 1)
    py = ((y_max - float(y)) / TILE_WORLD_SIZE) * float(tile_size - 1)
    return px, py


def world_bbox_to_tile_bbox_xyxy(
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    tile_x: int,
    tile_y: int,
    tile_size: int = 256,
    padding_px: int = 2,
) -> tuple[int, int, int, int] | None:
    """Project a world-space bbox into an inclusive tile-space bbox.

    Returns None when the projected box does not overlap the tile.
    """
    corners = [
        world_xy_to_tile_pixel(min_x, min_y, tile_x, tile_y, tile_size=tile_size),
        world_xy_to_tile_pixel(min_x, max_y, tile_x, tile_y, tile_size=tile_size),
        world_xy_to_tile_pixel(max_x, min_y, tile_x, tile_y, tile_size=tile_size),
        world_xy_to_tile_pixel(max_x, max_y, tile_x, tile_y, tile_size=tile_size),
    ]
    xs = [xy[0] for xy in corners]
    ys = [xy[1] for xy in corners]

    x0 = int(np.floor(min(xs))) - int(padding_px)
    y0 = int(np.floor(min(ys))) - int(padding_px)
    x1 = int(np.ceil(max(xs))) + int(padding_px)
    y1 = int(np.ceil(max(ys))) + int(padding_px)

    x0 = max(0, min(tile_size - 1, x0))
    y0 = max(0, min(tile_size - 1, y0))
    x1 = max(0, min(tile_size - 1, x1))
    y1 = max(0, min(tile_size - 1, y1))

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def d1_style_bbox_fallback(
    *,
    pos_x: float,
    pos_y: float,
    scale: float,
    tile_x: int,
    tile_y: int,
    tile_size: int = 256,
    base_radius_px: float = 6.0,
) -> tuple[int, int, int, int] | None:
    """Fallback bbox projection for placements without explicit world bbox.

    This is intentionally conservative and only used for MDDF-style placements
    that do not carry bounding extents.
    """
    px, py = world_xy_to_tile_pixel(pos_x, pos_y, tile_x, tile_y, tile_size=tile_size)
    radius = max(2.0, float(base_radius_px) * max(float(scale), 0.2))
    x0 = int(np.floor(px - radius))
    y0 = int(np.floor(py - radius))
    x1 = int(np.ceil(px + radius))
    y1 = int(np.ceil(py + radius))

    x0 = max(0, min(tile_size - 1, x0))
    y0 = max(0, min(tile_size - 1, y0))
    x1 = max(0, min(tile_size - 1, x1))
    y1 = max(0, min(tile_size - 1, y1))

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def pose_vector_from_placement(row: dict[str, object]) -> np.ndarray:
    """Pack pose metadata into a fixed-size float32 vector."""
    return np.asarray(
        [
            float(row.get("rotX", 0.0) or 0.0),
            float(row.get("rotY", 0.0) or 0.0),
            float(row.get("rotZ", 0.0) or 0.0),
            float(row.get("scale", 1.0) or 1.0),
            float(row.get("posX", 0.0) or 0.0),
            float(row.get("posY", 0.0) or 0.0),
            float(row.get("posZ", 0.0) or 0.0),
            float(row.get("uniqueId", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )


def crop_and_resize_rgb(rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int], crop_size: int) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    return _nearest_resize(crop, crop_size, crop_size).astype(np.uint8, copy=False)


def crop_and_resize_mask(mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int], crop_size: int) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size), dtype=np.float32)
    return _nearest_resize(crop, crop_size, crop_size).astype(np.float32, copy=False)


def _nearest_resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = np.linspace(0, arr.shape[0] - 1, h).astype(np.int32)
    xs = np.linspace(0, arr.shape[1] - 1, w).astype(np.int32)
    if arr.ndim == 2:
        return arr[np.ix_(ys, xs)]
    return arr[np.ix_(ys, xs, np.arange(arr.shape[2]))]


def is_probable_roof_asset(asset_path: str) -> bool:
    """Heuristic filter for building/roof families in the first implementation slice."""
    normalized = normalize_asset_path(asset_path)
    path = PurePosixPath(normalized)
    suffix = "".join(path.suffixes)
    if suffix not in {".wmo", ".wmo.mpq", ".mdx", ".m2"}:
        return False
    roof_like_tokens = ("/buildings/", "roof", "house", "inn", "tower", "city", "village")
    return any(token in normalized for token in roof_like_tokens)
