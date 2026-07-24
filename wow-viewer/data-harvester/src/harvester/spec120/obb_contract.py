"""Spec 120 Minimap OBB Object Detector & Metadata Sidecar Contract (T001).

Pure-function coordinate mappings, OBB bounding box derivations, and sidecar metadata schema definitions
shared by the dataset builder, detector trainer, inference CLI, and sidecar exporter.

No I/O here: callers pass placement dictionaries/rows and tile indices; this module owns
the World-to-Tile pixel coordinate math, OBB target encoding [class_id, cx, cy, w, h, angle],
and sidecar metadata schema formatting/validation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---- Stages & Output Signals -------------------------------------------------------------

STAGE_OBB_DETECTOR = "minimap_obb_detector"
STAGE_SIDECAR_EXPORTER = "minimap_sidecar_exporter"

OUTPUT_SIGNAL_OBB_BOXES = "minimap_detected_obb_boxes"
OUTPUT_SIGNAL_METADATA_SIDECAR = "minimap_metadata_sidecar"

# ---- Class Taxonomy -----------------------------------------------------------------------

COARSE_CLASSES = ("wmo", "mdx")
COARSE_CLASS_INDEX = {name: idx for idx, name in enumerate(COARSE_CLASSES)}
COARSE_INDEX_CLASS: dict[int, str] = {index: name for name, index in COARSE_CLASS_INDEX.items()}
NUM_COARSE_CLASSES = len(COARSE_CLASSES)

# ---- Map & Pixel Constants ----------------------------------------------------------------

# World of Warcraft ADT Tile size in yards (16 chunks * 33.3333 yards per chunk)
ADT_TILE_SIZE_YARDS: float = 533.3333333333333
DEFAULT_TILE_PIXELS: int = 256
YARDS_PER_PIXEL: float = ADT_TILE_SIZE_YARDS / DEFAULT_TILE_PIXELS  # ~2.08333 yd/px


class ObbContractError(ValueError):
    """Raised when a Spec 120 coordinate, OBB box, or sidecar schema is invalid."""


def world_to_tile_pixels(
    world_x: float,
    world_y: float,
    tile_x: int,
    tile_y: int,
    tile_pixels: int = DEFAULT_TILE_PIXELS,
) -> tuple[float, float]:
    """Convert world coordinates (world_x, world_y) to fractional pixel coordinates on a tile.

    WoW coordinate system convention:
    - Center of map is (0,0) at tile (32,32).
    - +X increases North, +Y increases West.
    - Tile (tx, ty) top-left in world space is ((32 - tx) * TILE_SIZE, (32 - ty) * TILE_SIZE).
    """
    fx = ((32.0 - float(tile_x)) * ADT_TILE_SIZE_YARDS - float(world_x)) / ADT_TILE_SIZE_YARDS
    fy = ((32.0 - float(tile_y)) * ADT_TILE_SIZE_YARDS - float(world_y)) / ADT_TILE_SIZE_YARDS

    px = fx * float(tile_pixels)
    py = fy * float(tile_pixels)

    return px, py


def tile_pixels_to_world(
    px: float,
    py: float,
    tile_x: int,
    tile_y: int,
    tile_pixels: int = DEFAULT_TILE_PIXELS,
) -> tuple[float, float]:
    """Convert tile pixel coordinates (px, py) back to world coordinates (world_x, world_y)."""
    fx = float(px) / float(tile_pixels)
    fy = float(py) / float(tile_pixels)

    world_x = (32.0 - float(tile_x)) * ADT_TILE_SIZE_YARDS - fx * ADT_TILE_SIZE_YARDS
    world_y = (32.0 - float(tile_y)) * ADT_TILE_SIZE_YARDS - fy * ADT_TILE_SIZE_YARDS

    return world_x, world_y


def is_pixel_on_tile(px: float, py: float, margin_px: float = 0.0, tile_pixels: int = DEFAULT_TILE_PIXELS) -> bool:
    """Check if pixel coordinates fall within tile bounds (with optional margin)."""
    return (-margin_px <= px <= float(tile_pixels) + margin_px) and (-margin_px <= py <= float(tile_pixels) + margin_px)


def derive_coarse_class(instance_type: str, asset_path: str) -> str:
    """Derive coarse class string ('wmo' vs 'mdx') from 0.5.3 placement metadata."""
    inst_lower = str(instance_type).lower().strip()
    path_lower = str(asset_path).lower().strip()

    if inst_lower == "modf" or "wmo" in path_lower or path_lower.endswith(".wmo"):
        return "wmo"
    return "mdx"


def placement_to_obb_target(
    world_x: float,
    world_y: float,
    tile_x: int,
    tile_y: int,
    extent_x_yards: float,
    extent_y_yards: float,
    rotation_deg: float,
    coarse_class: str,
    tile_pixels: int = DEFAULT_TILE_PIXELS,
) -> dict[str, Any]:
    """Encode a placement into a normalized OBB target dict.

    Returns dict with:
    - px, py: pixel coordinates on tile
    - cx_norm, cy_norm: normalized center coordinates [0.0, 1.0]
    - w_px, h_px: pixel dimensions
    - w_norm, h_norm: normalized dimensions [0.0, 1.0]
    - angle_deg: rotation angle in degrees
    - class_id: integer class id
    """
    px, py = world_to_tile_pixels(world_x, world_y, tile_x, tile_y, tile_pixels)

    cx_norm = px / float(tile_pixels)
    cy_norm = py / float(tile_pixels)

    w_px = max(2.0, extent_x_yards / YARDS_PER_PIXEL)
    h_px = max(2.0, extent_y_yards / YARDS_PER_PIXEL)

    w_norm = w_px / float(tile_pixels)
    h_norm = h_px / float(tile_pixels)

    class_id = COARSE_CLASS_INDEX.get(coarse_class, COARSE_CLASS_INDEX["mdx"])

    return {
        "px": px,
        "py": py,
        "cx_norm": cx_norm,
        "cy_norm": cy_norm,
        "w_px": w_px,
        "h_px": h_px,
        "w_norm": w_norm,
        "h_norm": h_norm,
        "angle_deg": float(rotation_deg) % 360.0,
        "class_id": class_id,
        "coarse_class": coarse_class,
    }


def format_sidecar_item(
    instance_id: int,
    position_px: tuple[float, float],
    world_pos: tuple[float, float, float],
    scale_px: tuple[float, float],
    scale_factor: float,
    rotation_deg: float,
    coarse_class: str,
    retrieved_asset: str,
    confidence: float,
    tile_x: int = 32,
    tile_y: int = 32,
) -> dict[str, Any]:
    """Format a single detection record into sidecar metadata contract schema."""
    return {
        "instance_id": int(instance_id),
        "tile_x": int(tile_x),
        "tile_y": int(tile_y),
        "position_px": [round(float(position_px[0]), 2), round(float(position_px[1]), 2)],
        "world_position": [round(float(world_pos[0]), 2), round(float(world_pos[1]), 2), round(float(world_pos[2]), 2)],
        "scale_px": [round(float(scale_px[0]), 2), round(float(scale_px[1]), 2)],
        "scale_factor": round(float(scale_factor), 3),
        "rotation_deg": round(float(rotation_deg), 1),
        "coarse_class": str(coarse_class),
        "retrieved_asset": str(retrieved_asset),
        "confidence": round(float(confidence), 4),
    }


def validate_sidecar_schema(items: list[dict[str, Any]]) -> bool:
    """Validate that a list of sidecar items complies with the required metadata schema."""
    required_keys = {
        "instance_id",
        "position_px",
        "world_position",
        "scale_px",
        "scale_factor",
        "rotation_deg",
        "coarse_class",
        "retrieved_asset",
        "confidence",
    }
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ObbContractError(f"Sidecar item #{i} is not a dictionary.")
        missing = required_keys - item.keys()
        if missing:
            raise ObbContractError(f"Sidecar item #{i} is missing keys: {missing}")
        if len(item["position_px"]) != 2:
            raise ObbContractError(f"Sidecar item #{i} position_px must have 2 elements.")
        if len(item["world_position"]) != 3:
            raise ObbContractError(f"Sidecar item #{i} world_position must have 3 elements.")
        if len(item["scale_px"]) != 2:
            raise ObbContractError(f"Sidecar item #{i} scale_px must have 2 elements.")
    return True
