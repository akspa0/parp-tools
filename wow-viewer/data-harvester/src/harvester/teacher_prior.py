"""Spec 077 §2 teacher deconstruction prior construction.

The teacher prior is a tile-level artifact that suppresses object pixels
in the raw minimap so downstream terrain models can train on a cleaner
signal. The policy is intentionally small and explicit so it stays
auditable.

Phase 1 channels
----------------
* ``raw_minimap_rgb_256``        ``(256, 256, 3)`` uint8  — passthrough
* ``teacher_object_mask_256``    ``(256, 256)``    uint8  — preferred object mask
* ``teacher_object_confidence_256`` ``(256, 256)``  uint8  — confidence
* ``processed_minimap_prior_256``  ``(256, 256, 5)`` uint8 — channel layout:

  - ``[..., 0:3]`` = object-suppressed RGB (per-tile median fill on
    object pixels)
  - ``[..., 3]``   = ``teacher_object_mask_256`` (uint8)
  - ``[..., 4]``   = ``teacher_object_confidence_256`` (uint8)

The mask preference chain (spec 077 FR-009) is:

1. ``object_filtered_mask``  (preferred: WMOs + filtered doodads, no trees)
2. ``object_precise_mask``   (fallback: precise silhouettes incl. trees)
3. ``object_mask``           (last resort: bounding-box footprint)

The confidence channel is a constant 255 for the chosen mask band. A
later slice may swap in a learned confidence model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class MaskSource(str, Enum):
    ObjectFiltered = "object_filtered_mask"
    ObjectPrecise = "object_precise_mask"
    ObjectMask = "object_mask"
    None_ = "none"


PRIOR_CHANNELS: tuple[str, ...] = (
    "suppressed_rgb_r",
    "suppressed_rgb_g",
    "suppressed_rgb_b",
    "teacher_object_mask",
    "teacher_object_confidence",
)


@dataclass(frozen=True)
class TeacherPriorTileRecord:
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    raw_minimap_key: str
    teacher_object_mask_key: str
    teacher_object_confidence_key: str
    processed_prior_key: str
    has_teacher_objects: bool
    teacher_object_cov: float
    filtered_mask_source: str


def pick_object_mask(
    *,
    object_filtered_mask: np.ndarray | None,
    object_precise_mask: np.ndarray | None,
    object_mask: np.ndarray | None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, MaskSource]:
    """Return ``(binary_mask_256, source)`` honoring the spec 077 preference chain.

    Each candidate is treated as a soft mask in ``[0, 1]``; a pixel is
    considered an object pixel when ``value >= threshold``. The first
    non-null candidate wins. A fully empty result means no teacher
    objects were observable on this tile.
    """
    if object_filtered_mask is not None and float(object_filtered_mask.max(initial=0.0)) > 0.0:
        return (object_filtered_mask >= threshold).astype(np.uint8), MaskSource.ObjectFiltered
    if object_precise_mask is not None and float(object_precise_mask.max(initial=0.0)) > 0.0:
        return (object_precise_mask >= threshold).astype(np.uint8), MaskSource.ObjectPrecise
    if object_mask is not None and float(object_mask.max(initial=0.0)) > 0.0:
        return (object_mask >= threshold).astype(np.uint8), MaskSource.ObjectMask
    return (np.zeros((256, 256), dtype=np.uint8), MaskSource.None_)


def suppress_object_pixels(
    minimap_rgb: np.ndarray,
    object_mask: np.ndarray,
) -> np.ndarray:
    """Replace object pixels with the per-tile median of non-object pixels.

    The fill is deterministic and uses the tile's own terrain color
    distribution, so a no-object tile passes through unchanged. Pixels
    with no observed terrain (all-object tile) fall back to neutral
    mid-gray.
    """
    if minimap_rgb.ndim != 3 or minimap_rgb.shape[2] != 3:
        raise ValueError(f"Expected (256, 256, 3) RGB; got {minimap_rgb.shape}")
    if minimap_rgb.dtype != np.uint8:
        minimap_rgb = minimap_rgb.astype(np.uint8)
    if object_mask.shape != minimap_rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {object_mask.shape} does not match minimap {minimap_rgb.shape[:2]}"
        )
    non_object = minimap_rgb[object_mask == 0]
    if non_object.size == 0:
        fill = np.full((1, 1, 3), 128, dtype=np.uint8)
    else:
        fill = np.median(non_object.reshape(-1, 3), axis=0, keepdims=True).astype(np.uint8)
        fill = np.broadcast_to(fill, minimap_rgb.shape)
    return np.where(object_mask[:, :, None] == 0, minimap_rgb, fill).astype(np.uint8, copy=False)


def build_prior_tensor(
    minimap_rgb: np.ndarray,
    object_filtered_mask: np.ndarray | None,
    object_precise_mask: np.ndarray | None,
    object_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MaskSource]:
    """Build the (suppressed, mask, confidence, source) tuple for one tile.

    The tensor layout is the five-channel ``processed_minimap_prior_256``
    described at the top of this module. The returned mask and
    confidence arrays are 256×256 uint8 for direct Zarr storage.
    """
    mask_uint8, source = pick_object_mask(
        object_filtered_mask=object_filtered_mask,
        object_precise_mask=object_precise_mask,
        object_mask=object_mask,
    )
    suppressed = suppress_object_pixels(minimap_rgb, mask_uint8)
    confidence = np.full(mask_uint8.shape, 255, dtype=np.uint8)
    tensor = np.concatenate(
        [suppressed, mask_uint8[:, :, None], confidence[:, :, None]],
        axis=2,
    ).astype(np.uint8, copy=False)
    return tensor, mask_uint8, confidence, source


def make_tile_record(
    *,
    build: str,
    map_name: str,
    tile_id: int,
    tile_x: int,
    tile_y: int,
    mask_uint8: np.ndarray,
    source: MaskSource,
    index: int,
) -> TeacherPriorTileRecord:
    coverage = float(mask_uint8.mean()) if mask_uint8.size else 0.0
    return TeacherPriorTileRecord(
        build=build,
        map_name=map_name,
        tile_id=tile_id,
        tile_x=tile_x,
        tile_y=tile_y,
        raw_minimap_key=f"raw_minimap_rgb_256/{index}",
        teacher_object_mask_key=f"teacher_object_mask_256/{index}",
        teacher_object_confidence_key=f"teacher_object_confidence_256/{index}",
        processed_prior_key=f"processed_minimap_prior_256/{index}",
        has_teacher_objects=bool(coverage > 0.0),
        teacher_object_cov=coverage,
        filtered_mask_source=source.value,
    )
