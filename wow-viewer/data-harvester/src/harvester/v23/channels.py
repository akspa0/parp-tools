"""Input-channel contract helpers for Spec 089 Phase 1.

The default V23 input tensor is a fixed 15-channel tensor:

- 0..2: minimap RGB
- 3..6: alpha_256
- 7..10: tileset identity planes derived from the first four retained
  prune-table ids
- 11..13: normal XYZ
- 14: terrain-valid mask
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

CHANNEL_ORDER: tuple[str, ...] = (
    "minimap_r",
    "minimap_g",
    "minimap_b",
    "alpha_0",
    "alpha_1",
    "alpha_2",
    "alpha_3",
    "tileset_pruned_0",
    "tileset_pruned_1",
    "tileset_pruned_2",
    "tileset_pruned_3",
    "normal_x",
    "normal_y",
    "normal_z",
    "terrain_valid_mask",
)


class InputMode(str, Enum):
    """Supported V23 input tensor subsets."""

    FULL = "full"
    MINIMAP_ONLY = "minimap_only"
    MINIMAP_ALPHA = "minimap_alpha"
    MINIMAP_ALPHA_NORMAL = "minimap_alpha_normal"

    @classmethod
    def coerce(cls, value: str | "InputMode") -> "InputMode":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


CHANNEL_INDICES: dict[InputMode, tuple[int, ...]] = {
    InputMode.FULL: tuple(range(15)),
    InputMode.MINIMAP_ONLY: (0, 1, 2),
    InputMode.MINIMAP_ALPHA: (0, 1, 2, 3, 4, 5, 6),
    InputMode.MINIMAP_ALPHA_NORMAL: (0, 1, 2, 3, 4, 5, 6, 11, 12, 13),
}


class MissingMinimapError(KeyError):
    """Raised when a V23 sample has no minimap input."""


def load_tileset_prune_table(
    source: str | Path | Mapping[str, Any] | Mapping[int, int] | None,
) -> dict[int, int]:
    """Load a tileset prune table from a mapping or JSON file.

    Accepted JSON layouts:

    - ``{"tileset_id_to_index": {"12": 0, "20": 1}, "oov_index": 256}``
    - ``{"12": 0, "20": 1}``
    """

    if source is None:
        return {}
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = dict(source)

    if "tileset_id_to_index" in payload:
        payload = payload["tileset_id_to_index"]

    table: dict[int, int] = {}
    for key, value in payload.items():
        table[int(key)] = int(value)
    return table


def _as_float32(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _has_array(tile: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> bool:
    if name not in tile:
        return False
    return np.asarray(tile[name]).shape == shape


def _normalize_minimap(minimap_rgb: np.ndarray) -> torch.Tensor:
    if minimap_rgb.shape != (256, 256, 3):
        raise ValueError(f"minimap_rgb must have shape (256, 256, 3), got {minimap_rgb.shape}")
    minimap = torch.from_numpy(np.ascontiguousarray(minimap_rgb)).permute(2, 0, 1).float() / 255.0
    return (minimap - IMAGENET_MEAN) / IMAGENET_STD


def _pad_256_to_257(mask_256: np.ndarray) -> np.ndarray:
    return np.pad(mask_256, ((0, 1), (0, 1)), mode="edge")


def _derive_object_presence_257(tile: Mapping[str, Any]) -> np.ndarray:
    if _has_array(tile, "object_precise_mask", (257, 257)):
        return (np.asarray(tile["object_precise_mask"], dtype=np.float32) >= 0.05).astype(np.float32)
    if _has_array(tile, "object_filtered_mask", (257, 257)):
        return (np.asarray(tile["object_filtered_mask"], dtype=np.float32) >= 0.05).astype(np.float32)
    if _has_array(tile, "object_mask", (257, 257)):
        return np.asarray(tile["object_mask"], dtype=np.float32)
    return np.zeros((257, 257), dtype=np.float32)


def derive_terrain_valid_mask_257(tile: Mapping[str, Any]) -> np.ndarray:
    """Compose the terrain-valid mask used by the target loss and channel 14."""

    normal_mask = (
        np.asarray(tile["mcnr_mask_257"], dtype=np.float32)
        if _has_array(tile, "mcnr_mask_257", (257, 257))
        else np.ones((257, 257), dtype=np.float32)
    )
    liquid_mask_256 = (
        np.clip(np.asarray(tile["liquid_mask"], dtype=np.float32), 0.0, 1.0)
        if _has_array(tile, "liquid_mask", (256, 256))
        else np.zeros((256, 256), dtype=np.float32)
    )
    liquid_mask_257 = _pad_256_to_257(liquid_mask_256)
    object_presence_257 = _derive_object_presence_257(tile)

    terrain_valid = normal_mask.astype(np.float32, copy=True)
    terrain_valid *= 1.0 - np.clip(liquid_mask_257, 0.0, 1.0)
    terrain_valid *= 1.0 - np.clip(object_presence_257, 0.0, 1.0)
    return terrain_valid.astype(np.float32, copy=False)


def _build_tileset_planes(
    tile: Mapping[str, Any],
    alpha_256: np.ndarray,
    tileset_prune_table: Mapping[int, int],
) -> tuple[torch.Tensor, bool]:
    """Build the four fixed tileset planes for indices 0..3 in the prune table.

    The spec's 15-channel contract reserves exactly four tileset planes, so the
    Phase 1 implementation uses the first four retained prune-table ids as the
    identity surface. Later phases can widen this surface without changing the
    minimap/alpha/normal/valid-mask blocks.
    """

    if not _has_array(tile, "mcly_tileset_ids", (16, 16, 4)):
        return torch.zeros((4, 256, 256), dtype=torch.float32), False

    mcly_tileset_ids = np.asarray(tile["mcly_tileset_ids"], dtype=np.int32)
    dominant_layer = np.asarray(alpha_256, dtype=np.float32).argmax(axis=2)

    cell_y = (np.arange(256, dtype=np.int32) // 16)[:, None]
    cell_x = (np.arange(256, dtype=np.int32) // 16)[None, :]
    dominant_tileset_ids = mcly_tileset_ids[cell_y, cell_x, dominant_layer]

    if tileset_prune_table:
        oov_index = max(tileset_prune_table.values(), default=3) + 1
        pruned = np.vectorize(lambda value: tileset_prune_table.get(int(value), oov_index), otypes=[np.int32])(
            dominant_tileset_ids
        )
    else:
        pruned = dominant_tileset_ids

    planes = np.zeros((4, 256, 256), dtype=np.float32)
    for channel_idx in range(4):
        planes[channel_idx] = (pruned == channel_idx).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(planes)), True


def build_channel_tensor(
    zarr_tile: Mapping[str, Any],
    mode: str | InputMode = InputMode.FULL,
    *,
    tileset_prune_table: str | Path | Mapping[str, Any] | Mapping[int, int] | None = None,
    return_channel_valid_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build a V23 input tensor from a V22 tile dict.

    Parameters
    ----------
    zarr_tile:
        Mapping returned by ``V22Dataset.__getitem__`` or a synthetic equivalent.
    mode:
        Requested channel subset.
    tileset_prune_table:
        Optional prune-table mapping or JSON path. The first four retained ids
        drive the fixed tileset planes in the 15-channel contract.
    return_channel_valid_mask:
        When ``True``, return ``(tensor, channel_valid_mask)`` where the second
        value is a boolean tensor aligned to the returned channel count.
    """

    input_mode = InputMode.coerce(mode)
    if not _has_array(zarr_tile, "minimap_rgb", (256, 256, 3)):
        tile_ref = zarr_tile.get("tile_id", "<unknown>")
        raise MissingMinimapError(f"V23 sample is missing minimap_rgb for tile {tile_ref}")

    minimap = _normalize_minimap(np.asarray(zarr_tile["minimap_rgb"], dtype=np.uint8))

    if _has_array(zarr_tile, "alpha_256", (256, 256, 4)):
        alpha_256 = np.clip(_as_float32(zarr_tile["alpha_256"]), 0.0, 1.0)
        alpha = torch.from_numpy(np.ascontiguousarray(alpha_256)).permute(2, 0, 1)
        alpha_valid = True
    else:
        alpha_256 = np.zeros((256, 256, 4), dtype=np.float32)
        alpha = torch.zeros((4, 256, 256), dtype=torch.float32)
        alpha_valid = False

    tileset_planes, tileset_valid = _build_tileset_planes(
        zarr_tile,
        alpha_256,
        load_tileset_prune_table(tileset_prune_table),
    )

    if _has_array(zarr_tile, "normal_xyz", (257, 257, 3)):
        normal_xyz = np.asarray(zarr_tile["normal_xyz"], dtype=np.float32)[:256, :256, :]
        normal = torch.from_numpy(np.ascontiguousarray(normal_xyz)).permute(2, 0, 1)
        normal_valid = True
    else:
        normal = torch.zeros((3, 256, 256), dtype=torch.float32)
        normal_valid = False

    terrain_valid_mask_257 = derive_terrain_valid_mask_257(zarr_tile)
    terrain_valid = torch.from_numpy(np.ascontiguousarray(terrain_valid_mask_257[:256, :256])).unsqueeze(0)

    full_tensor = torch.cat([minimap, alpha, tileset_planes, normal, terrain_valid], dim=0)
    full_valid_mask = torch.tensor(
        [True, True, True]
        + [alpha_valid] * 4
        + [tileset_valid] * 4
        + [normal_valid] * 3
        + [True],
        dtype=torch.bool,
    )

    indices = CHANNEL_INDICES[input_mode]
    selected_tensor = full_tensor[list(indices)]
    selected_valid_mask = full_valid_mask[list(indices)]
    if return_channel_valid_mask:
        return selected_tensor, selected_valid_mask
    return selected_tensor


__all__ = [
    "CHANNEL_INDICES",
    "CHANNEL_ORDER",
    "InputMode",
    "MissingMinimapError",
    "build_channel_tensor",
    "derive_terrain_valid_mask_257",
    "load_tileset_prune_table",
]
