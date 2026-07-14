"""Spec 103 — 13-channel v7 input assembler from the current clean signals.

Implements the pinned contract (specs/103-image-only-reconstruction/research-v7-contract.md):

    ch 0-2  minimap RGB   (recovery-attenuated ×0.85, ImageNet-normalized)
    ch 3-5  normal RGB    (recovery-attenuated ×0.70, ImageNet-normalized)
    ch 6    WDL prior     outer 17×17 = height_257[::16, ::16], normalized, bilinear
                          align_corners=True upsample; 0.5 constant when dropped/missing
    ch 7    height-min hint (constant plane)
    ch 8    height-max hint (constant plane)
    ch 9    liquid mask
    ch 10   liquid height prior (normalized, ×mask)
    ch 11   object footprint mask
    ch 12   brush imprint mask (zeros; V18 carries no brush imprints)

The model architecture is unchanged from v7 (13 channels). Tiles containing any objects are
dropped during curation (spec Principle #5: height under an object is occluded in the minimap,
an impossible target) — that is a data-selection change, not an architecture change. The object
mask channel stays in the input (it is zero on kept tiles, but the design is not altered).

All arrays are numpy in, torch out. Vertex-grid (257) signals resample to the 256 working
raster with align_corners=True; binary 257 masks with nearest. WDL-prior dropout fills ch 6
with 0.5 (v7's own missing-prior fallback) so one model serves prior-present and prior-absent
tiles. `wdl_height_33` is prohibited; only the verified ::16 / 8::16 pairing exists here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from harvester.spec103.v7_losses import build_recovery_mask

WORKING_SIZE = 256
HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
MASKED_RGB_ATTENUATION = 0.85
MASKED_NORMAL_ATTENUATION = 0.70
MISSING_PRIOR_FILL = 0.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HEIGHT_HINT_MODES = ("gt", "wdl", "none")


def wdl_lattice_from_height257(height_257: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The verified paired WDL lattice: outer 17×17 at ::16, inner 16×16 at 8::16."""
    grid = np.asarray(height_257, dtype=np.float32)
    if grid.shape != (257, 257):
        raise ValueError(f"height_257 must be (257, 257), got {grid.shape}")
    outer = grid[::16, ::16]
    inner = grid[8::16, 8::16]
    return outer, inner


def normalize_height(values: np.ndarray, global_min: float = HEIGHT_GLOBAL_MIN, global_max: float = HEIGHT_GLOBAL_MAX) -> np.ndarray:
    global_range = max(float(global_max - global_min), 1e-6)
    return np.clip((np.asarray(values, dtype=np.float32) - float(global_min)) / global_range, 0.0, 1.0)


def denormalize_height(values: np.ndarray, global_min: float = HEIGHT_GLOBAL_MIN, global_max: float = HEIGHT_GLOBAL_MAX) -> np.ndarray:
    global_range = float(global_max - global_min)
    return np.asarray(values, dtype=np.float32) * global_range + float(global_min)


def _resample_vertex_grid(grid: np.ndarray, size: int) -> torch.Tensor:
    """Vertex grid (corners are samples) → raster, corners map to corners."""
    tensor = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=True)
    return tensor.squeeze(0)


def render_wdl_prior_channel(
    outer_17: Optional[np.ndarray],
    size: int = WORKING_SIZE,
    global_min: float = HEIGHT_GLOBAL_MIN,
    global_max: float = HEIGHT_GLOBAL_MAX,
) -> torch.Tensor:
    """v7's `_render_wdl`: normalized outer grid upsampled align_corners=True; 0.5 when missing."""
    if outer_17 is None:
        return torch.full((1, size, size), MISSING_PRIOR_FILL, dtype=torch.float32)
    outer = np.asarray(outer_17, dtype=np.float32)
    if outer.shape != (17, 17) or not np.all(np.isfinite(outer)):
        return torch.full((1, size, size), MISSING_PRIOR_FILL, dtype=torch.float32)
    return _resample_vertex_grid(normalize_height(outer, global_min, global_max), size)


def _imagenet_normalize(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (rgb - mean) / std


def _binary_mask_channel(mask: Optional[np.ndarray], size: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros((1, size, size), dtype=torch.float32)
    data = np.asarray(mask, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0).unsqueeze(0)
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="nearest")
    return (tensor.squeeze(0) > 0.1).float()


def assemble_v7_input(
    minimap_rgb: np.ndarray,
    height_257: Optional[np.ndarray] = None,
    normal_xyz: Optional[np.ndarray] = None,
    liquid_mask: Optional[np.ndarray] = None,
    liquid_height: Optional[np.ndarray] = None,
    object_mask: Optional[np.ndarray] = None,
    brush_mask: Optional[np.ndarray] = None,
    wdl_outer_17: Optional[np.ndarray] = None,
    size: int = WORKING_SIZE,
    global_min: float = HEIGHT_GLOBAL_MIN,
    global_max: float = HEIGHT_GLOBAL_MAX,
    height_hints: str = "gt",
    drop_wdl_prior: bool = False,
) -> torch.Tensor:
    """Build the pinned (13, size, size) v7 input tensor from per-tile arrays.

    The model architecture is unchanged from v7 (13 channels including object mask ch 11 and
    brush ch 12). Tiles containing any objects are dropped during curation (spec Principle #5)
    — a data-selection change, not an architecture change. The object mask is zero on kept tiles
    but the channel stays so the design is not altered.

    The prior comes from `wdl_outer_17` when given, else is derived from `height_257`
    (the verified ::16 outer transform). `drop_wdl_prior` fills ch 6 with 0.5 and, in hint
    mode "wdl", neutralizes ch 7/8 — the WDL-prior-dropout augmentation.
    """
    if height_hints not in HEIGHT_HINT_MODES:
        raise ValueError(f"height_hints must be one of {HEIGHT_HINT_MODES}, got {height_hints!r}")

    rgb = np.asarray(minimap_rgb)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    minimap = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    if minimap.shape[-2:] != (size, size):
        minimap = F.interpolate(minimap.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)

    if normal_xyz is not None:
        normals = np.asarray(normal_xyz, dtype=np.float32)
        if np.abs(normals).max(initial=0.0) > 1.5:  # int8-scaled normals
            normals = normals / 127.0
        normal_rgb = np.clip((normals + 1.0) * 0.5, 0.0, 1.0).transpose(2, 0, 1)
        normal = torch.from_numpy(np.ascontiguousarray(normal_rgb)).unsqueeze(0)
        normal = F.interpolate(normal, size=(size, size), mode="bilinear", align_corners=True).squeeze(0)
    else:
        # v7's fallback: flat up normal (128, 128, 255)
        normal = torch.tensor([128.0, 128.0, 255.0], dtype=torch.float32).view(3, 1, 1).expand(3, size, size) / 255.0
        normal = normal.contiguous()

    if wdl_outer_17 is not None:
        outer = np.asarray(wdl_outer_17, dtype=np.float32)
    elif height_257 is not None:
        outer, _ = wdl_lattice_from_height257(height_257)
    else:
        outer = None
    if drop_wdl_prior:
        outer = None
    wdl_channel = render_wdl_prior_channel(outer, size, global_min, global_max)

    global_range = max(float(global_max - global_min), 1e-6)
    if height_hints == "gt" and height_257 is not None:
        heights = np.asarray(height_257, dtype=np.float32)
        hint_min = float(np.clip((heights.min() - global_min) / global_range, 0.0, 1.0))
        hint_max = float(np.clip((heights.max() - global_min) / global_range, 0.0, 1.0))
    elif height_hints == "wdl" and outer is not None:
        hint_min = float(np.clip((outer.min() - global_min) / global_range, 0.0, 1.0))
        hint_max = float(np.clip((outer.max() - global_min) / global_range, 0.0, 1.0))
    else:
        hint_min, hint_max = 0.0, 1.0
    height_min_channel = torch.full((1, size, size), hint_min, dtype=torch.float32)
    height_max_channel = torch.full((1, size, size), hint_max, dtype=torch.float32)

    liquid_channel = _binary_mask_channel(liquid_mask, size)
    if liquid_height is not None:
        liquid_height_channel = torch.from_numpy(
            np.ascontiguousarray(normalize_height(liquid_height, global_min, global_max))
        ).unsqueeze(0).unsqueeze(0)
        if liquid_height_channel.shape[-2:] != (size, size):
            liquid_height_channel = F.interpolate(liquid_height_channel, size=(size, size), mode="bilinear", align_corners=False)
        liquid_height_channel = liquid_height_channel.squeeze(0) * liquid_channel
    else:
        liquid_height_channel = torch.zeros((1, size, size), dtype=torch.float32)

    object_channel = _binary_mask_channel(object_mask, size)
    brush_channel = _binary_mask_channel(brush_mask, size)

    recovery = build_recovery_mask(
        object_mask=object_channel.unsqueeze(0),
        liquid_mask=liquid_channel.unsqueeze(0),
        brush_mask=brush_channel.unsqueeze(0),
    ).squeeze(0)
    minimap = _imagenet_normalize(minimap * (1.0 - recovery * MASKED_RGB_ATTENUATION))
    normal = _imagenet_normalize(normal * (1.0 - recovery * MASKED_NORMAL_ATTENUATION))

    return torch.cat(
        [
            minimap,
            normal,
            wdl_channel,
            height_min_channel,
            height_max_channel,
            liquid_channel,
            liquid_height_channel,
            object_channel,
            brush_channel,
        ],
        dim=0,
    )


def build_v7_targets(
    height_257: np.ndarray,
    size: int = WORKING_SIZE,
    global_min: float = HEIGHT_GLOBAL_MIN,
    global_max: float = HEIGHT_GLOBAL_MAX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Target (2, size, size) = [global absolute-normalized, local within-tile] + bounds (4,)."""
    heights = np.asarray(height_257, dtype=np.float32)
    global_range = max(float(global_max - global_min), 1e-6)
    tile_min = float(heights.min())
    tile_max = float(heights.max())
    tile_range = max(tile_max - tile_min, 1e-6)

    global_target = _resample_vertex_grid(normalize_height(heights, global_min, global_max), size)
    local_target = _resample_vertex_grid(np.clip((heights - tile_min) / tile_range, 0.0, 1.0), size)

    bounds = torch.tensor(
        [
            np.clip((tile_min - global_min) / global_range, 0.0, 1.0),
            np.clip((tile_max - global_min) / global_range, 0.0, 1.0),
            0.0,
            1.0,
        ],
        dtype=torch.float32,
    )
    return torch.cat([global_target, local_target], dim=0), bounds


def prediction_to_height257(
    predicted_global: np.ndarray,
    global_min: float = HEIGHT_GLOBAL_MIN,
    global_max: float = HEIGHT_GLOBAL_MAX,
) -> np.ndarray:
    """Predicted global channel (size×size, [0,1]) → world-unit 257×257 vertex grid."""
    raster = np.asarray(predicted_global, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(raster)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(257, 257), mode="bilinear", align_corners=True)
    return denormalize_height(tensor.squeeze().numpy(), global_min, global_max)
