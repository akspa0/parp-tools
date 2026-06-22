"""V20 signal patching core utility functions."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


def _flags_to_liquid_type_16(flags_16: np.ndarray) -> np.ndarray:
    """Convert MCNK flags into a coarse liquid-type grid.
    Classes:
      0 none
      1 water/river
      2 ocean
      3 magma
      4 slime
    """
    flags = flags_16.astype(np.int32, copy=False)
    out = np.zeros(flags.shape, dtype=np.uint8)
    out[(flags & 0x04) != 0] = 1
    out[(flags & 0x08) != 0] = 2
    out[(flags & 0x10) != 0] = 3
    out[(flags & 0x20) != 0] = 4
    return out


def inpaint_tile_heightmap(height: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint masked regions in height using griddata (cubic spline with fallback)."""
    h, w = height.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Valid (ground) points
    valid_mask = mask < 0.05
    if not np.any(valid_mask):
        return height.copy()

    points = np.stack([y_coords[valid_mask], x_coords[valid_mask]], axis=-1)
    values = height[valid_mask]

    # Missing (object) points
    missing_mask = mask >= 0.05
    if not np.any(missing_mask):
        return height.copy()

    # Interpolate using cubic spline, fallback to linear, then nearest
    try:
        grid_z = griddata(points, values, (y_coords, x_coords), method="cubic")
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
            grid_linear = griddata(points, values, (y_coords, x_coords), method="linear")
            grid_z[nan_mask] = grid_linear[nan_mask]
            
            nan_mask_2 = np.isnan(grid_z)
            if np.any(nan_mask_2):
                grid_nearest = griddata(points, values, (y_coords, x_coords), method="nearest")
                grid_z[nan_mask_2] = grid_nearest[nan_mask_2]
    except Exception:
        grid_z = griddata(points, values, (y_coords, x_coords), method="linear")
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
            grid_nearest = griddata(points, values, (y_coords, x_coords), method="nearest")
            grid_z[nan_mask] = grid_nearest[nan_mask]

    inpainted = height.copy()
    inpainted[missing_mask] = grid_z[missing_mask]
    return inpainted


def process_single_tile(args: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray]:
    """Process a single tile in a worker process."""
    idx, height, obj_mask, liquid_mask, mcnk_flags = args

    # 1. Compute liquid_type_256
    liq_type_16 = _flags_to_liquid_type_16(mcnk_flags)
    # block broadcast
    liq_type_256_raw = np.repeat(np.repeat(liq_type_16, 16, axis=0), 16, axis=1)
    liq_type_256 = (liq_type_256_raw * (liquid_mask > 0.1)).astype(np.uint8)

    # 2. Compute ground_intent_height_257
    ground_height = inpaint_tile_heightmap(height, obj_mask)

    return idx, liq_type_256, ground_height
