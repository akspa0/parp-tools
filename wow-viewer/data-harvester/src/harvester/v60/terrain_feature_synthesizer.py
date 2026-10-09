"""Terrain Feature Synthesizer (Spec 264 / Spec 266 / Spec 267).

Fuses low-frequency macro WDL elevation lattices with high-frequency
photometric shape-from-shading (Poisson), surface normal fields, and
ridge/crest contours to synthesize authentic, rich 3D terrain meshes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


from harvester.v60.building_foundation_carver import BuildingFoundationCarver


def compute_surface_normals(height_map: np.ndarray) -> np.ndarray:
    """Compute normalized 3D surface normal vectors (-dZ/dx, -dZ/dy, 1.0).

    Args:
        height_map: (H, W) array of elevation values.

    Returns:
        normals: (H, W, 3) float32 array of normalized normal vectors.
    """
    dz_dy, dz_dx = np.gradient(height_map.astype(np.float32))
    normals = np.zeros((*height_map.shape, 3), dtype=np.float32)
    normals[..., 0] = -dz_dx
    normals[..., 1] = -dz_dy
    normals[..., 2] = 1.0
    lens = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(lens, 1e-6)
    return normals


def synthesize_multiband_terrain(
    macro_wdl_257: np.ndarray,
    integrated_sfs_256: np.ndarray,
    ridge_mask_256: np.ndarray,
    water_mask_257: Optional[np.ndarray] = None,
    neural_residual_257: Optional[np.ndarray] = None,
    object_mask_256: Optional[np.ndarray] = None,
    alpha_mask_256: Optional[np.ndarray] = None,
    target_relief_yards: float = 3.5,
    ridge_boost_yards: float = 1.5,
    sfs_sigma_cut: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synthesize multi-band terrain fusing macro WDL, SfS relief, and ridge crests.

    Args:
        macro_wdl_257: (257, 257) coarse macro elevation in yards.
        integrated_sfs_256: (256, 256) Poisson heightfield from shadow residual.
        ridge_mask_256: (256, 256) binary or uint8 ridge/crest contours.
        water_mask_257: Optional (257, 257) boolean mask of ocean/water pixels.
        neural_residual_257: Optional (257, 257) neural model ΔZ refinement in yards.
        object_mask_256: Optional (256, 256) or (257, 257) mask of rooftops, doodads, and buildings.
        alpha_mask_256: Optional (256, 256) texture splat alpha map to suppress albedo bleed.
        target_relief_yards: Target local high-frequency relief amplitude in yards.
        ridge_boost_yards: Height displacement along sharp mountain spines in yards.
        sfs_sigma_cut: Gaussian filter scale separating macro vs meso frequencies.

    Returns:
        fused_h: (257, 257) final sculpted terrain elevation in yards.
        recovered_normals: (256, 256, 3) normal vector field of the sculpted terrain.
        metrics: Dictionary of synthesis diagnostics.
    """
    # 1. Band-pass decompose the Shape-from-Shading Poisson surface
    sfs_clean = integrated_sfs_256.copy().astype(np.float32)
    if object_mask_256 is not None:
        o_bool_256 = np.asarray(object_mask_256, dtype=bool)
        if np.any(o_bool_256):
            # Fill object regions with local perimeter median before filtering
            # to prevent high-pass edge ringing or trenches around buildings
            dil = ndimage.binary_dilation(o_bool_256, iterations=2)
            perim = dil & (~o_bool_256)
            if np.any(perim):
                fill_val = float(np.median(sfs_clean[perim]))
            else:
                fill_val = float(np.median(sfs_clean))
            sfs_clean[o_bool_256] = fill_val

    sfs_lp = ndimage.gaussian_filter(sfs_clean, sigma=sfs_sigma_cut)
    sfs_hp_256 = sfs_clean - sfs_lp

    # Resample high-pass relief and ridge mask to (257, 257)
    sfs_hp_257 = ndimage.zoom(sfs_hp_256, (257.0 / 256.0, 257.0 / 256.0), order=1)[:257, :257]
    # Filter out micro-texture variations (scale finer than ADT vertex lattice ~4.16 yards = 2 pixels)
    sfs_hp_257 = ndimage.gaussian_filter(sfs_hp_257, sigma=2.0)

    # Texture alpha splat attenuation: negate texture albedo bleeding into elevation
    if alpha_mask_256 is not None:
        a_np = np.clip(np.asarray(alpha_mask_256, dtype=np.float32), 0.0, 1.0)
        a_257 = ndimage.zoom(a_np, (257.0 / a_np.shape[0], 257.0 / a_np.shape[1]), order=1)[:257, :257]
        # Attenuate relief where texture splat is painted (up to 85% reduction)
        texture_suppression = 1.0 - 0.85 * a_257
        sfs_hp_257 *= texture_suppression

    r_bool_256 = (ridge_mask_256 > 0).astype(np.float32)
    if alpha_mask_256 is not None:
        # Suppress false ridges along texture boundaries
        r_bool_256 = r_bool_256 * (a_np < 0.25).astype(np.float32)

    # Object footprint masking
    obj_mask_257: Optional[np.ndarray] = None
    if object_mask_256 is not None:
        o_np = np.asarray(object_mask_256, dtype=np.float32) > 0.5
        if o_np.shape != (257, 257):
            obj_mask_257 = ndimage.zoom(o_np.astype(np.float32), (257.0 / o_np.shape[0], 257.0 / o_np.shape[1]), order=0)[:257, :257] > 0.5
        else:
            obj_mask_257 = o_np
        # Suppress SfS relief and ridges directly under objects
        sfs_hp_257[obj_mask_257] = 0.0
        r_bool_256[o_np[:256, :256]] = 0.0

    r257_ridge = ndimage.zoom(r_bool_256, (257.0 / 256.0, 257.0 / 256.0), order=1)[:257, :257]
    if obj_mask_257 is not None:
        r257_ridge[obj_mask_257] = 0.0

    # Identify land vs water
    land_mask = ~water_mask_257 if water_mask_257 is not None else np.ones((257, 257), dtype=bool)
    if obj_mask_257 is not None:
        land_sample = land_mask & (~obj_mask_257)
    else:
        land_sample = land_mask

    # 2. Scale SfS high-pass relief to physical yards based on local slope
    if np.any(land_sample):
        p95 = float(np.percentile(np.abs(sfs_hp_257[land_sample]), 95))
    elif np.any(land_mask):
        p95 = float(np.percentile(np.abs(sfs_hp_257[land_mask]), 95))
    else:
        p95 = float(np.percentile(np.abs(sfs_hp_257), 95))
    scale_yards = float(target_relief_yards / max(1e-5, p95))
    sfs_relief = sfs_hp_257 * scale_yards

    # 3. Apply convex ridge crest displacement along mountain spines
    ridge_smooth = ndimage.gaussian_filter(r257_ridge, sigma=1.0)
    ridge_boost = ridge_smooth * float(ridge_boost_yards)

    # 4. Fuse macro WDL anchor with meso SfS relief and ridge crests
    fused_h = macro_wdl_257.astype(np.float32) + sfs_relief + ridge_boost

    # Add optional neural residual refinement
    if neural_residual_257 is not None:
        res = neural_residual_257.astype(np.float32)
        if obj_mask_257 is not None:
            res[obj_mask_257] = 0.0
        fused_h += res

    # 4b. Level terrain where objects are masked out to median average height of surrounding terrain
    if obj_mask_257 is not None and np.any(obj_mask_257):
        carver = BuildingFoundationCarver(blend_margin=3)
        fused_h = carver.carve_masked_objects(fused_h, obj_mask_257)

    # 5. Enforce strict sea-level water clamping
    if water_mask_257 is not None:
        fused_h[water_mask_257] = 0.0
        # Prevent land dipping below sea level
        if np.any(land_mask):
            fused_h[land_mask] = np.maximum(fused_h[land_mask], 0.1)

    # 6. Recompute authentic surface normals of the sculpted terrain
    normals_257 = compute_surface_normals(fused_h)
    normals_256 = ndimage.zoom(normals_257, (256.0 / 257.0, 256.0 / 257.0, 1.0), order=1)[:256, :256, :]
    lens = np.linalg.norm(normals_256, axis=-1, keepdims=True)
    normals_256 /= np.maximum(lens, 1e-6)

    metrics = {
        "scale_yards": scale_yards,
        "p95_hp": p95,
        "relief_ptp": float(np.ptp(sfs_relief)),
        "ridge_max_boost": float(np.max(ridge_boost)),
        "fused_min": float(np.min(fused_h)),
        "fused_max": float(np.max(fused_h)),
        "fused_span": float(np.ptp(fused_h)),
    }
    return fused_h, normals_256, metrics
