"""Inches-Resolution Refiner & 3D Fractal Pastes/Scars Engine (Spec 268 Phase 4).

Operates in authentic inches-resolution coordinate space (1 yard = 36 inches),
discovers recurring 3D fractal editor brushes, multi-tile prefab pastes, and historical
brush scars across the continuous quilt canvas, and deterministically downsamples
to standard 145-vertex MCVT client yards (AC-005, AC-006).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from harvester.v60.fractal_brush_extractor import (
    FractalEditorBrush,
    build_archetypal_brush_library,
)
from harvester.v60.fractal_brush_fitter import (
    FractalBrushFitter,
    StampInvocation,
    compute_normalized_cross_correlation,
)

logger = logging.getLogger(__name__)

# Constants reflecting authentic authoring editor world metrics
INCHES_PER_YARD = 36.0
CHUNK_YARDS = 533.33333333 / 16.0  # 33.3333 yards
TILE_YARDS = 533.33333333  # 533.3333 yards
CHUNK_INCHES = CHUNK_YARDS * INCHES_PER_YARD  # 1200 inches
TILE_INCHES = TILE_YARDS * INCHES_PER_YARD  # 19200 inches


@dataclass
class TerrainBrushScar:
    center_xy: Tuple[float, float]
    radius_yards: float
    height_contrast: float
    alpha_correlation: float
    is_broken_relationship: bool  # True if height stamp exists without corresponding alpha pattern (fossil scar)


@dataclass
class RefinedTileGeometry:
    mcvt_heights_145: np.ndarray  # (16, 16, 145) float32 heights in yards
    mcnr_normals_145: np.ndarray  # (16, 16, 145, 3) float32 surface normals
    high_res_elevation_inches: np.ndarray  # High-density sculpted heightfield
    discovered_stamps: List[StampInvocation]
    discovered_scars: List[TerrainBrushScar]


class QuiltFractalRefiner:
    """Simulates inches-scale artist sculpting canvas across multi-tile quilts,

    detects v7-era fractal brushes, pastes, and scars, and downsamples to client ADT chunks.
    """

    def __init__(
        self,
        subcell_factor: int = 6,  # 6x sub-cell sampling = 36x 2D density
        brush_library: Optional[List[FractalEditorBrush]] = None,
    ) -> None:
        self.subcell_factor = max(1, subcell_factor)
        self.brush_library = brush_library or build_archetypal_brush_library()
        self.brush_fitter = FractalBrushFitter(self.brush_library)

    def sculpt_inches_canvas(
        self,
        base_elevation_257: np.ndarray,
        residual_shadow_256: np.ndarray,
        alpha_maps_256: Optional[List[np.ndarray]] = None,
        max_brush_stamps: int = 24,
    ) -> Tuple[np.ndarray, List[StampInvocation], List[TerrainBrushScar]]:
        """Sculpt micro-relief at inches-scale resolution across canvas (AC-005).

        Args:
            base_elevation_257: Coarse macro WDL elevation grid in yards.
            residual_shadow_256: Normalized bare photometric terrain shadow field in [0, 1].
            alpha_maps_256: Optional list of deciphered MCAL alpha splat maps in [0, 1].
            max_brush_stamps: Maximum number of matching pursuit stamps to fit.

        Returns:
            inches_canvas: High-resolution sculpted heightfield in yards.
            stamps: List of fitted 3D fractal editor brush stamps.
            scars: List of identified historical brush scars and broken relationships.
        """
        h_macro = np.asarray(base_elevation_257, dtype=np.float32)
        shadow = np.asarray(residual_shadow_256, dtype=np.float32)

        # 1. Upsample macro surface to high-density inches canvas
        # 256 pixels * subcell_factor
        grid_dim = 256 * self.subcell_factor
        scale = float(grid_dim) / 256.0

        h_resampled = ndimage.zoom(h_macro[:256, :256], scale, order=1)
        shadow_resampled = ndimage.zoom(shadow, scale, order=1)

        # 2. Extract high-frequency residual signal for brush fitting
        # De-mean and band-pass isolate micro-relief
        shadow_lp = ndimage.gaussian_filter(shadow_resampled, sigma=3.0 * self.subcell_factor)
        residual_hp = shadow_resampled - shadow_lp

        # 3. Fit 3D fractal brush stamps via matching pursuit on the canvas
        # Downsample residual for fast matching pursuit on archetypal library
        fit_target = ndimage.zoom(residual_hp, 1.0 / self.subcell_factor, order=1)
        stamps, _ = self.brush_fitter.fit_stamps(
            fit_target,
            max_stamps=max_brush_stamps,
        )

        # Apply stamps onto high-resolution inches canvas
        sculpted_canvas = h_resampled.copy()
        for stamp in stamps:
            # Scaled stamp center and radius in inches-canvas pixels
            cx = stamp.center_xy[0] * self.subcell_factor
            cy = stamp.center_xy[1] * self.subcell_factor
            r = stamp.radius * self.subcell_factor

            # Local stamping window
            x0 = max(0, int(cx - r))
            x1 = min(grid_dim, int(cx + r + 1))
            y0 = max(0, int(cy - r))
            y1 = min(grid_dim, int(cy + r + 1))

            if x1 > x0 and y1 > y0:
                y_sub, x_sub = np.mgrid[y0:y1, x0:x1]
                dist_sq = ((x_sub - cx) / max(1.0, r)) ** 2 + ((y_sub - cy) / max(1.0, r)) ** 2
                disp = stamp.amplitude * np.exp(-dist_sq * 2.0)
                sculpted_canvas[y0:y1, x0:x1] += disp

        # 4. Detect v7-era brush scars and correlate height stamps with alpha layers
        scars: List[TerrainBrushScar] = []
        composite_alpha = (
            np.mean(alpha_maps_256, axis=0)
            if alpha_maps_256 and len(alpha_maps_256) > 0
            else np.zeros((256, 256), dtype=np.float32)
        )

        for stamp in stamps:
            # Check local correlation between height residual and alpha splats
            bx = int(np.clip(stamp.center_xy[0], 0, 255))
            by = int(np.clip(stamp.center_xy[1], 0, 255))
            rad = int(np.clip(stamp.radius, 4, 32))

            bx0, bx1 = max(0, bx - rad), min(256, bx + rad)
            by0, by1 = max(0, by - rad), min(256, by + rad)

            h_crop = fit_target[by0:by1, bx0:bx1]
            a_crop = composite_alpha[by0:by1, bx0:bx1]

            ncc = compute_normalized_cross_correlation(h_crop, a_crop)
            is_broken = bool(abs(stamp.amplitude) > 1e-4 and ncc < 0.25)

            scars.append(
                TerrainBrushScar(
                    center_xy=stamp.center_xy,
                    radius_yards=stamp.radius * (TILE_YARDS / 256.0),
                    height_contrast=float(stamp.amplitude),
                    alpha_correlation=float(ncc),
                    is_broken_relationship=is_broken,
                )
            )

        return sculpted_canvas, stamps, scars

    def downsample_to_mcvt_chunks(
        self,
        sculpted_inches_canvas: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministically downsample high-resolution inches canvas to 16x16 ADT chunks

        with standard 145-vertex MCVT layouts (9x9 outer + 8x8 inner vertices) and MCNR normals (AC-006).

        Returns:
            mcvt_grid: (16, 16, 145) float32 heights in yards.
            mcnr_grid: (16, 16, 145, 3) float32 normalized surface normal vectors.
        """
        grid_dim = sculpted_inches_canvas.shape[0]
        chunk_pixels = grid_dim / 16.0

        mcvt = np.zeros((16, 16, 145), dtype=np.float32)
        mcnr = np.zeros((16, 16, 145, 3), dtype=np.float32)

        # Precompute gradient of the high-res canvas for anti-aliased normals
        dz_dy, dz_dx = np.gradient(sculpted_inches_canvas)
        normals_hi = np.zeros((*sculpted_inches_canvas.shape, 3), dtype=np.float32)
        normals_hi[..., 0] = -dz_dx
        normals_hi[..., 1] = -dz_dy
        normals_hi[..., 2] = 1.0
        lens = np.linalg.norm(normals_hi, axis=-1, keepdims=True)
        normals_hi /= np.maximum(lens, 1e-6)

        for cy in range(16):
            for cx in range(16):
                u0 = cx * chunk_pixels
                v0 = cy * chunk_pixels

                # 145 vertices per chunk:
                # First 81: 9x9 outer grid (stride = chunk_pixels / 8)
                # Next 64:  8x8 inner grid (stride = chunk_pixels / 8, offset = half stride)
                step = chunk_pixels / 8.0
                v_idx = 0

                # 9x9 outer grid
                for oy in range(9):
                    for ox in range(9):
                        px = int(np.clip(u0 + ox * step, 0, grid_dim - 1))
                        py = int(np.clip(v0 + oy * step, 0, grid_dim - 1))
                        mcvt[cy, cx, v_idx] = sculpted_inches_canvas[py, px]
                        mcnr[cy, cx, v_idx] = normals_hi[py, px]
                        v_idx += 1

                # 8x8 inner grid
                half_step = step * 0.5
                for iy in range(8):
                    for ix in range(8):
                        px = int(np.clip(u0 + half_step + ix * step, 0, grid_dim - 1))
                        py = int(np.clip(v0 + half_step + iy * step, 0, grid_dim - 1))
                        mcvt[cy, cx, v_idx] = sculpted_inches_canvas[py, px]
                        mcnr[cy, cx, v_idx] = normals_hi[py, px]
                        v_idx += 1

        return mcvt, mcnr
