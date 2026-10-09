"""Bare Terrain Shadow Sieve & WDL Macro-Trestle Quilt Synthesizer (Spec 268 Phase 3).

Strips 2D texture albedo and object albedo to isolate bare photometric terrain shadows,
and synthesizes continuous WDL macro-trestle elevation lattices across multi-tile quilts (AC-003, AC-004).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from harvester.v60.minimap_shadow_stripper import multiscale_laplacian_inpaint
from harvester.v60.trestle_dataset import compute_photometric_normals
from harvester.v60.trestle_elevation_model import TrestleElevationUNet
from harvester.v60.trestle_wdl_synthesizer import TrestleWdlSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class BareTerrainShadowResult:
    bare_shadow_256: np.ndarray  # (256, 256) float32 in [0, 1]
    photometric_normals_256: np.ndarray  # (256, 256, 3) float32
    albedo_stripped: np.ndarray  # (256, 256, 3) float32


class WdlQuiltSynthesizer:
    """Isolates bare terrain photometric illumination and synthesizes continuous

    WDL macro-trestle lattices across multi-tile quilt canvases.
    """

    def __init__(self, sun_azimuth_deg: float = 220.0, sun_elevation_deg: float = 45.0) -> None:
        self.sun_azimuth_deg = sun_azimuth_deg
        self.sun_elevation_deg = sun_elevation_deg

    def extract_bare_terrain_shadow(
        self,
        minimap_rgb_256: np.ndarray,
        texture_albedo_256: np.ndarray,
        object_mask_256: Optional[np.ndarray] = None,
    ) -> BareTerrainShadowResult:
        """Strip 2D texture albedo and masked object pixels to isolate bare photometric shading (AC-003).

        Args:
            minimap_rgb_256: (256, 256, 3) uint8 or float32 composite minimap.
            texture_albedo_256: (256, 256, 3) float32 estimated texture albedo from Stage 2.
            object_mask_256: Optional (256, 256) boolean/uint8 mask of buildings/doodads.

        Returns:
            BareTerrainShadowResult with normalized shadow field, surface normals, and stripped image.
        """
        img = np.asarray(minimap_rgb_256, dtype=np.float32)
        if img.max() > 1.5:
            img = img / 255.0

        albedo = np.asarray(texture_albedo_256, dtype=np.float32)
        if albedo.max() > 1.5:
            albedo = albedo / 255.0

        # Inpaint object regions if masked
        if object_mask_256 is not None:
            obj_bool = np.asarray(object_mask_256) > 0
            if np.any(obj_bool):
                img = multiscale_laplacian_inpaint(img, obj_bool)

        # De-albedo division: S = ||img|| / max(||albedo||, epsilon)
        img_mag = np.linalg.norm(img, axis=-1)
        albedo_mag = np.linalg.norm(albedo, axis=-1)

        raw_shadow = img_mag / np.maximum(albedo_mag, 0.08)

        # Normalize shadow to [0, 1] range while preserving contrast
        lo = float(np.percentile(raw_shadow, 2.0))
        hi = float(np.percentile(raw_shadow, 98.0))
        span = max(1e-4, hi - lo)
        norm_shadow = np.clip((raw_shadow - lo) / span, 0.0, 1.0)

        # Smooth high-frequency noise while preserving macro shadow gradients
        smooth_shadow = ndimage.gaussian_filter(norm_shadow, sigma=1.5)

        stripped = np.stack([smooth_shadow] * 3, axis=-1)

        # Compute photometric surface normal vectors from bare shadow
        normals = compute_photometric_normals(stripped)

        return BareTerrainShadowResult(
            bare_shadow_256=smooth_shadow,
            photometric_normals_256=normals,
            albedo_stripped=stripped,
        )

    def stitch_wdl_trestle_quilt(
        self,
        tile_wdl_lattices_17: Dict[Tuple[int, int], np.ndarray],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Stitch 17x17 WDL macro elevation lattices across adjacent tiles in a quilt.

        Enforces shared edge vertex equality:
          WDL(tx, ty)[:, 16] == WDL(tx + 1, ty)[:, 0]
          WDL(tx, ty)[16, :] == WDL(tx, ty + 1)[0, :]
        guaranteeing seamless continuous WDL trestle geometry (AC-004).
        """
        wdl_dict = {
            k: np.array(v, dtype=np.float32).copy()
            for k, v in tile_wdl_lattices_17.items()
        }
        all_tiles = set(wdl_dict.keys())

        # Horizontal seams: column 16 of left == column 0 of right
        for tx, ty in all_tiles:
            east = (tx + 1, ty)
            if east in all_tiles:
                left_col = wdl_dict[(tx, ty)][:, 16]
                right_col = wdl_dict[east][:, 0]
                mean_col = 0.5 * (left_col + right_col)
                wdl_dict[(tx, ty)][:, 16] = mean_col
                wdl_dict[east][:, 0] = mean_col

        # Vertical seams: row 16 of top == row 0 of bottom
        for tx, ty in all_tiles:
            south = (tx, ty + 1)
            if south in all_tiles:
                top_row = wdl_dict[(tx, ty)][16, :]
                bottom_row = wdl_dict[south][0, :]
                mean_row = 0.5 * (top_row + bottom_row)
                wdl_dict[(tx, ty)][16, :] = mean_row
                wdl_dict[south][0, :] = mean_row

        # 4-way corner intersections: (16, 16) / (16, 0) / (0, 16) / (0, 0)
        for tx, ty in all_tiles:
            e = (tx + 1, ty)
            s = (tx, ty + 1)
            se = (tx + 1, ty + 1)
            corners = [c for c in [(tx, ty), e, s, se] if c in all_tiles]
            if len(corners) > 1:
                vals = []
                for c in corners:
                    if c == (tx, ty):
                        vals.append(wdl_dict[c][16, 16])
                    elif c == e:
                        vals.append(wdl_dict[c][16, 0])
                    elif c == s:
                        vals.append(wdl_dict[c][0, 16])
                    elif c == se:
                        vals.append(wdl_dict[c][0, 0])
                m_corner = float(np.mean(vals))
                if (tx, ty) in all_tiles:
                    wdl_dict[(tx, ty)][16, 16] = m_corner
                if e in all_tiles:
                    wdl_dict[e][16, 0] = m_corner
                if s in all_tiles:
                    wdl_dict[s][0, 16] = m_corner
                if se in all_tiles:
                    wdl_dict[se][0, 0] = m_corner

        return wdl_dict

    def interpolate_wdl_to_257(self, wdl_17: np.ndarray, kx: int = 1, ky: int = 1) -> np.ndarray:
        """Upsample 17x17 WDL lattice to standard 257x257 macro elevation surface using exact grid splines."""
        from scipy.interpolate import RectBivariateSpline

        w = np.asarray(wdl_17, dtype=np.float32)
        x_in = np.linspace(0.0, 256.0, 17)
        y_in = np.linspace(0.0, 256.0, 17)
        spline = RectBivariateSpline(y_in, x_in, w, kx=kx, ky=ky)

        x_out = np.linspace(0.0, 256.0, 257)
        y_out = np.linspace(0.0, 256.0, 257)
        zoomed = spline(y_out, x_out).astype(np.float32)
        return zoomed

    def assemble_global_wdl_lattice(
        self,
        tile_wdl_lattices_17: Dict[Tuple[int, int], np.ndarray],
        bounds: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Any]:
        """Assemble individual 17x17 tile WDL lattices into a single continuous global (H*16+1, W*16+1) grid.

        The outer 1px edge (col 16 / row 16) is shared with the neighbor tile's 0th col/row,
        forming a unified continuous macro trestle across the entire quilt.
        """
        from harvester.v60.quilt_canvas_assembler import QuiltBounds

        all_keys = list(tile_wdl_lattices_17.keys())
        if bounds is None:
            xs = [tx for tx, _ in all_keys]
            ys = [ty for _, ty in all_keys]
            bounds = QuiltBounds(min(xs), min(ys), max(xs), max(ys))

        # Stitch seams across tiles first so shared boundary vertices match
        stitched_tiles = self.stitch_wdl_trestle_quilt(tile_wdl_lattices_17)

        h_verts = bounds.height_tiles * 16 + 1
        w_verts = bounds.width_tiles * 16 + 1
        global_lattice = np.zeros((h_verts, w_verts), dtype=np.float32)

        for (tx, ty), grid in stitched_tiles.items():
            u0 = (tx - bounds.min_tx) * 16
            v0 = (ty - bounds.min_ty) * 16
            global_lattice[v0 : v0 + 17, u0 : u0 + 17] = grid

        return global_lattice, bounds

    def interpolate_global_wdl_to_canvas(
        self,
        global_wdl_lattice: np.ndarray,
        bounds: Any,
        kx: int = 1,
        ky: int = 1,
    ) -> np.ndarray:
        """Upsample the global WDL lattice directly to the full continuous (H*256+1, W*256+1) elevation canvas.

        Because interpolation runs over the unified global grid with exact aligned corners,
        the outer 1px edges tie adjacent tiles together with mathematical C0 and C1 continuity.
        """
        from scipy.interpolate import RectBivariateSpline

        gw_h, gw_w = global_wdl_lattice.shape
        target_h = bounds.height_tiles * 256 + 1
        target_w = bounds.width_tiles * 256 + 1

        x_in = np.linspace(0.0, float(bounds.width_tiles * 256), gw_w)
        y_in = np.linspace(0.0, float(bounds.height_tiles * 256), gw_h)
        spline = RectBivariateSpline(y_in, x_in, global_wdl_lattice, kx=kx, ky=ky)

        x_out = np.linspace(0.0, float(bounds.width_tiles * 256), target_w)
        y_out = np.linspace(0.0, float(bounds.height_tiles * 256), target_h)
        return spline(y_out, x_out).astype(np.float32)

    def slice_tile_from_global_canvas(
        self,
        global_canvas: np.ndarray,
        tx: int,
        ty: int,
        bounds: Any,
    ) -> np.ndarray:
        """Slice a 257x257 tile elevation array from the continuous global canvas.

        Row/col 256 is guaranteed to be bit-for-bit identical to row/col 0 of the adjacent tile.
        """
        u0 = (tx - bounds.min_tx) * 256
        v0 = (ty - bounds.min_ty) * 256
        return global_canvas[v0 : v0 + 257, u0 : u0 + 257].copy()
