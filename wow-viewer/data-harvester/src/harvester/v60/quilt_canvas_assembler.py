"""Multi-Tile Quilt Canvas Assembler & Seam Boundary Solver (Spec 268 Phase 1).

Stitches adjacent minimap tiles into a continuous virtual canvas and solves
boundary vertices across adjacent tiles to enforce exact C0 height continuity
and C1 surface normal smoothness (AC-001).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QuiltBounds:
    min_tx: int
    min_ty: int
    max_tx: int
    max_ty: int

    @property
    def width_tiles(self) -> int:
        return self.max_tx - self.min_tx + 1

    @property
    def height_tiles(self) -> int:
        return self.max_ty - self.min_ty + 1

    @property
    def pixel_width(self) -> int:
        return self.width_tiles * 256

    @property
    def pixel_height(self) -> int:
        return self.height_tiles * 256


class QuiltCanvasAssembler:
    """Manages multi-tile quilt canvas stitching, global coordinate transformations,

    and seam boundary continuity enforcement across arbitrary collections of tiles.
    """

    def __init__(self, tiles: Optional[List[Tuple[int, int]]] = None) -> None:
        self.tiles: Set[Tuple[int, int]] = set(tiles or [])
        self._minimap_crops: Dict[Tuple[int, int], np.ndarray] = {}
        self._tile_elevations: Dict[Tuple[int, int], np.ndarray] = {}

    def add_tile(
        self,
        tx: int,
        ty: int,
        minimap_rgb: Optional[np.ndarray] = None,
        elevation_257: Optional[np.ndarray] = None,
    ) -> None:
        """Register a tile into the quilt canvas with optional minimap RGB and elevation grid."""
        coord = (tx, ty)
        self.tiles.add(coord)
        if minimap_rgb is not None:
            arr = np.asarray(minimap_rgb, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            self._minimap_crops[coord] = arr[:256, :256, :3]
        if elevation_257 is not None:
            self._tile_elevations[coord] = np.asarray(elevation_257, dtype=np.float32)[:257, :257]

    def compute_bounds(self) -> QuiltBounds:
        """Compute the axis-aligned bounding box enclosing all registered tiles."""
        if not self.tiles:
            return QuiltBounds(0, 0, 0, 0)
        xs = [tx for tx, _ in self.tiles]
        ys = [ty for _, ty in self.tiles]
        return QuiltBounds(min(xs), min(ys), max(xs), max(ys))

    def to_global_pixel_coords(
        self,
        tx: int,
        ty: int,
        local_u: float,
        local_v: float,
        bounds: Optional[QuiltBounds] = None,
    ) -> Tuple[float, float]:
        """Convert local tile coordinates (u, v) in [0, 256] to global quilt pixel coordinates."""
        b = bounds or self.compute_bounds()
        global_u = (tx - b.min_tx) * 256.0 + local_u
        global_v = (ty - b.min_ty) * 256.0 + local_v
        return global_u, global_v

    def to_local_tile_coords(
        self,
        global_u: float,
        global_v: float,
        bounds: Optional[QuiltBounds] = None,
    ) -> Tuple[int, int, float, float]:
        """Convert global quilt pixel coordinates to local tile coordinate (tx, ty) and (u, v)."""
        b = bounds or self.compute_bounds()
        tile_x_idx = int(np.floor(global_u / 256.0))
        tile_y_idx = int(np.floor(global_v / 256.0))
        tx = b.min_tx + tile_x_idx
        ty = b.min_ty + tile_y_idx
        local_u = global_u - tile_x_idx * 256.0
        local_v = global_v - tile_y_idx * 256.0
        return tx, ty, local_u, local_v

    def stitch_minimap_quilt(self, fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[np.ndarray, QuiltBounds]:
        """Stitch all registered tile minimap crops into a contiguous 2D RGB canvas."""
        bounds = self.compute_bounds()
        canvas = np.full(
            (bounds.pixel_height, bounds.pixel_width, 3),
            fill_color,
            dtype=np.uint8,
        )

        for (tx, ty), crop in self._minimap_crops.items():
            u0 = (tx - bounds.min_tx) * 256
            v0 = (ty - bounds.min_ty) * 256
            h, w = crop.shape[:2]
            canvas[v0 : v0 + h, u0 : u0 + w] = crop

        return canvas, bounds

    def stitch_2d_field(
        self,
        tile_fields_256: Dict[Tuple[int, int], np.ndarray],
        fill_val: float = 0.0,
    ) -> Tuple[np.ndarray, QuiltBounds]:
        """Stitch 2D (256, 256) float fields (e.g. bare shadows, albedos) into a continuous canvas."""
        bounds = self.compute_bounds()
        canvas = np.full(
            (bounds.pixel_height, bounds.pixel_width),
            fill_val,
            dtype=np.float32,
        )
        for (tx, ty), crop in tile_fields_256.items():
            u0 = (tx - bounds.min_tx) * 256
            v0 = (ty - bounds.min_ty) * 256
            canvas[v0 : v0 + 256, u0 : u0 + 256] = crop[:256, :256]
        return canvas, bounds

    def solve_seam_boundaries(
        self,
        tile_elevations_257: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
        margin: int = 4,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Solve and stitch boundary seams across all adjacent tile elevation lattices.

        Enforces exact C0 vertex continuity on shared boundary edges:
          Z(tx, ty)[y, 256] == Z(tx + 1, ty)[y, 0]
          Z(tx, ty)[256, x] == Z(tx, ty + 1)[0, x]
        and applies cosine-weighted Laplacian margin relaxation to achieve smooth
        C1 surface normal transitions (AC-001).

        Args:
            tile_elevations_257: Optional dictionary mapping (tx, ty) -> (257, 257) float32 arrays.
                                 Defaults to internally registered elevations.
            margin: Number of vertex steps adjacent to the seam over which to relax gradient steps.

        Returns:
            Dictionary of seamlessly stitched (257, 257) elevation grids.
        """
        elev_dict = (
            {k: np.array(v, dtype=np.float32).copy() for k, v in tile_elevations_257.items()}
            if tile_elevations_257 is not None
            else {k: np.array(v, dtype=np.float32).copy() for k, v in self._tile_elevations.items()}
        )

        all_tiles = set(elev_dict.keys())
        margin = max(1, min(margin, 16))

        # Cosine blending weights for margin relaxation: w(d) goes from 1.0 at seam (d=0) to 0.0 at d=margin
        d_indices = np.arange(1, margin + 1, dtype=np.float32)
        margin_weights = 0.5 * (1.0 + np.cos(np.pi * d_indices / (margin + 1)))

        # 1. Horizontal shared edges: between (tx, ty) and (tx + 1, ty)
        for tx, ty in all_tiles:
            neighbor = (tx + 1, ty)
            if neighbor in all_tiles:
                left_grid = elev_dict[(tx, ty)]
                right_grid = elev_dict[neighbor]

                # Shared border: east edge of left tile is column 256; west edge of right tile is column 0
                left_border = left_grid[:, 256]
                right_border = right_grid[:, 0]
                unified_border = 0.5 * (left_border + right_border)

                left_grid[:, 256] = unified_border
                right_grid[:, 0] = unified_border

                # Compute mean cross-boundary slope dZ/dx
                slope_left = left_border - left_grid[:, 255]
                slope_right = right_grid[:, 1] - right_border
                mean_slope = 0.5 * (slope_left + slope_right)

                # Relax interior margin columns
                for idx, d in enumerate(range(1, margin + 1)):
                    w = margin_weights[idx]
                    # Target expected height based on unified border and slope
                    target_left = unified_border - d * mean_slope
                    target_right = unified_border + d * mean_slope
                    left_grid[:, 256 - d] = (1.0 - w) * left_grid[:, 256 - d] + w * target_left
                    right_grid[:, d] = (1.0 - w) * right_grid[:, d] + w * target_right

        # 2. Vertical shared edges: between (tx, ty) and (tx, ty + 1)
        for tx, ty in all_tiles:
            neighbor = (tx, ty + 1)
            if neighbor in all_tiles:
                top_grid = elev_dict[(tx, ty)]
                bottom_grid = elev_dict[neighbor]

                # Shared border: south edge of top tile is row 256; north edge of bottom tile is row 0
                top_border = top_grid[256, :]
                bottom_border = bottom_grid[0, :]
                unified_border = 0.5 * (top_border + bottom_border)

                top_grid[256, :] = unified_border
                bottom_grid[0, :] = unified_border

                # Compute mean cross-boundary slope dZ/dy
                slope_top = top_border - top_grid[255, :]
                slope_bottom = bottom_grid[1, :] - bottom_border
                mean_slope = 0.5 * (slope_top + slope_bottom)

                # Relax interior margin rows
                for idx, d in enumerate(range(1, margin + 1)):
                    w = margin_weights[idx]
                    target_top = unified_border - d * mean_slope
                    target_bottom = unified_border + d * mean_slope
                    top_grid[256 - d, :] = (1.0 - w) * top_grid[256 - d, :] + w * target_top
                    bottom_grid[d, :] = (1.0 - w) * bottom_grid[d, :] + w * target_bottom

        # 3. 4-way corner intersection unification: (tx, ty), (tx+1, ty), (tx, ty+1), (tx+1, ty+1)
        for tx, ty in all_tiles:
            e = (tx + 1, ty)
            s = (tx, ty + 1)
            se = (tx + 1, ty + 1)
            corner_tiles = [coord for coord in [(tx, ty), e, s, se] if coord in all_tiles]
            if len(corner_tiles) > 1:
                corner_vals = []
                for coord in corner_tiles:
                    grid = elev_dict[coord]
                    if coord == (tx, ty):
                        corner_vals.append(grid[256, 256])
                    elif coord == e:
                        corner_vals.append(grid[256, 0])
                    elif coord == s:
                        corner_vals.append(grid[0, 256])
                    elif coord == se:
                        corner_vals.append(grid[0, 0])

                mean_corner = float(np.mean(corner_vals))
                if (tx, ty) in all_tiles:
                    elev_dict[(tx, ty)][256, 256] = mean_corner
                if e in all_tiles:
                    elev_dict[e][256, 0] = mean_corner
                if s in all_tiles:
                    elev_dict[s][0, 256] = mean_corner
                if se in all_tiles:
                    elev_dict[se][0, 0] = mean_corner

        return elev_dict

    def verify_seam_continuity(
        self,
        tile_elevations_257: Dict[Tuple[int, int], np.ndarray],
    ) -> Dict[str, float]:
        """Compute verification metrics on boundary seam continuity (C0 step & C1 normal cosine).

        Returns:
            Dictionary containing:
              - 'max_height_step_yards': Maximum absolute height step across any shared border edge.
              - 'mean_height_step_yards': Average absolute height step across all border vertices.
              - 'mean_normal_cosine_similarity': Average cosine similarity of surface normals along borders.
              - 'seams_evaluated': Count of evaluated shared border edges.
        """
        all_tiles = set(tile_elevations_257.keys())
        max_step = 0.0
        total_step = 0.0
        step_samples = 0
        normal_cosines: List[float] = []

        # Helper to compute boundary surface normal vectors
        def get_normals(h_arr: np.ndarray) -> np.ndarray:
            dz_dy, dz_dx = np.gradient(h_arr.astype(np.float32))
            norm = np.zeros((*h_arr.shape, 3), dtype=np.float32)
            norm[..., 0] = -dz_dx
            norm[..., 1] = -dz_dy
            norm[..., 2] = 1.0
            lens = np.linalg.norm(norm, axis=-1, keepdims=True)
            return norm / np.maximum(lens, 1e-6)

        tile_normals = {coord: get_normals(grid) for coord, grid in tile_elevations_257.items()}

        seam_count = 0

        # Evaluate horizontal seams
        for tx, ty in all_tiles:
            neighbor = (tx + 1, ty)
            if neighbor in all_tiles:
                seam_count += 1
                left_edge = tile_elevations_257[(tx, ty)][:, 256]
                right_edge = tile_elevations_257[neighbor][:, 0]
                diffs = np.abs(left_edge - right_edge)
                max_step = max(max_step, float(np.max(diffs)))
                total_step += float(np.sum(diffs))
                step_samples += len(diffs)

                norm_left = tile_normals[(tx, ty)][:, 256]
                norm_right = tile_normals[neighbor][:, 0]
                dots = np.sum(norm_left * norm_right, axis=-1)
                normal_cosines.extend(dots.tolist())

        # Evaluate vertical seams
        for tx, ty in all_tiles:
            neighbor = (tx, ty + 1)
            if neighbor in all_tiles:
                seam_count += 1
                top_edge = tile_elevations_257[(tx, ty)][256, :]
                bottom_edge = tile_elevations_257[neighbor][0, :]
                diffs = np.abs(top_edge - bottom_edge)
                max_step = max(max_step, float(np.max(diffs)))
                total_step += float(np.sum(diffs))
                step_samples += len(diffs)

                norm_top = tile_normals[(tx, ty)][256, :]
                norm_bottom = tile_normals[neighbor][0, :]
                dots = np.sum(norm_top * norm_bottom, axis=-1)
                normal_cosines.extend(dots.tolist())

        mean_step = total_step / max(1, step_samples)
        mean_cos = float(np.mean(normal_cosines)) if normal_cosines else 1.0

        return {
            "max_height_step_yards": max_step,
            "mean_height_step_yards": mean_step,
            "mean_normal_cosine_similarity": mean_cos,
            "seams_evaluated": float(seam_count),
        }

    def assemble_global_elevation_canvas(
        self,
        tile_elevations_257: Dict[Tuple[int, int], np.ndarray],
    ) -> Tuple[np.ndarray, QuiltBounds]:
        """Assemble seamlessly stitched tile grids into a single continuous 2D heightfield.

        Dimensions: (height_tiles * 256 + 1, width_tiles * 256 + 1)
        """
        all_keys = list(tile_elevations_257.keys())
        if all_keys:
            xs = [tx for tx, _ in all_keys]
            ys = [ty for _, ty in all_keys]
            bounds = QuiltBounds(min(xs), min(ys), max(xs), max(ys))
        else:
            bounds = self.compute_bounds()

        canvas_h = bounds.height_tiles * 256 + 1
        canvas_w = bounds.width_tiles * 256 + 1
        global_grid = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for (tx, ty), grid in tile_elevations_257.items():
            u0 = (tx - bounds.min_tx) * 256
            v0 = (ty - bounds.min_ty) * 256
            global_grid[v0 : v0 + 257, u0 : u0 + 257] = grid

        return global_grid, bounds
