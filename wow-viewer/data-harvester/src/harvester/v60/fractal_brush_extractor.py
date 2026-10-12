"""3D Fractal Editor Brush Discovery and Extraction Engine (Spec 262 Phase 5).

Extracts discrete Blizzard-style 3D sculpting brushes that couple spatial mesh
displacement Delta_Z(u, v) with texture alpha splatting footprints alpha_k(u, v).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class FractalEditorBrush:
    brush_id: str
    family: str  # ridge, peak, trench, plateau, terrace, fractal_mountain
    grid_size: int
    displacement: np.ndarray  # (N, N) float32 in [-1, 1] or normalized amplitude
    alpha_footprint: np.ndarray  # (N, N) float32 in [0, 1]
    roughness_fractal_dim: float  # D in [2.0, 3.0]
    peak_displacement: float  # Absolute max peak in meters
    symmetry: str  # radial, bilateral, asymmetric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brush_id": self.brush_id,
            "family": self.family,
            "grid_size": self.grid_size,
            "displacement": self.displacement.tolist(),
            "alpha_footprint": self.alpha_footprint.tolist(),
            "roughness_fractal_dim": self.roughness_fractal_dim,
            "peak_displacement": self.peak_displacement,
            "symmetry": self.symmetry,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FractalEditorBrush:
        return cls(
            brush_id=data["brush_id"],
            family=data["family"],
            grid_size=int(data["grid_size"]),
            displacement=np.array(data["displacement"], dtype=np.float32),
            alpha_footprint=np.array(data["alpha_footprint"], dtype=np.float32),
            roughness_fractal_dim=float(data["roughness_fractal_dim"]),
            peak_displacement=float(data["peak_displacement"]),
            symmetry=data["symmetry"],
        )


def compute_surface_fractal_dimension(surface: np.ndarray) -> float:
    """Estimate surface fractal dimension D in [2.0, 3.0] via multi-scale variogram analysis."""
    arr = np.asarray(surface, dtype=np.float32)
    h, w = arr.shape
    if min(h, w) < 4:
        return 2.1

    # Variogram slope log(Var(Delta z)) vs log(lag)
    lags = [1, 2, 4, 8]
    variances: List[float] = []
    valid_lags: List[float] = []

    for lag in lags:
        if lag < min(h, w) // 2:
            diff_x = arr[:, lag:] - arr[:, :-lag]
            diff_y = arr[lag:, :] - arr[:-lag, :]
            var_val = float(np.mean(diff_x * diff_x) + np.mean(diff_y * diff_y)) * 0.5
            if var_val > 1e-8:
                variances.append(var_val)
                valid_lags.append(float(lag))

    if len(valid_lags) >= 2:
        # log(Var) = 2 * H * log(lag) + C
        # D = 3 - H
        log_lags = np.log(valid_lags)
        log_vars = np.log(variances)
        slope, _ = np.polyfit(log_lags, log_vars, 1)
        hurst = np.clip(slope * 0.5, 0.05, 0.95)
        return float(np.clip(3.0 - hurst, 2.05, 2.95))

    return 2.3  # Standard natural terrain fractal dimension default


def detrend_planar_slope(height_patch: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Subtract best-fit planar slope z = a*x + b*y + c to isolate pure sculpting displacement."""
    arr = np.asarray(height_patch, dtype=np.float32)
    h, w = arr.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Least squares planar fit: A * [a, b, c]^T = z
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = arr.flatten()

    design = np.column_stack([x_flat, y_flat, np.ones_like(x_flat)])
    plane_params, _, _, _ = np.linalg.lstsq(design, z_flat, rcond=None)

    fitted_plane = (plane_params[0] * x_coords + plane_params[1] * y_coords + plane_params[2]).astype(np.float32)
    residual_displacement = arr - fitted_plane

    return residual_displacement, (float(plane_params[0]), float(plane_params[1]), float(plane_params[2]))


def extract_brush_from_patch(
    height_patch: np.ndarray,
    alpha_patch: Optional[np.ndarray] = None,
    target_grid_size: int = 33,
    family: str = "ridge",
) -> FractalEditorBrush:
    """Extract and normalize a discrete 3D brush from a height and alpha patch."""
    displacement_raw, _ = detrend_planar_slope(height_patch)

    # Resample to canonical grid size (e.g., 33x33)
    curr_h, curr_w = displacement_raw.shape
    zh = target_grid_size / curr_h
    zw = target_grid_size / curr_w

    resampled_disp = ndimage.zoom(displacement_raw, (zh, zw), order=1)[:target_grid_size, :target_grid_size]

    peak_val = float(np.max(np.abs(resampled_disp)))
    if peak_val > 1e-5:
        norm_disp = (resampled_disp / peak_val).astype(np.float32)
    else:
        norm_disp = np.zeros((target_grid_size, target_grid_size), dtype=np.float32)
        peak_val = 1.0

    # Alpha footprint
    if alpha_patch is not None:
        raw_alpha = np.asarray(alpha_patch, dtype=np.float32)
        if raw_alpha.max() > 1.0:
            raw_alpha /= 255.0
        norm_alpha = ndimage.zoom(raw_alpha, (zh, zw), order=1)[:target_grid_size, :target_grid_size]
        norm_alpha = np.clip(norm_alpha, 0.0, 1.0).astype(np.float32)
    else:
        # Default alpha footprint tracks absolute displacement
        norm_alpha = np.abs(norm_disp).astype(np.float32)

    # Classify symmetry
    diff_h = np.mean(np.abs(norm_disp - np.fliplr(norm_disp)))
    diff_v = np.mean(np.abs(norm_disp - np.flipud(norm_disp)))
    diff_diag = np.mean(np.abs(norm_disp - norm_disp.T))

    if diff_diag < 0.05 and diff_h < 0.05 and diff_v < 0.05:
        symmetry = "radial"
    elif diff_h < 0.08 or diff_v < 0.08:
        symmetry = "bilateral"
    else:
        symmetry = "asymmetric"

    fractal_dim = compute_surface_fractal_dimension(norm_disp)

    # Deterministic brush ID
    content_bytes = norm_disp.tobytes() + norm_alpha.tobytes()
    brush_id = f"brush_{family}_{hashlib.sha256(content_bytes).hexdigest()[:12]}"

    return FractalEditorBrush(
        brush_id=brush_id,
        family=family,
        grid_size=target_grid_size,
        displacement=norm_disp,
        alpha_footprint=norm_alpha,
        roughness_fractal_dim=fractal_dim,
        peak_displacement=peak_val,
        symmetry=symmetry,
    )


def build_archetypal_brush_library(grid_size: int = 33) -> List[FractalEditorBrush]:
    """Generate Blizzard 0.5.3 archetypal editor brushes from mathematical primitive profiles."""
    brushes: List[FractalEditorBrush] = []

    # Coordinate domain [-1, 1]
    coord = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coord, coord)
    radius_sq = xx * xx + yy * yy

    # 1. Bilateral Ridge Brush
    ridge_disp = np.maximum(0.0, 1.0 - np.abs(xx) * 1.5) * np.exp(-yy * yy * 2.0)
    brushes.append(
        extract_brush_from_patch(ridge_disp, target_grid_size=grid_size, family="ridge")
    )

    # 2. Conical Peak Brush
    peak_disp = np.maximum(0.0, 1.0 - np.sqrt(radius_sq))
    brushes.append(
        extract_brush_from_patch(peak_disp, target_grid_size=grid_size, family="peak")
    )

    # 3. Smooth Gaussian Bell Brush
    gauss_disp = np.exp(-radius_sq * 3.5)
    brushes.append(
        extract_brush_from_patch(gauss_disp, target_grid_size=grid_size, family="peak")
    )

    # 4. Flat-Topped Plateau Brush
    plateau_disp = np.clip(1.5 - np.sqrt(radius_sq) * 1.5, 0.0, 1.0)
    brushes.append(
        extract_brush_from_patch(plateau_disp, target_grid_size=grid_size, family="plateau")
    )

    # 5. Trench / Ravine Gouge Brush
    trench_disp = -np.maximum(0.0, 1.0 - np.abs(xx) * 1.5) * np.exp(-yy * yy * 2.0)
    brushes.append(
        extract_brush_from_patch(trench_disp, target_grid_size=grid_size, family="trench")
    )

    return brushes
