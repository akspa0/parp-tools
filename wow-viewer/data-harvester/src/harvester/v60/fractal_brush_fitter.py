"""3D Fractal Editor Brush Fitter and Matching Pursuit Engine (Spec 262 Phase 5).

Fits parameterized brush stamp invocations (B_j, p_i, sigma_i, theta_i, A_i) to
continuous terrain residual maps Delta_Z(x, y), reconstructing physical editor sculpting (AC-006).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from harvester.v60.fractal_brush_extractor import FractalEditorBrush

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StampInvocation:
    brush_id: str
    family: str
    center_xy: Tuple[float, float]
    radius: float  # scale sigma in canvas pixel units
    rotation_rad: float  # orientation theta
    amplitude: float  # displacement height A in meters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brush_id": self.brush_id,
            "family": self.family,
            "center_x": self.center_xy[0],
            "center_y": self.center_xy[1],
            "radius": self.radius,
            "rotation_rad": self.rotation_rad,
            "amplitude": self.amplitude,
        }


def compute_normalized_cross_correlation(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Compute Normalized Cross-Correlation (NCC) / Cosine similarity in [0, 1]."""
    s1 = np.asarray(signal1, dtype=np.float32).flatten()
    s2 = np.asarray(signal2, dtype=np.float32).flatten()

    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    dot = np.dot(s1, s2)
    return float(np.clip(dot / (norm1 * norm2), -1.0, 1.0))


def render_brush_stamp(
    brush: FractalEditorBrush,
    target_shape: Tuple[int, int],
    center_xy: Tuple[float, float],
    radius: float,
    rotation_rad: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Render a single transformed brush stamp onto target canvas."""
    h, w = target_shape
    cx, cy = center_xy
    r = max(1.0, radius)

    # Coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Translate to center
    dx = x_coords - cx
    dy = y_coords - cy

    # Rotate by -rotation_rad to sample unrotated brush domain
    cos_t = np.cos(-rotation_rad, dtype=np.float32)
    sin_t = np.sin(-rotation_rad, dtype=np.float32)
    rot_x = dx * cos_t - dy * sin_t
    rot_y = dx * sin_t + dy * cos_t

    # Scale to normalized brush coordinate domain [-1, 1]
    norm_u = rot_x / r
    norm_v = rot_y / r

    # Map [-1, 1] to brush array indices [0, N - 1]
    n = brush.grid_size
    idx_x = (norm_u + 1.0) * 0.5 * (n - 1)
    idx_y = (norm_v + 1.0) * 0.5 * (n - 1)

    # Bilinear sampling using scipy map_coordinates
    coords = np.array([idx_y, idx_x], dtype=np.float32)
    sampled = ndimage.map_coordinates(
        brush.displacement,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )

    # Outside radius r falloff cutoff
    dist_sq = norm_u * norm_u + norm_v * norm_v
    cutoff_mask = dist_sq <= 1.0
    sampled = sampled * cutoff_mask

    return (sampled * amplitude).astype(np.float32)


def render_stamps_composition(
    stamps: List[StampInvocation],
    brush_lookup: Dict[str, FractalEditorBrush],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Accumulate composition of multiple stamped brush invocations."""
    h, w = target_shape
    canvas = np.zeros((h, w), dtype=np.float32)

    for stamp in stamps:
        brush = brush_lookup.get(stamp.brush_id)
        if brush is None:
            continue
        canvas += render_brush_stamp(
            brush=brush,
            target_shape=target_shape,
            center_xy=stamp.center_xy,
            radius=stamp.radius,
            rotation_rad=stamp.rotation_rad,
            amplitude=stamp.amplitude,
        )

    return canvas


class FractalBrushFitter:
    """Matching pursuit engine for fitting discrete 3D fractal editor brushes."""

    def __init__(
        self,
        brush_library: List[FractalEditorBrush],
        candidate_scales: Optional[List[float]] = None,
        candidate_rotations: Optional[List[float]] = None,
    ):
        self.brush_library = brush_library
        self.brush_lookup = {b.brush_id: b for b in brush_library}
        self.candidate_scales = candidate_scales or [6.0, 10.0, 16.0, 24.0]
        # 4 rotation angles for symmetric/bilateral discovery
        self.candidate_rotations = candidate_rotations or [
            0.0,
            float(np.pi / 4.0),
            float(np.pi / 2.0),
            float(np.pi * 0.75),
        ]

    def fit_stamps(
        self,
        residual_target: np.ndarray,
        max_stamps: int = 8,
        min_residual_fraction: float = 0.15,
    ) -> Tuple[List[StampInvocation], float]:
        """Fit a sparse sequence of brush stamps using greedy matching pursuit.

        Returns:
            (stamps, final_ncc_correlation)
        """
        residual = np.asarray(residual_target, dtype=np.float32).copy()
        h, w = residual.shape
        stamps: List[StampInvocation] = []

        orig_energy = np.sum(residual * residual)
        if orig_energy < 1e-6:
            return [], 1.0

        for stamp_idx in range(max_stamps):
            # Locate current peak absolute residual error
            abs_res = np.abs(residual)
            flat_peak = np.argmax(abs_res)
            peak_y, peak_x = np.unravel_index(flat_peak, (h, w))
            peak_center = (float(peak_x), float(peak_y))

            best_stamp: Optional[StampInvocation] = None
            best_reduction = 0.0
            best_rendered: Optional[np.ndarray] = None

            # Search across library, scales, and rotations
            for brush in self.brush_library:
                rotations = [0.0] if brush.symmetry == "radial" else self.candidate_rotations

                for radius in self.candidate_scales:
                    for theta in rotations:
                        # Template candidate with amplitude 1.0
                        template = render_brush_stamp(
                            brush=brush,
                            target_shape=(h, w),
                            center_xy=peak_center,
                            radius=radius,
                            rotation_rad=theta,
                            amplitude=1.0,
                        )

                        denom = np.sum(template * template)
                        if denom < 1e-6:
                            continue

                        # Optimal least-squares amplitude
                        opt_amp = float(np.sum(residual * template) / denom)
                        if abs(opt_amp) < 1e-4:
                            continue

                        # Energy reduction: 2 * A * <res, t> - A^2 * <t, t> = A * <res, t>
                        reduction = opt_amp * np.sum(residual * template)

                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_stamp = StampInvocation(
                                brush_id=brush.brush_id,
                                family=brush.family,
                                center_xy=peak_center,
                                radius=radius,
                                rotation_rad=theta,
                                amplitude=opt_amp,
                            )
                            best_rendered = template * opt_amp

            if best_stamp is None or best_rendered is None or best_reduction <= 0.0:
                break

            stamps.append(best_stamp)
            residual -= best_rendered

            # Check convergence
            curr_energy = np.sum(residual * residual)
            if curr_energy / orig_energy < min_residual_fraction:
                break

        # Compute overall correlation against original target
        reconstructed = render_stamps_composition(stamps, self.brush_lookup, (h, w))
        ncc = compute_normalized_cross_correlation(residual_target, reconstructed)

        return stamps, ncc
