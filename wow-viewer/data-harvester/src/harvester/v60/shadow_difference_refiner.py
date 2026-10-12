"""Closed-Loop Shadow Difference & Directional Ridge Residual Synthesizer (Spec 262 Phase 4).

Computes photometric shadow difference:
    \\Delta S(x, y) = S_{real}(x, y) - S_{synth}(x, y)
extracts directional Hessian ridge/crest contours, evaluates ridge alignment fidelity (AC-005),
and inverts residual shading differences into normal perturbations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RidgeEvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    true_positive_pixels: int
    predicted_pixels: int
    ground_truth_pixels: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positive_pixels": float(self.true_positive_pixels),
            "predicted_pixels": float(self.predicted_pixels),
            "ground_truth_pixels": float(self.ground_truth_pixels),
        }


def compute_shadow_difference(
    real_shadow: np.ndarray,
    synth_shadow: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute normalized shadow difference delta_S = S_real - S_synth in [-1, 1]."""
    real = np.asarray(real_shadow, dtype=np.float32)
    synth = np.asarray(synth_shadow, dtype=np.float32)

    diff = real - synth
    diff = np.clip(diff, -1.0, 1.0)

    if valid_mask is not None:
        m = (np.asarray(valid_mask) > 0).astype(bool)
        diff[~m] = 0.0

    return diff


def extract_ridge_contours(
    shadow_map: np.ndarray,
    sigma: float = 1.5,
    curvature_threshold: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract physical terrain ridge crests using Hessian matrix principal curvature analysis.

    Returns:
        (ridge_binary_mask, ridge_orientations)
        - ridge_binary_mask: uint8 (H, W) array where 255 indicates a ridge crest.
        - ridge_orientations: float32 (H, W) array of principal ridge angles in radians [-pi, pi].
    """
    img = np.asarray(shadow_map, dtype=np.float32)
    if sigma > 0:
        smoothed = ndimage.gaussian_filter(img, sigma=sigma)
    else:
        smoothed = img

    # Compute Hessian matrix second partial derivatives
    # Ixx, Iyy, Ixy
    ixx = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=1) / 16.0
    iyy = ndimage.sobel(ndimage.sobel(smoothed, axis=0), axis=0) / 16.0
    ixy = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=0) / 16.0

    # Hessian eigenvalues:
    # trace = Ixx + Iyy
    # det = Ixx * Iyy - Ixy^2
    trace = ixx + iyy
    det = ixx * iyy - ixy * ixy
    discriminant = np.maximum(0.0, trace * trace - 4.0 * det)
    sqrt_disc = np.sqrt(discriminant)

    # Minimum eigenvalue (most negative indicates downward curvature across ridge crest)
    lambda_min = (trace - sqrt_disc) * 0.5

    # Ridge crest orientation: eigenvector perpendicular to strongest gradient curvature
    # theta = 0.5 * atan2(2 * Ixy, Ixx - Iyy)
    ridge_orientations = 0.5 * np.arctan2(2.0 * ixy, ixx - iyy).astype(np.float32)

    # Threshold for ridge detection
    ridge_mask = (lambda_min < -curvature_threshold).astype(np.uint8) * 255

    return ridge_mask, ridge_orientations


def evaluate_ridge_alignment(
    ridge_gt: np.ndarray,
    ridge_pred: np.ndarray,
    tolerance_radius: int = 2,
) -> RidgeEvaluationMetrics:
    """Evaluate ridge precision and recall with morphological tolerance buffer (AC-005)."""
    gt_bool = (np.asarray(ridge_gt) > 0).astype(bool)
    pred_bool = (np.asarray(ridge_pred) > 0).astype(bool)

    num_gt = int(np.sum(gt_bool))
    num_pred = int(np.sum(pred_bool))

    if num_gt == 0 and num_pred == 0:
        return RidgeEvaluationMetrics(1.0, 1.0, 1.0, 0, 0, 0)
    if num_gt == 0 or num_pred == 0:
        return RidgeEvaluationMetrics(0.0, 0.0, 0.0, 0, num_pred, num_gt)

    # Dilate GT to construct tolerance matching envelope
    if tolerance_radius > 0:
        struct = ndimage.generate_binary_structure(2, 2)
        gt_envelope = ndimage.binary_dilation(gt_bool, structure=struct, iterations=tolerance_radius)
        pred_envelope = ndimage.binary_dilation(pred_bool, structure=struct, iterations=tolerance_radius)
    else:
        gt_envelope = gt_bool
        pred_envelope = pred_bool

    # Matches
    pred_matches = np.logical_and(pred_bool, gt_envelope)
    gt_matches = np.logical_and(gt_bool, pred_envelope)

    tp_pred = int(np.sum(pred_matches))
    tp_gt = int(np.sum(gt_matches))

    precision = tp_pred / num_pred if num_pred > 0 else 0.0
    recall = tp_gt / num_gt if num_gt > 0 else 0.0

    if precision + recall > 1e-6:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return RidgeEvaluationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        true_positive_pixels=tp_pred,
        predicted_pixels=num_pred,
        ground_truth_pixels=num_gt,
    )


def estimate_normal_perturbations(
    delta_shadow: np.ndarray,
    solar_azimuth: float,
    solar_elevation: float,
    gain: float = 0.5,
) -> np.ndarray:
    """Invert photometric shadow difference into 3D terrain surface normal perturbations.

    Projects delta_S along the horizontal sun vector direction to steer surface normal
    inclination towards or away from the sun.
    """
    diff = np.asarray(delta_shadow, dtype=np.float32)
    h, w = diff.shape[:2]

    # Sun direction vector pointing towards light
    cos_el = np.cos(solar_elevation, dtype=np.float32)
    sin_el = np.sin(solar_elevation, dtype=np.float32)
    sun_x = -np.cos(solar_azimuth, dtype=np.float32) * cos_el
    sun_y = -np.sin(solar_azimuth, dtype=np.float32) * cos_el

    perturbations = np.zeros((h, w, 3), dtype=np.float32)
    # Shading brighter than synthetic (delta_S > 0) indicates slope facing sun
    perturbations[..., 0] = diff * sun_x * gain
    perturbations[..., 1] = diff * sun_y * gain
    perturbations[..., 2] = 0.0

    return perturbations


class ShadowDifferenceRefiner:
    """Orchestrates closed-loop shadow difference extraction and ridge synthesis."""

    def __init__(self, curvature_threshold: float = 0.02, tolerance_radius: int = 2):
        self.curvature_threshold = curvature_threshold
        self.tolerance_radius = tolerance_radius

    def analyze_residual(
        self,
        real_shadow: np.ndarray,
        synth_shadow: np.ndarray,
        solar_azimuth: float,
        solar_elevation: float,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute delta_S, extract ridge contours, evaluate alignment, and compute normal corrections."""
        delta_s = compute_shadow_difference(real_shadow, synth_shadow, valid_mask=valid_mask)

        # Ridge detection on real and synthetic shadow
        real_ridges, real_angles = extract_ridge_contours(
            real_shadow, curvature_threshold=self.curvature_threshold
        )
        synth_ridges, synth_angles = extract_ridge_contours(
            synth_shadow, curvature_threshold=self.curvature_threshold
        )

        metrics = evaluate_ridge_alignment(
            ridge_gt=real_ridges,
            ridge_pred=synth_ridges,
            tolerance_radius=self.tolerance_radius,
        )

        normal_delta = estimate_normal_perturbations(
            delta_s,
            solar_azimuth=solar_azimuth,
            solar_elevation=solar_elevation,
        )

        return {
            "delta_shadow": delta_s,
            "real_ridges": real_ridges,
            "synth_ridges": synth_ridges,
            "metrics": metrics,
            "normal_perturbations": normal_delta,
        }
