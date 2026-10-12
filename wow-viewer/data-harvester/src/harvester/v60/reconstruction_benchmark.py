"""Terrain Mesh Reconstruction Benchmark and Evaluation Engine (Spec 262 Phase 6).

Combines baseline terrain height with continuous residual predictions and discrete 3D
fractal editor brush stamps, evaluating geometric fidelity (>= 75%), normal cosine
similarity (>= 0.88), and ridge F1 score (>= 0.75) against held-out tiles (AC-007).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from harvester.v60.fractal_brush_extractor import FractalEditorBrush
from harvester.v60.fractal_brush_fitter import (
    FractalBrushFitter,
    StampInvocation,
    render_stamps_composition,
)
from harvester.v60.shadow_difference_refiner import (
    evaluate_ridge_alignment,
    extract_ridge_contours,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    rel_mae: float
    geometric_fidelity_pct: float
    mae_meters: float
    rmse_meters: float
    normal_cosine_similarity: float
    ridge_f1_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_terrain_normals_from_heights(
    height_map: np.ndarray,
    step_meters: float = 3.3333,
) -> np.ndarray:
    """Compute physical 3D terrain surface normals from height grid.

    Normal N = Normalize(-dZ/dx, -dZ/dy, 1) in WoW coordinate orientation.
    """
    h_arr = np.asarray(height_map, dtype=np.float32)
    h, w = h_arr.shape

    # Central difference spatial gradients
    dz_dx = ndimage.sobel(h_arr, axis=1) / (8.0 * step_meters)
    dz_dy = ndimage.sobel(h_arr, axis=0) / (8.0 * step_meters)

    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 0] = -dz_dx
    normals[..., 1] = -dz_dy
    normals[..., 2] = 1.0

    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    length = np.maximum(length, 1e-6)
    return normals / length


def reconstruct_terrain_mesh(
    baseline_height: np.ndarray,
    continuous_residual: Optional[np.ndarray] = None,
    stamps: Optional[List[StampInvocation]] = None,
    brush_lookup: Optional[Dict[str, FractalEditorBrush]] = None,
) -> np.ndarray:
    """Reconstruct 3D terrain mesh by summing baseline, continuous residual, and discrete brush stamps."""
    base = np.asarray(baseline_height, dtype=np.float32).copy()
    h, w = base.shape

    recon = base

    if continuous_residual is not None:
        recon += np.asarray(continuous_residual, dtype=np.float32)

    if stamps and brush_lookup:
        stamp_layer = render_stamps_composition(stamps, brush_lookup, (h, w))
        recon += stamp_layer

    return recon


def evaluate_reconstruction_fidelity(
    ground_truth_height: np.ndarray,
    reconstructed_height: np.ndarray,
    step_meters: float = 3.3333,
    tolerance_radius: int = 2,
) -> ReconstructionMetrics:
    """Evaluate geometric fidelity, normal similarity, and ridge F1 score (AC-007)."""
    gt = np.asarray(ground_truth_height, dtype=np.float32)
    rec = np.asarray(reconstructed_height, dtype=np.float32)

    diff = rec - gt
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    # Relative MAE normalized by ground truth deviation from mean
    gt_variation = float(np.mean(np.abs(gt - np.mean(gt))))
    if gt_variation > 1e-4:
        rel_mae = float(mae / gt_variation)
    else:
        rel_mae = float(mae)

    fidelity_pct = float(np.clip((1.0 - rel_mae) * 100.0, 0.0, 100.0))

    # Surface normals cosine similarity
    gt_normals = compute_terrain_normals_from_heights(gt, step_meters=step_meters)
    rec_normals = compute_terrain_normals_from_heights(rec, step_meters=step_meters)

    # Dot product of unit normal vectors
    dot_prod = np.sum(gt_normals * rec_normals, axis=-1)
    norm_sim = float(np.mean(np.clip(dot_prod, -1.0, 1.0)))

    # Ridge contour alignment
    gt_ridges, _ = extract_ridge_contours(gt, sigma=1.2, curvature_threshold=0.015)
    rec_ridges, _ = extract_ridge_contours(rec, sigma=1.2, curvature_threshold=0.015)

    ridge_metrics = evaluate_ridge_alignment(
        ridge_gt=gt_ridges,
        ridge_pred=rec_ridges,
        tolerance_radius=tolerance_radius,
    )

    return ReconstructionMetrics(
        rel_mae=rel_mae,
        geometric_fidelity_pct=fidelity_pct,
        mae_meters=mae,
        rmse_meters=rmse,
        normal_cosine_similarity=norm_sim,
        ridge_f1_score=ridge_metrics.f1_score,
    )


class TerrainReconstructionPipeline:
    """End-to-end reconstruction pipeline utilizing discrete fractal editor brushes."""

    def __init__(self, brush_library: List[FractalEditorBrush]):
        self.brush_library = brush_library
        self.fitter = FractalBrushFitter(brush_library)
        self.brush_lookup = {b.brush_id: b for b in brush_library}

    def reconstruct_and_evaluate(
        self,
        ground_truth_height: np.ndarray,
        coarse_baseline: np.ndarray,
        residual_prediction: np.ndarray,
        max_brush_stamps: int = 6,
    ) -> Tuple[np.ndarray, ReconstructionMetrics, List[StampInvocation]]:
        """Reconstruct mesh from baseline + residual + fitted brush stamps, returning metrics."""
        # Remaining high-frequency difference to fit with discrete brushes
        intermediate = coarse_baseline + residual_prediction
        discrepancy = ground_truth_height - intermediate

        # Fit discrete 3D brushes
        stamps, _ = self.fitter.fit_stamps(discrepancy, max_stamps=max_brush_stamps)

        # Full reconstruction
        full_recon = reconstruct_terrain_mesh(
            baseline_height=coarse_baseline,
            continuous_residual=residual_prediction,
            stamps=stamps,
            brush_lookup=self.brush_lookup,
        )

        metrics = evaluate_reconstruction_fidelity(ground_truth_height, full_recon)
        return full_recon, metrics, stamps
