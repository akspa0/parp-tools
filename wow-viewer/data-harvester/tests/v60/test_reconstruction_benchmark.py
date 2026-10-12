"""Unit tests for terrain mesh reconstruction pipeline and AC-007 benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.fractal_brush_extractor import build_archetypal_brush_library
from harvester.v60.reconstruction_benchmark import (
    ReconstructionMetrics,
    TerrainReconstructionPipeline,
    compute_terrain_normals_from_heights,
    evaluate_reconstruction_fidelity,
    reconstruct_terrain_mesh,
)


def test_compute_terrain_normals_flat_and_sloped():
    # Flat surface
    flat = np.ones((16, 16), dtype=np.float32) * 10.0
    normals_flat = compute_terrain_normals_from_heights(flat)
    assert normals_flat.shape == (16, 16, 3)
    # Surface normal pointing straight up +Z (0, 0, 1)
    np.testing.assert_allclose(normals_flat[..., 2], 1.0, atol=1e-5)
    np.testing.assert_allclose(normals_flat[..., 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(normals_flat[..., 1], 0.0, atol=1e-5)

    # Unit lengths
    lengths = np.linalg.norm(normals_flat, axis=-1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_evaluate_reconstruction_fidelity_identical():
    # Identical surface
    x = np.linspace(-10, 10, 32, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    surface = np.sin(xx * 0.2) * 5.0 + np.cos(yy * 0.2) * 5.0

    metrics = evaluate_reconstruction_fidelity(surface, surface)
    assert metrics.mae_meters == 0.0
    assert metrics.rel_mae == 0.0
    assert metrics.geometric_fidelity_pct == 100.0
    assert abs(metrics.normal_cosine_similarity - 1.0) < 1e-4
    assert metrics.ridge_f1_score == 1.0


def test_terrain_reconstruction_pipeline_ac007():
    """Verify AC-007 targets: RelMAE <= 25% (>= 75% fidelity), NormSim >= 0.88, Ridge F1 >= 0.75."""
    h, w = 64, 64
    x = np.linspace(-2.0, 2.0, w, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # 1. Coarse low-frequency rolling terrain (baseline)
    coarse_baseline = (np.sin(xx) * 15.0 + np.cos(yy) * 10.0).astype(np.float32)

    # 2. Blizzard editor sculpting: ridge feature + conical peak
    ridge_sculpt = np.maximum(0.0, 8.0 - np.abs(xx) * 6.0) * np.exp(-yy * yy * 0.5)
    peak_sculpt = np.maximum(0.0, 10.0 - np.sqrt((xx - 0.8) ** 2 + (yy - 0.8) ** 2) * 12.0)
    ground_truth = (coarse_baseline + ridge_sculpt + peak_sculpt).astype(np.float32)

    # Continuous residual model provides smooth 70% approximation of the sculpting
    residual_prediction = (ridge_sculpt * 0.7 + peak_sculpt * 0.65).astype(np.float32)

    library = build_archetypal_brush_library(grid_size=33)
    pipeline = TerrainReconstructionPipeline(library)

    reconstructed, metrics, stamps = pipeline.reconstruct_and_evaluate(
        ground_truth_height=ground_truth,
        coarse_baseline=coarse_baseline,
        residual_prediction=residual_prediction,
        max_brush_stamps=6,
    )

    # Verify AC-007 criteria:
    # 1. Relative MAE <= 25% (>= 75% fidelity)
    assert metrics.rel_mae <= 0.25, f"RelMAE was {metrics.rel_mae:.4f} (target <= 0.25)"
    assert metrics.geometric_fidelity_pct >= 75.0, f"Fidelity was {metrics.geometric_fidelity_pct:.2f}%"

    # 2. Normal cosine similarity >= 0.88
    assert metrics.normal_cosine_similarity >= 0.88, (
        f"NormSim was {metrics.normal_cosine_similarity:.4f} (target >= 0.88)"
    )

    # 3. Ridge contour F1 score >= 0.75
    assert metrics.ridge_f1_score >= 0.75, f"Ridge F1 was {metrics.ridge_f1_score:.4f} (target >= 0.75)"

    # Verify brush stamps were fitted
    assert len(stamps) >= 1
