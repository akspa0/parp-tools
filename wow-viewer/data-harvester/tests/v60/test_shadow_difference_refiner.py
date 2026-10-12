"""Unit tests for closed-loop shadow difference and ridge synthesizer (Spec 262 Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.shadow_difference_refiner import (
    ShadowDifferenceRefiner,
    compute_shadow_difference,
    estimate_normal_perturbations,
    evaluate_ridge_alignment,
    extract_ridge_contours,
)


def test_compute_shadow_difference_clamping_and_mask():
    real = np.ones((16, 16), dtype=np.float32) * 0.8
    synth = np.ones((16, 16), dtype=np.float32) * 0.3

    diff = compute_shadow_difference(real, synth)
    assert diff.shape == (16, 16)
    np.testing.assert_allclose(diff, 0.5, atol=1e-5)

    # Test with valid mask
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255

    masked_diff = compute_shadow_difference(real, synth, valid_mask=mask)
    assert np.all(masked_diff[:4, :] == 0.0)
    assert np.all(masked_diff[4:12, 4:12] == 0.5)


def test_extract_ridge_contours_parabolic_crest():
    # Parabolic ridge along Y axis: f(x, y) = 1.0 - (x - 16)^2 / 32
    h, w = 32, 32
    x = np.arange(w, dtype=np.float32)
    profile = 1.0 - ((x - 16.0) ** 2) / 64.0
    ridge_surface = np.tile(profile, (h, 1))

    ridges, orientations = extract_ridge_contours(ridge_surface, sigma=1.0, curvature_threshold=0.01)

    assert ridges.shape == (32, 32)
    # The center column (x = 16) must be identified as the ridge crest
    assert np.all(ridges[4:28, 16] == 255)
    # Flanks (x = 0 or 31) are slope flanks, not downward curvature crests
    assert np.all(ridges[:, 0] == 0)


def test_evaluate_ridge_alignment_ac005():
    """Verify >= 80% precision and recall with tolerance buffer (AC-005)."""
    gt = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)

    # Ground truth vertical ridge at x = 32
    gt[10:54, 32] = 255

    # Predicted ridge slightly offset by 1 pixel (x = 33) — within 2-pixel tolerance buffer
    pred[10:54, 33] = 255

    metrics = evaluate_ridge_alignment(gt, pred, tolerance_radius=2)

    # Both precision and recall should be 1.0 with 2-pixel tolerance
    assert metrics.precision >= 0.80, f"Precision was {metrics.precision}"
    assert metrics.recall >= 0.80, f"Recall was {metrics.recall}"
    assert metrics.f1_score >= 0.80


def test_estimate_normal_perturbations():
    diff = np.zeros((16, 16), dtype=np.float32)
    diff[4:12, 4:12] = 0.5  # Real is brighter than synthetic

    # Solar overhead azimuth = 0 (pointing East), elevation = pi/4
    perturbations = estimate_normal_perturbations(
        delta_shadow=diff,
        solar_azimuth=0.0,
        solar_elevation=float(np.pi / 4.0),
        gain=0.5,
    )

    assert perturbations.shape == (16, 16, 3)
    # Perturbation in X should be non-zero towards the sun
    assert np.any(perturbations[6, 6, 0] != 0.0)


def test_shadow_difference_refiner_pipeline():
    refiner = ShadowDifferenceRefiner(curvature_threshold=0.01, tolerance_radius=2)

    # Synthetic ridge
    h, w = 32, 32
    x = np.arange(w, dtype=np.float32)
    real_shadow = np.tile(1.0 - ((x - 16.0) ** 2) / 64.0, (h, 1))
    synth_shadow = np.tile(1.0 - ((x - 15.0) ** 2) / 64.0, (h, 1))

    results = refiner.analyze_residual(
        real_shadow=real_shadow,
        synth_shadow=synth_shadow,
        solar_azimuth=0.0,
        solar_elevation=float(np.pi / 4.0),
    )

    assert "delta_shadow" in results
    assert "real_ridges" in results
    assert "synth_ridges" in results
    assert "metrics" in results
    assert results["metrics"].f1_score >= 0.80
