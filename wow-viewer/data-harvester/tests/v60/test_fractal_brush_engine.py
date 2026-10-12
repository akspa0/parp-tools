"""Unit tests for 3D fractal editor brush discovery and fitting engine (Spec 262 Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.fractal_brush_extractor import (
    FractalEditorBrush,
    build_archetypal_brush_library,
    compute_surface_fractal_dimension,
    detrend_planar_slope,
    extract_brush_from_patch,
)
from harvester.v60.fractal_brush_fitter import (
    FractalBrushFitter,
    compute_normalized_cross_correlation,
    render_brush_stamp,
    render_stamps_composition,
)


def test_surface_fractal_dimension():
    # Flat surface
    flat = np.ones((16, 16), dtype=np.float32) * 5.0
    d_flat = compute_surface_fractal_dimension(flat)
    assert 2.0 <= d_flat <= 3.0

    # Rough random surface
    np.random.seed(42)
    rough = np.random.normal(0, 1, (32, 32)).astype(np.float32)
    d_rough = compute_surface_fractal_dimension(rough)
    assert 2.0 <= d_rough <= 3.0


def test_detrend_planar_slope():
    # Surface with explicit slope: z = 2.0 * x - 1.5 * y + 10.0 + Gaussian peak
    h, w = 32, 32
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    slope = 2.0 * x_idx - 1.5 * y_idx + 10.0
    peak = 5.0 * np.exp(-((x_idx - 16) ** 2 + (y_idx - 16) ** 2) / 20.0)
    surface = slope + peak

    residual, plane_params = detrend_planar_slope(surface)

    # Slopes should be closely recovered
    assert abs(plane_params[0] - 2.0) < 0.1
    assert abs(plane_params[1] - (-1.5)) < 0.1

    # Residual peak at center should remain positive and non-zero
    assert residual[16, 16] > 3.0
    # Far corners should be close to zero
    assert abs(residual[0, 0]) < 0.5


def test_build_archetypal_brush_library():
    library = build_archetypal_brush_library(grid_size=33)
    assert len(library) >= 5

    families = {b.family for b in library}
    assert "ridge" in families
    assert "peak" in families
    assert "plateau" in families
    assert "trench" in families

    for b in library:
        assert b.grid_size == 33
        assert b.displacement.shape == (33, 33)
        assert b.alpha_footprint.shape == (33, 33)


def test_render_brush_stamp_rotation_and_amplitude():
    library = build_archetypal_brush_library(grid_size=33)
    ridge_brush = next(b for b in library if b.family == "ridge")

    # Render horizontal vs rotated
    stamp_0 = render_brush_stamp(
        ridge_brush,
        target_shape=(64, 64),
        center_xy=(32.0, 32.0),
        radius=16.0,
        rotation_rad=0.0,
        amplitude=10.0,
    )
    assert stamp_0.shape == (64, 64)
    assert abs(stamp_0.max() - 10.0) < 0.5

    stamp_90 = render_brush_stamp(
        ridge_brush,
        target_shape=(64, 64),
        center_xy=(32.0, 32.0),
        radius=16.0,
        rotation_rad=float(np.pi / 2.0),
        amplitude=10.0,
    )
    # 90-degree rotated stamp should be close to transpose
    ncc = compute_normalized_cross_correlation(stamp_0.T, stamp_90)
    assert ncc > 0.90


def test_fractal_brush_fitter_ac006():
    """Verify >= 85% cross-correlation when fitting recurring 3D features (AC-006)."""
    library = build_archetypal_brush_library(grid_size=33)
    fitter = FractalBrushFitter(
        brush_library=library,
        candidate_scales=[10.0, 16.0, 24.0],
    )

    # Synthesize target terrain residual composed of two distinct stamped features:
    # 1. A ridge feature at (20, 25)
    # 2. A conical peak at (45, 40)
    ridge_brush = next(b for b in library if b.family == "ridge")
    peak_brush = next(b for b in library if b.family == "peak")

    stamp1 = render_brush_stamp(
        ridge_brush,
        (64, 64),
        center_xy=(20.0, 25.0),
        radius=16.0,
        rotation_rad=float(np.pi / 4.0),
        amplitude=8.0,
    )
    stamp2 = render_brush_stamp(
        peak_brush,
        (64, 64),
        center_xy=(45.0, 40.0),
        radius=12.0,
        rotation_rad=0.0,
        amplitude=6.0,
    )
    target_residual = stamp1 + stamp2

    # Fit stamps using matching pursuit
    stamps, ncc = fitter.fit_stamps(target_residual, max_stamps=4)

    assert len(stamps) >= 2
    # AC-006 target is >= 0.85 (85%) normalized cross-correlation
    assert ncc >= 0.85, f"Expected NCC >= 0.85, got {ncc:.4f}"
