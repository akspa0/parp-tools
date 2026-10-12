"""Unit tests for QuiltFractalRefiner (Spec 268 AC-005, AC-006)."""

import numpy as np
import pytest

from harvester.v60.quilt_fractal_refiner import (
    INCHES_PER_YARD,
    QuiltFractalRefiner,
    TerrainBrushScar,
)


def test_inches_resolution_sculpting_and_stamps_ac005():
    """Verify inches-scale canvas simulation and 3D fractal brush fitting (AC-005)."""
    refiner = QuiltFractalRefiner(subcell_factor=6)

    # Base elevation grid
    base_h = np.full((257, 257), 150.0, dtype=np.float32)

    # Residual shadow containing a deliberate ridge feature
    y_coords, x_coords = np.mgrid[0:256, 0:256].astype(np.float32)
    shadow = 0.5 + 0.3 * np.exp(-((x_coords - 128.0) ** 2 + (y_coords - 128.0) ** 2) / (32.0 ** 2))

    # Synthetic alpha map
    alpha_map = np.zeros((256, 256), dtype=np.float32)

    inches_canvas, stamps, scars = refiner.sculpt_inches_canvas(
        base_elevation_257=base_h,
        residual_shadow_256=shadow,
        alpha_maps_256=[alpha_map],
        max_brush_stamps=8,
    )

    # Grid dimension: 256 * 6 = 1536
    assert inches_canvas.shape == (1536, 1536)
    assert inches_canvas.dtype == np.float32

    # Verify at least one stamp was fitted
    assert len(stamps) >= 1

    # Verify scars detection evaluated broken relationships
    assert len(scars) == len(stamps)
    broken_count = sum(1 for s in scars if s.is_broken_relationship)
    # Since alpha_map is empty while height stamp was fitted, relationship is broken (fossil scar)
    assert broken_count >= 1


def test_deterministic_downsample_to_mcvt_ac006():
    """Verify anti-aliased downsampling from inches canvas to 145-vertex MCVT chunks (AC-006)."""
    refiner = QuiltFractalRefiner(subcell_factor=4)

    # High-res canvas: 256 * 4 = 1024
    y_coords, x_coords = np.mgrid[0:1024, 0:1024].astype(np.float32)
    hi_res_canvas = 200.0 + 50.0 * np.sin(x_coords / 100.0)

    mcvt, mcnr = refiner.downsample_to_mcvt_chunks(hi_res_canvas)

    # Exact shapes required by ADT MCNK specification
    assert mcvt.shape == (16, 16, 145)
    assert mcnr.shape == (16, 16, 145, 3)

    # Height values should remain bounded in yards
    assert mcvt.min() >= 140.0
    assert mcvt.max() <= 260.0

    # Normal vectors should all point upwards (+Z component > 0.0)
    assert np.all(mcnr[..., 2] > 0.0)

    # Normal vectors must be normalized unit vectors
    lens = np.linalg.norm(mcnr, axis=-1)
    assert np.allclose(lens, 1.0, atol=1e-3)
