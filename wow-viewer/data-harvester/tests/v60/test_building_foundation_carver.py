"""Tests for BuildingFoundationCarver (Spec 264 Phase 3)."""

import numpy as np
import pytest

from harvester.v60.building_foundation_carver import (
    BuildingFoundationCarver,
    FoundationCarveResult,
)
from harvester.v60.development_ground_truth import WmoPlacement


def test_carver_levels_foundation_footprint():
    """Verify that sloped terrain becomes level inside the building footprint."""
    carver = BuildingFoundationCarver(blend_margin=3)

    # 45-degree slope terrain: Z = X * 0.5 + Y * 0.5
    y_coords, x_coords = np.mgrid[0:257, 0:257]
    terrain = (x_coords * 0.2 + y_coords * 0.2).astype(np.float32)

    # Building placed at pixel box [100, 100, 140, 140]
    wmo = WmoPlacement(
        name="TEST_INN.WMO",
        unique_id=101,
        pos=(0.0, 0.0, 25.0),
        rot=(0.0, 0.0, 0.0),
        bounds_min=(-10.0, -10.0, 0.0),
        bounds_max=(10.0, 10.0, 20.0),
        pixel_box=(100, 100, 140, 140),
    )

    result = carver.carve_foundations(terrain, [wmo])

    assert result.carved_height_257.shape == (257, 257)
    assert np.any(result.foundation_mask)

    # Inside footprint, elevation should be level at 25.0
    footprint = result.foundation_mask
    footprint_heights = result.carved_height_257[footprint]

    slope_variance = float(np.var(footprint_heights))
    assert slope_variance <= 1e-4, f"Foundation must be flat/level, got variance {slope_variance}"
    assert np.allclose(footprint_heights, 25.0, atol=1e-3)


def test_carver_smooth_perimeter_transition():
    """Verify that transition between foundation and natural terrain has no steep spikes."""
    carver = BuildingFoundationCarver(blend_margin=4)

    # Steep terrain
    terrain = np.zeros((257, 257), dtype=np.float32)
    terrain[150:, :] = 40.0

    wmo = WmoPlacement(
        name="TEST_BARRACKS.WMO",
        unique_id=102,
        pos=(0.0, 0.0, 15.0),
        rot=(0.0, 0.0, 0.0),
        bounds_min=(-15.0, -15.0, 0.0),
        bounds_max=(15.0, 15.0, 30.0),
        pixel_box=(50, 50, 90, 90),
    )

    result = carver.carve_foundations(terrain, [wmo])

    # Check finite and continuous
    assert np.all(np.isfinite(result.carved_height_257))

    # Gradient should not have extreme spikes
    gy, gx = np.gradient(result.carved_height_257)
    max_grad = float(np.max(np.sqrt(gx**2 + gy**2)))
    assert max_grad < 100.0, f"Smoothstep blending should avoid cliffs, max grad: {max_grad}"


def test_carver_empty_placements():
    """Verify carver is a no-op when no placements are provided."""
    carver = BuildingFoundationCarver()
    terrain = np.ones((257, 257), dtype=np.float32) * 42.0

    result = carver.carve_foundations(terrain, [])

    assert np.array_equal(result.carved_height_257, terrain)
    assert not np.any(result.foundation_mask)
    assert len(result.plateau_elevations) == 0
