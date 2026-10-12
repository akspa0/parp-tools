"""Tests for Terrain Feature Synthesizer (Spec 264 / Spec 266 / Spec 267)."""

import numpy as np
import pytest

from harvester.v60.terrain_feature_synthesizer import (
    compute_surface_normals,
    synthesize_multiband_terrain,
)


def test_synthesizer_object_mask_levels_to_median_no_towers():
    """Verify that masked object footprints are flattened to the median surrounding terrain height, preventing towers."""
    # Create sloped terrain: base Z = 50.0 + linear gradient
    y, x = np.mgrid[0:257, 0:257]
    macro_wdl = (50.0 + x * 0.1 + y * 0.1).astype(np.float32)

    # Synthetic Poisson heightfield with a huge artificial spike (tower) inside the object footprint
    sfs = np.zeros((256, 256), dtype=np.float32)
    sfs[100:120, 100:120] = 50.0  # Simulated 50-yard tower from bright building roof

    ridge_mask = np.zeros((256, 256), dtype=np.uint8)
    ridge_mask[100:120, 100:120] = 255  # Simulated ridge outline

    neural_residual = np.zeros((257, 257), dtype=np.float32)
    neural_residual[100:120, 100:120] = 30.0  # Simulated neural spike

    # Object mask marking the building footprint
    object_mask = np.zeros((256, 256), dtype=bool)
    object_mask[100:120, 100:120] = True

    fused_h, normals, metrics = synthesize_multiband_terrain(
        macro_wdl_257=macro_wdl,
        integrated_sfs_256=sfs,
        ridge_mask_256=ridge_mask,
        neural_residual_257=neural_residual,
        object_mask_256=object_mask,
    )

    # Footprint should NOT have the 50-yard or 30-yard tower
    footprint_heights = fused_h[102:118, 102:118]
    # Local macro elevation around (110, 110) is 50 + 11 + 11 = ~72 yards
    assert np.all(footprint_heights < 80.0), f"Tower was not leveled! Max footprint height: {np.max(footprint_heights)}"
    assert np.all(footprint_heights > 65.0), f"Footprint fell below surrounding terrain: {np.min(footprint_heights)}"

    # Footprint should be level/flat (low variance)
    assert np.var(footprint_heights) < 0.1, f"Footprint must be a level plateau, got variance: {np.var(footprint_heights)}"


def test_synthesizer_alpha_mask_attenuates_texture_relief():
    """Verify that texture alpha splats attenuate high-pass relief and suppress false ridges."""
    macro_wdl = np.full((257, 257), 100.0, dtype=np.float32)

    # High frequency noise simulating rock texture crevices
    np.random.seed(42)
    noise_sfs = np.random.randn(256, 256).astype(np.float32) * 5.0
    ridge_mask = (noise_sfs > 2.0).astype(np.uint8) * 255

    # Alpha mask covering the right half
    alpha_mask = np.zeros((256, 256), dtype=np.float32)
    alpha_mask[:, 128:] = 1.0  # 100% painted texture

    fused_with_alpha, _, _ = synthesize_multiband_terrain(
        macro_wdl_257=macro_wdl,
        integrated_sfs_256=noise_sfs,
        ridge_mask_256=ridge_mask,
        alpha_mask_256=alpha_mask,
    )

    fused_without_alpha, _, _ = synthesize_multiband_terrain(
        macro_wdl_257=macro_wdl,
        integrated_sfs_256=noise_sfs,
        ridge_mask_256=ridge_mask,
        alpha_mask_256=None,
    )

    # Right half (with alpha suppression) should have significantly lower relief variance than without
    var_with = float(np.var(fused_with_alpha[:, 140:240]))
    var_without = float(np.var(fused_without_alpha[:, 140:240]))
    assert var_with < var_without * 0.4, f"Alpha mask must attenuate texture relief: {var_with} vs {var_without}"
