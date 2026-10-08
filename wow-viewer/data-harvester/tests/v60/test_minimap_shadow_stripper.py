"""Unit tests for minimap shadow stripper and multi-scale inpainting."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.minimap_shadow_stripper import (
    MinimapShadowStripper,
    compute_relative_high_frequency_energy,
    multiscale_laplacian_inpaint,
)


def test_multiscale_laplacian_inpaint_smooth_continuation():
    # Linear ramp gradient across a 64x64 patch: f(x, y) = x / 64
    x = np.linspace(0.2, 0.8, 64, dtype=np.float32)
    ramp = np.tile(x, (64, 1))

    # Center hole to inpaint (20x20 in the middle)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[22:42, 22:42] = 255

    corrupted = ramp.copy()
    corrupted[22:42, 22:42] = 0.0  # Zero out masked region

    inpainted = multiscale_laplacian_inpaint(corrupted, mask, num_scales=3, iterations_per_scale=50)

    # Inpainted pixels should match the linear ramp smoothly with low error
    err = np.abs(inpainted[22:42, 22:42] - ramp[22:42, 22:42])
    assert float(np.mean(err)) < 0.05
    assert float(np.max(err)) < 0.10


def test_shadow_stripper_energy_attenuation_ac004():
    """Verify >= 90% object high-frequency energy attenuation (AC-004)."""
    # Smooth baseline terrain shading
    h, w = 64, 64
    x = np.linspace(0.4, 0.6, w, dtype=np.float32)
    terrain = np.tile(x, (h, 1))

    # Add high-frequency checkerboard / sharp roof edge object in center
    object_region = np.zeros((h, w), dtype=np.float32)
    for y in range(20, 44):
        for x_idx in range(20, 44):
            if (x_idx + y) % 2 == 0:
                object_region[y, x_idx] = 0.5

    contaminated = terrain + object_region

    # Mask covering the object
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[18:46, 18:46] = 255

    stripper = MinimapShadowStripper(inpaint_scales=3, iterations=60)
    cleaned_shadow, attenuation = stripper.strip_and_inpaint(
        contaminated,
        mask,
        normalize_albedo=False,
    )

    # AC-004 requires >= 90% attenuation (0.90) of object high-frequency contrast
    assert attenuation >= 0.90, f"Expected >= 90% attenuation, got {attenuation * 100:.2f}%"
    assert cleaned_shadow.shape == (h, w)
