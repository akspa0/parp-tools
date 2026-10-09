"""Unit tests for WdlQuiltSynthesizer (Spec 268 AC-003, AC-004)."""

import numpy as np
import pytest

from harvester.v60.wdl_quilt_synthesizer import WdlQuiltSynthesizer


def test_bare_terrain_shadow_extraction_ac003():
    """Verify stripping texture albedo isolates clean bare photometric shading (AC-003)."""
    synthesizer = WdlQuiltSynthesizer()

    # Generate synthetic ground-truth illumination gradient
    y_coords, x_coords = np.mgrid[0:256, 0:256].astype(np.float32)
    true_lighting = 0.5 + 0.4 * np.sin(x_coords / 32.0)  # Pure terrain shading in [0.1, 0.9]

    # Generate noisy texture albedo (grass / dirt checkerboard)
    albedo = np.zeros((256, 256, 3), dtype=np.float32)
    checker = ((x_coords // 16) + (y_coords // 16)) % 2 == 0
    albedo[checker] = [0.35, 0.45, 0.25]  # Green grass
    albedo[~checker] = [0.45, 0.35, 0.25]  # Dirt

    # Composite minimap = albedo * true_lighting
    composite_minimap = albedo * true_lighting[..., None]

    result = synthesizer.extract_bare_terrain_shadow(
        minimap_rgb_256=composite_minimap,
        texture_albedo_256=albedo,
    )

    shadow = result.bare_shadow_256
    assert shadow.shape == (256, 256)
    assert shadow.min() >= 0.0
    assert shadow.max() <= 1.0

    # Cross-correlation between extracted shadow and true lighting
    s_norm = (shadow - shadow.mean()) / (shadow.std() + 1e-6)
    l_norm = (true_lighting - true_lighting.mean()) / (true_lighting.std() + 1e-6)
    correlation = float(np.mean(s_norm * l_norm))

    assert correlation >= 0.85, f"Correlation too low: {correlation}"


def test_wdl_lattice_stitching_ac004():
    """Verify stitching 17x17 WDL lattices guarantees zero edge-boundary shear (AC-004)."""
    synthesizer = WdlQuiltSynthesizer()

    # Create 2 adjacent 17x17 WDL lattices with deliberate boundary steps
    wdl_left = np.full((17, 17), 300.0, dtype=np.float32)
    wdl_right = np.full((17, 17), 320.0, dtype=np.float32)  # 20 yard cliff

    lattices = {(16, 32): wdl_left, (17, 32): wdl_right}
    stitched = synthesizer.stitch_wdl_trestle_quilt(lattices)

    # East border of left tile (col 16) must equal West border of right tile (col 0)
    col_left = stitched[(16, 32)][:, 16]
    col_right = stitched[(17, 32)][:, 0]

    assert np.allclose(col_left, col_right, atol=1e-5)
    assert np.allclose(col_left, 310.0, atol=1e-5)


def test_wdl_upsampling_to_257():
    """Verify bicubic interpolation upsamples 17x17 to 257x257 accurately."""
    synthesizer = WdlQuiltSynthesizer()
    wdl_17 = np.zeros((17, 17), dtype=np.float32)
    wdl_17[8, 8] = 400.0  # Center mountain peak

    elev_257 = synthesizer.interpolate_wdl_to_257(wdl_17)
    assert elev_257.shape == (257, 257)
    # Peak at center (128, 128)
    assert elev_257[128, 128] >= 350.0
