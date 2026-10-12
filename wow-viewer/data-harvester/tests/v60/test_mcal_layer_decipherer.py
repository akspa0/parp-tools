"""Unit tests for McalLayerDecipherer and dynamic layer stacking (Spec 268 AC-002)."""

import numpy as np
import pytest

from harvester.v60.mcal_layer_decipherer import (
    DEFAULT_TILESET_PALETTE,
    McalLayerDecipherer,
    TilesetTexture,
)


def test_albedo_demixing_synthetic_rgb():
    """Verify decomposing synthetic minimap blends into texture weights and bare illumination."""
    decipherer = McalLayerDecipherer()

    # Create a 256x256 image with grass (top half) and dirt (bottom half)
    # plus a gradient illumination ramp
    img = np.zeros((256, 256, 3), dtype=np.float32)

    grass_rgb = DEFAULT_TILESET_PALETTE[0].rgb_array
    dirt_rgb = DEFAULT_TILESET_PALETTE[1].rgb_array

    img[:128, :] = grass_rgb
    img[128:, :] = dirt_rgb

    # Apply lighting ramp
    ramp = np.linspace(0.6, 1.2, 256)[:, None, None]
    img_lit = np.clip(img * ramp, 0.0, 1.0)

    weights, recon_albedo, bare_illum = decipherer.demix_pixel_albedo(img_lit)

    assert weights.shape == (256, 256, len(DEFAULT_TILESET_PALETTE))
    assert recon_albedo.shape == (256, 256, 3)
    assert bare_illum.shape == (256, 256)

    # Top half should be predominantly grass (id 0)
    top_dominant = np.argmax(np.mean(weights[:128, :], axis=(0, 1)))
    assert top_dominant == 0

    # Bottom half should be predominantly dirt (id 1)
    bottom_dominant = np.argmax(np.mean(weights[128:, :], axis=(0, 1)))
    assert bottom_dominant == 1

    # Bare illumination should follow the ramp
    assert np.mean(bare_illum[-32:, :]) > np.mean(bare_illum[:32, :])


def test_chunk_layer_budget_and_mcal_ac002():
    """Verify every MCNK chunk obeys the <= 4 active layers constraint and generates 64x64 MCAL splats."""
    decipherer = McalLayerDecipherer()

    # Create a multi-texture noisy minimap
    np.random.seed(42)
    synthetic_minimap = np.random.uniform(0.2, 0.8, (256, 256, 3)).astype(np.float32)

    weights, _, _ = decipherer.demix_pixel_albedo(synthetic_minimap, top_k=6)
    chunks = decipherer.build_chunk_layers(weights)

    # Must contain exactly 16x16 = 256 chunks
    assert len(chunks) == 256

    for chunk in chunks:
        # Hard constraint: at most 4 active textures per chunk (L0 to L3)
        assert 1 <= len(chunk.active_texture_ids) <= 4, f"Chunk exceeded layer budget: {len(chunk.active_texture_ids)}"

        # MCAL layers should equal num_textures - 1 (since L0 base layer has no explicit alpha)
        expected_mcal_count = len(chunk.active_texture_ids) - 1
        assert len(chunk.mcal_alpha_layers) == expected_mcal_count

        for mcal_alpha in chunk.mcal_alpha_layers:
            assert mcal_alpha.shape == (64, 64)
            assert mcal_alpha.dtype == np.uint8
            assert mcal_alpha.min() >= 0
            assert mcal_alpha.max() <= 255


def test_mtex_manifest_generation():
    """Verify unique MTEX texture list generated from active chunk layers."""
    decipherer = McalLayerDecipherer()
    weights = np.zeros((256, 256, len(DEFAULT_TILESET_PALETTE)), dtype=np.float32)

    # Set grass (0) and cobblestone (3) active
    weights[..., 0] = 0.7
    weights[..., 3] = 0.3

    chunks = decipherer.build_chunk_layers(weights)
    mtex = decipherer.generate_mtex_manifest(chunks)

    assert len(mtex) >= 2
    assert "Tileset/Base/GreenGrass.blp" in mtex
    assert "Tileset/Base/CobblestoneRoad.blp" in mtex


def test_palette_color_matching():
    """Verify closest palette texture matching for RGB color vectors."""
    decipherer = McalLayerDecipherer()

    # Query with exact grass color
    grass_rgb = DEFAULT_TILESET_PALETTE[0].rgb_array
    tid, name, dist = decipherer.match_color_to_palette(grass_rgb)
    assert tid == 0
    assert name == "Tileset/Base/GreenGrass.blp"
    assert dist == pytest.approx(0.0, abs=1e-5)

    # Query with slight variation of mountain rock (id 2)
    rock_query = np.array([0.47, 0.49, 0.48], dtype=np.float32)
    tid, name, dist = decipherer.match_color_to_palette(rock_query)
    assert tid == 2
    assert name == "Tileset/Base/MountainRock.blp"


def test_d1_neural_layer_prediction():
    """Verify Model D1 neural inference recovers base and overlay layers from minimap."""
    from pathlib import Path
    ckpt_path = Path("checkpoints/d1_best.pt")
    if not ckpt_path.is_file():
        pytest.skip("checkpoints/d1_best.pt not present")

    decipherer = McalLayerDecipherer()
    # Create test image
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    weights, albedo, illum, meta = decipherer.predict_d1_neural_layers(img, checkpoint_path=ckpt_path)

    assert meta["model"] == "D1UNet"
    assert "layer_1_texture_id" in meta
    assert "layer_2_texture_id" in meta
    assert weights.shape == (256, 256, len(DEFAULT_TILESET_PALETTE))
    assert albedo.shape == (256, 256, 3)
    assert illum.shape == (256, 256)

