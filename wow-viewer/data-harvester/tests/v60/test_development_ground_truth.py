"""Tests for DevelopmentGroundTruthExtractor (Spec 264 Phase 1)."""

from pathlib import Path
import numpy as np
import pytest

from harvester.v60.development_ground_truth import (
    DevelopmentGroundTruthExtractor,
    DevelopmentTileData,
    M2Placement,
    WmoPlacement,
)

MAPS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_data" / "original_development" / "World" / "Maps" / "development"
TEXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_data" / "original_development" / "World" / "Textures" / "Minimap"


def test_extractor_scanned_tiles():
    """Verify scanner finds authentic sculpted tiles in original_development."""
    if not MAPS_DIR.is_dir():
        pytest.skip(f"Original development maps directory not present: {MAPS_DIR}")

    extractor = DevelopmentGroundTruthExtractor(MAPS_DIR, TEXTURES_DIR)
    sculpted = extractor.scan_all_sculpted_tiles()

    assert len(sculpted) >= 100, f"Expected >= 100 sculpted tiles, found {len(sculpted)}"
    tile_coords = set(sculpted)
    assert (16, 33) in tile_coords, "Tile (16, 33) should be present and sculpted"
    assert (15, 34) in tile_coords, "Tile (15, 34) should be present and sculpted"


def test_extract_height_257_relief():
    """Verify height_257 extraction yields valid non-flat elevation in yards."""
    adt_16_33 = MAPS_DIR / "development_16_33.adt"
    if not adt_16_33.is_file():
        pytest.skip("development_16_33.adt not found")

    extractor = DevelopmentGroundTruthExtractor(MAPS_DIR, TEXTURES_DIR)
    h257, is_sculpted = extractor.extract_height_257(adt_16_33)

    assert is_sculpted is True
    assert h257.shape == (257, 257)
    assert h257.dtype == np.float32

    relief = float(np.max(h257) - np.min(h257))
    assert relief >= 50.0, f"Expected significant vertical relief, got {relief:.2f} yards"


def test_extract_wmo_and_m2_objects():
    """Verify object parsing extracts real WMOs and M2s with bounding boxes."""
    obj_16_38 = MAPS_DIR / "development_16_38_obj0.adt"
    if not obj_16_38.is_file():
        pytest.skip("development_16_38_obj0.adt not found")

    extractor = DevelopmentGroundTruthExtractor(MAPS_DIR, TEXTURES_DIR)
    m2s, wmos, bmask = extractor.extract_objects(obj_16_38, 16, 38)

    assert len(wmos) >= 3, f"Expected >= 3 WMOs (Goldshire Inn, etc), got {len(wmos)}"
    wmo_names = [w.name.upper() for w in wmos]
    assert any("GOLDSHIRE" in n or "FARM" in n for n in wmo_names)

    assert bmask.shape == (256, 256)
    assert bmask.dtype == bool
    assert np.any(bmask), "Building mask should contain rasterized building footprints"


def test_load_tile_full_stack():
    """Verify load_tile packages height, minimap RGB, and objects coherently."""
    adt_16_35 = MAPS_DIR / "development_16_35.adt"
    if not adt_16_35.is_file():
        pytest.skip("development_16_35.adt not found")

    extractor = DevelopmentGroundTruthExtractor(MAPS_DIR, TEXTURES_DIR)
    tile_data = extractor.load_tile(16, 35)

    assert tile_data is not None
    assert tile_data.height_257.shape == (257, 257)
    assert tile_data.minimap_rgb.shape == (256, 256, 3)
    assert np.min(tile_data.minimap_rgb) >= 0.0
    assert np.max(tile_data.minimap_rgb) <= 1.0
    assert len(tile_data.wmo_placements) >= 1
    assert any("STORMWINDHARBOR" in w.name.upper() for w in tile_data.wmo_placements)


def test_extract_alpha_splats_tex0():
    """Verify that _tex0.adt alpha masks decode to valid 256x256 composite splat maps."""
    tex_16_33 = MAPS_DIR / "development_16_33_tex0.adt"
    if not tex_16_33.is_file():
        pytest.skip("development_16_33_tex0.adt not found")

    extractor = DevelopmentGroundTruthExtractor(MAPS_DIR, TEXTURES_DIR)
    alpha_mask = extractor.extract_alpha_splats(tex_16_33)

    assert alpha_mask is not None
    assert alpha_mask.shape == (256, 256)
    assert alpha_mask.dtype == np.float32
    assert float(np.min(alpha_mask)) >= 0.0
    assert float(np.max(alpha_mask)) <= 1.0
    assert np.any(alpha_mask > 0.0), "Tile 16_33 must have active texture splat layers"

    # Also verify load_tile loads alpha_mask automatically
    tile = extractor.load_tile(16, 33)
    assert tile is not None
    assert tile.alpha_mask is not None
    assert tile.alpha_mask.shape == (256, 256)


def test_extract_pm4_objects_and_fallback():
    """Verify that PM4 collision geometry extracts building footprints on tiles without _obj0.adt."""
    pm4_16_33 = MAPS_DIR / "development_16_33.pm4"
    if not pm4_16_33.is_file():
        pytest.skip("development_16_33.pm4 not found")

    extractor = DevelopmentGroundTruthExtractor(MAPS_DIR, TEXTURES_DIR)
    placements, bmask = extractor.extract_pm4_objects(pm4_16_33, 16, 33)

    assert len(placements) >= 10, f"Expected >= 10 PM4 collision structures, got {len(placements)}"
    assert bmask.shape == (256, 256)
    assert bmask.dtype == bool
    assert np.any(bmask), "PM4 collision structures must generate building footprints"

    first_obj = placements[0]
    assert first_obj.name.startswith("PM4_Structure_")
    assert first_obj.bounds_min[2] >= 90.0, f"Expected physical Z height >= 90 yds, got {first_obj.bounds_min[2]}"


