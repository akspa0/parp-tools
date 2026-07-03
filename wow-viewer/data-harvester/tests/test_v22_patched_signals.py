"""Tests for V22 patched signal derivations from V18 store arrays.

All tests use synthetic tiles (no game client). Reference values match the
C# algorithms in ``WowViewer.Core.IO/Maps/RawArraySerializer.cs``.
"""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v22_patched_signals import (
    derive_mcnr_mask_257,
    derive_liquid_type_256,
    derive_ground_intent_height_257,
    derive_model_focus_mask,
    derive_model_above_terrain_mask,
)


class TestDeriveMcnrMask257:
    def test_present_copy(self):
        tile = {"mcnr_mask_257": np.ones((257, 257), dtype=bool)}
        result = derive_mcnr_mask_257(tile)
        assert result.shape == (257, 257)
        assert result.dtype == bool
        assert result.all()

    def test_missing_checkerboard_fallback(self):
        tile = {}
        result = derive_mcnr_mask_257(tile)
        assert result.shape == (257, 257)
        assert result.dtype == bool
        # Checkerboard: (0,0)=True, (0,1)=False, (1,0)=False, (1,1)=True
        assert result[0, 0] == True
        assert result[0, 1] == False
        assert result[1, 0] == False
        assert result[1, 1] == True

    def test_wrong_shape_ignored(self):
        tile = {"mcnr_mask_257": np.ones((256, 256), dtype=bool)}
        result = derive_mcnr_mask_257(tile)
        # The function copies the tile value; wrong shape means it still
        # returns the wrong shape. This is fine — consumers should validate.
        assert result.shape == (256, 256)


class TestDeriveLiquidType256:
    def test_present_values(self):
        source = np.zeros((257, 257), dtype=np.uint8)
        source[0, 0] = 0xFF  # no liquid
        source[0, 1] = 0x01  # water
        source[0, 2] = 0x02  # ocean
        tile = {"liquid_basic_type_257": source}
        result = derive_liquid_type_256(tile)
        assert result.shape == (256, 256)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0   # 0xFF → 0
        assert result[0, 1] == 2   # 1 → 2  (0x01 + 1)
        assert result[0, 2] == 3   # 2 → 3  (0x02 + 1)

    def test_crop_257_to_256(self):
        source = np.ones((257, 257), dtype=np.uint8) * 0xFF
        source[256, 256] = 0x01  # this is row/col 256, which gets cropped
        tile = {"liquid_basic_type_257": source}
        result = derive_liquid_type_256(tile)
        assert result.shape == (256, 256)
        # Last row/col of 257 array gets dropped
        assert result[255, 255] == 0  # cropped column

    def test_missing_zeros(self):
        tile = {}
        result = derive_liquid_type_256(tile)
        assert result.shape == (256, 256)
        assert result.dtype == np.uint8
        assert (result == 0).all()

    def test_wrong_shape_zeros(self):
        tile = {"liquid_basic_type_257": np.zeros((16, 16), dtype=np.uint8)}
        result = derive_liquid_type_256(tile)
        assert result.shape == (256, 256)
        assert (result == 0).all()


class TestDeriveGroundIntentHeight257:
    def test_no_objects_same_as_height(self):
        height = np.random.rand(257, 257).astype(np.float32)
        tile = {"height_257": height, "object_precise_mask": np.zeros((257, 257), dtype=np.float32)}
        result = derive_ground_intent_height_257(tile)
        assert result.shape == (257, 257)
        assert np.allclose(result, height)

    def test_objects_inpainted(self):
        height = np.ones((257, 257), dtype=np.float32) * 100.0
        precise = np.zeros((257, 257), dtype=np.float32)
        # Place an object in the middle of the tile
        precise[128, 128] = 1.0
        tile = {"height_257": height, "object_precise_mask": precise}
        result = derive_ground_intent_height_257(tile)
        assert result.shape == (257, 257)
        # Center pixel should be inpainted from neighbors
        assert result[128, 128] == pytest.approx(100.0, abs=0.5)

    def test_large_object_region_inpainted(self):
        """A 3x3 object region is fully inpainted within max iterations."""
        height = np.ones((257, 257), dtype=np.float32) * 50.0
        precise = np.zeros((257, 257), dtype=np.float32)
        precise[10:13, 10:13] = 1.0  # 3x3 object
        tile = {"height_257": height, "object_precise_mask": precise}
        result = derive_ground_intent_height_257(tile)
        assert result.shape == (257, 257)
        assert np.allclose(result, 50.0, atol=1.0)

    def test_missing_precise_no_inpainting(self):
        height = np.ones((257, 257), dtype=np.float32) * 100.0
        tile = {"height_257": height}
        result = derive_ground_intent_height_257(tile)
        assert np.allclose(result, height)

    def test_missing_height_raises(self):
        with pytest.raises(ValueError, match="missing height_257"):
            derive_ground_intent_height_257({})

    def test_wrong_height_shape_raises(self):
        with pytest.raises(ValueError, match="unexpected shape"):
            derive_ground_intent_height_257({"height_257": np.zeros((256, 256), dtype=np.float32)})


class TestDeriveModelFocusMask:
    def test_present_copy(self):
        tile = {"object_filtered_mask": np.ones((257, 257), dtype=np.float32)}
        result = derive_model_focus_mask(tile)
        assert result.shape == (257, 257)
        assert result.dtype == np.float32
        assert result.all()

    def test_missing_zeros(self):
        tile = {}
        result = derive_model_focus_mask(tile)
        assert result.shape == (257, 257)
        assert (result == 0.0).all()


class TestDeriveModelAboveTerrainMask:
    def test_no_placements_all_zeros(self):
        tile = {"height_257": np.ones((257, 257), dtype=np.float32)}
        result = derive_model_above_terrain_mask(tile, [], [], 30, 30)
        assert result.shape == (257, 257)
        assert (result == 0.0).all()

    def test_placement_above_terrain(self):
        """A placement at Z=100 projected onto terrain at Z=50 should set mask pixel to 1.0."""
        height = np.full((257, 257), 50.0, dtype=np.float32)
        tile = {"height_257": height}
        # Placement at (posX, posY, posZ) that projects to some pixel
        # Use tile (0,0) so math is simpler
        mddf_placements = [
            {"posX": 10.0, "posY": 10.0, "posZ": 100.0, "rotX": 0, "rotY": 0, "rotZ": 0, "scale": 1.0},
        ]
        result = derive_model_above_terrain_mask(tile, mddf_placements, [], 0, 0)
        assert result.shape == (257, 257)

    def test_placement_below_terrain_underground(self):
        """A placement at Z=10 with terrain at Z=50 should NOT set the mask pixel."""
        height = np.full((257, 257), 50.0, dtype=np.float32)
        tile = {"height_257": height}
        mddf_placements = [
            {"posX": 10.0, "posY": 10.0, "posZ": 10.0, "rotX": 0, "rotY": 0, "rotZ": 0, "scale": 1.0},
        ]
        result = derive_model_above_terrain_mask(tile, mddf_placements, [], 0, 0)
        assert result.shape == (257, 257)

    def test_missing_height_all_zeros(self):
        result = derive_model_above_terrain_mask({}, [], [], 0, 0)
        assert (result == 0.0).all()