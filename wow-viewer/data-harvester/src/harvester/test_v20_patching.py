import numpy as np
import pytest
from harvester.patch_utils import _flags_to_liquid_type_16, inpaint_tile_heightmap, process_single_tile
# We can import process_single_tile directly from scripts by adjusting paths,
# but since it's in a script, let's copy/test the core functions here or test the script functions.


def test_flags_to_liquid_type_16():
    flags = np.zeros((16, 16), dtype=np.int32)
    flags[2, 2] = 0x04  # water
    flags[4, 4] = 0x08  # ocean
    flags[6, 6] = 0x10  # magma
    flags[8, 8] = 0x20  # slime

    liq_type = _flags_to_liquid_type_16(flags)
    assert liq_type[2, 2] == 1
    assert liq_type[4, 4] == 2
    assert liq_type[6, 6] == 3
    assert liq_type[8, 8] == 4
    assert np.all(liq_type[flags == 0] == 0)


def test_inpaint_tile_heightmap():
    # Create a smooth inclined plane heightmap
    h, w = 257, 257
    y, x = np.mgrid[0:h, 0:w]
    height = (y * 0.1 + x * 0.2).astype(np.float32)

    # Place a large flat building in the middle
    mask = np.zeros((h, w), dtype=np.float32)
    mask[100:150, 100:150] = 1.0

    # Corrupt the height map under the building to simulate a roof
    height_corrupted = height.copy()
    height_corrupted[100:150, 100:150] = 50.0  # Spike

    # Run inpainter
    inpainted = inpaint_tile_heightmap(height_corrupted, mask)

    # The inpainting should recover the smooth inclined plane with low error
    # since griddata cubic/linear interpolation on a plane is exact.
    diff = np.abs(inpainted[100:150, 100:150] - height[100:150, 100:150])
    assert np.max(diff) < 1.0  # very small error on a linear plane
    assert not np.any(np.isnan(inpainted))


def test_inpaint_nan_prevention():
    # If the boundary has all object values or edge cases, ensure it falls back gracefully and does not produce NaNs.
    h, w = 257, 257
    height = np.ones((h, w), dtype=np.float32) * 10.0
    mask = np.ones((h, w), dtype=np.float32)  # fully masked
    
    # Fully masked should return original height copy cleanly without NaN
    inpainted = inpaint_tile_heightmap(height, mask)
    assert not np.any(np.isnan(inpainted))
    assert np.all(inpainted == 10.0)


def test_process_single_tile():
    height = np.random.randn(257, 257).astype(np.float32)
    obj_mask = np.zeros((257, 257), dtype=np.float32)
    obj_mask[10, 10] = 1.0  # small spike
    liquid_mask = np.random.rand(256, 256).astype(np.float32)
    liquid_mask[liquid_mask < 0.5] = 0.0
    mcnk_flags = np.zeros((16, 16), dtype=np.int32)
    mcnk_flags[0, 0] = 0x08  # ocean

    args = (42, height, obj_mask, liquid_mask, mcnk_flags)
    idx, liq_type_256, ground_h = process_single_tile(args)

    assert idx == 42
    assert liq_type_256.shape == (256, 256)
    assert ground_h.shape == (257, 257)
    assert not np.any(np.isnan(liq_type_256))
    assert not np.any(np.isnan(ground_h))
    
    # Liquid type at (0, 0) chunk (which is pixels 0-15) should be 2 (ocean) if liquid_mask is set there
    if liquid_mask[5, 5] > 0.1:
        assert liq_type_256[5, 5] == 2
