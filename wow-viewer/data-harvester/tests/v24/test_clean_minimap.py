"""Minimap cleaner tests (FR-010): no-object, all-object, partial-object,
no_object_minimap preference."""

import numpy as np
import pytest

from harvester.v24.clean_minimap import clean_minimap, object_mask_256


def _minimap() -> np.ndarray:
    rng = np.random.default_rng(94)
    return rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)


@pytest.mark.v24
def test_no_object_identity():
    minimap = _minimap()
    mask = np.zeros((257, 257), dtype=np.float32)
    cleaned, meta = clean_minimap(minimap, mask)
    np.testing.assert_allclose(cleaned, minimap.astype(np.float32) / 255.0)
    assert meta["source"] == "identity"
    assert not meta["cleaned_minimap_unavailable"]


@pytest.mark.v24
def test_all_object_global_mean():
    minimap = _minimap()
    mask = np.ones((257, 257), dtype=np.float32)
    cleaned, meta = clean_minimap(minimap, mask)
    expected = (minimap.astype(np.float32) / 255.0).reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(cleaned[0, 0], expected, atol=1e-5)
    np.testing.assert_allclose(cleaned[128, 200], expected, atol=1e-5)
    assert meta["cleaned_minimap_unavailable"]


@pytest.mark.v24
def test_partial_object_replaced():
    minimap = np.full((256, 256, 3), 100, dtype=np.uint8)
    minimap[100:120, 100:120] = 255  # bright "roof" block
    mask = np.zeros((257, 257), dtype=np.float32)
    mask[100:121, 100:121] = 1.0
    cleaned, meta = clean_minimap(minimap, mask)
    # Roof pixels are replaced by their terrain surroundings...
    assert cleaned[110, 110, 0] == pytest.approx(100 / 255.0, abs=1e-4)
    # ...and non-object pixels are untouched.
    assert cleaned[10, 10, 0] == pytest.approx(100 / 255.0, abs=1e-6)
    assert meta["source"] == "median_fill"


@pytest.mark.v24
def test_prefers_no_object_minimap():
    minimap = _minimap()
    mask = np.ones((257, 257), dtype=np.float32)
    rendered = np.full((256, 256, 3), 42, dtype=np.uint8)
    cleaned, meta = clean_minimap(minimap, mask, rendered)
    np.testing.assert_allclose(cleaned, 42 / 255.0, atol=1e-5)
    assert meta["source"] == "no_object_minimap"


@pytest.mark.v24
def test_object_mask_downsample_corners():
    mask = np.zeros((257, 257), dtype=np.float32)
    mask[10, 10] = 1.0
    m256 = object_mask_256(mask)
    assert m256.shape == (256, 256)
    # A single masked corner marks its four adjacent cells.
    assert m256[9, 9] and m256[9, 10] and m256[10, 9] and m256[10, 10]
    assert not m256[12, 12]
