"""Unit tests for MCCV terrain shadow comparator and synthesis bridge (Spec 262)."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.mccv_shadow_comparator import (
    MccvComparisonMetrics,
    compare_residual_to_mccv,
    get_chunk_vertex_uvs,
    rasterize_mccv_tile_to_grid,
    synthesize_mccv_from_residual,
)


def test_get_chunk_vertex_uvs():
    uvs = get_chunk_vertex_uvs()
    assert uvs.shape == (145, 2)
    # Check boundaries [0, 1]
    assert np.all(uvs >= 0.0)
    assert np.all(uvs <= 1.0)
    # Vertex 0 is (0, 0), vertex 80 is (1, 1)
    np.testing.assert_allclose(uvs[0], [0.0, 0.0])
    np.testing.assert_allclose(uvs[80], [1.0, 1.0])
    # Vertex 81 is inner grid (0.0625, 0.0625)
    np.testing.assert_allclose(uvs[81], [0.0625, 0.0625])


def test_synthesize_and_rasterize_roundtrip():
    # Synthetic shadow ramp across 256x256
    x = np.linspace(0.2, 0.8, 256, dtype=np.float32)
    y = np.linspace(0.2, 0.8, 256, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    orig_shadow = (xx + yy) * 0.5

    # 1. Synthesize 145-vertex MCCV chunks
    mccv_dict = synthesize_mccv_from_residual(orig_shadow)
    assert len(mccv_dict) == 256  # 16x16 chunks

    # Check that chunk 0,0 has 580 bytes (145 * 4)
    chunk_00 = mccv_dict[(0, 0)]
    assert len(chunk_00) == 580
    assert chunk_00.dtype == np.uint8

    # 2. Rasterize back to 256x256 grid
    raster = rasterize_mccv_tile_to_grid(mccv_dict, tile_res=256)
    assert raster.shape == (256, 256)

    # 3. Compare correlation
    metrics = compare_residual_to_mccv(orig_shadow, raster)
    assert metrics.normalized_cross_correlation > 0.90
    assert metrics.correlation_passed is True


def test_compare_residual_to_mccv_divergent():
    s1 = np.ones((256, 256), dtype=np.float32) * 0.5
    s2 = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)

    metrics = compare_residual_to_mccv(s1, s2)
    assert metrics.correlation_passed is False
