"""Unit tests for MCCV terrain shadow comparator and synthesis bridge (Spec 262)."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.mccv_shadow_comparator import (
    MccvComparisonMetrics,
    compare_residual_to_mccv,
    get_chunk_vertex_uvs,
    rasterize_mccv_tile_to_grid,
    read_mccv_from_adt,
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


def test_read_mccv_from_adt_synthetic():
    import struct

    # Build synthetic MCNK with MCCV
    mccv_payload = np.full((145, 4), 127, dtype=np.uint8).tobytes()
    mccv_chunk = b"MCCV" + struct.pack("<I", len(mccv_payload)) + mccv_payload

    mcnk_header = bytearray(128)
    struct.pack_into("<I", mcnk_header, 4, 3)  # IndexX = 3
    struct.pack_into("<I", mcnk_header, 8, 5)  # IndexY = 5
    struct.pack_into("<I", mcnk_header, 0x74, 128)  # ofsMccv = 128

    mcnk_payload = bytes(mcnk_header) + mccv_chunk
    mcnk_chunk = b"MCNK" + struct.pack("<I", len(mcnk_payload)) + mcnk_payload

    # Build MCIN
    mcin_entries = bytearray(256 * 16)
    mcin_chunk_len = 8 + (256 * 16)
    struct.pack_into("<I", mcin_entries, 0, mcin_chunk_len)  # Entry 0 points right after MCIN
    mcin_chunk = b"MCIN" + struct.pack("<I", len(mcin_entries)) + bytes(mcin_entries)

    adt_data = mcin_chunk + mcnk_chunk

    chunks = read_mccv_from_adt(adt_data)
    assert (3, 5) in chunks
    assert chunks[(3, 5)].shape == (145, 4)
    assert np.all(chunks[(3, 5)] == 127)
