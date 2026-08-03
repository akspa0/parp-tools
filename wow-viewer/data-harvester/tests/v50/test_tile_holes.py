"""Hole masks decode to the right nesting: tile -> chunk (16x16) -> quad (4x4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvester.v50.tile_holes import (
    QUADS_PER_CHUNK,
    bitmask_census,
    chunk_relief,
    hidden_chunk_records,
    hole_pixel_mask,
    load_hole_masks,
    quad_count,
    quad_grid,
    render_hidden_tile,
    tile_hole_metrics,
)


def test_quad_decoding_is_bit_exact() -> None:
    assert quad_count(0x0000) == 0
    assert quad_count(0xFFFF) == QUADS_PER_CHUNK
    assert quad_count(0x0001) == 1
    assert quad_count(0x8001) == 2

    # Bit i -> row i//4, col i%4. Bit 0 is top-left, bit 15 is bottom-right.
    assert quad_grid(0x0001)[0, 0] and quad_grid(0x0001).sum() == 1
    assert quad_grid(0x8000)[3, 3] and quad_grid(0x8000).sum() == 1
    assert quad_grid(0xFFFF).all()
    # A whole top row is bits 0-3.
    assert np.array_equal(quad_grid(0x000F)[0], np.ones(4, dtype=bool))
    assert not quad_grid(0x000F)[1:].any()


def test_tile_metrics_separate_full_from_partial_cutouts() -> None:
    grid = np.zeros((16, 16), dtype=np.uint16)
    grid[0, 0] = 0xFFFF   # fully hidden chunk
    grid[5, 7] = 0x000F   # a shaped cutout: one row of quads
    metrics = tile_hole_metrics(grid)

    assert metrics["hole_chunk_count"] == 2
    assert metrics["hole_quad_count"] == 16 + 4
    assert metrics["fully_holed_chunk_count"] == 1
    assert metrics["partial_holed_chunk_count"] == 1
    assert metrics["hole_quad_fraction"] == pytest.approx(20 / 4096)
    assert tile_hole_metrics(np.zeros((16, 16), dtype=np.uint16))["hole_chunk_count"] == 0


def test_pixel_mask_preserves_the_quad_shape() -> None:
    """A partially-holed chunk must render as the CUTOUT SHAPE, not a solid square."""
    grid = np.zeros((16, 16), dtype=np.uint16)
    grid[0, 0] = 0x000F  # top row of quads only
    pixels = hole_pixel_mask(grid, size=257)

    assert pixels[0:4, 0:16].all()      # top quad row of chunk 0 spans 4px tall, 16px wide
    assert not pixels[4:16, 0:16].any() # the rest of that chunk stays visible
    assert not pixels[:, 16:].any()     # no bleed into the neighbouring chunk

    # A fully-holed chunk covers its whole 16x16 pixel footprint.
    full = np.zeros((16, 16), dtype=np.uint16); full[1, 1] = 0xFFFF
    assert hole_pixel_mask(full, size=257)[16:32, 16:32].all()
    assert not hole_pixel_mask(np.zeros((16, 16), dtype=np.uint16), size=257).any()


def test_hidden_chunk_records_carry_what_is_under_the_hole() -> None:
    height = np.zeros((257, 257), dtype=np.float32)
    height[16:33, 16:33] = np.linspace(0, 50, 17 * 17, dtype=np.float32).reshape(17, 17)
    grid = np.zeros((16, 16), dtype=np.uint16)
    grid[1, 1] = 0x00FF

    rows = hidden_chunk_records("Azeroth", 31, 28, grid, height)
    assert len(rows) == 1
    row = rows[0]
    assert row["tile_key"] == "Azeroth_31_28"
    assert (row["chunk_x"], row["chunk_y"]) == (1, 1)
    assert row["hole_mask_hex"] == "0x00FF" and row["hole_quads"] == 8
    assert row["fully_holed"] is False
    assert row["height_range"] == pytest.approx(50.0)   # real relief under the hole
    assert row["height_levels"] > 100


def test_chunk_relief_reads_the_right_block() -> None:
    height = np.zeros((257, 257), dtype=np.float32)
    height[32:49, 48:65] = 9.0
    assert chunk_relief(height, chunk_x=3, chunk_y=2)["height_max"] == 9.0
    assert chunk_relief(height, chunk_x=0, chunk_y=0)["height_max"] == 0.0


def test_bitmask_census_detects_structure_versus_freehand() -> None:
    """If a few patterns dominate, the masks came from a brush/template, not freehand painting."""
    structured = [{"hole_mask": 0xFFFF} for _ in range(90)] + [{"hole_mask": 0x000F} for _ in range(10)]
    census = bitmask_census(structured)
    assert census["holed_chunks"] == 100
    assert census["distinct_masks"] == 2
    assert census["coverage_of_top_8"] == 1.0
    assert census["top_masks"][0]["hex"] == "0xFFFF"
    assert census["top_masks"][0]["grid"] == [[1] * 4] * 4

    spread = [{"hole_mask": v} for v in range(1, 101)]
    assert bitmask_census(spread)["coverage_of_top_8"] == pytest.approx(0.08)


def test_load_rejects_a_non_uint16_export(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hole_field": "bool_per_chunk", "maps": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a uint16 mask export"):
        load_hole_masks(bad)

    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "hole_field": "mcnk_holes_uint16_row_major_yx",
        "maps": {"Azeroth": [{"x": 1, "y": 2, "holes": [0] * 256}]},
    }), encoding="utf-8")
    loaded = load_hole_masks(good)
    assert loaded["Azeroth"][(1, 2)].shape == (16, 16)

    short = tmp_path / "short.json"
    short.write_text(json.dumps({
        "hole_field": "mcnk_holes_uint16_row_major_yx",
        "maps": {"Azeroth": [{"x": 0, "y": 0, "holes": [0] * 12}]},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 256 chunk masks"):
        load_hole_masks(short)


def test_hidden_tile_sheet_is_written(tmp_path: Path) -> None:
    from PIL import Image

    rng = np.random.default_rng(0)
    height = (rng.random((257, 257)) * 60).astype(np.float32)
    grid = np.zeros((16, 16), dtype=np.uint16)
    grid[2, 3] = 0xFFFF
    render_hidden_tile(height_257=height, mask=grid, minimap=None,
                       title="Azeroth_31_28", output=tmp_path / "t.png")
    with Image.open(tmp_path / "t.png") as image:
        assert image.width == 4 * 257
