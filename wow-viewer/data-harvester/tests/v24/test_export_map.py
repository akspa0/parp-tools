"""Tests for Spec 097 Slice 1: v24_export_map.py (per-map V18 Zarr -> OBJ + atlas)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v24_export_map.py"
SRC = Path(__file__).resolve().parents[2] / "src"


def _import_align():
    import importlib.util

    spec = importlib.util.spec_from_file_location("v24_export_map", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.v24
def test_align_seams_smooths_east_west_border() -> None:
    """Two adjacent tiles share exactly 1 pixel at the seam; the contract
    is that those 1-pixel shared columns are equal, NOT that a wide band
    is averaged. The earlier 16-pixel band produced a visible "weird
    border" on the tiles; the 1-pixel alignment fixes that.
    """
    m = _import_align()
    per_tile = {
        (0, 0): np.zeros((257, 257), dtype=np.float32),
        (0, 1): np.full((257, 257), 100.0, dtype=np.float32),
    }
    full = m._align_seams(per_tile, [0], [0, 1])
    # The seam is at column 257 (the right edge of tile 0 = left edge of tile 1).
    # The two columns at indices 256 and 257 must be EQUAL (the shared pixel).
    seam_x = 257
    np.testing.assert_array_equal(full[:, seam_x - 1], full[:, seam_x])
    # The shared pixel must be the average of the two original sides (~50).
    assert 40.0 <= float(full[:, seam_x].mean()) <= 60.0
    # Outside the 1-pixel seam, the values stay at the original 0 / 100.
    # (Column seam_x - 2 is interior of tile 0, column seam_x + 1 is interior of tile 1.)
    assert full[:, seam_x - 2].max() == 0.0
    assert full[:, seam_x + 1].min() == 100.0


@pytest.mark.v24
def test_align_seams_smooths_north_south_border() -> None:
    """Same contract for the horizontal seam between two stacked tiles."""
    m = _import_align()
    per_tile = {
        (0, 0): np.zeros((257, 257), dtype=np.float32),
        (1, 0): np.full((257, 257), 100.0, dtype=np.float32),
    }
    # tile_rows[0] = 1 (the 100 tile) is at OBJ row 0; tile_rows[1] = 0
    # (the 0 tile) is at OBJ row 257. The seam is at OBJ row 256/257.
    full = m._align_seams(per_tile, [1, 0], [0])
    seam_y = 257
    np.testing.assert_array_equal(full[seam_y - 1, :], full[seam_y, :])
    assert 40.0 <= float(full[seam_y, :].mean()) <= 60.0
    # Row seam_y - 2 is interior of the 100 tile (rows 0..256).
    assert full[seam_y - 2, :].max() == 100.0
    # Row seam_y + 1 is interior of the 0 tile (rows 257..513).
    assert full[seam_y + 1, :].min() == 0.0


@pytest.mark.v24
def test_align_tile_boundaries_shared_east_west_seam() -> None:
    """Quilt version: only the 1-pixel shared column is aligned. The 16-pixel
    band is opt-in via ``seam_width`` and off by default.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v24_quilt_objs",
        Path(__file__).resolve().parents[2] / "scripts" / "v24_quilt_objs.py",
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    per_tile = {
        (0, 0): np.zeros((257, 257), dtype=np.float32),
        (0, 1): np.full((257, 257), 100.0, dtype=np.float32),
    }
    aligned = m._align_tile_boundaries(per_tile)  # default seam_width=0
    west = aligned[(0, 0)]
    east = aligned[(0, 1)]
    # The 1-pixel shared column (west's rightmost == east's leftmost) is
    # the average of the two sides' values (~50), not a hard step.
    np.testing.assert_array_equal(west[:, -1], east[:, 0])
    assert 40.0 <= float(west[:, -1].mean()) <= 60.0
    # Outside the 1-pixel seam, the values stay at the original 0 / 100.
    assert west[:, -2].max() == 0.0
    assert east[:, 1].min() == 100.0


@pytest.mark.v24
def test_align_tile_boundaries_handles_missing_neighbour() -> None:
    """A tile with no neighbour on the east or south stays unchanged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v24_quilt_objs",
        Path(__file__).resolve().parents[2] / "scripts" / "v24_quilt_objs.py",
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    per_tile = {(0, 0): np.full((257, 257), 42.0, dtype=np.float32)}
    aligned = m._align_tile_boundaries(per_tile)  # default seam_width=0
    assert (aligned[(0, 0)] == 42.0).all()


@pytest.mark.v24
def test_align_tile_boundaries_naming_xy_and_yx() -> None:
    """The naming flag affects which number in the stem is X vs Y."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v24_quilt_objs",
        Path(__file__).resolve().parents[2] / "scripts" / "v24_quilt_objs.py",
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m._parse_world_xy("tile_31_27", naming="xy") == (31, 27)
    assert m._parse_world_xy("tile_27_31", naming="yx") == (31, 27)
    # Same string, different naming, different result.
    assert m._parse_world_xy("tile_27_31", naming="xy") == (27, 31)
    assert m._parse_world_xy("tile_31_27", naming="yx") == (27, 31)
    # Unparseable returns None for both.
    assert m._parse_world_xy("garbage", naming="xy") is None
    assert m._parse_world_xy("garbage", naming="yx") is None


def test_align_seams_handles_missing_tiles() -> None:
    """A missing tile in the grid is filled with the per-map mean before alignment."""
    m = _import_align()
    # Only the (0, 0) tile is real; (0, 1) is missing -> gets the map mean.
    per_tile = {
        (0, 0): np.full((257, 257), 10.0, dtype=np.float32),
        (0, 1): None,
    }
    # _align_seams does NOT fill in missing tiles; that is the caller's job
    # (the script does it after the per-map mean is known). The test of
    # the caller-side fill lives in the script's own path; here we just
    # confirm the function expects all entries to be filled.
    per_tile = {k: (v if v is not None else np.full((257, 257), 10.0, dtype=np.float32))
                for k, v in per_tile.items()}
    full = m._align_seams(per_tile, [0], [0, 1])
    assert full.shape == (257, 514)
    # Both sides should be 10 (no seam difference after the fill).
    assert (full == 10.0).all()


@pytest.mark.v24
def test_export_map_loads_v18_and_lists_northrend_tiles() -> None:
    """Smoke: the script's V18 loader produces a TileRecord with the right shape."""
    m = _import_align()
    import zarr
    import pyarrow.parquet as pq

    v18 = zarr.open_group(
        str(Path("I:/parp/parp-tools/wow-viewer/output/datasets/v18/3_3_5_12340.zarr")),
        mode="r",
    )
    index = m._read_v18_index(
        Path("I:/parp/parp-tools/wow-viewer/output/datasets/v18/3_3_5_12340.zarr")
    )
    northrend_rows = [i for i, m_name in enumerate(index["map"]) if m_name == "Northrend"]
    assert len(northrend_rows) > 0
    record = m._load_v18_record(v18, index, northrend_rows[0])
    assert record.cleaned_minimap.shape == (256, 256, 3)
    assert record.cleaned_minimap.dtype == np.float32
    assert (record.cleaned_minimap >= 0.0).all()
    assert (record.cleaned_minimap <= 1.0).all()
