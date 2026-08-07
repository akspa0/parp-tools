#!/usr/bin/env python3
"""Quick smoke test of the v60 dedup merge with tiny synthetic stores.

Run: uv run python scripts/_test_v60_dedup.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from harvester.v50.contracts import DEFAULT_RELEASE_V60  # noqa: E402
from v60_build_dataset import _merge_into_unified_dedup  # noqa: E402


def _make_store(path: Path, build: str, map_name: str, n: int, height_val: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(path), mode="w")
    # height_257: identical across stores when height_val matches
    heights = np.full((n, 4, 4), height_val, dtype=np.float32)
    g.create_array("height_257", data=heights, overwrite=True)
    # normal_xyz: identical across stores
    normals = np.full((n, 4, 4, 3), 0.5, dtype=np.float32)
    g.create_array("normal_xyz", data=normals, overwrite=True)
    # minimap_rgb: identical across stores
    minimap = np.full((n, 4, 4, 3), 128, dtype=np.uint8)
    g.create_array("minimap_rgb", data=minimap, overwrite=True)
    # A signal that is UNAVAILABLE for some rows (index longer than array)
    # -> simulate by writing an array shorter than the index for one store
    idx = [{
        "build_id": build, "map": map_name, "tile_x": i, "tile_y": i,
        "tile_id": i, "surviving_height_levels": 1, "signal_class": "na",
        "signal_class_evidence": "",
    } for i in range(n)]
    pq.write_table(pa.Table.from_pylist(idx), str(path / "index.parquet"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="v60test-"))
    try:
        work = tmp / "work"
        work.mkdir()
        # Two stores with identical data (should dedup to 1 unique each)
        _make_store(work / "0_5_3_3368-Kalimdor.zarr", "0_5_3_3368", "Kalimdor", 3, 10.0)
        _make_store(work / "4_0_0_11927-Kalimdor.zarr", "4_0_0_11927", "Kalimdor", 3, 10.0)
        # One store with DIFFERENT height (should be a separate unique)
        _make_store(work / "1_0_0_3980-Azeroth.zarr", "1_0_0_3980", "Azeroth", 2, 99.0)

        stores = sorted(work.glob("*.zarr"))
        out = tmp / "unified.zarr"
        result = _merge_into_unified_dedup(stores, out, DEFAULT_RELEASE_V60)

        print(f"\nrows={result['row_count']} signals={result['signal_count']} "
              f"unique={result['unique_arrays']} naive={result['naive_arrays']}")
        assert result["row_count"] == 8, result["row_count"]
        assert result["signal_count"] == 3, result["signal_count"]
        # height: 2 unique (10.0 and 99.0), normal/minimap: 1 unique each
        assert result["unique_arrays"] == 4, result["unique_arrays"]
        print("SMOKE TEST PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())