"""Smoke test for the v61 on-the-fly dedup store (DedupStore)."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from harvester.v50.contracts import DEFAULT_RELEASE_V60  # noqa: E402
from v61_harvest import DedupStore  # noqa: E402


def test_dedup_store_deduplicates_identical_arrays():
    tmp = Path(tempfile.mkdtemp(prefix="v61test-"))
    try:
        out = tmp / "v61.zarr"
        store = DedupStore(out, DEFAULT_RELEASE_V60)

        # Two tiles with identical height/normal/minimap (should dedup to 1 each)
        for build in ("0_5_3_3368", "4_0_0_11927"):
            store.add_tile({
                "_build_id": build, "_map": "Kalimdor", "tile_x": 1, "tile_y": 1,
                "height_257": np.full((4, 4), 10.0, dtype=np.float32),
                "normal_xyz": np.full((4, 4, 3), 0.5, dtype=np.float32),
                "minimap_rgb": np.full((4, 4, 3), 128, dtype=np.uint8),
            })
        # One tile with DIFFERENT height (separate unique)
        store.add_tile({
            "_build_id": "1_0_0_3980", "_map": "Azeroth", "tile_x": 2, "tile_y": 2,
            "height_257": np.full((4, 4), 99.0, dtype=np.float32),
            "normal_xyz": np.full((4, 4, 3), 0.5, dtype=np.float32),
            "minimap_rgb": np.full((4, 4, 3), 128, dtype=np.uint8),
        })

        result = store.finalize(out)

        assert result["row_count"] == 3, result["row_count"]
        assert result["signal_count"] == 3, result["signal_count"]
        # height: 2 unique (10.0, 99.0); normal/minimap: 1 unique each
        assert result["unique_arrays"] == 4, result["unique_arrays"]

        # Verify the store is readable and pointers are correct.
        g = zarr.open_group(str(out), mode="r")
        idx = pq.read_table(out / "index.parquet").to_pylist()
        assert len(idx) == 3
        assert g["height_257"]["canonical"].shape[0] == 2
        assert g["normal_xyz"]["canonical"].shape[0] == 1
        assert g["minimap_rgb"]["canonical"].shape[0] == 1
        assert g["height_257"]["row_index"].shape[0] == 3
        print("SMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)