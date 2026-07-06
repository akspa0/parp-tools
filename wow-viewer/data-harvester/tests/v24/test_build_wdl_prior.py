"""V24 store build test (FR-006/FR-007) on a synthetic 5-tile V18 store."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v24 import store

from .conftest import requires_shim

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _make_synthetic_v18(path: Path, synthetic_height: np.ndarray) -> None:
    n = 5
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    heights = np.stack(
        [
            synthetic_height,
            synthetic_height + 25.0,
            synthetic_height * 0.5,
            np.zeros((257, 257), dtype=np.float32),  # audit-empty tile
            synthetic_height - 10.0,
        ]
    )
    group.create_array(name="height_257", shape=(n, 257, 257), dtype="float32")
    group["height_257"][:] = heights
    group.create_array(name="liquid_mask", shape=(n, 256, 256), dtype="float32")
    group["liquid_mask"][:] = 0.0

    table = pa.table(
        {
            "tile_id": list(range(n)),
            "build": ["synthetic"] * n,
            "map": ["TestMap"] * n,
            "tile_x": [30, 31, 32, 33, 34],
            "tile_y": [30, 30, 30, 30, 30],
            "has_height_257": [True, True, True, False, True],
        }
    )
    pq.write_table(table, str(path / "index.parquet"))


@pytest.mark.v24
@requires_shim
def test_build_wdl_prior_synthetic_store(tmp_path, synthetic_height):
    v18_path = tmp_path / "v18.zarr"
    v24_path = tmp_path / "v24.zarr"
    _make_synthetic_v18(v18_path, synthetic_height)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_wdl_prior.py"),
            "build",
            "--v18-store", str(v18_path),
            "--staged-client", str(tmp_path),  # no WDLs -> all synthetic
            "--output", str(v24_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    group = store.open_v24_store(v24_path)
    assert group["wdl_prior_outer"].shape == (5, 17, 17)
    assert group["wdl_prior_inner"].shape == (5, 16, 16)

    source_outer = np.asarray(group["wdl_prior_source_outer"][:])
    assert set(np.unique(source_outer)) <= {0, 1, 2}
    # Tiles 0-2 and 4 are synthetic (no real WDL); tile 3 is audit-empty.
    assert (source_outer[0] == 1).all()
    assert (source_outer[3] == 2).all()

    confidence = np.asarray(group["wdl_prior_confidence_outer"][:])
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0
    assert (confidence[3] == 0.0).all()

    stats = store.coverage_stats(group)
    assert stats["synthetic_cell_ratio"] > 0.7
    assert bool(np.asarray(group["wdl_prior_audit_empty"][:])[3])

    index = store.read_index(v24_path)
    assert index["v18_row"] == [0, 1, 2, 3, 4]

    # inspect summary runs cleanly on the produced store
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "inspect_v24_dataset.py"),
            "summary",
            "--store", str(v24_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "synthetic_cell_ratio" in result.stdout
