"""Spec 118 US3 multi-feature-store: the geometry chain concatenates MORE THAN ONE
``v115-feature-map-v1`` prior so an object-segmentation map can AUGMENT the Spec 115 terrain-feature
deconfounding rather than evict it.

The real multi-store logic lives in ``harvester.v50.feature_stores`` (loaded/validated/concatenated
once, reused by both trainers and the materializer), so it is unit-tested directly here. The thin
CLI scripts are proven to advertise the repeatable flag via ``--help``, matching the project's
"verify the documented CLI surface" convention.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v50.feature_stores import (
    FeatureStoreError,
    feature_channels_for_row,
    load_feature_stores,
    plan_entries,
    road_feature_binding,
    total_class_count,
)

_SCRIPTS = Path(__file__).parents[2] / "scripts"


def _write_feature_store(
    path: Path,
    *,
    rows: int,
    class_count: int,
    fill: float,
    taxonomy_revision: str | None = None,
    source_signal: str = "object_geometry_visible",
    row_indices: list[int] | None = None,
) -> Path:
    """Write a minimal but schema-valid v115-feature-map-v1 store whose channels are a constant."""
    group = zarr.open_group(str(path), mode="w")
    group.create_array(
        "feature_map",
        data=np.full((rows, class_count, 256, 256), fill, dtype=np.float16),
    )
    attrs = {
        "schema": "v115-feature-map-v1",
        "class_count": class_count,
        "source_signal": source_signal,
        "checkpoint_sha256": "a" * 64,
    }
    if taxonomy_revision is not None:
        attrs["taxonomy_revision"] = taxonomy_revision
    group.attrs.update(attrs)
    indices = row_indices if row_indices is not None else list(range(rows))
    index_rows = [{"source_row_index": int(i), "map": "Kalimdor"} for i in indices]
    pq.write_table(pa.Table.from_pylist(index_rows), path / "index.parquet")
    return path


def test_two_stores_concatenate_channels_in_cli_order(tmp_path: Path) -> None:
    terrain = _write_feature_store(
        tmp_path / "terrain.zarr", rows=3, class_count=4, fill=0.25,
        taxonomy_revision="v115.1",
    )
    objects = _write_feature_store(
        tmp_path / "objects.zarr", rows=3, class_count=2, fill=0.75,
    )
    bindings = load_feature_stores([terrain, objects], selected_rows=[0, 1, 2])

    assert total_class_count(bindings) == 6
    feats = feature_channels_for_row(bindings, 1)
    assert feats is not None
    assert feats.shape == (6, 256, 256)
    # First 4 channels are the terrain prior (0.25), the next 2 are the object prior (0.75) --
    # i.e. CLI order, terrain then objects, is preserved.
    assert np.allclose(feats[:4], 0.25)
    assert np.allclose(feats[4:], 0.75)


def test_single_store_returns_its_own_channels_unconcatenated(tmp_path: Path) -> None:
    only = _write_feature_store(tmp_path / "only.zarr", rows=2, class_count=2, fill=0.5)
    bindings = load_feature_stores([only], selected_rows=[0, 1])
    feats = feature_channels_for_row(bindings, 0)
    assert feats is not None
    assert feats.shape == (2, 256, 256)


def test_no_stores_is_none_channels(tmp_path: Path) -> None:
    assert load_feature_stores(None, selected_rows=[0]) == []
    assert load_feature_stores([], selected_rows=[0]) == []
    assert feature_channels_for_row([], 0) is None


def test_missing_row_in_any_store_is_refused_at_load(tmp_path: Path) -> None:
    # The object store only covers rows {0, 1}; row 2 is selected -> refuse, so a silent
    # partial-coverage augmentation can never happen.
    terrain = _write_feature_store(tmp_path / "t.zarr", rows=3, class_count=4, fill=0.1,
                                   taxonomy_revision="v115.1")
    objects = _write_feature_store(tmp_path / "o.zarr", rows=2, class_count=2, fill=0.2,
                                   row_indices=[0, 1])
    with pytest.raises(FeatureStoreError, match="missing 1 selected curriculum rows"):
        load_feature_stores([terrain, objects], selected_rows=[0, 1, 2])


def test_wrong_schema_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zarr"
    group = zarr.open_group(str(bad), mode="w")
    group.create_array("feature_map", data=np.zeros((1, 2, 256, 256), dtype=np.float16))
    group.attrs.update({"schema": "something-else", "class_count": 2})
    pq.write_table(pa.Table.from_pylist([{"source_row_index": 0}]), bad / "index.parquet")
    with pytest.raises(FeatureStoreError, match="not a v115-feature-map-v1 store"):
        load_feature_stores([bad], selected_rows=[0])


def test_road_binding_selects_the_taxonomy_carrying_store(tmp_path: Path) -> None:
    terrain = _write_feature_store(tmp_path / "terrain.zarr", rows=2, class_count=4, fill=0.25,
                                   taxonomy_revision="v115.1", source_signal="terrain_feature")
    objects = _write_feature_store(tmp_path / "objects.zarr", rows=2, class_count=2, fill=0.75)
    # Object prior first, terrain second: road binding must still find the terrain one by its
    # taxonomy_revision attr, regardless of order.
    bindings = load_feature_stores([objects, terrain], selected_rows=[0, 1])
    road = road_feature_binding(bindings)
    assert road is not None
    assert road.path == terrain
    assert road.class_count == 4
    # An all-object binding list has no road-capable prior.
    assert road_feature_binding(load_feature_stores([objects], selected_rows=[0, 1])) is None


def test_plan_entries_records_each_store(tmp_path: Path) -> None:
    terrain = _write_feature_store(tmp_path / "terrain.zarr", rows=2, class_count=4, fill=0.25,
                                   taxonomy_revision="v115.1")
    objects = _write_feature_store(tmp_path / "objects.zarr", rows=2, class_count=2, fill=0.75)
    entries = plan_entries(load_feature_stores([terrain, objects], selected_rows=[0, 1]))
    assert [e["class_count"] for e in entries] == [4, 2]
    assert entries[0]["taxonomy_revision"] == "v115.1"
    # The object prior carries no taxonomy, so the key is simply absent (not a null).
    assert "taxonomy_revision" not in entries[1]
    assert entries[1]["source_signal"] == "object_geometry_visible"


@pytest.mark.parametrize(
    "script",
    ["v50_train_direct_geometry.py", "v50_train_geometry_detailer.py", "v50_materialize_coarse_relief.py"],
)
def test_cli_advertises_repeatable_feature_store(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / script), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"{script} --help failed: {result.stderr}"
    assert "REPEATABLE" in result.stdout
