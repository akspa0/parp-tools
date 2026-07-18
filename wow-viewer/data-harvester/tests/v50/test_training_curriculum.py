"""The curriculum builder must copy only reviewed keep-rows, split by whole map, and emit a store
both canonical trainers' ``require_store_release`` gate actually accepts -- never a store that
re-derives its own quality policy or leaks a map across the holdout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v50.contracts import require_store_release
from harvester.v50.training_curriculum import CurriculumBuildError, build_training_curriculum


def _write_complete_store(path: Path, map_name: str, rows: int) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update({"schema": "v50-complete-store-v1", "model_family": "v50", "release": "v50.1"})
    rng = np.random.default_rng(hash(map_name) % (2**32))
    group.create_array("minimap_rgb", data=rng.integers(0, 255, size=(rows, 8, 8, 3), dtype=np.uint8))
    group.create_array("height_257", data=rng.random((rows, 9, 9), dtype=np.float32))
    group.create_array("object_precise_mask", data=np.zeros((rows, 9, 9), dtype=np.float32))
    index = [
        {"tile_id": i, "build": "0_5_3_3368", "map": map_name, "tile_x": i, "tile_y": 0}
        for i in range(rows)
    ]
    pq.write_table(pa.Table.from_pylist(index), path / "index.parquet")


def _write_manifest(path: Path, map_name: str, keeps: list[bool]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "tile_id": i,
            "build": "0_5_3_3368",
            "map": map_name,
            "tile_x": i,
            "tile_y": 0,
            "keep": keep,
            "reason": "kept" if keep else "object_contaminated",
            "height_regime": "flat",
        }
        for i, keep in enumerate(keeps)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path / "curation_manifest.parquet")


def _build_two_map_fixture(tmp_path: Path) -> dict:
    store_a, store_b = tmp_path / "A.zarr", tmp_path / "B.zarr"
    manifest_a, manifest_b = tmp_path / "curation-A", tmp_path / "curation-B"
    _write_complete_store(store_a, "MapA", rows=4)
    _write_complete_store(store_b, "MapB", rows=3)
    _write_manifest(manifest_a, "MapA", keeps=[True, False, True, True])
    _write_manifest(manifest_b, "MapB", keeps=[True, True, False])
    return {"stores": [store_a, store_b], "manifests": [manifest_a, manifest_b]}


def test_builder_copies_only_keep_rows_and_splits_by_whole_map(tmp_path: Path):
    fixture = _build_two_map_fixture(tmp_path)
    output = tmp_path / "curriculum.zarr"

    summary = build_training_curriculum(
        stores=fixture["stores"], curation_manifests=fixture["manifests"],
        output=output, val_map="MapB", release="v50.1",
    )

    assert summary["total_rows"] == 5  # 3 kept from MapA + 2 kept from MapB
    assert summary["splits"] == {"train": 3, "val": 2}
    index = pq.read_table(output / "index.parquet").to_pylist()
    assert all(row["split"] == ("val" if row["map"] == "MapB" else "train") for row in index)
    # dropped row (MapA tile 1) must not appear
    assert {(row["map"], row["source_tile_id"]) for row in index} == {("MapA", 0), ("MapA", 2), ("MapA", 3), ("MapB", 0), ("MapB", 1)}


def test_builder_output_passes_the_trainer_release_gate_and_preserves_bytes(tmp_path: Path):
    fixture = _build_two_map_fixture(tmp_path)
    output = tmp_path / "curriculum.zarr"
    build_training_curriculum(
        stores=fixture["stores"], curation_manifests=fixture["manifests"],
        output=output, val_map="MapB", release="v50.1",
    )

    group = zarr.open_group(str(output), mode="r")
    require_store_release(group, "v50.1", store=output)  # must not raise

    source = zarr.open_group(str(fixture["stores"][0]), mode="r")
    np.testing.assert_array_equal(group["minimap_rgb"][0], source["minimap_rgb"][0])
    np.testing.assert_array_equal(group["height_257"][0], source["height_257"][0])


def test_builder_refuses_overwrite_all_val_and_mismatched_pairing(tmp_path: Path):
    fixture = _build_two_map_fixture(tmp_path)
    output = tmp_path / "curriculum.zarr"
    output.mkdir()
    with pytest.raises(CurriculumBuildError, match="refusing to overwrite"):
        build_training_curriculum(stores=fixture["stores"], curation_manifests=fixture["manifests"],
                                  output=output, val_map="MapB", release="v50.1")

    with pytest.raises(CurriculumBuildError, match="one --curation-manifest per --store"):
        build_training_curriculum(stores=fixture["stores"], curation_manifests=fixture["manifests"][:1],
                                  output=tmp_path / "c2.zarr", val_map="MapB", release="v50.1")

    only_a = {"stores": fixture["stores"][:1], "manifests": fixture["manifests"][:1]}
    with pytest.raises(CurriculumBuildError, match="whole corpus"):
        build_training_curriculum(stores=only_a["stores"], curation_manifests=only_a["manifests"],
                                  output=tmp_path / "c3.zarr", val_map="MapA", release="v50.1")

    with pytest.raises(CurriculumBuildError, match="matched no kept rows"):
        build_training_curriculum(stores=fixture["stores"], curation_manifests=fixture["manifests"],
                                  output=tmp_path / "c4.zarr", val_map="Nowhere", release="v50.1")
