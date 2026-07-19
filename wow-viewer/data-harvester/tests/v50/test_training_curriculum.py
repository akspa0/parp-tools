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


def _write_complete_store(path: Path, map_name: str, rows: int, authored_rows: set[int] | None = None) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update({"schema": "v50-complete-store-v1", "model_family": "v50", "release": "v50.1"})
    rng = np.random.default_rng(hash(map_name) % (2**32))
    group.create_array("minimap_rgb", data=rng.integers(1, 255, size=(rows, 8, 8, 3), dtype=np.uint8))
    group.create_array("height_257", data=rng.random((rows, 9, 9), dtype=np.float32))
    group.create_array("object_precise_mask", data=np.zeros((rows, 9, 9), dtype=np.float32))
    if authored_rows is not None:
        authored = np.zeros((rows, 8, 8, 3), dtype=np.uint8)
        for r in authored_rows:
            authored[r] = rng.integers(1, 255, size=(8, 8, 3), dtype=np.uint8)
        group.create_array("minimap_rgb_authored", data=authored)
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


def test_stratified_split_holds_out_rows_from_every_map_deterministically(tmp_path: Path):
    # The standard regime: the WDL target is absolute elevation, so val tiles must come from the
    # same maps as train -- a whole-map holdout measures an unseen altitude offset instead (this
    # was observed on the real v50.1 corpus: val loss *worsened* as training progressed).
    store_a, store_b = tmp_path / "A.zarr", tmp_path / "B.zarr"
    manifest_a, manifest_b = tmp_path / "curation-A", tmp_path / "curation-B"
    _write_complete_store(store_a, "MapA", rows=10)
    _write_complete_store(store_b, "MapB", rows=10)
    _write_manifest(manifest_a, "MapA", keeps=[True] * 10)
    _write_manifest(manifest_b, "MapB", keeps=[True] * 10)

    first = build_training_curriculum(
        stores=[store_a, store_b], curation_manifests=[manifest_a, manifest_b],
        output=tmp_path / "s1.zarr", val_map=None, val_fraction=0.2, release="v50.1",
    )
    second = build_training_curriculum(
        stores=[store_a, store_b], curation_manifests=[manifest_a, manifest_b],
        output=tmp_path / "s2.zarr", val_map=None, val_fraction=0.2, release="v50.1",
    )

    assert first["splits"] == {"train": 16, "val": 4}
    assert first["val_rows_per_map"] == {"MapA": 2, "MapB": 2}  # every map contributes val rows
    first_index = pq.read_table(tmp_path / "s1.zarr" / "index.parquet").to_pylist()
    second_index = pq.read_table(tmp_path / "s2.zarr" / "index.parquet").to_pylist()
    assert [row["split"] for row in first_index] == [row["split"] for row in second_index]


def test_dual_minimap_source_emits_two_rows_per_authored_tile(tmp_path: Path):
    # Spec 112: a kept tile with BOTH a synthetic and an authored minimap contributes two rows
    # (one per source) sharing one height target; a tile with only synthetic contributes one.
    store = tmp_path / "A.zarr"
    manifest = tmp_path / "curation-A"
    _write_complete_store(store, "MapA", rows=4, authored_rows={0, 2})  # tiles 0,2 have authored
    _write_manifest(manifest, "MapA", keeps=[True, True, True, True])
    output = tmp_path / "curriculum.zarr"

    summary = build_training_curriculum(
        stores=[store], curation_manifests=[manifest],
        output=output, val_fraction=0.25, release="v50.1",
    )

    # 4 kept tiles -> 4 synthetic rows + 2 authored rows = 6
    assert summary["total_rows"] == 6
    assert summary["minimap_source_counts"] == {"authored": 2, "synthetic": 4}

    index = pq.read_table(output / "index.parquet").to_pylist()
    out = zarr.open_group(str(output), mode="r")
    src = zarr.open_group(str(store), mode="r")

    # authored row's minimap_rgb equals the source authored image; synthetic row's equals synthetic
    for row in index:
        tid = row["source_tile_id"]
        want = src["minimap_rgb_authored"][tid] if row["minimap_source"] == "authored" else src["minimap_rgb"][tid]
        np.testing.assert_array_equal(out["minimap_rgb"][row["tile_id"]], want)

    # neither the authored source array nor the 1024 upscaler target is copied as its own column
    assert "minimap_rgb_authored" not in out.array_keys()
    assert "minimap_rgb_1024" not in out.array_keys()


def test_per_source_blank_check_recovers_authored_when_synthetic_is_blank(tmp_path: Path):
    # Spec 112 (user-directed): a tile whose synthetic minimap is blank/uniform but whose authored
    # minimap is valid must still contribute an AUTHORED row -- the ~275-tile authored-recovery case
    # on the real corpus. The synthetic row is correctly skipped for that tile.
    store = tmp_path / "A.zarr"
    manifest = tmp_path / "curation-A"
    group = zarr.open_group(str(store), mode="w")
    group.attrs.update({"schema": "v50-complete-store-v1", "model_family": "v50", "release": "v50.1"})
    rng = np.random.default_rng(1)
    synthetic = rng.integers(1, 255, size=(3, 8, 8, 3), dtype=np.uint8)
    synthetic[1] = 128  # tile 1: uniform (nonzero) => blank by std, unusable as a synthetic input
    authored = rng.integers(1, 255, size=(3, 8, 8, 3), dtype=np.uint8)  # all three authored valid
    group.create_array("minimap_rgb", data=synthetic)
    group.create_array("minimap_rgb_authored", data=authored)
    group.create_array("height_257", data=rng.random((3, 9, 9), dtype=np.float32))
    index = [{"tile_id": i, "build": "0_5_3_3368", "map": "MapA", "tile_x": i, "tile_y": 0} for i in range(3)]
    pq.write_table(pa.Table.from_pylist(index), store / "index.parquet")
    _write_manifest(manifest, "MapA", keeps=[True, True, True])

    summary = build_training_curriculum(
        stores=[store], curation_manifests=[manifest],
        output=tmp_path / "curriculum.zarr", val_fraction=0.34, release="v50.1", min_rgb_std=1.0,
    )

    # tile 1 loses its synthetic row (blank) but keeps its authored row
    assert summary["minimap_source_counts"] == {"authored": 3, "synthetic": 2}
    index_out = pq.read_table(tmp_path / "curriculum.zarr" / "index.parquet").to_pylist()
    tile1 = {r["minimap_source"] for r in index_out if r["source_tile_id"] == 1}
    assert tile1 == {"authored"}


def test_both_rows_of_a_tile_share_split_no_leakage(tmp_path: Path):
    store = tmp_path / "A.zarr"
    manifest = tmp_path / "curation-A"
    _write_complete_store(store, "MapA", rows=8, authored_rows=set(range(8)))  # every tile dual
    _write_manifest(manifest, "MapA", keeps=[True] * 8)
    output = tmp_path / "curriculum.zarr"

    build_training_curriculum(
        stores=[store], curation_manifests=[manifest],
        output=output, val_fraction=0.25, release="v50.1",
    )
    index = pq.read_table(output / "index.parquet").to_pylist()

    split_by_group: dict[str, set[str]] = {}
    for row in index:
        split_by_group.setdefault(row["source_group_id"], set()).add(row["split"])
    # every tile's two rows land in exactly one split -- the leak-safety invariant the trainer rechecks
    assert all(len(splits) == 1 for splits in split_by_group.values())
    assert {row["split"] for row in index} == {"train", "val"}


def test_split_mode_selection_is_exactly_one_of_val_map_or_val_fraction(tmp_path: Path):
    fixture = _build_two_map_fixture(tmp_path)
    with pytest.raises(CurriculumBuildError, match="exactly one"):
        build_training_curriculum(stores=fixture["stores"], curation_manifests=fixture["manifests"],
                                  output=tmp_path / "x.zarr", val_map="MapB", val_fraction=0.2, release="v50.1")
    with pytest.raises(CurriculumBuildError, match="exactly one"):
        build_training_curriculum(stores=fixture["stores"], curation_manifests=fixture["manifests"],
                                  output=tmp_path / "y.zarr", val_map=None, val_fraction=None, release="v50.1")
    with pytest.raises(CurriculumBuildError, match=r"in \(0, 1\)"):
        build_training_curriculum(stores=fixture["stores"], curation_manifests=fixture["manifests"],
                                  output=tmp_path / "z.zarr", val_map=None, val_fraction=1.5, release="v50.1")


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
