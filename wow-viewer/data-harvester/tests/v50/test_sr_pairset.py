"""Spec 113 T012: pairs exist only for tiles with BOTH sources (excluded tiles counted, never
zero-filled); the US1 gate is enforced before any pairing; the corrective transform is applied to
the HR; the split is deterministic and leak-free."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v50.minimap_alignment import apply_dihedral, apply_translation
from harvester.v50.sr_pairset import PairSetBuildError, build_sr_pairset


def _write_store(path: Path, map_name: str, rows: int, authored_rows: set[int], detail_rows: set[int]) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(
        {
            "schema": "v50-complete-store-v1",
            "release": "v50.1",
            "minimap_rgb_1024_render_mode": "detail",
        }
    )
    rng = np.random.default_rng(hash(map_name) % (2**32))
    authored = np.zeros((rows, 8, 8, 3), dtype=np.uint8)
    detail = np.zeros((rows, 32, 32, 3), dtype=np.uint8)
    for r in authored_rows:
        authored[r] = rng.integers(1, 255, size=(8, 8, 3), dtype=np.uint8)
    for r in detail_rows:
        detail[r] = rng.integers(1, 255, size=(32, 32, 3), dtype=np.uint8)
    group.create_array("minimap_rgb_authored", data=authored)
    group.create_array("minimap_rgb_1024", data=detail)
    group.create_array("height_257", data=rng.random((rows, 5, 5), dtype=np.float32))
    index = [{"tile_id": i, "build": "0_5_3_3368", "map": map_name, "tile_x": i, "tile_y": 0} for i in range(rows)]
    pq.write_table(pa.Table.from_pylist(index), path / "index.parquet")


def _alignment_report(
    path: Path,
    gate: str = "pass_identity",
    corrective: str | None = None,
    offset: list[int] | None = None,
) -> Path:
    report = path / "alignment.json"
    report.write_text(
        json.dumps(
            {
                "gate": gate,
                "corrective_transform": corrective,
                "corrective_offset_lr": offset or [0, 0],
            }
        ),
        encoding="utf-8",
    )
    return report


def _visual_review(path: Path, stores: list[Path]) -> Path:
    report = path / "visual-review.json"
    report.write_text(
        json.dumps(
            [
                {
                    "schema": "v50-store-visual-review-v1",
                    "store": str(store.resolve()),
                    "output": str((path / f"{store.stem}.png").resolve()),
                    "map": "Kalimdor",
                    "authored_object_policy": "may_contain_client_baked_objects",
                    "synthetic_object_policy": "terrain_only_no_objects",
                    "pixel_equality_required": False,
                    "rows": [{"row": 0}],
                }
                for store in stores
            ]
        ),
        encoding="utf-8",
    )
    return report


def test_pairs_only_for_tiles_with_both_sources_and_exclusions_counted(tmp_path: Path):
    store = tmp_path / "A.zarr"
    # 6 tiles: 0-3 have authored; 2-5 have detail => pairs are exactly {2, 3}
    _write_store(store, "Kalimdor", rows=6, authored_rows={0, 1, 2, 3}, detail_rows={2, 3, 4, 5})

    summary = build_sr_pairset(
        stores=[store], output=tmp_path / "pairs.zarr",
        alignment_report=_alignment_report(tmp_path), val_fraction=0.5,
    )

    assert summary["total_pairs"] == 2
    assert summary["excluded"] == {"missing_authored": 2, "missing_detail_render": 2}
    index = pq.read_table(tmp_path / "pairs.zarr" / "index.parquet").to_pylist()
    assert {r["source_tile_id"] for r in index} == {2, 3}


def test_gate_failure_refuses_pairing(tmp_path: Path):
    store = tmp_path / "A.zarr"
    _write_store(store, "Kalimdor", rows=2, authored_rows={0, 1}, detail_rows={0, 1})

    with pytest.raises(PairSetBuildError, match="alignment gate"):
        build_sr_pairset(
            stores=[store], output=tmp_path / "pairs.zarr",
            alignment_report=_alignment_report(tmp_path, gate="fail_inconsistent"), val_fraction=0.5,
        )


def test_failed_raw_pixel_gate_can_build_explicit_terrain_only_pairs_after_visual_review(tmp_path: Path):
    store = tmp_path / "A.zarr"
    _write_store(store, "Kalimdor", rows=2, authored_rows={0, 1}, detail_rows={0, 1})

    summary = build_sr_pairset(
        stores=[store],
        output=tmp_path / "pairs.zarr",
        alignment_report=_alignment_report(tmp_path, gate="fail_inconsistent"),
        val_fraction=0.5,
        terrain_only_cross_domain=True,
        visual_review_report=_visual_review(tmp_path, [store]),
    )

    pairs = zarr.open_group(str(tmp_path / "pairs.zarr"), mode="r")
    assert summary["pairing_mode"] == "terrain_only_cross_domain_same_tile"
    assert summary["corrective_transform"] == "identity"
    assert summary["synthetic_object_policy"] == "terrain_only_no_objects"
    assert pairs.attrs["pairing_mode"] == "terrain_only_cross_domain_same_tile"


def test_corrective_transform_is_applied_to_hr(tmp_path: Path):
    store = tmp_path / "A.zarr"
    _write_store(store, "Azeroth", rows=2, authored_rows={0, 1}, detail_rows={0, 1})

    build_sr_pairset(
        stores=[store], output=tmp_path / "pairs.zarr",
        alignment_report=_alignment_report(tmp_path, gate="pass_with_transform", corrective="flip_v"),
        val_fraction=0.5,
    )

    source = zarr.open_group(str(store), mode="r")
    pairs = zarr.open_group(str(tmp_path / "pairs.zarr"), mode="r")
    index = pq.read_table(tmp_path / "pairs.zarr" / "index.parquet").to_pylist()
    for row in index:
        expected = apply_dihedral(np.asarray(source["minimap_rgb_1024"][row["source_tile_id"]]), "flip_v")
        np.testing.assert_array_equal(pairs["hr"][row["pair_id"]], expected)
    assert pairs.attrs["corrective_transform"] == "flip_v"


def test_corrective_lr_offset_is_scaled_and_applied_to_hr(tmp_path: Path):
    store = tmp_path / "A.zarr"
    _write_store(store, "Azeroth", rows=2, authored_rows={0, 1}, detail_rows={0, 1})

    build_sr_pairset(
        stores=[store], output=tmp_path / "pairs.zarr",
        alignment_report=_alignment_report(
            tmp_path, gate="pass_with_transform", corrective="identity", offset=[1, -1]
        ),
        val_fraction=0.5,
    )

    source = zarr.open_group(str(store), mode="r")
    pairs = zarr.open_group(str(tmp_path / "pairs.zarr"), mode="r")
    index = pq.read_table(tmp_path / "pairs.zarr" / "index.parquet").to_pylist()
    for row in index:
        expected = apply_translation(
            np.asarray(source["minimap_rgb_1024"][row["source_tile_id"]]), (4, -4)
        )
        np.testing.assert_array_equal(pairs["hr"][row["pair_id"]], expected)
    assert list(pairs.attrs["corrective_offset_lr"]) == [1, -1]


def test_out_of_scope_map_is_refused(tmp_path: Path):
    store = tmp_path / "P.zarr"
    _write_store(store, "PVPZone02", rows=2, authored_rows={0, 1}, detail_rows={0, 1})

    with pytest.raises(PairSetBuildError, match="PVPZone02"):
        build_sr_pairset(
            stores=[store], output=tmp_path / "pairs.zarr",
            alignment_report=_alignment_report(tmp_path), val_fraction=0.5,
        )


def test_split_is_deterministic_and_leak_free(tmp_path: Path):
    store = tmp_path / "A.zarr"
    _write_store(store, "Kalimdor", rows=10, authored_rows=set(range(10)), detail_rows=set(range(10)))

    first = build_sr_pairset(
        stores=[store], output=tmp_path / "p1.zarr",
        alignment_report=_alignment_report(tmp_path), val_fraction=0.2,
    )
    second = build_sr_pairset(
        stores=[store], output=tmp_path / "p2.zarr",
        alignment_report=_alignment_report(tmp_path), val_fraction=0.2,
    )

    idx1 = pq.read_table(tmp_path / "p1.zarr" / "index.parquet").to_pylist()
    idx2 = pq.read_table(tmp_path / "p2.zarr" / "index.parquet").to_pylist()
    assert [r["split"] for r in idx1] == [r["split"] for r in idx2]
    assert first["splits"]["train"] == 8 and first["splits"]["val"] == 2
    # one pair per tile, so tile-level disjointness is structural; verify anyway
    train = {r["source_group_id"] for r in idx1 if r["split"] == "train"}
    val = {r["source_group_id"] for r in idx1 if r["split"] == "val"}
    assert not (train & val)


def test_pairset_summary_conforms_to_the_spec_schema(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    store = tmp_path / "A.zarr"
    _write_store(store, "Kalimdor", rows=8, authored_rows=set(range(8)), detail_rows=set(range(8)))
    summary = build_sr_pairset(
        stores=[store], output=tmp_path / "pairs.zarr",
        alignment_report=_alignment_report(tmp_path), val_fraction=0.25,
    )

    schema_path = (
        Path(__file__).parents[3]
        / "specs"
        / "113-minimap-superres"
        / "contracts"
        / "sr-pairset-and-run.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(summary, schema)
