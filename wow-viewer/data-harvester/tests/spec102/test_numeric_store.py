from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY
from harvester.spec102.numeric_store import (
    CURATED_SELECTION_MODE,
    RAW_V18_IDENTITY_SELECTION_MODE,
    build_numeric_store,
)
from harvester.spec102.strict_target_contract import (
    REQUIRED_STRICT_OBJECT_TARGET_VERSION,
    StrictFragmentTrace,
    StrictFragmentTraceSidecar,
    canonical_json,
    compute_strict_fragment_trace_sha256,
)


def _materialized_assets() -> tuple[dict, ...]:
    return ({"asset_index": 0, "source": "M2Triangle", "normalized_asset_path": "world/x.m2"},)


def _materialized_trace() -> StrictFragmentTrace:
    """One real, hash-consistent v3 fragment for the materialized fixture tile."""
    arrays = {
        "object_geometry_fragment_raster_xy": np.array([[8, 8]], dtype="<i4"),
        "object_geometry_fragment_world_xyze": np.array([[1.0, 2.0, 11.0, 1.0]], dtype="<f4"),
        "object_geometry_fragment_source_ids": np.array([[100, 0, 0]], dtype="<i4"),
        "object_geometry_fragment_source_classification": np.array([[1, 1]], dtype="u1"),
        "object_geometry_fragment_terrain_vertex_dense_xy": np.array([[8, 8, 9, 8, 8, 9]], dtype="<i4"),
        "object_geometry_fragment_terrain_vertex_z": np.array([[10.0, 10.0, 10.0]], dtype="<f4"),
        "object_geometry_fragment_terrain_vertex_present": np.array([[1, 1, 1]], dtype="u1"),
        "object_geometry_fragment_terrain_weights": np.array([[0.25, 0.25, 0.5]], dtype="<f4"),
        "object_geometry_fragment_terrain_liquid_elevation": np.array([[0.0, 0.0]], dtype="<f4"),
    }
    assets = _materialized_assets()
    return StrictFragmentTrace(
        arrays=arrays,
        count=1,
        sha256=compute_strict_fragment_trace_sha256(arrays, assets),
        assets=assets,
        unresolved_placements=(),
    )


def _write_fragment_trace_sidecar(store: Path, tile_count: int) -> None:
    """Materialize the ordinal-0 tile trace exactly like the strict harvester."""
    sidecar = StrictFragmentTraceSidecar.open(store, tile_capacity=tile_count)
    trace = _materialized_trace()
    for ordinal in range(tile_count):
        sidecar.append(ordinal, trace if ordinal == 0 else None)
    sidecar.finalize(tile_count)
    sidecar.close()


def _write_v18(path, *, tile_ids: list[int] | None = None) -> list[dict]:
    tile_ids = tile_ids or [0, 1]
    count = len(tile_ids)
    group = zarr.open_group(str(path), mode="w")
    group.create_array("minimap_rgb", data=np.full((count, 256, 256, 3), 17, dtype=np.uint8))
    strict_mask = np.zeros((count, 257, 257), dtype=np.float32)
    strict_mask[0, 8:12, 8:12] = 1.0
    group.create_array("object_geometry_visible_mask", data=strict_mask)
    strict_top = np.zeros((count, 257, 257), dtype=np.float32)
    strict_top[0, 8:12, 8:12] = 11.0
    group.create_array("object_geometry_visible_top_elevation", data=strict_top)
    strict_terrain = np.zeros((count, 257, 257), dtype=np.float32)
    strict_terrain[0, 8:12, 8:12] = 10.0
    group.create_array("object_geometry_visible_terrain_elevation", data=strict_terrain)
    strict_source = np.zeros((count, 257, 257), dtype=np.uint8)
    strict_source[0, 8:12, 8:12] = 1
    group.create_array("object_geometry_visible_source", data=strict_source)
    group.create_array("liquid_mask", data=np.zeros((count, 256, 256), dtype=np.float32))
    group.create_array("liquid_height", data=np.zeros((count, 256, 256), dtype=np.float32))
    group.create_array("mcnk_flags_16", data=np.ones((count, 16, 16), dtype=np.int32))
    group.create_array("normal_xyz", data=np.ones((count, 257, 257, 3), dtype=np.float32))
    group.create_array("height_257", data=np.arange(count * 257 * 257, dtype=np.float32).reshape(count, 257, 257))
    # The persisted V18 index only carries the canonical JSON form of the asset
    # table, exactly like `build_v18_dataset`; the sidecar validator must accept
    # that shape.  Ordinal 0 is the one materialized CompleteVisible tile.
    materialized_sha256 = _materialized_trace().sha256
    materialized_assets_json = canonical_json(list(_materialized_assets()))

    def _fragment_trace_provenance(ordinal: int) -> dict:
        if ordinal == 0:
            return {
                "object_geometry_fragment_trace_schema": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
                "object_geometry_fragment_count": 1,
                "object_geometry_fragment_sha256": materialized_sha256,
                "object_geometry_target_assets_json": materialized_assets_json,
                "object_geometry_target_unresolved_placements_json": "[]",
                "object_geometry_fragment_trace_arrays_present": True,
                "object_geometry_fragment_trace_validated": True,
                "object_geometry_fragment_trace_sidecar_start": 0,
                "object_geometry_fragment_trace_sidecar_end": 1,
                "object_geometry_target_placement_count": 1,
                "object_geometry_target_geometry_resolved_placement_count": 1,
            }
        return {
            "object_geometry_fragment_trace_schema": "missing",
            "object_geometry_fragment_count": 0,
            "object_geometry_fragment_sha256": "missing",
            "object_geometry_target_assets_json": "[]",
            "object_geometry_target_unresolved_placements_json": "[]",
            "object_geometry_fragment_trace_arrays_present": False,
            "object_geometry_fragment_trace_validated": False,
            "object_geometry_fragment_trace_sidecar_start": 1,
            "object_geometry_fragment_trace_sidecar_end": 1,
            "object_geometry_target_placement_count": 0,
            "object_geometry_target_geometry_resolved_placement_count": 0,
        }

    rows = [
        {
            "tile_id": tile_id,
            "build": "3_3_5_12340",
            "map": "Azeroth" if ordinal == 0 else "Northrend",
            "tile_x": ordinal,
            "tile_y": 0,
            "has_minimap_rgb": True,
            "has_height_257": True,
            "has_normal_xyz": True,
            "has_mcnk_flags_16": True,
            "has_liquid_mask": False,
            "has_liquid_height": False,
            "has_liquid_source_mh2o": False,
            "has_extra_provenance": ordinal == 0,
            "object_geometry_target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "object_geometry_target_status": "CompleteVisible" if ordinal == 0 else "IncompleteGeometry",
            "object_geometry_target_materialized": ordinal == 0,
            "object_geometry_target_arrays_present": True,
            "object_geometry_target_geometry_unresolved_placement_count": 0 if ordinal == 0 else 1,
            "object_geometry_target_fallback_required_placement_count": 0,
            "object_geometry_target_terrain_unknown_pixel_count": 0,
            "object_geometry_target_liquid_evidence_status": "Dry" if ordinal == 0 else "LiquidCovered",
            "object_geometry_target_liquid_covered_pixel_count": 0 if ordinal == 0 else 1,
            "object_geometry_target_liquid_surface_unknown_pixel_count": 0,
            "object_geometry_target_liquid_covered_fragment_count": 0 if ordinal == 0 else 1,
            "object_geometry_target_liquid_hidden_fragment_count": 0 if ordinal == 0 else 1,
            "object_geometry_target_liquid_above_surface_fragment_count": 0,
            "object_geometry_target_liquid_unknown_fragment_count": 0,
            **_fragment_trace_provenance(ordinal),
        }
        for ordinal, tile_id in enumerate(tile_ids)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path / "index.parquet")
    _write_fragment_trace_sidecar(path, count)
    (path / "signal_validation.json").write_text(
        json.dumps({"passed": False, "failures": ["unrelated roof signal"]}), encoding="utf-8"
    )
    return rows


def test_raw_v18_identity_mode_preserves_every_identity_and_target_provenance(tmp_path) -> None:
    source = tmp_path / "raw.zarr"
    rows = _write_v18(source)
    output = tmp_path / "numeric.zarr"
    build_numeric_store(
        selection_store=source,
        v18_stores=[source],
        output=output,
        selection_mode=RAW_V18_IDENTITY_SELECTION_MODE,
    )
    copied = pq.read_table(output / "index.parquet").to_pylist()
    assert len(copied) == len(rows)
    assert [row["source_v18_row"] for row in copied] == [0, 1]
    assert [row["source_v18_tile_id"] for row in copied] == [0, 1]
    assert copied[0]["has_extra_provenance"] is True
    assert copied[0]["strict_target_materialized"] is True
    assert copied[0]["strict_target_version"] == REQUIRED_STRICT_OBJECT_TARGET_VERSION
    assert copied[0]["strict_target_status"] == "CompleteVisible"
    assert copied[1]["strict_target_materialized"] is False
    assert copied[1]["strict_target_status"] == "IncompleteGeometry"
    assert copied[1]["strict_target_liquid_evidence_status"] == "LiquidCovered"
    assert copied[1]["strict_target_liquid_covered_fragment_count"] == 1
    assert zarr.open_group(str(output), mode="r")["normal_xyz_257"][0, 0, 0, 0] == 127
    assert np.array_equal(
        zarr.open_group(str(output), mode="r")[STRICT_OBJECT_TARGET_KEY][1],
        zarr.open_group(str(source), mode="r")["object_geometry_visible_mask"][1],
    )
    contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    assert contract["schema"] == "spec102-numeric-store-v4"
    assert contract["selection_mode"] == RAW_V18_IDENTITY_SELECTION_MODE
    assert contract["source_v18_contracts"]["3_3_5_12340"]["signal_validation"]["passed"] is False
    assert contract["object_target_provenance"]["target_array"] == STRICT_OBJECT_TARGET_KEY
    assert contract["object_target_provenance"]["legacy_precise_mask_policy"] == "prohibited"
    assert contract["object_target_provenance"]["target_version"] == REQUIRED_STRICT_OBJECT_TARGET_VERSION


def test_numeric_store_refuses_implicit_or_nonordinal_raw_source_identity(tmp_path) -> None:
    source = tmp_path / "raw.zarr"
    _write_v18(source, tile_ids=[0, 2])
    with pytest.raises(RuntimeError, match="contiguous ordinal"):
        build_numeric_store(
            selection_store=source,
            v18_stores=[source],
            output=tmp_path / "bad.zarr",
            selection_mode=RAW_V18_IDENTITY_SELECTION_MODE,
        )
    with pytest.raises(RuntimeError, match="explicit v18_row"):
        build_numeric_store(
            selection_store=source,
            v18_stores=[source],
            output=tmp_path / "implicit.zarr",
            selection_mode=CURATED_SELECTION_MODE,
        )


def test_numeric_store_refuses_legacy_precise_mask_as_a_strict_target_substitute(tmp_path) -> None:
    source = tmp_path / "legacy.zarr"
    _write_v18(source)
    group = zarr.open_group(str(source), mode="a")
    del group["object_geometry_visible_mask"]
    group.create_array("object_precise_mask", data=np.zeros((2, 257, 257), dtype=np.float32))
    with pytest.raises(RuntimeError, match="object_geometry_visible_mask"):
        build_numeric_store(
            selection_store=source,
            v18_stores=[source],
            output=tmp_path / "should-not-build.zarr",
            selection_mode=RAW_V18_IDENTITY_SELECTION_MODE,
        )


def test_numeric_store_requires_the_configured_object_geometry_target_version(tmp_path) -> None:
    source = tmp_path / "wrong-version.zarr"
    rows = _write_v18(source)
    rows[0]["object_geometry_target_version"] = "v1"
    pq.write_table(pa.Table.from_pylist(rows), source / "index.parquet")
    with pytest.raises(RuntimeError, match="target version"):
        build_numeric_store(
            selection_store=source,
            v18_stores=[source],
            output=tmp_path / "should-not-build-v1.zarr",
            selection_mode=RAW_V18_IDENTITY_SELECTION_MODE,
        )


def test_raw_identity_copy_resumes_only_from_a_hash_bound_partial_output(tmp_path) -> None:
    source = tmp_path / "raw.zarr"
    _write_v18(source)
    output = tmp_path / "numeric.zarr"
    build_numeric_store(
        selection_store=source,
        v18_stores=[source],
        output=output,
        selection_mode=RAW_V18_IDENTITY_SELECTION_MODE,
        max_rows=1,
    )
    progress = json.loads((output / "numeric_build_progress.json").read_text(encoding="utf-8"))
    assert progress["next_row"] == 1
    assert not (output / "contract.json").exists()
    build_numeric_store(
        selection_store=source,
        v18_stores=[source],
        output=output,
        selection_mode=RAW_V18_IDENTITY_SELECTION_MODE,
        resume=True,
        max_rows=1,
    )
    assert (output / "contract.json").exists()
    assert json.loads((output / "numeric_build_progress.json").read_text(encoding="utf-8"))["complete"]
