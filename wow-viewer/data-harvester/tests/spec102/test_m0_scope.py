from __future__ import annotations

import pytest

from harvester.spec102.m0_scope import (
    ALPHA_EXCLUSION_REASON,
    M0_BUILD_LOCAL_SCHEMA,
    build_m0_build_local_manifest,
    validate_m0_build_local_scope,
)
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION


def _index() -> list[dict]:
    return [
        {"build": "0_5_3_3368", "map": "Azeroth", "tile_id": 0, "tile_x": 0, "tile_y": 0},
        *_strict_rows(),
    ]


def _strict_rows() -> list[dict]:
    return [
        {
            "build": "3_3_5_12340", "map": map_name, "tile_id": tile_id,
            "tile_x": tile_id, "tile_y": 0,
            "strict_target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "strict_target_materialized": True,
            "strict_target_status": "CompleteVisible",
            "strict_target_arrays_present": True,
            "strict_target_geometry_unresolved_placement_count": 0,
            "strict_target_fallback_required_placement_count": 0,
            "strict_target_terrain_unknown_pixel_count": 0,
            "strict_target_liquid_evidence_status": "Dry",
            "strict_target_liquid_covered_pixel_count": 0,
            "strict_target_liquid_surface_unknown_pixel_count": 0,
            "strict_target_liquid_covered_fragment_count": 0,
            "strict_target_liquid_hidden_fragment_count": 0,
            "strict_target_liquid_above_surface_fragment_count": 0,
            "strict_target_liquid_unknown_fragment_count": 0,
            "strict_target_fragment_trace_schema": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "strict_target_fragment_count": 1,
            "strict_target_fragment_sha256": "a" * 64,
            "strict_target_assets_json": "[]",
            "strict_target_unresolved_placements_json": "[]",
            "strict_target_fragment_trace_arrays_present": True,
            "strict_target_fragment_trace_validated": True,
            "strict_target_fragment_trace_sidecar_start": ordinal,
            "strict_target_fragment_trace_sidecar_end": ordinal + 1,
            "strict_target_fragment_trace_source_sidecar_sha256": "b" * 64,
        }
        for ordinal, (tile_id, map_name) in enumerate(
            ((1, "Azeroth"), (2, "Azeroth"), (3, "Northrend"), (4, "Kalimdor"))
        )
    ]


def _source_manifest() -> dict:
    rows = []
    for row, metadata in enumerate(_index()):
        rows.append({"row": row, **metadata, "split": "train", "eligible_m0": True, "rejection_reasons": []})
    return {"schema": "spec102-curated-split-v5", "rows": rows}


def test_build_local_scope_keeps_every_row_and_excludes_alpha() -> None:
    manifest = build_m0_build_local_manifest(
        _source_manifest(), source_index=_index(), validation_map="Northrend", test_map="Kalimdor",
        source_manifest_path="source.json", source_manifest_sha256="a", source_store="store", source_index_sha256="b",
    )
    scope = validate_m0_build_local_scope(manifest, source_index=_index())
    assert manifest["schema"] == M0_BUILD_LOCAL_SCHEMA
    assert scope.rows_by_split == {"train": [1, 2], "validation_map": [3], "test_build_local": [4]}
    assert manifest["rows"][0]["split"] == "excluded_build"
    assert manifest["rows"][0]["eligible_m0"] is False
    assert ALPHA_EXCLUSION_REASON in manifest["rows"][0]["rejection_reasons"]
    assert scope.audit_binding["validation_map"] == "Northrend"
    assert scope.audit_binding["test_map"] == "Kalimdor"


def test_build_local_scope_rejects_map_split_leakage() -> None:
    manifest = build_m0_build_local_manifest(
        _source_manifest(), source_index=_index(), validation_map="Northrend", test_map="Kalimdor",
        source_manifest_path="source.json", source_manifest_sha256="a", source_store="store", source_index_sha256="b",
    )
    manifest["rows"][2]["split"] = "validation_map"
    with pytest.raises(RuntimeError, match="crosses scoped M0 splits"):
        validate_m0_build_local_scope(manifest, source_index=_index())


def test_build_local_scope_rejects_unmaterialized_strict_target() -> None:
    index = _index()
    index[3]["strict_target_materialized"] = False
    with pytest.raises(RuntimeError, match="no materialized strict geometry target"):
        build_m0_build_local_manifest(
            _source_manifest(), source_index=index, validation_map="Northrend", test_map="Kalimdor",
            source_manifest_path="source.json", source_manifest_sha256="a", source_store="store", source_index_sha256="b",
        )


def test_build_local_scope_rejects_incomplete_or_unknown_terrain_strict_target() -> None:
    index = _index()
    index[3]["strict_target_status"] = "TerrainSurfaceUnavailable"
    index[3]["strict_target_terrain_unknown_pixel_count"] = 1
    with pytest.raises(RuntimeError, match="incomplete strict geometry-target status"):
        build_m0_build_local_manifest(
            _source_manifest(), source_index=index, validation_map="Northrend", test_map="Kalimdor",
            source_manifest_path="source.json", source_manifest_sha256="a", source_store="store", source_index_sha256="b",
        )


def test_build_local_scope_rejects_non_dry_liquid_evidence() -> None:
    index = _index()
    index[3]["strict_target_liquid_evidence_status"] = "LiquidCovered"
    index[3]["strict_target_liquid_covered_pixel_count"] = 1
    with pytest.raises(RuntimeError, match="Dry liquid-evidence status"):
        build_m0_build_local_manifest(
            _source_manifest(), source_index=index, validation_map="Northrend", test_map="Kalimdor",
            source_manifest_path="source.json", source_manifest_sha256="a", source_store="store", source_index_sha256="b",
        )
