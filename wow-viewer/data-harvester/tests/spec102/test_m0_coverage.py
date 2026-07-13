from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from harvester.spec102.m0_coverage import (
    build_m0_coverage_report,
    parse_discovery_inventory,
    sha256_file,
    validate_m0_coverage_audit,
)
from harvester.spec102.m0_scope import M0_BUILD_LOCAL_SCHEMA, STRICT_LIQUID_EVIDENCE_RULE
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION

SCOPE = {
    "schema": M0_BUILD_LOCAL_SCHEMA,
    "kind": "build_local",
    "allowed_builds": ["3_3_5_12340"],
    "required_splits": ["train", "validation_map", "test_build_local"],
    "cross_era_claim": False,
    "target_quality_basis": "strict_transformed_geometry_terrain_visible",
    "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
    "terrain_visibility_proof": "per_fragment_transformed_geometry_vs_raw_mcvt_z",
    "liquid_evidence_rule": STRICT_LIQUID_EVIDENCE_RULE,
    "validation_map": "Northrend",
    "test_map": "Kalimdor",
}


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fragment_trace_fields(ordinal: int) -> dict:
    """Numeric-store fragment-trace provenance for one strict v3 target row.

    Curation and numeric rows must agree on every field, so both are built from
    this single source; offsets are per-row cumulative so each row is coherent.
    """
    return {
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


def _row(ordinal: int, map_name: str) -> dict:
    return {
        "tile_id": ordinal,
        "build": "3_3_5_12340",
        "map": map_name,
        "tile_x": ordinal,
        "tile_y": 0,
    }


def _fixture(tmp_path):
    raw = tmp_path / "raw.zarr"
    numeric = tmp_path / "numeric.zarr"
    raw.mkdir()
    numeric.mkdir()
    rows = [_row(0, "Azeroth"), _row(1, "Northrend"), _row(2, "Kalimdor")]
    for row in rows:
        row.update({
            "object_geometry_target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "object_geometry_target_status": "CompleteVisible",
            "object_geometry_target_materialized": True,
            "object_geometry_target_arrays_present": True,
            "object_geometry_target_geometry_unresolved_placement_count": 0,
            "object_geometry_target_fallback_required_placement_count": 0,
            "object_geometry_target_terrain_unknown_pixel_count": 0,
            "object_geometry_target_liquid_evidence_status": "Dry",
            "object_geometry_target_liquid_covered_pixel_count": 0,
            "object_geometry_target_liquid_surface_unknown_pixel_count": 0,
            "object_geometry_target_liquid_covered_fragment_count": 0,
            "object_geometry_target_liquid_hidden_fragment_count": 0,
            "object_geometry_target_liquid_above_surface_fragment_count": 0,
            "object_geometry_target_liquid_unknown_fragment_count": 0,
        })
    pq.write_table(pa.Table.from_pylist(rows), raw / "index.parquet")
    for name, value in {
        "harvest_metrics.json": {"rejected_missing_required_tiles": 0},
        "_resume_state.json": {"finalized": True},
        "finalization.json": {"finalized": True, "valid_tiles": 3, "rejected_tile_count": 0},
        "signal_validation.json": {"passed": False, "failures": ["roof-only issue"]},
        "zarr.json": {"zarr_format": 3},
    }.items():
        _write_json(raw / name, value)
    numeric_rows = [
        {
            **row,
            "row": ordinal,
            "source_v18_row": ordinal,
            "source_v18_tile_id": ordinal,
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
            **_fragment_trace_fields(ordinal),
        }
        for ordinal, row in enumerate(rows)
    ]
    pq.write_table(pa.Table.from_pylist(numeric_rows), numeric / "index.parquet")
    _write_json(numeric / "contract.json", {
        "schema": "spec102-numeric-store-v3",
        "selection_mode": "raw_v18_identity",
        "source_selection_store": str(raw.resolve()),
    })
    curation_path = tmp_path / "curation.json"
    curation_rows = [
        {
            **row,
            "row": ordinal,
            "source_v18_row": ordinal,
            "source_v18_tile_id": ordinal,
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
            **_fragment_trace_fields(ordinal),
            "liquid_coverage": 0.0,
            "eligible_m0": True,
            "rejection_reasons": [],
        }
        for ordinal, row in enumerate(rows)
    ]
    _write_json(curation_path, {
        "schema": "spec102-m0-full-3_3_5-curation-v4",
        "max_liquid_coverage": 0.0,
        "m0_liquid_policy": "dry_only_no_per_pixel_valid_loss_mask",
        "store_index_sha256": sha256_file(numeric / "index.parquet"),
        "raw_v18_index_sha256": sha256_file(raw / "index.parquet"),
        "raw_v18_fragment_trace_sidecar_sha256": None,
        "rows": curation_rows,
    })
    split_path = tmp_path / "split.json"
    split_rows = [
        {**row, "row": ordinal, "eligible_m0": True, "split": split}
        for ordinal, (row, split) in enumerate(
            zip(rows, ("train", "validation_map", "test_build_local"), strict=True)
        )
    ]
    _write_json(split_path, {
        "schema": M0_BUILD_LOCAL_SCHEMA,
        "m0_training_scope": SCOPE,
        "source_manifest_sha256": sha256_file(curation_path),
        "rows": split_rows,
    })
    inventory = [
        {"map": row["map"], "include": True, "hasReadableTile": True, "tilesWithData": 1}
        for row in rows
    ]
    return raw, numeric, curation_path, split_path, inventory


def test_coverage_requires_full_identity_provenance_and_binds_source_files(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert report["safe_for_requested_3_3_5_m0_training"]
    report_path = tmp_path / "coverage.json"
    _write_json(report_path, report)
    assert validate_m0_coverage_audit(
        report_path, raw_v18_store=raw, store=numeric, split_manifest=split, expected_scope=SCOPE,
    )["coverage"]["full_row_identity_coverage"]
    (raw / "harvest_metrics.json").write_text("{}", encoding="utf-8")
    try:
        validate_m0_coverage_audit(
            report_path, raw_v18_store=raw, store=numeric, split_manifest=split, expected_scope=SCOPE,
        )
    except RuntimeError as error:
        assert "harvest_metrics" in str(error)
    else:
        raise AssertionError("coverage validator accepted a changed raw V18 source")


def test_coverage_rejects_numeric_map_or_row_omission(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    rows = pq.read_table(numeric / "index.parquet").to_pylist()[:-1]
    pq.write_table(pa.Table.from_pylist(rows), numeric / "index.parquet")
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert not report["safe_for_requested_3_3_5_m0_training"]
    assert any("numeric corpus" in failure for failure in report["hard_failures"])


def test_discovery_parser_tolerates_trailing_cli_log_but_not_non_json() -> None:
    parsed = parse_discovery_inventory('[{"map":"Azeroth","include":true}]\nLoaded auxiliary table')
    assert parsed[0]["map"] == "Azeroth"


def test_coverage_hard_fails_when_any_readable_requested_map_is_not_tensor_pack_ready(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    inventory.append({
        "map": "TrialOfTheChampion", "include": False, "hasReadableTile": True,
        "tilesWithData": 1, "reason": "missing tensor-pack signal",
    })
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert not report["safe_for_requested_3_3_5_m0_training"]
    assert any("readable but not tensor-pack ready" in failure for failure in report["hard_failures"])


def test_coverage_hard_fails_when_a_requested_map_has_no_coherent_m0_rows(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    curation_payload = json.loads(curation.read_text(encoding="utf-8"))
    curation_payload["rows"][2]["eligible_m0"] = False
    curation_payload["rows"][2]["rejection_reasons"] = ["terrain_occluded_by_liquid"]
    _write_json(curation, curation_payload)
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    split_payload["rows"][2]["eligible_m0"] = False
    split_payload["source_manifest_sha256"] = sha256_file(curation)
    _write_json(split, split_payload)
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert not report["safe_for_requested_3_3_5_m0_training"]
    assert report["coverage"]["requested_maps_without_eligible_m0"] == ["Kalimdor"]
    assert any("zero strict/liquid-coherent eligible M0 rows" in failure for failure in report["hard_failures"])


def test_coverage_hard_fails_when_raw_harvest_rejected_required_tiles(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    _write_json(raw / "harvest_metrics.json", {"rejected_missing_required_tiles": 1})
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert not report["safe_for_requested_3_3_5_m0_training"]
    assert any("missing required signals" in failure for failure in report["hard_failures"])


def test_coverage_rejects_a_nonzero_initial_m0_liquid_ceiling(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    payload = json.loads(curation.read_text(encoding="utf-8"))
    payload["max_liquid_coverage"] = 0.20
    _write_json(curation, payload)
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    split_payload["source_manifest_sha256"] = sha256_file(curation)
    _write_json(split, split_payload)
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert not report["safe_for_requested_3_3_5_m0_training"]
    assert any("dry-only M0 liquid policy" in failure for failure in report["hard_failures"])


def test_coverage_rejects_an_eligible_row_with_any_liquid_coverage(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    payload = json.loads(curation.read_text(encoding="utf-8"))
    payload["rows"][0]["liquid_coverage"] = 1.0 / (256 * 256)
    _write_json(curation, payload)
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    split_payload["source_manifest_sha256"] = sha256_file(curation)
    _write_json(split, split_payload)
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert not report["safe_for_requested_3_3_5_m0_training"]
    assert any("liquid-covered row" in failure for failure in report["hard_failures"])


def test_coverage_rejects_an_eligible_row_with_non_dry_strict_target_evidence(tmp_path) -> None:
    raw, numeric, curation, split, inventory = _fixture(tmp_path)
    raw_rows = pq.read_table(raw / "index.parquet").to_pylist()
    raw_rows[0]["object_geometry_target_liquid_evidence_status"] = "LiquidCovered"
    raw_rows[0]["object_geometry_target_liquid_covered_fragment_count"] = 1
    pq.write_table(pa.Table.from_pylist(raw_rows), raw / "index.parquet")
    numeric_rows = pq.read_table(numeric / "index.parquet").to_pylist()
    numeric_rows[0]["strict_target_liquid_evidence_status"] = "LiquidCovered"
    numeric_rows[0]["strict_target_liquid_covered_fragment_count"] = 1
    pq.write_table(pa.Table.from_pylist(numeric_rows), numeric / "index.parquet")
    payload = json.loads(curation.read_text(encoding="utf-8"))
    payload["raw_v18_index_sha256"] = sha256_file(raw / "index.parquet")
    payload["store_index_sha256"] = sha256_file(numeric / "index.parquet")
    payload["rows"][0]["strict_target_liquid_evidence_status"] = "LiquidCovered"
    payload["rows"][0]["strict_target_liquid_covered_fragment_count"] = 1
    _write_json(curation, payload)
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    split_payload["source_manifest_sha256"] = sha256_file(curation)
    _write_json(split, split_payload)
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=tmp_path,
        raw_v18_store=raw,
        numeric_store=numeric,
        curation_manifest=curation,
        split_manifest=split,
        expected_scope=SCOPE,
    )
    assert not report["safe_for_requested_3_3_5_m0_training"]
    assert any("non-Dry liquid evidence" in failure for failure in report["hard_failures"])
