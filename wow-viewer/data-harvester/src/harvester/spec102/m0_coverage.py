"""Full-corpus provenance gate for the 3.3.5-only Spec 102 M0 decision."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from harvester.spec102.curation import (
    DEFAULT_M0_MAX_LIQUID_COVERAGE,
    STRICT_LIQUID_EVIDENCE_DRY,
    STRICT_TARGET_FRAGMENT_TRACE_FIELDS,
    STRICT_TARGET_LIQUID_COUNTER_FIELDS,
)
from harvester.spec102.m0_scope import (
    M0_ALLOWED_BUILD,
    STRICT_FRAGMENT_TRACE_PROOF,
    STRICT_LIQUID_EVIDENCE_RULE,
)
from harvester.spec102.strict_target_contract import (
    REQUIRED_STRICT_OBJECT_TARGET_VERSION,
    STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY,
    sha256_tree,
)

M0_COVERAGE_AUDIT_SCHEMA = "spec102-m0-coverage-audit-v4"
M0_FULL_CURATION_SCHEMA = "spec102-m0-full-3_3_5-curation-v4"
IDENTITY_FIELDS = ("build", "map", "tile_id", "tile_x", "tile_y")
RAW_BINDING_FILES = (
    "index.parquet",
    "harvest_metrics.json",
    "_resume_state.json",
    "finalization.json",
    "signal_validation.json",
    "zarr.json",
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_discovery_inventory(raw_output: str) -> list[dict[str, Any]]:
    """Extract the CLI's JSON array without accepting arbitrary trailing logs."""
    start = raw_output.find("[")
    end = raw_output.rfind("]")
    if start < 0 or end < start:
        raise RuntimeError("discover-maps did not emit a JSON array")
    try:
        inventory = json.loads(raw_output[start:end + 1])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"discover-maps emitted invalid JSON: {error}") from error
    if not isinstance(inventory, list) or not all(isinstance(item, dict) for item in inventory):
        raise RuntimeError("discover-maps JSON must be an array of map objects")
    return inventory


def _identity(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    try:
        return (
            str(row["build"]), str(row["map"]), int(row["tile_id"]),
            int(row["tile_x"]), int(row["tile_y"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"row has invalid map/tile identity: {row}") from error


def _identity_sets(rows: list[dict[str, Any]], *, label: str, failures: list[str]) -> dict[tuple[str, str, int, int, int], int]:
    identities: dict[tuple[str, str, int, int, int], int] = {}
    for ordinal, row in enumerate(rows):
        try:
            identity = _identity(row)
        except RuntimeError as error:
            failures.append(f"{label} {ordinal}: {error}")
            continue
        if identity in identities:
            failures.append(f"{label} contains duplicate identity {identity}")
            continue
        identities[identity] = ordinal
    return identities


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _binding_files(store: Path, names: tuple[str, ...]) -> dict[str, str | None]:
    return {name: sha256_file(store / name) if (store / name).is_file() else None for name in names}


def _scope_contains(actual: Any, expected: dict[str, Any]) -> bool:
    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def _map_summary(
    raw_rows: list[dict[str, Any]],
    numeric_rows: list[dict[str, Any]],
    curation_rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for _name, rows, key in (
        ("raw_rows", raw_rows, "raw_rows"),
        ("numeric_rows", numeric_rows, "numeric_rows"),
    ):
        for row in rows:
            map_name = str(row["map"])
            result.setdefault(map_name, {"raw_rows": 0, "numeric_rows": 0, "eligible_m0": 0, "rejected_m0": 0})
            result[map_name][key] += 1
    for row in curation_rows:
        map_name = str(row["map"])
        result.setdefault(map_name, {"raw_rows": 0, "numeric_rows": 0, "eligible_m0": 0, "rejected_m0": 0})
        result[map_name]["eligible_m0" if row.get("eligible_m0") is True else "rejected_m0"] += 1
    return {name: result[name] for name in sorted(result)}


def build_m0_coverage_report(
    *,
    inventory: list[dict[str, Any]],
    client_root: Path,
    raw_v18_store: Path,
    numeric_store: Path,
    curation_manifest: Path,
    split_manifest: Path,
    expected_scope: dict[str, Any],
) -> dict[str, Any]:
    """Return a fail-closed report; caller owns writing it after inspection."""
    failures: list[str] = []
    raw_index = pq.read_table(raw_v18_store / "index.parquet").to_pylist()
    numeric_index = pq.read_table(numeric_store / "index.parquet").to_pylist()
    curation = _read_json(curation_manifest)
    split = _read_json(split_manifest)
    raw_metrics = _read_json(raw_v18_store / "harvest_metrics.json")
    raw_finalization = _read_json(raw_v18_store / "finalization.json")
    raw_signal_validation = _read_json(raw_v18_store / "signal_validation.json")
    numeric_contract = _read_json(numeric_store / "contract.json")
    # Bind the raw V18 fragment-trace sidecar for staleness detection.  Deep
    # per-row trace validation happens once, upstream, in the numeric-store
    # build; this report re-binds its hash so the validator can prove the raw
    # source has not changed.  A missing sidecar hashes to None here and in the
    # validator, keeping the two symmetric.
    try:
        raw_fragment_trace_sha256: str | None = sha256_tree(
            raw_v18_store / STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY
        )
    except Exception:
        raw_fragment_trace_sha256 = None

    terrain_inventory = [item for item in inventory if item.get("include") is True]
    readable_not_ready = [
        item for item in inventory
        if item.get("hasReadableTile") is True and item.get("include") is not True
    ]
    # `include` says what the current harvest path can tensor-pack.  It does
    # not authorize a smaller corpus: every readable 3.3.5 terrain map is in
    # scope for this requested model and must be accounted for before CUDA.
    requested_inventory = [
        item for item in inventory
        if item.get("hasReadableTile") is True or item.get("include") is True
    ]
    inventory_maps = {str(item.get("map")) for item in terrain_inventory}
    requested_maps = {str(item.get("map")) for item in requested_inventory}
    raw_maps = {str(row.get("map")) for row in raw_index}
    numeric_maps = {str(row.get("map")) for row in numeric_index}
    if {str(row.get("build")) for row in raw_index} != {M0_ALLOWED_BUILD}:
        failures.append("raw V18 source is not exactly the required 3.3.5 build")
    if {str(row.get("build")) for row in numeric_index} != {M0_ALLOWED_BUILD}:
        failures.append("numeric M0 corpus is not exactly the required 3.3.5 build")
    if numeric_contract.get("selection_mode") != "raw_v18_identity":
        failures.append("numeric corpus was not constructed through raw_v18_identity mode")
    if Path(numeric_contract.get("source_selection_store", "")).resolve() != raw_v18_store.resolve():
        failures.append("numeric corpus is not bound to the supplied raw V18 source")
    if readable_not_ready:
        gaps = ", ".join(sorted(str(item.get("map")) for item in readable_not_ready))
        failures.append("requested 3.3.5 maps are readable but not tensor-pack ready: " + gaps)
    if raw_maps != requested_maps:
        failures.append("raw V18 map identities do not exactly cover every requested readable 3.3.5 map")
    if numeric_maps != raw_maps:
        failures.append("numeric corpus map identities do not exactly cover raw V18 maps")
    if curation.get("schema") != M0_FULL_CURATION_SCHEMA:
        failures.append("curation manifest is not the full raw-3.3.5 M0 curation schema")
    if curation.get("max_liquid_coverage") != DEFAULT_M0_MAX_LIQUID_COVERAGE:
        failures.append("curation manifest does not enforce the initial dry-only M0 liquid policy")
    if curation.get("m0_liquid_policy") != "dry_only_no_per_pixel_valid_loss_mask":
        failures.append("curation manifest does not declare the initial dry-only M0 liquid policy")
    if curation.get("store_index_sha256") != sha256_file(numeric_store / "index.parquet"):
        failures.append("curation manifest is bound to a different numeric index")
    if curation.get("raw_v18_index_sha256") != sha256_file(raw_v18_store / "index.parquet"):
        failures.append("curation manifest is bound to a different raw V18 index")
    if curation.get("raw_v18_fragment_trace_sidecar_sha256") != raw_fragment_trace_sha256:
        failures.append("curation manifest is bound to a different raw V18 fragment-trace sidecar")
    split_scope = dict(split.get("m0_training_scope") or {})
    split_scope["schema"] = split.get("schema")
    if not _scope_contains(split_scope, expected_scope):
        failures.append("split manifest M0 scope differs from the expected build-local scope")
    if split.get("source_manifest_sha256") != sha256_file(curation_manifest):
        failures.append("split manifest is not bound to this full-corpus curation manifest")

    raw_identities = _identity_sets(raw_index, label="raw V18", failures=failures)
    numeric_identities = _identity_sets(numeric_index, label="numeric", failures=failures)
    curation_rows = curation.get("rows") if isinstance(curation.get("rows"), list) else []
    if not isinstance(curation.get("rows"), list):
        failures.append("curation manifest has no rows list")
    curation_identities = _identity_sets(curation_rows, label="curation", failures=failures)
    split_rows = split.get("rows") if isinstance(split.get("rows"), list) else []
    if not isinstance(split.get("rows"), list):
        failures.append("split manifest has no rows list")
    split_identities = _identity_sets(split_rows, label="split", failures=failures)
    if set(numeric_identities) != set(raw_identities):
        failures.append("numeric corpus does not contain every raw V18 row exactly once")
    if set(curation_identities) != set(raw_identities):
        failures.append("curation manifest does not account for every raw V18 row exactly once")
    if set(split_identities) != set(raw_identities):
        failures.append("split manifest does not account for every raw V18 row exactly once")

    curation_by_row: dict[int, dict[str, Any]] = {}
    for row in curation_rows:
        try:
            ordinal = int(row["row"])
            source_row = int(row["source_v18_row"])
        except (KeyError, TypeError, ValueError):
            failures.append("curation row lacks numeric/source V18 row identity")
            continue
        if ordinal in curation_by_row:
            failures.append(f"curation has duplicate numeric row {ordinal}")
            continue
        curation_by_row[ordinal] = row
        if ordinal != source_row:
            failures.append(f"curation row {ordinal} is not a direct source-row identity copy")
    split_by_row: dict[int, dict[str, Any]] = {}
    for row in split_rows:
        try:
            ordinal = int(row["row"])
        except (KeyError, TypeError, ValueError):
            failures.append("split row lacks numeric row identity")
            continue
        if ordinal in split_by_row:
            failures.append(f"split has duplicate numeric row {ordinal}")
            continue
        split_by_row[ordinal] = row
    if set(curation_by_row) != set(range(len(raw_index))):
        failures.append("curation row IDs do not exactly cover raw V18 ordinal rows")
    if set(split_by_row) != set(range(len(raw_index))):
        failures.append("split row IDs do not exactly cover raw V18 ordinal rows")

    m0_eligible_by_split: dict[str, int] = {"train": 0, "validation_map": 0, "test_build_local": 0}
    for ordinal, raw in enumerate(raw_index):
        numeric = numeric_index[ordinal] if ordinal < len(numeric_index) else None
        curated = curation_by_row.get(ordinal)
        split_row = split_by_row.get(ordinal)
        if numeric is None or curated is None or split_row is None:
            continue
        for label, row in (("numeric", numeric), ("curation", curated), ("split", split_row)):
            try:
                if _identity(row) != _identity(raw):
                    failures.append(f"{label} row {ordinal} identity differs from raw V18")
            except RuntimeError as error:
                failures.append(f"{label} row {ordinal}: {error}")
        materialized = bool(numeric.get("strict_target_materialized", False))
        if bool(curated.get("strict_target_materialized", False)) != materialized:
            failures.append(f"curation row {ordinal} strict-target materialization differs from numeric store")
        for field in (
            "strict_target_version",
            "strict_target_status",
            "strict_target_arrays_present",
            "strict_target_geometry_unresolved_placement_count",
            "strict_target_fallback_required_placement_count",
            "strict_target_terrain_unknown_pixel_count",
            "strict_target_liquid_evidence_status",
            *STRICT_TARGET_LIQUID_COUNTER_FIELDS,
            *STRICT_TARGET_FRAGMENT_TRACE_FIELDS,
        ):
            if curated.get(field) != numeric.get(field):
                failures.append(f"curation row {ordinal} strict-target {field} differs from numeric store")
        if curated.get("eligible_m0") is True:
            if numeric.get("strict_target_version") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
                failures.append(
                    f"curation admitted non-{REQUIRED_STRICT_OBJECT_TARGET_VERSION} strict target "
                    f"at raw V18 row {ordinal}"
                )
            if not materialized:
                failures.append(f"curation admitted unmaterialized strict target at raw V18 row {ordinal}")
            if str(numeric.get("strict_target_status", "")) not in {"CompleteEmpty", "CompleteVisible"}:
                failures.append(f"curation admitted incomplete strict target at raw V18 row {ordinal}")
            if numeric.get("strict_target_arrays_present") is not True:
                failures.append(f"curation admitted strict target without all required arrays at raw V18 row {ordinal}")
            if numeric.get("strict_target_liquid_evidence_status") != STRICT_LIQUID_EVIDENCE_DRY:
                failures.append(f"curation admitted non-Dry liquid evidence at raw V18 row {ordinal}")
            if numeric.get("strict_target_fragment_trace_schema") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
                failures.append(f"curation admitted wrong fragment-trace schema at raw V18 row {ordinal}")
            if numeric.get("strict_target_fragment_trace_arrays_present") is not True:
                failures.append(f"curation admitted target without the nine fragment-trace arrays at raw V18 row {ordinal}")
            if numeric.get("strict_target_fragment_trace_validated") is not True:
                failures.append(f"curation admitted unvalidated fragment trace at raw V18 row {ordinal}")
            try:
                trace_count = int(numeric.get("strict_target_fragment_count", -1))
                trace_start = int(numeric.get("strict_target_fragment_trace_sidecar_start", -1))
                trace_end = int(numeric.get("strict_target_fragment_trace_sidecar_end", -1))
            except (TypeError, ValueError):
                trace_count = trace_start = trace_end = -1
            if trace_count < 0 or trace_start < 0 or trace_end - trace_start != trace_count:
                failures.append(f"curation admitted incoherent fragment trace at raw V18 row {ordinal}")
            for field in (
                "strict_target_geometry_unresolved_placement_count",
                "strict_target_fallback_required_placement_count",
                "strict_target_terrain_unknown_pixel_count",
            ):
                try:
                    nonzero = int(numeric.get(field, 0) or 0) != 0
                except (TypeError, ValueError):
                    nonzero = True
                if nonzero:
                    failures.append(
                        f"curation admitted strict target with nonzero {field} at raw V18 row {ordinal}"
                    )
            for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS:
                try:
                    nonzero = int(numeric.get(field, 0) or 0) != 0
                except (TypeError, ValueError):
                    nonzero = True
                if nonzero:
                    failures.append(
                        f"curation admitted liquid-evidence row with nonzero {field} at raw V18 row {ordinal}"
                    )
            try:
                liquid_coverage = float(curated["liquid_coverage"])
            except (KeyError, TypeError, ValueError):
                liquid_coverage = math.nan
            if not math.isfinite(liquid_coverage) or liquid_coverage != 0.0:
                failures.append(
                    f"curation admitted liquid-covered row {ordinal} into dry-only initial M0"
                )
            if split_row.get("eligible_m0") is not True:
                failures.append(f"split dropped M0 eligibility at raw V18 row {ordinal}")
            split_name = str(split_row.get("split"))
            if split_name not in m0_eligible_by_split:
                failures.append(f"eligible M0 row {ordinal} has invalid split {split_name}")
            else:
                m0_eligible_by_split[split_name] += 1
        elif split_row.get("eligible_m0") is True:
            failures.append(f"split promoted rejected M0 row {ordinal}")
    if any(count <= 0 for count in m0_eligible_by_split.values()):
        failures.append("full M0 coverage has no eligible row in one required build-local split")

    map_summary = _map_summary(raw_index, numeric_index, curation_rows)
    requested_maps_without_eligible_m0 = sorted(
        map_name
        for map_name in requested_maps
        if map_summary.get(map_name, {}).get("eligible_m0", 0) <= 0
    )
    if requested_maps_without_eligible_m0:
        failures.append(
            "requested 3.3.5 map(s) have zero strict/liquid-coherent eligible M0 rows: "
            + ", ".join(requested_maps_without_eligible_m0)
        )
    terrain_tiles = sum(int(item.get("tilesWithData") or 0) for item in terrain_inventory)
    requested_terrain_tiles = sum(int(item.get("tilesWithData") or 0) for item in requested_inventory)
    missing_required_tiles = int(raw_metrics.get("rejected_missing_required_tiles") or 0)
    raw_location_gap = requested_terrain_tiles - len(raw_index)
    if missing_required_tiles:
        failures.append(
            f"raw V18 rejected {missing_required_tiles} requested tile(s) for missing required signals"
        )
    if raw_location_gap != 0:
        failures.append(
            "raw V18 row count does not equal requested readable WDT locations "
            f"({len(raw_index)} != {requested_terrain_tiles})"
        )
    report = {
        "schema": M0_COVERAGE_AUDIT_SCHEMA,
        "build": M0_ALLOWED_BUILD,
        "client_root": str(client_root.resolve()),
        "inventory_sha256": canonical_json_sha256(inventory),
        "inventory": {
            "map_dbc_records": len(inventory),
            "requested_readable_maps": sorted(requested_maps),
            "requested_readable_map_count": len(requested_maps),
            "requested_readable_wdt_locations": requested_terrain_tiles,
            "terrain_ready_maps": sorted(inventory_maps),
            "terrain_ready_map_count": len(inventory_maps),
            "terrain_ready_wdt_locations": terrain_tiles,
            "readable_not_tensor_pack_ready": [
                {key: item.get(key) for key in ("map", "displayName", "reason", "tilesWithData")}
                for item in readable_not_ready
            ],
        },
        "raw_v18_store": str(raw_v18_store.resolve()),
        "raw_v18_bindings": _binding_files(raw_v18_store, RAW_BINDING_FILES),
        "raw_v18_fragment_trace_sidecar_sha256": raw_fragment_trace_sha256,
        "numeric_store": str(numeric_store.resolve()),
        "numeric_store_contract_sha256": sha256_file(numeric_store / "contract.json"),
        "numeric_store_index_sha256": sha256_file(numeric_store / "index.parquet"),
        "curation_manifest": str(curation_manifest.resolve()),
        "curation_manifest_sha256": sha256_file(curation_manifest),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest),
        "m0_liquid_policy": "dry_only_no_per_pixel_valid_loss_mask",
        "m0_liquid_evidence_rule": STRICT_LIQUID_EVIDENCE_RULE,
        "m0_fragment_trace_proof": STRICT_FRAGMENT_TRACE_PROOF,
        "m0_training_scope": expected_scope,
        "coverage": {
            "raw_v18_rows": len(raw_index),
            "raw_v18_maps": len(raw_maps),
            "numeric_rows": len(numeric_index),
            "numeric_maps": len(numeric_maps),
            "curation_rows": len(curation_rows),
            "split_rows": len(split_rows),
            "full_map_identity_coverage": raw_maps == requested_maps == numeric_maps,
            "full_row_identity_coverage": (
                set(raw_identities) == set(numeric_identities) == set(curation_identities) == set(split_identities)
            ),
            "eligible_m0_by_split": m0_eligible_by_split,
            "map_summary": map_summary,
            "strict_liquid_evidence_status_counts": {
                status: sum(
                    int(row.get("strict_target_liquid_evidence_status") == status)
                    for row in curation_rows
                )
                for status in sorted({str(row.get("strict_target_liquid_evidence_status", "missing")) for row in curation_rows})
            },
            "strict_liquid_non_dry_rows": sum(
                int(row.get("strict_target_liquid_evidence_status") != STRICT_LIQUID_EVIDENCE_DRY)
                for row in curation_rows
            ),
            "requested_maps_without_eligible_m0": requested_maps_without_eligible_m0,
            "source_gaps": {
                "harvest_rejected_missing_required_tiles": missing_required_tiles,
                "terrain_ready_wdt_locations_minus_raw_rows": terrain_tiles - len(raw_index),
                "requested_readable_wdt_locations_minus_raw_rows": raw_location_gap,
                "readable_not_tensor_pack_ready_maps": len(readable_not_ready),
            },
            "raw_finalization": {
                "finalized": raw_finalization.get("finalized"),
                "valid_tiles": raw_finalization.get("valid_tiles"),
                "rejected_tile_count": raw_finalization.get("rejected_tile_count"),
            },
            "raw_global_signal_validation": {
                "passed": raw_signal_validation.get("passed"),
                "failures": raw_signal_validation.get("failures", []),
                "note": "Recorded for provenance; the seven M0 signals receive their own exact audit.",
            },
        },
        "hard_failures": list(dict.fromkeys(failures)),
        "safe_for_requested_3_3_5_m0_training": not failures,
    }
    return report


def validate_m0_coverage_audit(
    report_path: Path,
    *,
    raw_v18_store: Path,
    store: Path,
    split_manifest: Path,
    expected_scope: dict[str, Any],
) -> dict[str, Any]:
    """Reject stale, partial, or source-mismatched coverage before CUDA allocation."""
    report = _read_json(report_path)
    if report.get("schema") != M0_COVERAGE_AUDIT_SCHEMA:
        raise RuntimeError("M0 coverage report has an unsupported schema")
    if report.get("safe_for_requested_3_3_5_m0_training") is not True:
        failure = (report.get("hard_failures") or ["coverage report did not certify the requested corpus"])[0]
        raise RuntimeError(f"M0 blocked by coverage audit: {failure}")
    if report.get("m0_training_scope") != expected_scope:
        raise RuntimeError("M0 coverage report was produced for a different build-local scope")
    if report.get("m0_liquid_policy") != "dry_only_no_per_pixel_valid_loss_mask":
        raise RuntimeError("M0 coverage report does not enforce the initial dry-only liquid policy")
    if report.get("m0_liquid_evidence_rule") != STRICT_LIQUID_EVIDENCE_RULE:
        raise RuntimeError("M0 coverage report does not enforce strict Dry liquid-evidence provenance")
    if report.get("m0_fragment_trace_proof") != STRICT_FRAGMENT_TRACE_PROOF:
        raise RuntimeError("M0 coverage report does not enforce the exact C# v3 fragment-trace proof")
    expected = {
        "raw_v18_store": str(raw_v18_store.resolve()),
        "numeric_store": str(store.resolve()),
        "numeric_store_contract_sha256": sha256_file(store / "contract.json"),
        "numeric_store_index_sha256": sha256_file(store / "index.parquet"),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest),
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    bindings = report.get("raw_v18_bindings")
    if not isinstance(bindings, dict):
        mismatches.append("raw_v18_bindings")
    else:
        for name in RAW_BINDING_FILES:
            current = sha256_file(raw_v18_store / name) if (raw_v18_store / name).is_file() else None
            if bindings.get(name) != current:
                mismatches.append(f"raw_v18_bindings.{name}")
    try:
        current_trace_sha256 = sha256_tree(
            raw_v18_store / STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY
        )
    except Exception:
        current_trace_sha256 = None
    if report.get("raw_v18_fragment_trace_sidecar_sha256") != current_trace_sha256:
        mismatches.append("raw_v18_fragment_trace_sidecar_sha256")
    coverage = report.get("coverage") or {}
    if coverage.get("full_map_identity_coverage") is not True:
        mismatches.append("full_map_identity_coverage")
    if coverage.get("full_row_identity_coverage") is not True:
        mismatches.append("full_row_identity_coverage")
    if coverage.get("requested_maps_without_eligible_m0"):
        mismatches.append("requested_maps_without_eligible_m0")
    inventory = report.get("inventory") or {}
    if inventory.get("readable_not_tensor_pack_ready"):
        mismatches.append("readable_not_tensor_pack_ready")
    source_gaps = coverage.get("source_gaps") or {}
    for field in (
        "harvest_rejected_missing_required_tiles",
        "requested_readable_wdt_locations_minus_raw_rows",
        "readable_not_tensor_pack_ready_maps",
    ):
        try:
            clean = int(source_gaps.get(field, -1)) == 0
        except (TypeError, ValueError):
            clean = False
        if not clean:
            mismatches.append(f"source_gaps.{field}")
    if mismatches:
        raise RuntimeError("M0 coverage report is stale, partial, or bound to different inputs: " + ", ".join(mismatches))
    return report
