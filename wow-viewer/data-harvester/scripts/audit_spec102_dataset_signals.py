"""Fail-closed numeric and visual audit of every signal in a Spec 102 store."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr

from harvester.spec102.curation import (
    STRICT_LIQUID_EVIDENCE_DRY,
    STRICT_TARGET_LIQUID_COUNTER_FIELDS,
)
from harvester.spec102.m0 import PRECISE_MASK_KEY
from harvester.spec102.m0_scope import (
    FUTURE_VISIBILITY_RULE,
    M0_ALLOWED_BUILD,
    M0_BUILD_LOCAL_SCHEMA,
    M0_REQUIRED_SPLITS,
    STRICT_LIQUID_EVIDENCE_RULE,
    STRICT_TARGET_QUALITY_BASIS,
    STRICT_TERRAIN_VISIBILITY_PROOF,
    validate_m0_build_local_scope,
)
from harvester.spec102.numeric_store import (
    SPECS,
    STRICT_LIQUID_COUNTER_FIELDS,
    STRICT_LIQUID_EVIDENCE_STATUS_FIELD,
    STRICT_OBJECT_TARGET_VERSION_FIELD,
    _i8_normals,
    _u8_unit,
)
from harvester.spec102.signal_audit import (
    AUDIT_SCHEMA,
    M0_AUDITED_SIGNAL_KEYS,
    combine_audited_signal_row_fingerprints,
    fingerprint_audited_signal_rows,
    project_placement_to_terrain,
    render_signal_panel,
    sha256_file,
)
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION

SOURCE_KEYS = {
    "minimap_rgb": "minimap_rgb",
    PRECISE_MASK_KEY: "object_geometry_visible_mask",
    "object_geometry_visible_top_elevation_257": "object_geometry_visible_top_elevation",
    "object_geometry_visible_terrain_elevation_257": "object_geometry_visible_terrain_elevation",
    "object_geometry_visible_source_257": "object_geometry_visible_source",
    "liquid_mask_256": "liquid_mask",
    "liquid_height_256": "liquid_height",
    "mcnk_flags_16": "mcnk_flags_16",
    "normal_xyz_257": "normal_xyz",
    "height_257": "height_257",
}
IDENTITY_FIELDS = ("build", "map", "tile_id", "tile_x", "tile_y")


def _add_failure(failures: list[str], message: str) -> None:
    if message not in failures:
        failures.append(message)


def _coerce_source_value(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if name == "liquid_mask_256":
        return _u8_unit(value)
    if name == "normal_xyz_257":
        return _i8_normals(value)
    return value.astype(SPECS[name][0], copy=False)


def _hash_or_fail(path: Path, *, label: str, failures: list[str]) -> str | None:
    if not path.is_file():
        _add_failure(failures, f"missing {label}: {path}")
        return None
    try:
        return sha256_file(path)
    except OSError as error:
        _add_failure(failures, f"cannot hash {label}: {error}")
        return None


def _array_schema(
    array: Any,
    *,
    count: int,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> dict:
    actual_shape = tuple(array.shape)
    expected_shape = (count, *shape)
    actual_dtype = np.dtype(array.dtype)
    expected_dtype = np.dtype(dtype)
    return {
        "shape": list(actual_shape),
        "expected_shape": list(expected_shape),
        "dtype": str(actual_dtype),
        "expected_dtype": str(expected_dtype),
        "shape_ok": actual_shape == expected_shape,
        "dtype_ok": actual_dtype == expected_dtype,
    }


def _raw_source_issues(name: str, value: np.ndarray) -> set[str]:
    """Validate raw V18 values before conversion can clip or coerce them."""
    data = np.asarray(value)
    issues: set[str] = set()
    if np.issubdtype(data.dtype, np.floating):
        finite = np.isfinite(data)
        if not finite.all():
            issues.add("nonfinite")
            data = data[finite]
        else:
            data = data.reshape(-1)
    else:
        data = data.reshape(-1)
    if data.size == 0:
        return issues
    low = float(data.min())
    high = float(data.max())
    if name == PRECISE_MASK_KEY and (low < 0.0 or high > 1.0):
        issues.add("range")
    elif name == "object_geometry_visible_source_257" and (low < 0.0 or high > 2.0):
        issues.add("range")
    elif name == "liquid_mask_256":
        if data.dtype == np.uint8:
            valid = low >= 0.0 and high <= 255.0
        else:
            valid = low >= 0.0 and high <= 1.0
        if not valid:
            issues.add("range")
    elif name == "normal_xyz_257":
        limit = 127.0 if data.dtype == np.int8 else 1.0
        if low < -limit or high > limit:
            issues.add("range")
    elif name == "minimap_rgb" and (low < 0.0 or high > 255.0):
        issues.add("range")
    return issues


def _raw_source_tile_health(name: str, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-tile raw nonfinite and range failures without coercing inputs."""
    data = np.asarray(value)
    flat = data.reshape(data.shape[0], -1)
    if np.issubdtype(data.dtype, np.floating):
        finite = np.isfinite(flat)
        nonfinite = ~finite.all(axis=1)
        low = np.where(finite, flat, np.inf).min(axis=1)
        high = np.where(finite, flat, -np.inf).max(axis=1)
    else:
        nonfinite = np.zeros(data.shape[0], dtype=bool)
        low = flat.min(axis=1)
        high = flat.max(axis=1)
    if name == PRECISE_MASK_KEY:
        range_failure = (low < 0.0) | (high > 1.0)
    elif name == "object_geometry_visible_source_257":
        range_failure = (low < 0.0) | (high > 2.0)
    elif name == "liquid_mask_256":
        maximum = 255.0 if data.dtype == np.uint8 else 1.0
        range_failure = (low < 0.0) | (high > maximum)
    elif name == "normal_xyz_257":
        maximum = 127.0 if data.dtype == np.int8 else 1.0
        range_failure = (low < -maximum) | (high > maximum)
    elif name == "minimap_rgb":
        range_failure = (low < 0.0) | (high > 255.0)
    else:
        range_failure = np.zeros(data.shape[0], dtype=bool)
    return nonfinite, range_failure


def _batch_mismatch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return an exact copy-integrity failure bit for every batch row."""
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
        equal = (left == right) | (np.isnan(left) & np.isnan(right))
    else:
        equal = left == right
    return ~equal.reshape(equal.shape[0], -1).all(axis=1)


def _choose_panel_rows(metrics: list[dict], split: str, count: int) -> list[int]:
    candidates = [item for item in metrics if item["split"] == split and item["eligible_m0"]]
    ordered: list[int] = []
    rankings = (
        sorted(candidates, key=lambda item: item["object_coverage"], reverse=True),
        sorted(candidates, key=lambda item: item["liquid_coverage"], reverse=True),
        sorted(candidates, key=lambda item: item["relief"], reverse=True),
        sorted(candidates, key=lambda item: item["object_coverage"]),
    )
    cursor = 0
    while len(ordered) < min(count, len(candidates)):
        ranking = rankings[cursor % len(rankings)]
        position = cursor // len(rankings)
        if position < len(ranking):
            row = int(ranking[position]["row"])
            if row not in ordered:
                ordered.append(row)
        cursor += 1
    return ordered


def _read_rows(array: Any, rows: list[int]) -> np.ndarray:
    """Read a selected manifest scope without assuming its rows start at zero."""
    if rows[-1] - rows[0] + 1 == len(rows):
        return np.asarray(array[rows[0]:rows[-1] + 1])
    return np.asarray(array.oindex[np.asarray(rows, dtype=np.int64)])


def _write_report(args: argparse.Namespace, report: dict) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path.resolve()),
        "audit_shard": report.get("audit_shard"),
        "safe_for_m0_build_local_training": report.get("safe_for_m0_build_local_training"),
        "shard_clean": report.get("shard_clean"),
        "hard_failures": report.get("hard_failures"),
        "scoped_signal_fingerprint": report.get("scoped_signal_fingerprint"),
    }, indent=2))
    return report_path


def _base_report(args: argparse.Namespace, *, hashes: dict[str, str | None], tile_count: int) -> dict:
    return {
        "schema": AUDIT_SCHEMA,
        "store": str(args.store.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "v18_stores": [str(args.v18_store.resolve())],
        "tile_count": tile_count,
        "store_tile_count": tile_count,
        "selected_builds": [],
        "evaluation_splits": {},
        **hashes,
        "safe_for_m0_training": False,
        "safe_for_m0_build_local_training": False,
        "shard_clean": False,
        "partial_scope": False,
        "hard_failures": [],
        "schema_checks": {},
        "source_schema_checks": {},
        "source_copy_mismatch_tiles": {},
        "source_raw_nonfinite_tiles": {},
        "source_raw_range_failure_tiles": {},
        "nonfinite_tiles": {},
        "range_failure_tiles": {},
        "signal_counts": {},
        "normal_nonzero_mean_length": None,
        "rgb_edge_mean_inside_object_target": None,
        "rgb_edge_mean_outside_object_target": None,
        "placement_terrain_audit": {},
        "placement_warning": (
            "Centroid projection mirrors the placement fallback only. It is diagnostic evidence, "
            "not an instance-pixel mapping; exact partial clipping requires transformed source geometry."
        ),
        "object_target_provenance": {
            "target_array": PRECISE_MASK_KEY,
            "terrain_occlusion_clipped": False,
            "per_pixel_object_top_elevation": False,
            "build_local_strict_target_accepted": False,
            "future_visibility_rule": FUTURE_VISIBILITY_RULE,
            "training_disposition": "not yet audited",
        },
        "panels": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit every Spec 102 dataset signal before training")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument(
        "--v18-store",
        required=True,
        type=Path,
        help="the one trusted 3.3.5 V18 source; Alpha inputs are refused in this build-local audit",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--panel-count", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--include-placement-diagnostic",
        action="store_true",
        help="run the slow centroid/bounds diagnostic; it never changes or certifies mask pixels",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.panel_count < 1:
        raise ValueError("--panel-count must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hard_failures: list[str] = []
    hashes = {
        "store_contract_sha256": _hash_or_fail(args.store / "contract.json", label="store contract", failures=hard_failures),
        "store_index_sha256": _hash_or_fail(args.store / "index.parquet", label="store index", failures=hard_failures),
        "split_manifest_sha256": _hash_or_fail(args.split_manifest, label="split manifest", failures=hard_failures),
    }
    report = _base_report(args, hashes=hashes, tile_count=0)

    group = None
    index: list[dict] = []
    contract: dict = {}
    manifest: dict = {}
    manifest_by_row: dict[int, dict] = {}
    selected_row_numbers: list[int] = []
    evaluation_splits: tuple[str, str, str] = M0_REQUIRED_SPLITS
    audit_scope: dict[str, Any] = {}
    try:
        group = zarr.open_group(str(args.store), mode="r")
    except Exception as error:  # Zarr emits several implementation-specific errors.
        _add_failure(hard_failures, f"cannot open numeric store: {error}")
    try:
        index = pq.read_table(args.store / "index.parquet").to_pylist()
        report["tile_count"] = len(index)
        report["store_tile_count"] = len(index)
    except Exception as error:
        _add_failure(hard_failures, f"cannot read numeric store index: {error}")
    try:
        contract = json.loads((args.store / "contract.json").read_text(encoding="utf-8"))
        if contract.get("schema") != "spec102-numeric-store-v1":
            _add_failure(hard_failures, "numeric store contract has an unsupported schema")
    except (OSError, json.JSONDecodeError) as error:
        _add_failure(hard_failures, f"cannot read numeric store contract: {error}")
    try:
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        m0_scope = validate_m0_build_local_scope(manifest, source_index=index)
        manifest_by_row = m0_scope.metadata_by_row
        all_scoped_rows = m0_scope.scoped_rows
        selected_row_numbers = all_scoped_rows[args.shard_index::args.shard_count]
        report["tile_count"] = len(selected_row_numbers)
        report["scope_tile_count"] = len(all_scoped_rows)
        report["audit_shard"] = {
            "index": args.shard_index,
            "count": args.shard_count,
            "scope_tile_count": len(all_scoped_rows),
        }
        report["selected_builds"] = [M0_ALLOWED_BUILD]
        report["evaluation_splits"] = {
            "train": "train",
            "validation": "validation_map",
            "test": "test_build_local",
        }
        audit_scope = m0_scope.audit_binding
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        _add_failure(hard_failures, f"cannot read split manifest: {error}")

    if group is not None and index:
        for name, (dtype, shape) in SPECS.items():
            if name not in group:
                _add_failure(hard_failures, f"missing numeric array: {name}")
                continue
            check = _array_schema(group[name], count=len(index), dtype=np.dtype(dtype), shape=shape)
            report["schema_checks"][name] = check
            if not check["shape_ok"] or not check["dtype_ok"]:
                _add_failure(hard_failures, f"numeric schema mismatch: {name}")
    elif group is not None:
        _add_failure(hard_failures, "numeric store index is empty")

    sources: dict[str, dict[str, Any]] = {}
    for path in (args.v18_store,):
        source_label = str(path.resolve())
        try:
            source_group = zarr.open_group(str(path), mode="r")
            source_index = pq.read_table(path / "index.parquet").to_pylist()
        except Exception as error:
            _add_failure(hard_failures, f"cannot open V18 source {source_label}: {error}")
            continue
        if not source_index:
            _add_failure(hard_failures, f"V18 source index is empty: {source_label}")
            continue
        builds = {str(row.get("build")) for row in source_index}
        if len(builds) != 1:
            _add_failure(hard_failures, f"V18 source must contain exactly one build: {source_label}")
            continue
        build = next(iter(builds))
        if build != M0_ALLOWED_BUILD:
            _add_failure(
                hard_failures,
                f"M0 build-local audit accepts only {M0_ALLOWED_BUILD}, not V18 build {build}",
            )
            continue
        if build in sources:
            _add_failure(hard_failures, f"duplicate V18 source for build {build}")
            continue
        source_checks: dict[str, dict] = {}
        for name, (_dtype, shape) in SPECS.items():
            source_name = SOURCE_KEYS[name]
            if source_name not in source_group:
                _add_failure(hard_failures, f"missing V18 source array {source_name} for build {build}")
                continue
            array = source_group[source_name]
            actual_shape = tuple(array.shape)
            expected_shape = (len(source_index), *shape)
            source_checks[name] = {
                "source_array": source_name,
                "shape": list(actual_shape),
                "expected_shape": list(expected_shape),
                "shape_ok": actual_shape == expected_shape,
                "dtype": str(array.dtype),
            }
            if actual_shape != expected_shape:
                _add_failure(hard_failures, f"V18 source shape mismatch {source_name} for build {build}")
        placements_by_tile: dict[int, list[dict]] = defaultdict(list)
        placements_path = path / "placements.parquet"
        if not placements_path.is_file():
            _add_failure(hard_failures, f"missing V18 placements provenance: {placements_path}")
        else:
            try:
                for placement in pq.read_table(placements_path).to_pylist():
                    placements_by_tile[int(placement["tile_id"])].append(placement)
            except Exception as error:
                _add_failure(hard_failures, f"cannot read V18 placements provenance {placements_path}: {error}")
        report["source_schema_checks"][build] = source_checks
        sources[build] = {
            "group": source_group,
            "index": source_index,
            "placements_by_tile": placements_by_tile,
        }

    if M0_ALLOWED_BUILD not in sources:
        _add_failure(hard_failures, f"M0 build-local audit requires V18 source {M0_ALLOWED_BUILD}")

    metadata_failures = Counter()
    if selected_row_numbers and sources:
        for row_number in selected_row_numbers:
            metadata = index[row_number]
            try:
                if int(metadata.get("row", -1)) != row_number:
                    metadata_failures["row numbering"] += 1
                if metadata.get("identity_verified") is not True:
                    metadata_failures["identity_verified"] += 1
                build = str(metadata["build"])
                if build not in sources:
                    metadata_failures["missing V18 build"] += 1
                    continue
                source_index = sources[build]["index"]
                source_row = int(metadata["v18_row"])
                if not 0 <= source_row < len(source_index):
                    metadata_failures["V18 row range"] += 1
                    continue
                origin = source_index[source_row]
                for field in IDENTITY_FIELDS:
                    if str(metadata[field]) != str(origin[field]):
                        metadata_failures[f"identity:{field}"] += 1
                materialized = bool(origin.get("object_geometry_target_materialized", False))
                if str(metadata.get("strict_target_version", "")) != str(
                    origin.get(STRICT_OBJECT_TARGET_VERSION_FIELD, "")
                ):
                    metadata_failures["strict target version provenance"] += 1
                if bool(metadata.get("strict_target_materialized", False)) != materialized:
                    metadata_failures["strict target materialization provenance"] += 1
                if str(metadata.get("strict_target_status", "")) != str(
                    origin.get("object_geometry_target_status", "")
                ):
                    metadata_failures["strict target status provenance"] += 1
                if str(metadata.get("strict_target_liquid_evidence_status", "")) != str(
                    origin.get(STRICT_LIQUID_EVIDENCE_STATUS_FIELD, "")
                ):
                    metadata_failures["strict target liquid-evidence status provenance"] += 1
                for source_field in STRICT_LIQUID_COUNTER_FIELDS:
                    numeric_field = f"strict_target_{source_field.removeprefix('object_geometry_target_')}"
                    if metadata.get(numeric_field) != origin.get(source_field):
                        metadata_failures[f"strict target liquid-evidence:{source_field}"] += 1
                if materialized and (
                    int(metadata.get("strict_target_geometry_unresolved_placement_count", 0) or 0) != 0
                    or int(metadata.get("strict_target_fallback_required_placement_count", 0) or 0) != 0
                    or int(metadata.get("strict_target_terrain_unknown_pixel_count", 0) or 0) != 0
                ):
                    metadata_failures["materialized strict target has incomplete geometry/terrain provenance"] += 1
                if manifest_by_row[row_number].get("eligible_m0") is True and not materialized:
                    metadata_failures["M0 eligible without materialized strict target"] += 1
                if manifest_by_row[row_number].get("eligible_m0") is True:
                    if metadata.get("strict_target_version") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
                        metadata_failures[
                            "M0 eligible without required strict target version"
                        ] += 1
                    if metadata.get("strict_target_liquid_evidence_status") != STRICT_LIQUID_EVIDENCE_DRY:
                        metadata_failures["M0 eligible without Dry liquid evidence"] += 1
                    for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS:
                        try:
                            nonzero = int(metadata.get(field, 0) or 0) != 0
                        except (TypeError, ValueError):
                            nonzero = True
                        if nonzero:
                            metadata_failures[f"M0 eligible nonzero {field}"] += 1
            except (KeyError, TypeError, ValueError):
                metadata_failures["required metadata fields"] += 1
    for reason, count in metadata_failures.items():
        _add_failure(hard_failures, f"numeric/V18 identity preflight failed ({reason}): {count} rows")

    # A malformed contract or source is reported, not allowed to explode midway through a scan.
    if hard_failures:
        report["hard_failures"] = hard_failures
        report["preflight_failed"] = True
        _write_report(args, report)
        return 2

    build_local_strict_target = (
        audit_scope.get("schema") == M0_BUILD_LOCAL_SCHEMA
        and audit_scope.get("kind") == "build_local"
        and audit_scope.get("allowed_builds") == [M0_ALLOWED_BUILD]
        and audit_scope.get("required_splits") == list(M0_REQUIRED_SPLITS)
        and audit_scope.get("cross_era_claim") is False
        and manifest["m0_training_scope"].get("target_quality_basis") == STRICT_TARGET_QUALITY_BASIS
        and manifest["m0_training_scope"].get("target_version")
        == REQUIRED_STRICT_OBJECT_TARGET_VERSION
        and manifest["m0_training_scope"].get("terrain_visibility_proof") == STRICT_TERRAIN_VISIBILITY_PROOF
        and manifest["m0_training_scope"].get("liquid_evidence_rule") == STRICT_LIQUID_EVIDENCE_RULE
        and audit_scope.get("validation_map") == manifest["m0_training_scope"].get("validation_map")
        and audit_scope.get("test_map") == manifest["m0_training_scope"].get("test_map")
    )
    configured_provenance = contract.get("object_target_provenance")
    if not isinstance(configured_provenance, dict):
        configured_provenance = {}
    target_provenance = {
        "target_array": configured_provenance.get("target_array", PRECISE_MASK_KEY),
        "target_version": configured_provenance.get("target_version"),
        "terrain_occlusion_clipped": configured_provenance.get("terrain_occlusion_clipped") is True,
        "per_pixel_object_top_elevation": configured_provenance.get("per_pixel_object_top_elevation") is True,
        "top_elevation_array": configured_provenance.get("top_elevation_array"),
        "source_top_elevation_array": configured_provenance.get("source_top_elevation_array"),
        "terrain_elevation_array": configured_provenance.get("terrain_elevation_array"),
        "source_terrain_elevation_array": configured_provenance.get("source_terrain_elevation_array"),
        "source_array": configured_provenance.get("source_array"),
        "source_source_array": configured_provenance.get("source_source_array"),
        "liquid_evidence_status_field": configured_provenance.get("liquid_evidence_status_field"),
        "liquid_evidence_required_for_m0": configured_provenance.get("liquid_evidence_required_for_m0"),
        "liquid_evidence_source_status_field": configured_provenance.get(
            "liquid_evidence_source_status_field"
        ),
        "liquid_evidence_source_counter_fields": configured_provenance.get(
            "liquid_evidence_source_counter_fields"
        ),
        "liquid_evidence_dry_only": (
            manifest["m0_training_scope"].get("liquid_evidence_rule") == STRICT_LIQUID_EVIDENCE_RULE
        ),
        "clip_rule": configured_provenance.get("clip_rule"),
        "build_local_strict_target_accepted": build_local_strict_target,
        "terrain_visibility_proof": manifest["m0_training_scope"]["terrain_visibility_proof"],
        "future_visibility_rule": FUTURE_VISIBILITY_RULE,
        "training_disposition": "eligible only if every strict geometry/terrain audit check passes",
    }
    report["object_target_provenance"] = target_provenance
    expected_target_provenance = {
        "target_array": PRECISE_MASK_KEY,
        "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
        "top_elevation_array": "object_geometry_visible_top_elevation_257",
        "source_top_elevation_array": "object_geometry_visible_top_elevation",
        "terrain_elevation_array": "object_geometry_visible_terrain_elevation_257",
        "source_terrain_elevation_array": "object_geometry_visible_terrain_elevation",
        "source_array": "object_geometry_visible_source_257",
        "source_source_array": "object_geometry_visible_source",
        "liquid_evidence_status_field": "strict_target_liquid_evidence_status",
        "liquid_evidence_required_for_m0": "Dry with every liquid counter equal to zero",
        "liquid_evidence_source_status_field": STRICT_LIQUID_EVIDENCE_STATUS_FIELD,
        "liquid_evidence_source_counter_fields": list(STRICT_LIQUID_COUNTER_FIELDS),
    }
    for field, expected in expected_target_provenance.items():
        if target_provenance.get(field) != expected:
            _add_failure(hard_failures, f"strict object target provenance mismatch: {field}")

    source_mismatches = Counter()
    source_raw_nonfinite = Counter()
    source_raw_range_failures = Counter()
    nonfinite_tiles = Counter()
    range_failures = Counter()
    metrics: list[dict] = []
    aggregate = Counter()
    normal_norm_sum = normal_norm_count = 0.0
    rgb_edge_inside_sum = rgb_edge_outside_sum = 0.0
    rgb_edge_inside_count = rgb_edge_outside_count = 0
    top_elevation_nonfinite = terrain_elevation_nonfinite = 0
    strict_source_invalid = strict_clearance_violation = 0
    signal_row_fingerprints: dict[int, str] = {}

    for start in range(0, len(selected_row_numbers), args.batch_size):
        row_numbers = selected_row_numbers[start:start + args.batch_size]
        batch = {name: _read_rows(group[name], row_numbers) for name in SPECS}
        signal_row_fingerprints.update(fingerprint_audited_signal_rows(row_numbers, batch))
        batch_builds = {str(index[row]["build"]) for row in row_numbers}
        if len(batch_builds) != 1:
            _add_failure(hard_failures, "M0 build-local audit mixed builds in one source batch")
            continue
        source_data = sources[next(iter(batch_builds))]
        source = source_data["group"]
        source_rows = [int(index[row]["v18_row"]) for row in row_numbers]
        raw_source_batch = {
            name: _read_rows(source[SOURCE_KEYS[name]], source_rows)
            for name in SPECS
        }
        for name in SPECS:
            raw_nonfinite, raw_range = _raw_source_tile_health(name, raw_source_batch[name])
            source_raw_nonfinite[name] += int(raw_nonfinite.sum())
            source_raw_range_failures[name] += int(raw_range.sum())
            value = batch[name]
            if np.issubdtype(value.dtype, np.floating):
                numeric_bad = ~np.isfinite(value).reshape(value.shape[0], -1).all(axis=1)
                nonfinite_tiles[name] += int(numeric_bad.sum())
            try:
                copied = _coerce_source_value(name, raw_source_batch[name])
            except (TypeError, ValueError, FloatingPointError) as error:
                _add_failure(hard_failures, f"cannot convert raw V18 batch {name}: {error}")
                continue
            source_mismatches[name] += int(_batch_mismatch(value, copied).sum())
        rgb = batch["minimap_rgb"]
        precise = batch[PRECISE_MASK_KEY]
        liquid = batch["liquid_mask_256"] > 127
        liquid_height = batch["liquid_height_256"]
        normals = batch["normal_xyz_257"].astype(np.float32) / 127.0
        height = batch["height_257"]
        target = np.maximum.reduce((
            precise[:, :-1, :-1], precise[:, 1:, :-1],
            precise[:, :-1, 1:], precise[:, 1:, 1:],
        )) > 0.5
        precise_flat = precise.reshape(precise.shape[0], -1)
        precise_finite = np.isfinite(precise_flat)
        precise_low = np.where(precise_finite, precise_flat, np.inf).min(axis=1)
        precise_high = np.where(precise_finite, precise_flat, -np.inf).max(axis=1)
        range_failures[PRECISE_MASK_KEY] += int((~precise_finite.all(axis=1) | (precise_low < 0.0) | (precise_high > 1.0)).sum())
        liquid_values = batch["liquid_mask_256"]
        range_failures["liquid_mask_256"] += int((~np.isin(liquid_values, (0, 255))).reshape(liquid_values.shape[0], -1).any(axis=1).sum())
        liquid_any = liquid.reshape(liquid.shape[0], -1).any(axis=1)
        object_coverage = target.mean(axis=(1, 2))
        liquid_coverage = liquid.mean(axis=(1, 2))
        for offset, row_number in enumerate(row_numbers):
            metadata = index[row_number]
            aggregate["liquid_metadata_disagreement_tiles"] += int(
                bool(metadata.get("has_liquid_mask", False)) != bool(liquid_any[offset])
            )
            if manifest_by_row[row_number].get("eligible_m0") is True and liquid_any[offset]:
                aggregate["m0_eligible_with_liquid_tiles"] += 1
        liquid_nonfinite = (~np.isfinite(liquid_height) & liquid).reshape(liquid.shape[0], -1).any(axis=1)
        aggregate["liquid_height_nonfinite_inside_mask_tiles"] += int(liquid_nonfinite.sum())
        strict_positive = precise > 0.5
        strict_top = batch["object_geometry_visible_top_elevation_257"]
        strict_terrain = batch["object_geometry_visible_terrain_elevation_257"]
        strict_source = batch["object_geometry_visible_source_257"]
        bad_top = strict_positive & ~np.isfinite(strict_top)
        bad_terrain = strict_positive & ~np.isfinite(strict_terrain)
        bad_source = strict_positive & ~np.isin(strict_source, (1, 2))
        clearance_violation = strict_positive & np.isfinite(strict_top) & np.isfinite(strict_terrain) & (
            strict_top <= strict_terrain + 0.25
        )
        top_elevation_nonfinite += int(bad_top.reshape(bad_top.shape[0], -1).any(axis=1).sum())
        terrain_elevation_nonfinite += int(bad_terrain.reshape(bad_terrain.shape[0], -1).any(axis=1).sum())
        strict_source_invalid += int(bad_source.reshape(bad_source.shape[0], -1).any(axis=1).sum())
        strict_clearance_violation += int(
            clearance_violation.reshape(clearance_violation.shape[0], -1).any(axis=1).sum()
        )
        aggregate["empty_object_target_tiles"] += int((~target.reshape(target.shape[0], -1).any(axis=1)).sum())
        aggregate["object_coverage_ge_50pct_tiles"] += int((object_coverage >= 0.5).sum())
        aggregate["liquid_coverage_nonzero_tiles"] += int(liquid_any.sum())
        aggregate["object_pixels_on_liquid"] += int((target & liquid).sum())
        aggregate["object_pixels"] += int(target.sum())
        aggregate["liquid_pixels"] += int(liquid.sum())
        aggregate["pixels"] += int(liquid.size)
        aggregate["placeholder_rgb_tiles"] += int((rgb.reshape(rgb.shape[0], -1).std(axis=1) < 1.0).sum())
        flags = batch["mcnk_flags_16"]
        aggregate["mcnk_flag_cells"] += int(flags.size)
        aggregate["mcnk_flag_nonzero_cells"] += int(np.count_nonzero(flags))
        aggregate["mcnk_flag_all_zero_tiles"] += int((~np.any(flags.reshape(flags.shape[0], -1), axis=1)).sum())
        normal_lengths = np.linalg.norm(normals, axis=-1)
        valid_normal_lengths = normal_lengths[normal_lengths > 0.05]
        normal_norm_sum += float(valid_normal_lengths.sum())
        normal_norm_count += int(valid_normal_lengths.size)
        gray = rgb.astype(np.float32).mean(axis=-1)
        edge = np.zeros_like(gray)
        edge[:, :, 1:] += np.abs(np.diff(gray, axis=2))
        edge[:, 1:, :] += np.abs(np.diff(gray, axis=1))
        rgb_edge_inside_sum += float(edge[target].sum())
        rgb_edge_inside_count += int(target.sum())
        rgb_edge_outside_sum += float(edge[~target].sum())
        rgb_edge_outside_count += int((~target).sum())
        relief = np.ptp(height, axis=(1, 2))
        metrics.extend({
            "row": row_number,
            "split": str(manifest_by_row[row_number]["split"]),
            "object_coverage": float(object_coverage[offset]),
            "liquid_coverage": float(liquid_coverage[offset]),
            "relief": float(relief[offset]),
            "eligible_m0": manifest_by_row[row_number].get("eligible_m0") is True,
        } for offset, row_number in enumerate(row_numbers))

    for name, count in nonfinite_tiles.items():
        if count:
            _add_failure(hard_failures, f"nonfinite numeric {name}: {count} tiles")
    for name, count in source_mismatches.items():
        if count:
            _add_failure(hard_failures, f"source copy mismatch {name}: {count} tiles")
    for name, count in source_raw_nonfinite.items():
        if count:
            _add_failure(hard_failures, f"raw V18 nonfinite {name}: {count} tiles")
    for name, count in source_raw_range_failures.items():
        if count:
            _add_failure(hard_failures, f"raw V18 range violation {name}: {count} tiles")
    for name, count in range_failures.items():
        if count:
            _add_failure(hard_failures, f"numeric range violation {name}: {count} tiles")
    normal_mean_length = normal_norm_sum / max(normal_norm_count, 1)
    if args.shard_count == 1:
        if aggregate["mcnk_flag_nonzero_cells"] <= 0:
            _add_failure(hard_failures, "3.3.5 MCNK flags are all zero across the audited scope")
        if normal_norm_count <= 0:
            _add_failure(hard_failures, "native normals have no nonzero vectors across the audited scope")
        elif not 0.75 <= normal_mean_length <= 1.25:
            _add_failure(
                hard_failures,
                f"native normal mean length {normal_mean_length:.6f} is outside the 0.75-1.25 contract",
            )
    if top_elevation_nonfinite:
        _add_failure(hard_failures, f"object top elevation is nonfinite on {top_elevation_nonfinite} positive-mask tiles")
    if terrain_elevation_nonfinite:
        _add_failure(hard_failures, f"object terrain elevation is nonfinite on {terrain_elevation_nonfinite} positive-mask tiles")
    if strict_source_invalid:
        _add_failure(hard_failures, f"strict object target has invalid geometry source on {strict_source_invalid} positive-mask tiles")
    if strict_clearance_violation:
        _add_failure(hard_failures, f"strict object target violates terrain-Z clearance on {strict_clearance_violation} positive-mask tiles")
    if aggregate["m0_eligible_with_liquid_tiles"]:
        _add_failure(
            hard_failures,
            "initial M0 dry-only policy violated: "
            f"{aggregate['m0_eligible_with_liquid_tiles']} eligible row(s) contain liquid pixels",
        )

    placement_counts: dict[str, Any] = {"status": "not_run"}
    if args.include_placement_diagnostic:
        measured = Counter()
        for row_number in selected_row_numbers:
            metadata = index[row_number]
            source_data = sources[str(metadata["build"])]
            placements = source_data["placements_by_tile"].get(int(metadata["tile_id"]), [])
            if placements:
                measured["tiles_with_placements"] += 1
            height = np.asarray(group["height_257"][int(metadata["row"])], dtype=np.float32)
            for placement in placements:
                measured["total"] += 1
                try:
                    relation = project_placement_to_terrain(
                        placement,
                        tile_x=int(metadata["tile_x"]),
                        tile_y=int(metadata["tile_y"]),
                        terrain_height_257=height,
                    )
                except (KeyError, TypeError, ValueError):
                    measured["invalid"] += 1
                    continue
                if relation is None:
                    measured["unprojectable"] += 1
                    continue
                measured[f"top_source:{relation.top_source}"] += 1
                if relation.clearance < -0.5:
                    kind = str(placement.get("instance_type", "unknown"))
                    measured[f"below_terrain:{kind}"] += 1
                    measured["below_terrain_total"] += 1
        placement_counts = dict(measured)
        if measured["invalid"]:
            _add_failure(hard_failures, f"invalid placement terrain diagnostics: {measured['invalid']} placements")

    if target_provenance["target_array"] != PRECISE_MASK_KEY:
        _add_failure(hard_failures, "object target provenance names a non-canonical target array")
    if not target_provenance["terrain_occlusion_clipped"]:
        _add_failure(
            hard_failures,
            "strict object target terrain-visibility is unproven: no Z-clipped target provenance exists",
        )
    if not target_provenance["per_pixel_object_top_elevation"]:
        _add_failure(
            hard_failures,
            "strict object target terrain-visibility is unproven: no per-pixel object top-elevation provenance exists",
        )
    if not target_provenance["clip_rule"]:
        _add_failure(hard_failures, "object target provenance records no terrain-Z clipping rule")

    panel_paths: dict[str, str] = {}
    for split in (evaluation_splits[1], evaluation_splits[2]):
        selected = _choose_panel_rows(metrics, split, args.panel_count)
        if not selected:
            _add_failure(hard_failures, f"cannot render signal panel: split {split} has no rows")
            continue
        samples = [
            {
                "metadata": index[row],
                **{name: np.asarray(group[name][row]) for name in SPECS},
            }
            for row in selected
        ]
        panel_path = args.output_dir / f"signals_{split}.png"
        render_signal_panel(samples, split=split, source_label=args.store.name).save(panel_path)
        panel_paths[split] = str(panel_path.resolve())

    report.update({
        "safe_for_m0_training": False,
        "safe_for_m0_build_local_training": not hard_failures and args.shard_count == 1,
        "shard_clean": not hard_failures,
        "partial_scope": args.shard_count > 1,
        "m0_training_scope": audit_scope,
        "hard_failures": hard_failures,
        "preflight_failed": False,
        "source_copy_mismatch_tiles": dict(source_mismatches),
        "source_raw_nonfinite_tiles": dict(source_raw_nonfinite),
        "source_raw_range_failure_tiles": dict(source_raw_range_failures),
        "nonfinite_tiles": dict(nonfinite_tiles),
        "range_failure_tiles": dict(range_failures),
        "signal_counts": dict(aggregate),
        "normal_nonzero_mean_length": normal_mean_length,
        "normal_nonzero_sum": normal_norm_sum,
        "normal_nonzero_count": normal_norm_count,
        "rgb_edge_mean_inside_object_target": rgb_edge_inside_sum / max(rgb_edge_inside_count, 1),
        "rgb_edge_mean_outside_object_target": rgb_edge_outside_sum / max(rgb_edge_outside_count, 1),
        "rgb_edge_inside_sum": rgb_edge_inside_sum,
        "rgb_edge_inside_count": rgb_edge_inside_count,
        "rgb_edge_outside_sum": rgb_edge_outside_sum,
        "rgb_edge_outside_count": rgb_edge_outside_count,
        "placement_terrain_audit": dict(placement_counts),
        "audited_signal_keys": list(M0_AUDITED_SIGNAL_KEYS),
        "signal_row_fingerprints": {str(row): digest for row, digest in signal_row_fingerprints.items()},
        "scoped_signal_fingerprint": (
            combine_audited_signal_row_fingerprints(signal_row_fingerprints)
            if args.shard_count == 1 and len(signal_row_fingerprints) == len(all_scoped_rows)
            else None
        ),
        "panels": panel_paths,
    })
    _write_report(args, report)
    return 0 if report["shard_clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
