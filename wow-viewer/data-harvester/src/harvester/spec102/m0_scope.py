"""M0-only build-local split contract for the strict 3.3.5 geometry target."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from harvester.spec102.curation import (
    STRICT_LIQUID_EVIDENCE_DRY,
    STRICT_TARGET_FRAGMENT_TRACE_FIELDS,
    STRICT_TARGET_LIQUID_COUNTER_FIELDS,
)
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION

M0_BUILD_LOCAL_SCHEMA = "spec102-m0-build-local-split-v4"
M0_CURATED_SOURCE_SCHEMAS = (
    "spec102-curated-split-v5",
    "spec102-m0-full-3_3_5-curation-v4",
)
M0_ALLOWED_BUILD = "3_3_5_12340"
M0_REQUIRED_SPLITS = ("train", "validation_map", "test_build_local")
ALPHA_EXCLUSION_REASON = "excluded_initial_m0_build_scope"
FUTURE_VISIBILITY_RULE = (
    "per_fragment_geometry_vs_terrain_only; placement_or_instance_wipeout_prohibited"
)
STRICT_TARGET_QUALITY_BASIS = "strict_transformed_geometry_terrain_visible"
STRICT_TERRAIN_VISIBILITY_PROOF = "per_fragment_transformed_geometry_vs_raw_mcvt_z"
STRICT_FRAGMENT_TRACE_PROOF = "lossless_uncollapsed_transformed_mesh_raw_mcvt_liquid_fragment_trace_v3"
STRICT_TARGET_COMPLETE_STATUSES = ("CompleteEmpty", "CompleteVisible")
STRICT_LIQUID_EVIDENCE_RULE = "Dry with every liquid evidence counter equal to zero"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class M0BuildLocalScope:
    manifest: dict[str, Any]
    metadata_by_row: dict[int, dict[str, Any]]
    rows_by_split: dict[str, list[int]]
    scoped_rows: list[int]
    excluded_rows: list[int]

    @property
    def audit_binding(self) -> dict[str, Any]:
        scope = self.manifest["m0_training_scope"]
        return {
            "schema": M0_BUILD_LOCAL_SCHEMA,
            "kind": scope["kind"],
            "allowed_builds": list(scope["allowed_builds"]),
            "required_splits": list(scope["required_splits"]),
            "cross_era_claim": scope["cross_era_claim"],
            "target_quality_basis": scope["target_quality_basis"],
            "target_version": scope["target_version"],
            "terrain_visibility_proof": scope["terrain_visibility_proof"],
            "fragment_trace_proof": scope["fragment_trace_proof"],
            "liquid_evidence_rule": scope["liquid_evidence_rule"],
            "validation_map": scope["validation_map"],
            "test_map": scope["test_map"],
        }

    def artifact_binding(
        self,
        *,
        store: Path,
        split_manifest: Path,
        coverage_report: Path | None = None,
    ) -> dict[str, Any]:
        """Bind a checkpoint to the exact scope, store snapshot, and split file."""
        binding = {
            "schema": "spec102-m0-artifact-binding-v1",
            "scope": self.audit_binding,
            "store_contract_sha256": sha256_file(store / "contract.json"),
            "store_index_sha256": sha256_file(store / "index.parquet"),
            "split_manifest_sha256": sha256_file(split_manifest),
        }
        if coverage_report is not None:
            binding["coverage_report_sha256"] = sha256_file(coverage_report)
        return binding


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raise_if_invalid(errors: list[str]) -> None:
    if errors:
        raise RuntimeError("invalid Spec 102 M0 build-local split:\n- " + "\n- ".join(errors))


def build_m0_build_local_manifest(
    source_manifest: dict[str, Any],
    *,
    source_index: list[dict[str, Any]],
    validation_map: str,
    test_map: str,
    source_manifest_path: str,
    source_manifest_sha256: str,
    source_store: str,
    source_index_sha256: str,
) -> dict[str, Any]:
    """Derive the M0-only 3.3.5 scope without silently dropping any store row."""
    if source_manifest.get("schema") not in M0_CURATED_SOURCE_SCHEMAS:
        raise RuntimeError(
            "M0 build-local scope requires a frozen Spec 102 curation manifest; "
            f"accepted schemas are {M0_CURATED_SOURCE_SCHEMAS}"
        )
    if validation_map == test_map:
        raise ValueError("validation and test map must be different complete maps")
    source_rows = source_manifest.get("rows")
    if not isinstance(source_rows, list):
        raise RuntimeError("frozen Spec 102 curation source manifest has no rows list")
    source_by_row = {int(row["row"]): dict(row) for row in source_rows}
    if set(source_by_row) != set(range(len(source_index))):
        raise RuntimeError("frozen Spec 102 curation source manifest does not exactly cover the numeric store")
    available_maps = {
        str(item["map"]) for item in source_index
        if str(item["build"]) == M0_ALLOWED_BUILD
    }
    if validation_map not in available_maps or test_map not in available_maps:
        raise RuntimeError("M0 build-local holdout maps are absent from the 3.3.5 numeric store")

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = dict.fromkeys(M0_REQUIRED_SPLITS, 0)
    eligible_counts: dict[str, int] = dict.fromkeys(M0_REQUIRED_SPLITS, 0)
    excluded_counts: dict[str, int] = {}
    for row_number, source_metadata in enumerate(source_index):
        row = source_by_row[row_number]
        build = str(source_metadata["build"])
        if build == M0_ALLOWED_BUILD:
            map_name = str(source_metadata["map"])
            split = (
                "validation_map" if map_name == validation_map
                else "test_build_local" if map_name == test_map
                else "train"
            )
            row["split"] = split
            counts[split] += 1
            eligible_counts[split] += int(row.get("eligible_m0") is True)
        else:
            row["split"] = "excluded_build"
            row["eligible_m0"] = False
            reasons = list(row.get("rejection_reasons") or [])
            if ALPHA_EXCLUSION_REASON not in reasons:
                reasons.append(ALPHA_EXCLUSION_REASON)
            row["rejection_reasons"] = reasons
            excluded_counts[build] = excluded_counts.get(build, 0) + 1
        rows.append(row)

    manifest = {
        "schema": M0_BUILD_LOCAL_SCHEMA,
        "source_store": source_store,
        "source_index_sha256": source_index_sha256,
        "source_manifest": source_manifest_path,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_schema": source_manifest["schema"],
        "m0_training_scope": {
            "kind": "build_local",
            "allowed_builds": [M0_ALLOWED_BUILD],
            "required_splits": list(M0_REQUIRED_SPLITS),
            "cross_era_claim": False,
            "target_quality_basis": STRICT_TARGET_QUALITY_BASIS,
            "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "terrain_visibility_proof": STRICT_TERRAIN_VISIBILITY_PROOF,
            "fragment_trace_proof": STRICT_FRAGMENT_TRACE_PROOF,
            "liquid_evidence_rule": STRICT_LIQUID_EVIDENCE_RULE,
            "future_visibility_rule": FUTURE_VISIBILITY_RULE,
            "validation_map": validation_map,
            "test_map": test_map,
        },
        "counts": {
            "rows_by_split": counts,
            "eligible_m0_by_split": eligible_counts,
            "excluded_rows_by_build": excluded_counts,
        },
        "rows": rows,
    }
    validate_m0_build_local_scope(manifest, source_index=source_index)
    return manifest


def validate_m0_build_local_scope(
    manifest: dict[str, Any],
    *,
    source_index: list[dict[str, Any]],
) -> M0BuildLocalScope:
    """Validate the narrow M0 contract before any CUDA allocation or metric claim."""
    errors: list[str] = []
    if manifest.get("schema") != M0_BUILD_LOCAL_SCHEMA:
        errors.append(f"schema must be {M0_BUILD_LOCAL_SCHEMA}")
    scope = manifest.get("m0_training_scope")
    if not isinstance(scope, dict):
        errors.append("m0_training_scope must be an object")
        scope = {}
    if scope.get("kind") != "build_local":
        errors.append("scope kind must be build_local")
    if scope.get("allowed_builds") != [M0_ALLOWED_BUILD]:
        errors.append(f"allowed_builds must be [{M0_ALLOWED_BUILD}]")
    if tuple(scope.get("required_splits") or ()) != M0_REQUIRED_SPLITS:
        errors.append(f"required_splits must be {list(M0_REQUIRED_SPLITS)}")
    if scope.get("cross_era_claim") is not False:
        errors.append("cross_era_claim must be false")
    if scope.get("target_quality_basis") != STRICT_TARGET_QUALITY_BASIS:
        errors.append("target_quality_basis must record the strict geometry-visible contract")
    if scope.get("target_version") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
        errors.append(f"target_version must be {REQUIRED_STRICT_OBJECT_TARGET_VERSION}")
    if scope.get("terrain_visibility_proof") != STRICT_TERRAIN_VISIBILITY_PROOF:
        errors.append("terrain_visibility_proof must record per-fragment raw-MCVT Z proof")
    if scope.get("fragment_trace_proof") != STRICT_FRAGMENT_TRACE_PROOF:
        errors.append("fragment_trace_proof must record the exact lossless C# v3 trace contract")
    if scope.get("liquid_evidence_rule") != STRICT_LIQUID_EVIDENCE_RULE:
        errors.append("liquid evidence rule must require Dry status and zero counters")
    if scope.get("future_visibility_rule") != FUTURE_VISIBILITY_RULE:
        errors.append("future visibility rule must prohibit whole-placement or whole-instance wipeout")

    rows = manifest.get("rows")
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        rows = []
    metadata_by_row: dict[int, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            errors.append("rows must contain objects")
            continue
        try:
            row = int(item["row"])
        except (KeyError, TypeError, ValueError):
            errors.append("rows must contain integer row IDs")
            continue
        if row in metadata_by_row:
            errors.append("rows contain duplicate row IDs")
            continue
        metadata_by_row[row] = item
    expected_rows = set(range(len(source_index)))
    if set(metadata_by_row) != expected_rows:
        errors.append("rows must exactly cover the numeric store, including excluded builds")

    rows_by_split = {split: [] for split in M0_REQUIRED_SPLITS}
    scoped_rows: list[int] = []
    excluded_rows: list[int] = []
    maps_by_split: dict[str, set[str]] = {split: set() for split in M0_REQUIRED_SPLITS}
    map_owner: dict[str, str] = {}
    for row_number, source_metadata in enumerate(source_index):
        item = metadata_by_row.get(row_number)
        if item is None:
            continue
        for field in ("build", "map", "tile_id", "tile_x", "tile_y"):
            if str(item.get(field)) != str(source_metadata.get(field)):
                errors.append(f"row {row_number} does not match numeric-store {field}")
                break
        build = str(source_metadata.get("build"))
        split = item.get("split")
        if build == M0_ALLOWED_BUILD:
            scoped_rows.append(row_number)
            if split not in M0_REQUIRED_SPLITS:
                errors.append(f"3.3.5 row {row_number} has invalid split {split}")
                continue
            map_name = str(source_metadata.get("map"))
            owner = map_owner.setdefault(map_name, str(split))
            if owner != split:
                errors.append(f"map {map_name} crosses scoped M0 splits")
            maps_by_split[str(split)].add(map_name)
            if item.get("eligible_m0") is True:
                if source_metadata.get("strict_target_version") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
                    errors.append(
                        f"M0-eligible row {row_number} does not use strict target "
                        f"{REQUIRED_STRICT_OBJECT_TARGET_VERSION}"
                    )
                if source_metadata.get("strict_target_materialized") is not True:
                    errors.append(
                        f"M0-eligible row {row_number} has no materialized strict geometry target"
                    )
                if source_metadata.get("strict_target_status") not in STRICT_TARGET_COMPLETE_STATUSES:
                    errors.append(
                        f"M0-eligible row {row_number} has incomplete strict geometry-target status"
                    )
                if source_metadata.get("strict_target_arrays_present") is not True:
                    errors.append(
                        f"M0-eligible row {row_number} lacks strict geometry target arrays"
                    )
                for field in (
                    "strict_target_geometry_unresolved_placement_count",
                    "strict_target_fallback_required_placement_count",
                    "strict_target_terrain_unknown_pixel_count",
                ):
                    try:
                        nonzero = int(source_metadata.get(field, 0) or 0) != 0
                    except (TypeError, ValueError):
                        nonzero = True
                    if nonzero:
                        errors.append(
                            f"M0-eligible row {row_number} has nonzero {field}"
                        )
                if source_metadata.get("strict_target_liquid_evidence_status") != STRICT_LIQUID_EVIDENCE_DRY:
                    errors.append(
                        f"M0-eligible row {row_number} lacks Dry liquid-evidence status"
                    )
                for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS:
                    try:
                        nonzero = int(source_metadata.get(field, 0) or 0) != 0
                    except (TypeError, ValueError):
                        nonzero = True
                    if nonzero:
                        errors.append(
                            f"M0-eligible row {row_number} has nonzero {field}"
                        )
                for field in STRICT_TARGET_FRAGMENT_TRACE_FIELDS:
                    if field not in source_metadata:
                        errors.append(f"M0-eligible row {row_number} lacks {field}")
                if source_metadata.get("strict_target_fragment_trace_schema") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
                    errors.append(f"M0-eligible row {row_number} has wrong fragment-trace schema")
                if source_metadata.get("strict_target_fragment_trace_arrays_present") is not True:
                    errors.append(f"M0-eligible row {row_number} lacks the nine fragment-trace arrays")
                if source_metadata.get("strict_target_fragment_trace_validated") is not True:
                    errors.append(f"M0-eligible row {row_number} has unvalidated fragment-trace data")
                try:
                    trace_count = int(source_metadata.get("strict_target_fragment_count", -1))
                    trace_start = int(source_metadata.get("strict_target_fragment_trace_sidecar_start", -1))
                    trace_end = int(source_metadata.get("strict_target_fragment_trace_sidecar_end", -1))
                except (TypeError, ValueError):
                    trace_count = trace_start = trace_end = -1
                if trace_count < 0 or trace_start < 0 or trace_end - trace_start != trace_count:
                    errors.append(f"M0-eligible row {row_number} has incoherent fragment-trace offsets")
                if not _SHA256_RE.fullmatch(str(source_metadata.get("strict_target_fragment_sha256", "") or "")):
                    errors.append(f"M0-eligible row {row_number} has invalid fragment-trace SHA-256")
                if not _SHA256_RE.fullmatch(
                    str(source_metadata.get("strict_target_fragment_trace_source_sidecar_sha256", "") or "")
                ):
                    errors.append(f"M0-eligible row {row_number} lacks source fragment-trace sidecar binding")
                try:
                    assets = json.loads(str(source_metadata.get("strict_target_assets_json", "")))
                    unresolved = json.loads(str(source_metadata.get("strict_target_unresolved_placements_json", "")))
                except (TypeError, ValueError, json.JSONDecodeError):
                    assets = unresolved = None
                if not isinstance(assets, list) or not isinstance(unresolved, list):
                    errors.append(f"M0-eligible row {row_number} has invalid fragment-trace assets/unresolved provenance")
                elif unresolved:
                    errors.append(f"M0-eligible row {row_number} retains unresolved fragment-trace placements")
                rows_by_split[str(split)].append(row_number)
        else:
            excluded_rows.append(row_number)
            if split != "excluded_build":
                errors.append(f"non-3.3.5 row {row_number} must be excluded_build")
            if item.get("eligible_m0") is not False:
                errors.append(f"excluded row {row_number} must be ineligible for M0")
            if ALPHA_EXCLUSION_REASON not in (item.get("rejection_reasons") or []):
                errors.append(f"excluded row {row_number} lacks its literal exclusion reason")
    for split, selected in rows_by_split.items():
        if not selected:
            errors.append(f"scope has no M0-eligible {split} rows")
    if any(maps_by_split[left] & maps_by_split[right] for left in M0_REQUIRED_SPLITS for right in M0_REQUIRED_SPLITS if left != right):
        errors.append("a complete map appears in more than one scoped split")
    _raise_if_invalid(errors)
    return M0BuildLocalScope(
        manifest=manifest,
        metadata_by_row=metadata_by_row,
        rows_by_split=rows_by_split,
        scoped_rows=scoped_rows,
        excluded_rows=excluded_rows,
    )
