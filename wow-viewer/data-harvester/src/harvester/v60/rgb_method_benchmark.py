"""Prepare an evidence-gated RGB-only terrain-method benchmark plan.

The planner reads manifests only. It does not materialize arrays, train models, download external
weights, or treat source-side object masks as runtime inputs. Authored raw RGB is marked
runtime-compatible; the object-library sieve is retained as a synthetic luma/object-control source
because its input is ``objectified_terrain_shadow_256``, not ``minimap_rgb``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from harvester.v60.clean_signal_corpus import load_clean_signal_manifest
from harvester.v60.object_library_sieve import (
    CLEAN_SIGNAL,
    INPUT_SIGNAL,
    MASK_SIGNAL,
    load_object_library_sieve_manifest,
)
from harvester.v60.terrain_method_translation import FORBIDDEN_INPUTS

RGB_METHOD_BENCHMARK_SCHEMA = "v60-rgb-method-benchmark-v1"
SOURCE_SELECTIONS = frozenset({"authored", "object_library", "both"})
CONDITION_IDS = ("no_mask", "predicted_mask", "withheld_mask")

BASELINE_DEFINITIONS = (
    {
        "baseline_id": "tile_mean_height_v1",
        "scope": "final_height",
        "description": "Per-tile target mean height baseline.",
        "requires_target_for_evaluation": True,
    },
    {
        "baseline_id": "identity_observation_v1",
        "scope": "observation_or_clean_head",
        "description": "Pass-through observation baseline for clean/contaminated image comparison.",
        "requires_target_for_evaluation": True,
    },
    {
        "baseline_id": "zero_predicted_mask_v1",
        "scope": "predicted_mask",
        "description": "No-object predicted-mask baseline; must not be confused with a ground-truth mask.",
        "requires_target_for_evaluation": False,
    },
)

CONDITION_DEFINITIONS = {
    "no_mask": {
        "model_input_policy": "observation_only",
        "mask_role": "not_provided",
        "required_predicted_artifact": False,
    },
    "predicted_mask": {
        "model_input_policy": "observation_plus_predicted_mask",
        "mask_role": "predicted_only",
        "required_predicted_artifact": True,
    },
    "withheld_mask": {
        "model_input_policy": "observation_only",
        "mask_role": "evaluation_only_withheld_from_model",
        "required_predicted_artifact": False,
    },
}


class RGBMethodBenchmarkError(ValueError):
    """Raised when a benchmark source cannot satisfy the RGB method contract."""


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _split_report(rows: list[dict[str, Any]], *, group_key: str) -> dict[str, Any]:
    failures: list[str] = []
    split_counts = Counter()
    group_splits: dict[str, str] = {}
    identity_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, row in enumerate(rows):
        row_id = str(row.get("row_id", ""))
        split = str(row.get("split", ""))
        group_id = str(row.get(group_key, ""))
        if not row_id:
            failures.append(f"row[{position}] missing row_id")
        elif row_id in seen_ids:
            failures.append(f"duplicate row_id {row_id!r}")
        seen_ids.add(row_id)
        if split not in {"train", "validation", "test"}:
            failures.append(f"row[{position}] invalid split {split!r}")
        else:
            split_counts[split] += 1
        if not group_id:
            failures.append(f"row[{position}] missing {group_key}")
        elif split in {"train", "validation", "test"}:
            prior = group_splits.setdefault(group_id, split)
            if prior != split:
                failures.append(f"group {group_id!r} crosses {prior!r}/{split!r}")
        identity_rows.append(
            {
                "row_id": row_id,
                "group_id": group_id,
                "split": split,
                "family": str(row.get("family", row.get("terrain_control_family", ""))),
                "map": str(row.get("map", "")),
                "tile_x": row.get("tile_x"),
                "tile_y": row.get("tile_y"),
            }
        )
    identity_rows.sort(key=lambda item: (item["row_id"], item["group_id"]))
    return {
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "group_count": len(group_splits),
        "split_identity_sha256": _sha256_json(identity_rows),
        "failures": failures,
        "valid": not failures,
    }


def _require_no_forbidden_reads(
    values: list[str], *, context: str, failures: list[str]
) -> None:
    canonical = {str(value).strip().lower() for value in values if str(value).strip()}
    forbidden = sorted(canonical & set(FORBIDDEN_INPUTS))
    if forbidden:
        failures.append(f"{context} declares forbidden model inputs: {forbidden}")


def _authored_source_report(root: Path) -> dict[str, Any]:
    try:
        manifest = load_clean_signal_manifest(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RGBMethodBenchmarkError(f"invalid authored clean-signal corpus {root}: {exc}") from exc

    failures: list[str] = []
    if manifest.get("source_schema") != "v7-clean-signal-real-minimap-rgb-v1":
        failures.append("authored source_schema is not the raw minimap RGB contract")
    if manifest.get("source_filter") != "authored":
        failures.append("authored benchmark requires source_filter='authored'")
    if manifest.get("input_contract") != "minimap_rgb_to_raw_luma_diagnostic_v1":
        failures.append("authored corpus input_contract is not minimap_rgb_to_raw_luma_diagnostic_v1")

    rows = manifest["rows"]
    for position, row in enumerate(rows):
        if row.get("source_kind") != "real_minimap_diagnostic":
            failures.append(f"row[{position}] is not real_minimap_diagnostic")
        if row.get("observation_status", "accepted") != "accepted":
            failures.append(f"row[{position}] observation is not accepted")
        if row.get("forbidden_signals", []):
            failures.append(f"row[{position}] declares forbidden_signals")
        provenance = row.get("observation_provenance", {})
        if provenance.get("inference_target_reads", []) != []:
            failures.append(f"row[{position}] has inference_target_reads")
        _require_no_forbidden_reads(
            ["clean_observation_luma_256", "clean_observation_gradient_256", "clean_observation_confidence_256"],
            context=f"row[{position}]",
            failures=failures,
        )

    split = _split_report(rows, group_key="source_group_id")
    failures.extend(split["failures"])
    maps = sorted({str(row.get("map", "")) for row in rows if str(row.get("map", ""))})
    source_report = {
        "source_key": "authored",
        "source_root": str(root.resolve()),
        "manifest_schema": manifest.get("schema"),
        "source_schema": manifest.get("source_schema"),
        "source_modality": "rgb",
        "runtime_compatible": not failures,
        "row_count": len(rows),
        "maps": maps,
        "split": split,
        "model_input_arrays": [
            "clean_observation_luma_256",
            "clean_observation_gradient_256",
            "clean_observation_confidence_256",
        ],
        "source_observation": "minimap_rgb",
        "evaluation_only_arrays": ["height_257", "relative_height_257", "coarse_relief_257", "detail_residual_257"],
        "available_conditions": {
            "no_mask": len(rows),
            "predicted_mask": 0,
            "withheld_mask": 0,
        },
        "forbidden_reads": [],
        "failures": failures,
        "valid": not failures,
    }
    return source_report


def _object_library_source_report(root: Path) -> dict[str, Any]:
    try:
        manifest = load_object_library_sieve_manifest(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RGBMethodBenchmarkError(f"invalid object-library sieve corpus {root}: {exc}") from exc

    failures: list[str] = []
    if manifest.get("source_policy") != "real_v50_object_library_over_project_control_terrain":
        failures.append("object-library source_policy is not the approved real-library-over-control policy")
    rows = manifest["rows"]
    for position, row in enumerate(rows):
        if row.get("source_kind") != "real_v50_object_library_composite":
            failures.append(f"row[{position}] is not a real_v50_object_library_composite")
        if row.get("input") != INPUT_SIGNAL:
            failures.append(f"row[{position}] input is not {INPUT_SIGNAL}")
        targets = {str(value) for value in row.get("targets", [])}
        if CLEAN_SIGNAL not in targets or MASK_SIGNAL not in targets:
            failures.append(f"row[{position}] lacks clean terrain or contamination supervision")
        if row.get("forbidden_signals", []):
            failures.append(f"row[{position}] declares forbidden_signals")
        _require_no_forbidden_reads([str(row.get("input", ""))], context=f"row[{position}]", failures=failures)

    # Boundary-crossing sieve rows are intentionally forced into validation even when their
    # source control row also has ordinary train-regime derivatives. Preserve that manifest split
    # rather than treating the control-row reuse as accidental leakage; the control ID remains in
    # the row-level provenance identity below.
    split = _split_report(rows, group_key="row_id")
    split["group_policy"] = "manifest_row_with_source_control_provenance"
    failures.extend(split["failures"])
    source_report = {
        "source_key": "object_library",
        "source_root": str(root.resolve()),
        "manifest_schema": manifest.get("schema"),
        "source_schema": manifest.get("source_control_schema"),
        "source_modality": "synthetic_luma_object_control",
        "runtime_compatible": False,
        "runtime_compatibility_reason": "objectified_terrain_shadow_256 is not minimap_rgb",
        "row_count": len(rows),
        "maps": [],
        "split": split,
        "model_input_arrays": [INPUT_SIGNAL],
        "source_observation": INPUT_SIGNAL,
        "evaluation_only_arrays": [CLEAN_SIGNAL, MASK_SIGNAL, "height_257", "mcnr_normal_xyz"],
        "available_conditions": {
            "no_mask": len(rows),
            "predicted_mask": 0,
            "withheld_mask": len(rows),
        },
        "forbidden_reads": [],
        "failures": failures,
        "valid": not failures,
    }
    return source_report


def _condition_reports(source_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for condition_id in CONDITION_IDS:
        definition = CONDITION_DEFINITIONS[condition_id]
        by_source = {
            report["source_key"]: {
                "eligible_row_count": int(report["available_conditions"][condition_id]),
                "runtime_compatible": bool(report["runtime_compatible"]),
                "status": "available"
                if report["available_conditions"][condition_id] and report["valid"]
                else "blocked",
                "reason": (
                    "available"
                    if report["available_conditions"][condition_id] and report["valid"]
                    else "no compatible source rows or source validation failed"
                ),
            }
            for report in source_reports
        }
        eligible = sum(item["eligible_row_count"] for item in by_source.values())
        runtime_eligible = sum(
            item["eligible_row_count"] for item in by_source.values() if item["runtime_compatible"]
        )
        reports.append(
            {
                "condition_id": condition_id,
                **definition,
                "source_reports": by_source,
                "eligible_row_count": eligible,
                "runtime_eligible_row_count": runtime_eligible,
                "status": "available" if eligible else "blocked",
            }
        )
    return reports


def build_rgb_method_benchmark_plan(
    *,
    source: str,
    authored_corpus: str | Path | None = None,
    object_library_sieve: str | Path | None = None,
) -> dict[str, Any]:
    """Build a no-write RGB method benchmark plan from existing manifests."""

    if source not in SOURCE_SELECTIONS:
        raise RGBMethodBenchmarkError(f"source must be one of {sorted(SOURCE_SELECTIONS)}")
    if source in {"authored", "both"} and authored_corpus is None:
        raise RGBMethodBenchmarkError(f"source={source!r} requires --authored-corpus")
    if source in {"object_library", "both"} and object_library_sieve is None:
        raise RGBMethodBenchmarkError(f"source={source!r} requires --object-library-sieve")

    source_reports: list[dict[str, Any]] = []
    if source in {"authored", "both"}:
        source_reports.append(_authored_source_report(Path(authored_corpus)))
    if source in {"object_library", "both"}:
        source_reports.append(_object_library_source_report(Path(object_library_sieve)))

    invalid_sources = [report for report in source_reports if not report["valid"]]
    if invalid_sources:
        failures = [failure for report in invalid_sources for failure in report["failures"]]
        raise RGBMethodBenchmarkError("benchmark source validation failed: " + "; ".join(failures[:8]))

    conditions = _condition_reports(source_reports)
    failures = [failure for report in source_reports for failure in report["failures"]]
    runtime_conditions = [
        condition["condition_id"]
        for condition in conditions
        if condition["runtime_eligible_row_count"] > 0
    ]
    plan = {
        "schema": RGB_METHOD_BENCHMARK_SCHEMA,
        "planner": "harvester.v60.rgb_method_benchmark.build_rgb_method_benchmark_plan",
        "source_selection": source,
        "source_reports": source_reports,
        "conditions": conditions,
        "baselines": list(BASELINE_DEFINITIONS),
        "metric_groups": [
            "final_height",
            "clean_identity",
            "contaminated_input",
            "object_mask",
            "cross_tile",
            "family",
        ],
        "forbidden_reads": sorted({value for report in source_reports for value in report["forbidden_reads"]}),
        "runtime_eligible_conditions": runtime_conditions,
        "next_gate": "user_review_dry_plan_before_any_training",
        "failures": failures,
        "valid": not failures and not any(not report["valid"] for report in source_reports),
        "dry_run": True,
    }
    return plan


__all__ = [
    "BASELINE_DEFINITIONS",
    "CONDITION_DEFINITIONS",
    "CONDITION_IDS",
    "RGB_METHOD_BENCHMARK_SCHEMA",
    "RGBMethodBenchmarkError",
    "SOURCE_SELECTIONS",
    "build_rgb_method_benchmark_plan",
]
