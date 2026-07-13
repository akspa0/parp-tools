"""Merge complete, hash-identical Spec 102 signal-audit shards fail-closed."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from harvester.spec102.signal_audit import (
    AUDIT_SCHEMA,
    combine_audited_signal_row_fingerprints,
)

IDENTITY_KEYS = (
    "schema",
    "store",
    "split_manifest",
    "store_contract_sha256",
    "store_index_sha256",
    "split_manifest_sha256",
    "m0_training_scope",
    "object_target_provenance",
    "audited_signal_keys",
)
COUNTER_KEYS = (
    "source_copy_mismatch_tiles",
    "source_raw_nonfinite_tiles",
    "source_raw_range_failure_tiles",
    "nonfinite_tiles",
    "range_failure_tiles",
    "signal_counts",
)


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read audit shard {path}: {error}") from error


def _sum_counters(reports: list[dict[str, Any]], key: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    for report in reports:
        total.update({name: int(value) for name, value in (report.get(key) or {}).items()})
    return dict(total)


def merge(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise RuntimeError("at least one audit shard is required")
    base = reports[0]
    errors: list[str] = []
    if base.get("schema") != AUDIT_SCHEMA:
        errors.append("unsupported audit schema")
    for report in reports:
        for key in IDENTITY_KEYS:
            if report.get(key) != base.get(key):
                errors.append(f"shards disagree on {key}")
        if report.get("preflight_failed") is True:
            errors.append("a shard failed preflight")
        if report.get("partial_scope") is not True:
            errors.append("all inputs must be partial audit shards")
        if report.get("shard_clean") is not True:
            errors.append("a shard has hard failures")
    shard_metadata = [report.get("audit_shard") for report in reports]
    if any(not isinstance(item, dict) for item in shard_metadata):
        errors.append("a shard has no audit_shard metadata")
        shard_count = 0
        scope_tile_count = 0
    else:
        shard_count = int(shard_metadata[0]["count"])
        scope_tile_count = int(shard_metadata[0]["scope_tile_count"])
        if any(int(item["count"]) != shard_count or int(item["scope_tile_count"]) != scope_tile_count for item in shard_metadata):
            errors.append("shards disagree on expected coverage")
        if sorted(int(item["index"]) for item in shard_metadata) != list(range(shard_count)):
            errors.append("shards do not cover every deterministic shard index exactly once")
    covered = sum(int(report.get("tile_count", 0)) for report in reports)
    if covered != scope_tile_count:
        errors.append(f"shards cover {covered} rows, expected {scope_tile_count}")

    signal_row_fingerprints: dict[int, str] = {}
    for report in reports:
        fingerprints = report.get("signal_row_fingerprints")
        if not isinstance(fingerprints, dict):
            errors.append("a shard lacks signal row fingerprints")
            continue
        if len(fingerprints) != int(report.get("tile_count", 0)):
            errors.append("a shard signal fingerprint count does not match its row count")
        for row_text, digest in fingerprints.items():
            try:
                row = int(row_text)
            except (TypeError, ValueError):
                errors.append("a shard has a non-integer signal fingerprint row")
                continue
            if row in signal_row_fingerprints:
                errors.append(f"duplicate signal fingerprint row {row}")
                continue
            if not isinstance(digest, str):
                errors.append(f"signal fingerprint for row {row} is not text")
                continue
            signal_row_fingerprints[row] = digest
    if len(signal_row_fingerprints) != scope_tile_count:
        errors.append(
            f"signal fingerprints cover {len(signal_row_fingerprints)} rows, expected {scope_tile_count}"
        )

    hard_failures: list[str] = []
    for report in reports:
        for failure in report.get("hard_failures") or []:
            if failure not in hard_failures:
                hard_failures.append(failure)
    for error in errors:
        if error not in hard_failures:
            hard_failures.append(error)
    normal_sum = sum(float(report.get("normal_nonzero_sum", 0.0)) for report in reports)
    normal_count = sum(int(report.get("normal_nonzero_count", 0)) for report in reports)
    edge_inside_sum = sum(float(report.get("rgb_edge_inside_sum", 0.0)) for report in reports)
    edge_inside_count = sum(int(report.get("rgb_edge_inside_count", 0)) for report in reports)
    edge_outside_sum = sum(float(report.get("rgb_edge_outside_sum", 0.0)) for report in reports)
    edge_outside_count = sum(int(report.get("rgb_edge_outside_count", 0)) for report in reports)
    panels: dict[str, str] = {}
    for report, shard in zip(reports, shard_metadata, strict=True):
        for split, path in (report.get("panels") or {}).items():
            panels[f"{split}_shard_{int(shard['index'])}"] = path

    merged_counters = {key: _sum_counters(reports, key) for key in COUNTER_KEYS}
    total_mcnk_nonzero = int(merged_counters["signal_counts"].get("mcnk_flag_nonzero_cells", 0))
    if total_mcnk_nonzero <= 0:
        if "3.3.5 MCNK flags are all zero across the audited scope" not in hard_failures:
            hard_failures.append("3.3.5 MCNK flags are all zero across the audited scope")
    normal_mean = normal_sum / max(normal_count, 1)
    if normal_count <= 0:
        if "native normals have no nonzero vectors across the audited scope" not in hard_failures:
            hard_failures.append("native normals have no nonzero vectors across the audited scope")
    elif not 0.75 <= normal_mean <= 1.25:
        normal_failure = f"native normal mean length {normal_mean:.6f} is outside the 0.75-1.25 contract"
        if normal_failure not in hard_failures:
            hard_failures.append(normal_failure)

    merged = dict(base)
    merged.update({
        "tile_count": covered,
        "scope_tile_count": scope_tile_count,
        "safe_for_m0_training": False,
        "safe_for_m0_build_local_training": not hard_failures,
        "shard_clean": not hard_failures,
        "partial_scope": False,
        "audit_shards": sorted(shard_metadata, key=lambda item: int(item["index"])),
        "hard_failures": hard_failures,
        "preflight_failed": False,
        "normal_nonzero_sum": normal_sum,
        "normal_nonzero_count": normal_count,
        "normal_nonzero_mean_length": normal_mean,
        "rgb_edge_inside_sum": edge_inside_sum,
        "rgb_edge_inside_count": edge_inside_count,
        "rgb_edge_mean_inside_object_target": edge_inside_sum / max(edge_inside_count, 1),
        "rgb_edge_outside_sum": edge_outside_sum,
        "rgb_edge_outside_count": edge_outside_count,
        "rgb_edge_mean_outside_object_target": edge_outside_sum / max(edge_outside_count, 1),
        "panels": panels,
        "placement_terrain_audit": {"status": "not_run"},
        "signal_row_fingerprints": {str(row): digest for row, digest in signal_row_fingerprints.items()},
        "scoped_signal_fingerprint": (
            combine_audited_signal_row_fingerprints(signal_row_fingerprints)
            if len(signal_row_fingerprints) == scope_tile_count
            else None
        ),
    })
    for key, value in merged_counters.items():
        merged[key] = value
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge deterministic Spec 102 signal-audit shards")
    parser.add_argument("--shard-report", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    merged = merge([_load(path) for path in args.shard_report])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output.resolve()),
        "safe_for_m0_build_local_training": merged["safe_for_m0_build_local_training"],
        "hard_failures": merged["hard_failures"],
        "scoped_signal_fingerprint": merged.get("scoped_signal_fingerprint"),
    }, indent=2))
    return 0 if merged["safe_for_m0_build_local_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
