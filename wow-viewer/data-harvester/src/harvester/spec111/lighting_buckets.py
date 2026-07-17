"""Spec 111: ingest the shading-match fields streamed into a build's decoded metadata sidecar
(``decoded_metadata.parquet``, written by the C# harvester's streaming pathway -- see
``WowViewer.Tool.Harvest/Program.cs``'s ``AnalyzeAuthoredMinimapLighting``) and produce a per-map
and overall lighting-bucket distribution report.

Lane note: the active dataset lane is **v50** (Spec 109). The C# harvester computes shading-match
provenance at extraction time and attaches it to each tile's metadata regardless of lane; this
sidecar layout is the transport the harvester writes *today*, because Spec 109's clean-room V50
store builder does not exist yet (``v50_build_dataset.py`` fails closed). When that builder lands,
it must carry ``minimap_lighting`` (including these shading-match fields) as one of its
DatasetSignals, and this reader gains a V50-store path alongside -- not instead of -- the sidecar
it can already read. No new primary storage format is introduced (constitution principle V); the
report is a derived summary artifact, analogous to Spec 110's ``synthesis-manifest.json``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Fixed 3-hour bucket edges over one day; matches data-model.md's
# LightingBucketDistributionReport.BucketCounts contract.
BUCKET_EDGES_HOURS: list[float] = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]

MATCHED_STATUS = "matched"
LOW_CONFIDENCE_STATUSES = {"low_confidence_ambiguous", "low_confidence_flat_terrain"}
NOT_EVALUATED_STATUS = "not_evaluated"


def bucket_label(hours: float) -> str:
    for start, end in zip(BUCKET_EDGES_HOURS[:-1], BUCKET_EDGES_HOURS[1:]):
        if start <= hours < end:
            return f"{start:02.0f}-{end:02.0f}"
    return f"{BUCKET_EDGES_HOURS[-2]:02.0f}-{BUCKET_EDGES_HOURS[-1]:02.0f}"


def iter_decoded_metadata(store_path: Path) -> list[dict[str, Any]]:
    metadata_path = store_path / "decoded_metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No decoded_metadata.parquet under {store_path}. This report reads the sidecar "
            "written by Full/V22 streaming exports; run one of those exports for this build first."
        )

    table = pq.read_table(str(metadata_path))
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    rows: list[dict[str, Any]] = []
    for row_index in range(table.num_rows):
        row = {name: values[row_index] for name, values in columns.items()}
        payload = row.get("decoded_metadata_json", "")
        if not payload:
            continue
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        rows.append(decoded)
    return rows


class MapAccumulator:
    def __init__(self) -> None:
        self.bucket_counts: dict[str, int] = defaultdict(int)
        self.not_evaluated_count = 0
        self.low_confidence_count = 0
        self.total_eligible_tiles = 0
        self.build_fingerprints: set[str] = set()

    def add(self, shading_match: dict[str, Any]) -> None:
        status = shading_match.get("shading_match_status", NOT_EVALUATED_STATUS)
        build_fingerprint = shading_match.get("shading_match_build_fingerprint")

        if status == NOT_EVALUATED_STATUS:
            # Not scoped to this feature's 0.5.3.3368 target, or genuinely missing ground truth --
            # excluded from total_eligible_tiles per data-model.md, not counted as an outcome.
            return

        self.total_eligible_tiles += 1
        if build_fingerprint:
            self.build_fingerprints.add(str(build_fingerprint))

        if status == MATCHED_STATUS:
            hours = shading_match.get("shading_matched_time_of_day_hours")
            if hours is None:
                self.low_confidence_count += 1
                return
            self.bucket_counts[bucket_label(float(hours))] += 1
        elif status in LOW_CONFIDENCE_STATUSES:
            self.low_confidence_count += 1
        else:
            # Unrecognized status: treat conservatively as low-confidence rather than silently
            # dropping it from the reconciliation total.
            self.low_confidence_count += 1

    def to_report_row(self, map_name: str) -> dict[str, Any]:
        return {
            "build_fingerprint": sorted(self.build_fingerprints)[0] if self.build_fingerprints else None,
            "map_name": map_name,
            "bucket_counts": dict(sorted(self.bucket_counts.items())),
            "not_evaluated_count": self.not_evaluated_count,
            "low_confidence_count": self.low_confidence_count,
            "total_eligible_tiles": self.total_eligible_tiles,
        }


def build_report(store_path: Path, only_map: str | None = None) -> dict[str, Any]:
    overall = MapAccumulator()
    per_map: dict[str, MapAccumulator] = defaultdict(MapAccumulator)
    tiles_seen = 0
    tiles_without_shading_field = 0

    for decoded in iter_decoded_metadata(store_path):
        map_name = str(decoded.get("map_name") or "unknown")
        if only_map and map_name != only_map:
            continue

        minimap_lighting = decoded.get("minimap_lighting")
        tiles_seen += 1
        if not isinstance(minimap_lighting, dict) or "shading_match_status" not in minimap_lighting:
            # A tile from before this field existed, or with no minimap_lighting at all. Distinct
            # from an explicit not_evaluated status -- surfaced separately so a stale/partial store
            # is visible rather than silently folded into "not evaluated".
            tiles_without_shading_field += 1
            continue

        overall.add(minimap_lighting)
        per_map[map_name].add(minimap_lighting)

    report = {
        "store_path": str(store_path),
        "tiles_seen": tiles_seen,
        "tiles_without_shading_match_field": tiles_without_shading_field,
        "overall": overall.to_report_row("__all__"),
        "per_map": [per_map[name].to_report_row(name) for name in sorted(per_map)],
    }
    validate_report(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    """Enforce data-model.md's reconciliation invariant: every counted tile lands in exactly one
    of bucket_counts / not_evaluated_count / low_confidence_count."""
    for row in [report["overall"], *report["per_map"]]:
        bucket_total = sum(row["bucket_counts"].values())
        reconciled = bucket_total + row["not_evaluated_count"] + row["low_confidence_count"]
        if reconciled != row["total_eligible_tiles"]:
            raise AssertionError(
                f"Lighting bucket report failed to reconcile for map={row['map_name']!r}: "
                f"bucket_counts+not_evaluated+low_confidence={reconciled} != "
                f"total_eligible_tiles={row['total_eligible_tiles']}"
            )
