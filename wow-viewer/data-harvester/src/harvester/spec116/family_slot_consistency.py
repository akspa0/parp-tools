"""Spec 116 US1: measure how consistently each surface family occupies each layer slot.

This is the first user story and the feature's MVP: a single analysis, no model trained, that fixes
the output vocabulary of every later model. For every surface family ``f`` we measure the
distribution of the slot ordinal ``s`` it occupies across all chunk/slot rows in the corpus, then
compute a summary **consistency score** = the mean over families of ``max_s P(s | f)`` (a family is
"consistent" if it almost always lands in one slot). If the score is at/above a configurable
threshold (default 0.70), slot-keyed prediction is viable; if below, heads MUST key on family and
slot becomes a training-time grouping only (spec US1 acceptance 2).

The resulting recommendation is written to a durable ``v116-analysis-report-v1`` artifact (hash-bound
to the store + taxonomy) consumed verbatim by US3 (FR-002 / SC-001).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec116.relational_extract import extract_layer_entries
from harvester.spec116.structure_contract import sha256_file
from harvester.v50.terrain_feature_labels import FAMILY_NAMES, MAX_LAYERS

DEFAULT_THRESHOLD = 0.70
ANALYSIS_REPORT_SCHEMA = "v116-analysis-report-v1"


class FamilySlotConsistencyError(ValueError):
    """Raised when the family->slot consistency measurement cannot be produced honestly."""


def measure_family_slot_consistency(
    *,
    store: Path,
    dumps: Iterable[Path],
    threshold: float = DEFAULT_THRESHOLD,
    build_id: str = "",
) -> dict:
    """Measure family->slot consistency and return the ``v116-analysis-report-v1`` artifact.

    The artifact is returned in-memory; the CLI decides whether to write it. ``build_id`` is
    recorded for provenance and may be empty for a dry run.
    """
    if not 0.0 <= threshold <= 1.0:
        raise FamilySlotConsistencyError(f"threshold must be in [0, 1], got {threshold!r}")

    result = extract_layer_entries(store=store, dumps=dumps)
    counts = result.family_slot_counts()  # (CLASS_COUNT, MAX_LAYERS)

    per_family: list[dict] = []
    max_probs: list[float] = []
    for family in range(len(FAMILY_NAMES)):
        total = int(counts[family, :].sum())
        if total <= 0:
            # A family absent from the corpus contributes no slot evidence; skip it from the score
            # but still report its (zero) distribution.
            per_family.append({
                "family": FAMILY_NAMES[family],
                "slot_distribution": [0.0] * MAX_LAYERS,
                "max_slot_probability": 0.0,
            })
            continue
        dist = counts[family, :].astype(np.float64) / total
        max_p = float(dist.max())
        per_family.append({
            "family": FAMILY_NAMES[family],
            "slot_distribution": [float(x) for x in dist],
            "max_slot_probability": max_p,
        })
        max_probs.append(max_p)

    if not max_probs:
        raise FamilySlotConsistencyError("no family with any layer-entry rows; cannot score")

    summary = float(np.mean(max_probs))
    recommendation = "slot_keyed" if summary >= threshold else "family_keyed"

    dump_paths = [Path(d) for d in dumps]
    report = {
        "schema": ANALYSIS_REPORT_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report_kind": "family_slot_consistency",
        "identity": {
            "store": {"path": str(store.resolve()), "sha256": sha256_file(store / "index.parquet")},
            "taxonomy_revision": result.taxonomy_revision,
            "rule_set_sha256": result.rule_set_sha256,
            "texture_name_dumps": [
                {"path": str(d.resolve()), "sha256": sha256_file(d)} for d in dump_paths
            ],
        },
        "family_slot_consistency": {
            "per_family": per_family,
            "summary_consistency_score": summary,
            "threshold": float(threshold),
            "recommendation": recommendation,
        },
        "decision": {"kind": "vocabulary", "value": recommendation},
        # Provenance not validated by the schema but useful for the report:
        "row_count": result.row_count,
        "layer_entry_count": len(result.rows),
        "excluded": dict(result.excluded),
        "build_id": build_id,
    }
    # Self-check: the artifact we just built must satisfy the contract.
    from harvester.spec116.structure_contract import validate_analysis_report
    validate_analysis_report(report)
    return report


def recommendation_from_report(report: dict) -> str:
    """Read the durable vocabulary decision out of a US1 report (consumed by US3)."""
    return str(report["decision"]["value"])


__all__ = [
    "FamilySlotConsistencyError",
    "measure_family_slot_consistency",
    "recommendation_from_report",
    "DEFAULT_THRESHOLD",
    "ANALYSIS_REPORT_SCHEMA",
]
