"""Spec 111 US2: turn a real lighting-bucket distribution (harvester.spec111.lighting_buckets)
into a synthetic-lighting-variant sampling plan, so training-time lighting matches the observed
real 0.5.3.3368 distribution instead of an arbitrary/uniform sweep.

The existing synthetic-variant generator (``scripts/spec103_build_synthetic_store.py``,
``build_synthetic_store(..., lighting_times=[...])``) already accepts an explicit list of
normalized game_time values and already owns all source-group/variant leak-safety tagging. This
module only computes *which* game_time values to request and in what proportion -- it has no
access to, and never touches, per-tile source-grouping fields, so it cannot introduce train/eval
leakage by construction (contracts/minimap-lighting-calibration-contract.md, Rebalancing contract
item 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harvester.spec111.lighting_buckets import BUCKET_EDGES_HOURS

NO_REAL_BASELINE_POLICY = "retain_existing_synthetic_coverage_no_real_baseline"


@dataclass(frozen=True)
class RebalancedTrainingSamplingPlan:
    """See specs/111-minimap-lighting-calibration/data-model.md#RebalancedTrainingSamplingPlan."""

    source_build_fingerprint: str | None
    bucket_weights: dict[str, float]
    sparse_bucket_policy: str
    sparse_buckets: list[str] = field(default_factory=list)
    leak_safety_tags_preserved: bool = True  # see module docstring: structurally guaranteed here

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_build_fingerprint": self.source_build_fingerprint,
            "bucket_weights": dict(self.bucket_weights),
            "sparse_bucket_policy": self.sparse_bucket_policy,
            "sparse_buckets": list(self.sparse_buckets),
            "leak_safety_tags_preserved": self.leak_safety_tags_preserved,
        }


def _all_bucket_labels() -> list[str]:
    return [
        f"{start:02.0f}-{end:02.0f}"
        for start, end in zip(BUCKET_EDGES_HOURS[:-1], BUCKET_EDGES_HOURS[1:])
    ]


def _bucket_midpoint_game_time(bucket_label: str) -> float:
    start_hours, end_hours = (float(part) for part in bucket_label.split("-"))
    return ((start_hours + end_hours) / 2.0) / 24.0


def compute_sampling_plan(report: dict[str, Any]) -> RebalancedTrainingSamplingPlan:
    """Derive bucket sampling weights from a ``harvester.spec111.lighting_buckets.build_report``
    result's ``overall`` row. Buckets with zero real examples get no fabricated weight; they are
    recorded in ``sparse_buckets`` under the explicit ``NO_REAL_BASELINE_POLICY`` instead."""
    overall = report["overall"]
    bucket_counts: dict[str, int] = overall.get("bucket_counts", {})
    total_real_examples = sum(bucket_counts.values())

    all_buckets = _all_bucket_labels()
    sparse_buckets = [label for label in all_buckets if bucket_counts.get(label, 0) == 0]

    if total_real_examples == 0:
        # No real coverage at all: every bucket is sparse. Do not fabricate a uniform weighting --
        # callers must fall back to their own pre-existing (documented arbitrary) sweep policy.
        return RebalancedTrainingSamplingPlan(
            source_build_fingerprint=overall.get("build_fingerprint"),
            bucket_weights={},
            sparse_bucket_policy=NO_REAL_BASELINE_POLICY,
            sparse_buckets=sparse_buckets,
        )

    weights = {
        label: count / total_real_examples
        for label, count in bucket_counts.items()
        if count > 0
    }

    return RebalancedTrainingSamplingPlan(
        source_build_fingerprint=overall.get("build_fingerprint"),
        bucket_weights=weights,
        sparse_bucket_policy=NO_REAL_BASELINE_POLICY,
        sparse_buckets=sparse_buckets,
    )


def rebalanced_lighting_times(plan: RebalancedTrainingSamplingPlan, variant_count: int) -> list[float]:
    """Expand a sampling plan into ``variant_count`` normalized game_time values (each in [0, 1)),
    proportioned by ``bucket_weights``, suitable for passing directly as
    ``build_synthetic_store(..., lighting_times=...)``. Each bucket contributes its own midpoint
    game_time, repeated according to its weighted share of ``variant_count``."""
    if variant_count <= 0:
        raise ValueError("variant_count must be positive")
    if not plan.bucket_weights:
        raise ValueError(
            "No real-example bucket weights available (every bucket is sparse); rebalancing "
            "cannot proceed without a documented explicit fallback policy from the caller."
        )

    # Largest-remainder allocation so integer variant counts sum exactly to variant_count instead
    # of drifting from naive per-bucket rounding.
    labels = sorted(plan.bucket_weights)
    raw_allocations = {label: plan.bucket_weights[label] * variant_count for label in labels}
    base_counts = {label: int(raw_allocations[label]) for label in labels}
    remaining = variant_count - sum(base_counts.values())
    remainders = sorted(labels, key=lambda label: raw_allocations[label] - base_counts[label], reverse=True)
    for label in remainders[:remaining]:
        base_counts[label] += 1

    times: list[float] = []
    for label in labels:
        times.extend([_bucket_midpoint_game_time(label)] * base_counts[label])
    return times
