"""Spec 111 US2 coverage (T010/T014): sampling-weight computation from a real distribution report,
no_real_baseline flagging for sparse buckets, and the input-contract guarantee that a rebalanced
lighting-times list can only ever carry bare floats into the existing synthetic-variant generator
-- never a lighting-bucket label, status, or other field the model could consume as an input
feature."""

from __future__ import annotations

import pytest

from harvester.spec111.rebalance_lighting_variants import (
    NO_REAL_BASELINE_POLICY,
    compute_sampling_plan,
    rebalanced_lighting_times,
)


def _report(bucket_counts: dict[str, int], build_fingerprint: str | None = "0.5.3.3368") -> dict:
    total = sum(bucket_counts.values())
    return {
        "overall": {
            "build_fingerprint": build_fingerprint,
            "bucket_counts": bucket_counts,
            "not_evaluated_count": 0,
            "low_confidence_count": 0,
            "total_eligible_tiles": total,
        },
    }


def test_compute_sampling_plan_weights_buckets_proportional_to_real_counts():
    report = _report({"06-09": 3, "12-15": 1})

    plan = compute_sampling_plan(report)

    assert plan.bucket_weights == pytest.approx({"06-09": 0.75, "12-15": 0.25})
    assert plan.source_build_fingerprint == "0.5.3.3368"
    assert plan.sparse_bucket_policy == NO_REAL_BASELINE_POLICY


def test_compute_sampling_plan_flags_zero_coverage_buckets_as_sparse_without_fabricating_weight():
    report = _report({"12-15": 4})

    plan = compute_sampling_plan(report)

    assert plan.bucket_weights == {"12-15": 1.0}
    # Every other 3-hour bucket has zero real examples and must be listed, not silently dropped
    # or given an invented uniform share.
    assert "00-03" in plan.sparse_buckets
    assert "12-15" not in plan.sparse_buckets
    assert len(plan.sparse_buckets) == 7


def test_compute_sampling_plan_with_no_real_examples_produces_no_weights():
    report = _report({})

    plan = compute_sampling_plan(report)

    assert plan.bucket_weights == {}
    assert len(plan.sparse_buckets) == 8


def test_rebalanced_lighting_times_allocates_exactly_the_requested_count():
    plan = compute_sampling_plan(_report({"06-09": 3, "12-15": 1}))

    times = rebalanced_lighting_times(plan, variant_count=8)

    assert len(times) == 8
    # 0.75/0.25 split of 8 -> 6 and 2 midpoint samples.
    assert times.count(7.5 / 24.0) == 6
    assert times.count(13.5 / 24.0) == 2


def test_rebalanced_lighting_times_output_is_bare_floats_only():
    # Input-contract guarantee (T014): the values handed to the existing synthetic-variant
    # generator's lighting_times= parameter can only ever be plain floats -- there is no code path
    # by which a bucket label, status string, or confidence value could reach that list, so the
    # rebalanced training data cannot carry ground-truth lighting/time as anything other than the
    # same bare game_time float an arbitrary/uniform sweep always used.
    plan = compute_sampling_plan(_report({"06-09": 1}))

    times = rebalanced_lighting_times(plan, variant_count=3)

    assert all(type(value) is float for value in times)
    assert all(0.0 <= value < 1.0 for value in times)


def test_rebalanced_lighting_times_rejects_a_plan_with_no_weights_instead_of_guessing():
    plan = compute_sampling_plan(_report({}))

    with pytest.raises(ValueError, match="sparse"):
        rebalanced_lighting_times(plan, variant_count=4)


def test_leak_safety_tags_preserved_is_true_by_construction():
    # This module never receives or touches per-tile source_group_id/lighting_variant_id fields --
    # it only ever sees aggregate bucket counts, so it structurally cannot introduce leakage.
    plan = compute_sampling_plan(_report({"06-09": 1}))

    assert plan.leak_safety_tags_preserved is True
