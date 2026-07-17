"""Spec 111 US3 coverage (T018): the checkpoint-comparison promotion rule. A regression can never
promote, and an inconclusive result also keeps the deployed checkpoint."""

from __future__ import annotations

import pytest

from harvester.spec111.checkpoint_comparison import (
    OUTCOME_IMPROVED,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_REGRESSED,
    compare_checkpoints,
)


def _compare(baseline_metric: float, candidate_metric: float):
    return compare_checkpoints(
        baseline_checkpoint="baseline.pt",
        candidate_checkpoint="candidate.pt",
        held_out_split="spec108-group-holdout",
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
    )


def test_clear_improvement_promotes():
    result = _compare(baseline_metric=1.0, candidate_metric=0.8)

    assert result.outcome == OUTCOME_IMPROVED
    assert result.promotion_decision is True


def test_regression_never_promotes():
    result = _compare(baseline_metric=1.0, candidate_metric=1.3)

    assert result.outcome == OUTCOME_REGRESSED
    assert result.promotion_decision is False


def test_within_noise_margin_is_inconclusive_and_does_not_promote():
    result = _compare(baseline_metric=1.0, candidate_metric=0.995)

    assert result.outcome == OUTCOME_INCONCLUSIVE
    assert result.promotion_decision is False


def test_missing_checkpoint_identity_is_rejected():
    with pytest.raises(ValueError, match="checkpoint identities"):
        compare_checkpoints(
            baseline_checkpoint="",
            candidate_checkpoint="candidate.pt",
            held_out_split="spec108-group-holdout",
            baseline_metric=1.0,
            candidate_metric=0.5,
        )


def test_metadata_round_trip_records_every_decision_field():
    result = _compare(baseline_metric=2.0, candidate_metric=1.0)

    metadata = result.to_metadata()

    assert metadata["outcome"] == OUTCOME_IMPROVED
    assert metadata["promotion_decision"] is True
    assert metadata["held_out_split"] == "spec108-group-holdout"
    assert metadata["baseline_metric"] == 2.0
    assert metadata["candidate_metric"] == 1.0
