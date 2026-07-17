"""Spec 111 US3: the checkpoint-comparison record and its promotion rule.

See specs/111-minimap-lighting-calibration/data-model.md#ReconstructionCheckpointComparison and
contracts/minimap-lighting-calibration-contract.md (Training/evaluation execution contract items
3-4). The rule this module owns: a candidate checkpoint trained on rebalanced data is compared
against the currently deployed checkpoint on the same held-out split, an explicit outcome is
recorded before any promotion decision, and a regression can never promote.

Metric computation itself belongs to the existing Spec 108 trainer/eval flow; this module only
turns two already-computed held-out metric values into the recorded outcome, so the decision
logic is pure and testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OUTCOME_IMPROVED = "improved"
OUTCOME_REGRESSED = "regressed"
OUTCOME_INCONCLUSIVE = "inconclusive"

# Relative-change threshold below which a difference in held-out loss is treated as noise rather
# than a real improvement or regression.
DEFAULT_INCONCLUSIVE_RELATIVE_MARGIN = 0.01


@dataclass(frozen=True)
class ReconstructionCheckpointComparison:
    baseline_checkpoint: str
    candidate_checkpoint: str
    held_out_split: str
    baseline_metric: float
    candidate_metric: float
    outcome: str
    promotion_decision: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "baseline_checkpoint": self.baseline_checkpoint,
            "candidate_checkpoint": self.candidate_checkpoint,
            "held_out_split": self.held_out_split,
            "baseline_metric": self.baseline_metric,
            "candidate_metric": self.candidate_metric,
            "outcome": self.outcome,
            "promotion_decision": self.promotion_decision,
        }


def compare_checkpoints(
    *,
    baseline_checkpoint: str,
    candidate_checkpoint: str,
    held_out_split: str,
    baseline_metric: float,
    candidate_metric: float,
    inconclusive_relative_margin: float = DEFAULT_INCONCLUSIVE_RELATIVE_MARGIN,
) -> ReconstructionCheckpointComparison:
    """Compare two held-out loss values (lower is better) and record the promotion decision.

    ``promotion_decision`` is True only for a clear improvement; both regressed and inconclusive
    outcomes keep the currently deployed checkpoint (contract: a regression MUST NOT promote, and
    an inconclusive result gives no evidence the retrain helped).
    """
    if not baseline_checkpoint or not candidate_checkpoint:
        raise ValueError("Both baseline and candidate checkpoint identities are required.")
    if not (baseline_metric > 0.0) or not (candidate_metric >= 0.0):
        raise ValueError("Held-out metrics must be finite, non-negative loss values.")

    relative_change = (baseline_metric - candidate_metric) / baseline_metric
    if abs(relative_change) < inconclusive_relative_margin:
        outcome = OUTCOME_INCONCLUSIVE
    elif relative_change > 0.0:
        outcome = OUTCOME_IMPROVED
    else:
        outcome = OUTCOME_REGRESSED

    return ReconstructionCheckpointComparison(
        baseline_checkpoint=baseline_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        held_out_split=held_out_split,
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        outcome=outcome,
        promotion_decision=outcome == OUTCOME_IMPROVED,
    )
