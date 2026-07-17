"""V18 per-signal audit planning and known-defect rejection (Spec 109 T018, FR-016/FR-017).

FR-016: every signal is audited independently, per row. Passing one signal must never promote a
different signal in the same row -- so the unit of result here is (signal, row), not a single
verdict for a whole signal or a whole store. A blacklisted signal (``holes_16``: "uncorrected hole
masks" per FR-017) is rejected in every row without inspecting its content; every other signal is
checked per row for the concrete defects the prior audit found: missing/None payloads, non-finite
values, and a false ``has_*`` truthfulness flag (a row claims a signal is present while its array is
actually empty, or claims absence while the array actually carries content).

This module has no opinion on *which* signals exist or what their blacklist/has-flag wiring is --
that catalog is Spec 109 T002 (not yet frozen as of this pass) and is supplied by the caller as
``V18SignalSpec`` values, so nothing here hardcodes an assumption T002 hasn't made yet.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from harvester.v50.contracts import MigrationPolicy, migration_policy_for_signal


@dataclass(frozen=True)
class V18SignalSpec:
    name: str
    blacklisted: bool = False
    blacklist_reason: str = ""
    has_flag_name: str | None = None

    def __post_init__(self) -> None:
        if self.blacklisted and not self.blacklist_reason:
            raise ValueError(f"blacklisted signal {self.name!r} must state a reason")


@dataclass(frozen=True)
class V18RowSignalObservation:
    """One row's raw material for auditing one signal. ``value=None`` means the row has no payload
    for this signal at all (a lineage gap, not just an empty/zero array)."""

    row_id: int
    value: np.ndarray | None
    has_flag_value: bool | None = None


@dataclass(frozen=True)
class V18RowAuditResult:
    row_id: int
    passed: bool
    reason: str | None = None


@dataclass(frozen=True)
class V18SignalAuditResult:
    signal_name: str
    migration_policy: str
    blacklisted: bool
    row_results: tuple[V18RowAuditResult, ...] = field(default_factory=tuple)

    @property
    def passed_row_ids(self) -> tuple[int, ...]:
        return tuple(result.row_id for result in self.row_results if result.passed)

    @property
    def rejected_row_ids(self) -> tuple[int, ...]:
        return tuple(result.row_id for result in self.row_results if not result.passed)


def _audit_row(spec: V18SignalSpec, observation: V18RowSignalObservation) -> V18RowAuditResult:
    if observation.value is None:
        return V18RowAuditResult(observation.row_id, passed=False, reason="lineage_gap_missing_payload")

    if not np.all(np.isfinite(observation.value)):
        return V18RowAuditResult(observation.row_id, passed=False, reason="non_finite_values")

    if spec.has_flag_name is not None and observation.has_flag_value is not None:
        has_content = bool(np.any(observation.value != 0))
        if observation.has_flag_value and not has_content:
            return V18RowAuditResult(
                observation.row_id, passed=False, reason=f"false_{spec.has_flag_name}_claims_present_but_empty"
            )
        if not observation.has_flag_value and has_content:
            return V18RowAuditResult(
                observation.row_id, passed=False, reason=f"false_{spec.has_flag_name}_claims_absent_but_populated"
            )

    return V18RowAuditResult(observation.row_id, passed=True, reason=None)


def audit_v18_signal(
    spec: V18SignalSpec, observations: Iterable[V18RowSignalObservation]
) -> V18SignalAuditResult:
    """Audit one signal across all supplied rows. Passing this signal for a row says nothing about
    any other signal in that same row (FR-016) -- callers must not aggregate across signals here."""
    policy = migration_policy_for_signal(spec.name)

    if spec.blacklisted:
        results = tuple(
            V18RowAuditResult(observation.row_id, passed=False, reason=spec.blacklist_reason)
            for observation in observations
        )
        return V18SignalAuditResult(
            signal_name=spec.name,
            migration_policy=MigrationPolicy.UNAVAILABLE.value,
            blacklisted=True,
            row_results=results,
        )

    results = tuple(_audit_row(spec, observation) for observation in observations)
    return V18SignalAuditResult(
        signal_name=spec.name,
        migration_policy=policy,
        blacklisted=False,
        row_results=results,
    )
