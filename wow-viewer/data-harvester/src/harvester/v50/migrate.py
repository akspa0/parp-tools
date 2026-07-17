"""Bit-preserving verified V18 copy, fresh-signal slots, row lineage, and a resume ledger
(Spec 109 T031, FR-006/research.md Decision 2).

Two responsibilities kept deliberately separate:

- ``plan_signal_migration`` turns one ``verify_v18.V18SignalAuditResult`` into a per-row decision
  (``copy`` only for rows that both passed audit *and* whose signal policy is
  ``copy-if-verified``; every other case -- rejected, blacklisted, or ``fresh-only`` policy
  regardless of audit outcome -- becomes ``fresh_extract_needed`` or ``unavailable``). This is pure
  decision logic: passing audit never overrides a ``fresh-only`` policy (FR-006/FR-017).
- ``copy_signal_row``/``MigrationLedger`` perform and track the actual bit-preserving copy for one
  (row, signal) pair, with resumability: a (row, signal) already recorded in the ledger is returned
  from the ledger rather than recomputed, so an interrupted migration can continue without redoing
  completed work or silently duplicating it.

Fresh extraction itself (invoking the existing C# harvester) is ``build.py``'s responsibility, not
this module's -- a migration plan can mark a signal ``fresh_extract_needed`` without this module
ever needing a client root or subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from harvester.v50.contracts import MigrationPolicy
from harvester.v50.identity import hash_array
from harvester.v50.verify_v18 import V18SignalAuditResult

ACTION_COPY = "copy"
ACTION_FRESH_EXTRACT_NEEDED = "fresh_extract_needed"
ACTION_UNAVAILABLE = "unavailable"


def plan_signal_migration(audit_result: V18SignalAuditResult) -> dict[int, str]:
    """Per-row migration decision for one signal. A row that passed the V18 audit only becomes
    ``copy`` eligible when the signal's own migration policy is ``copy-if-verified`` -- passing
    audit can never override ``fresh-only`` (liquid_mask/liquid_height) or a blacklist
    (holes_16), matching FR-006/FR-017 exactly."""
    if audit_result.blacklisted:
        return {row.row_id: ACTION_UNAVAILABLE for row in audit_result.row_results}

    decisions: dict[int, str] = {}
    for row in audit_result.row_results:
        if row.passed and audit_result.migration_policy == MigrationPolicy.COPY_IF_VERIFIED.value:
            decisions[row.row_id] = ACTION_COPY
        else:
            decisions[row.row_id] = ACTION_FRESH_EXTRACT_NEEDED
    return decisions


@dataclass(frozen=True)
class MigrationLedgerEntry:
    row_id: int
    signal_name: str
    action: str
    source_hash: str | None = None
    destination_hash: str | None = None


@dataclass(frozen=True)
class MigrationLedger:
    entries: tuple[MigrationLedgerEntry, ...] = field(default_factory=tuple)

    def completed_keys(self) -> frozenset[tuple[int, str]]:
        return frozenset((entry.row_id, entry.signal_name) for entry in self.entries)

    def remaining(self, all_keys: frozenset[tuple[int, str]]) -> frozenset[tuple[int, str]]:
        return all_keys - self.completed_keys()

    def entry_for(self, row_id: int, signal_name: str) -> MigrationLedgerEntry | None:
        for entry in self.entries:
            if entry.row_id == row_id and entry.signal_name == signal_name:
                return entry
        return None

    def append(self, entry: MigrationLedgerEntry) -> MigrationLedger:
        return MigrationLedger(entries=(*self.entries, entry))


def copy_signal_row(
    row_id: int,
    signal_name: str,
    source_array: np.ndarray,
    ledger: MigrationLedger,
) -> tuple[np.ndarray, MigrationLedger]:
    """Bit-for-bit copy of one row's signal array. Resumable: if (row_id, signal_name) is already
    in the ledger, returns the already-recorded result instead of recomputing -- an interrupted
    migration can pick back up without redoing (or double-recording) completed work."""
    existing = ledger.entry_for(row_id, signal_name)
    if existing is not None:
        return np.ascontiguousarray(source_array), ledger

    copied = np.array(source_array, copy=True)
    content_hash = hash_array(copied)
    entry = MigrationLedgerEntry(
        row_id=row_id,
        signal_name=signal_name,
        action="copied",
        source_hash=content_hash,
        destination_hash=content_hash,  # identical by construction: a bit-preserving copy
    )
    return copied, ledger.append(entry)
