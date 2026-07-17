"""Spec 109 T028: selective-migration and resumability tests. Passing audit never overrides a
fresh-only policy or a blacklist (FR-006/FR-017); an interrupted migration can resume without
redoing or double-recording completed work."""

from __future__ import annotations

import numpy as np

from harvester.v50.migrate import (
    ACTION_COPY,
    ACTION_FRESH_EXTRACT_NEEDED,
    ACTION_UNAVAILABLE,
    MigrationLedger,
    copy_signal_row,
    plan_signal_migration,
)
from harvester.v50.verify_v18 import V18RowSignalObservation, V18SignalSpec, audit_v18_signal


def test_a_passed_copy_if_verified_row_is_planned_for_copy():
    spec = V18SignalSpec(name="height_257")
    audit = audit_v18_signal(spec, [V18RowSignalObservation(row_id=0, value=np.ones((4, 4), dtype=np.float32))])

    plan = plan_signal_migration(audit)

    assert plan == {0: ACTION_COPY}


def test_a_rejected_row_is_planned_for_fresh_extraction_not_copy():
    spec = V18SignalSpec(name="height_257")
    bad = np.full((4, 4), np.nan, dtype=np.float32)
    audit = audit_v18_signal(spec, [V18RowSignalObservation(row_id=0, value=bad)])

    plan = plan_signal_migration(audit)

    assert plan == {0: ACTION_FRESH_EXTRACT_NEEDED}


def test_a_passed_fresh_only_row_is_still_planned_for_fresh_extraction_not_copy():
    # FR-017: liquid_mask/liquid_height are fresh-only regardless of how clean the row looks.
    spec = V18SignalSpec(name="liquid_mask")
    audit = audit_v18_signal(spec, [V18RowSignalObservation(row_id=0, value=np.ones((4, 4), dtype=np.float32))])

    plan = plan_signal_migration(audit)

    assert plan == {0: ACTION_FRESH_EXTRACT_NEEDED}


def test_a_blacklisted_signal_is_planned_unavailable_for_every_row_regardless_of_content():
    spec = V18SignalSpec(name="holes_16", blacklisted=True, blacklist_reason="known-defective")
    audit = audit_v18_signal(
        spec,
        [
            V18RowSignalObservation(row_id=0, value=np.zeros((4, 4), dtype=np.float32)),
            V18RowSignalObservation(row_id=1, value=np.ones((4, 4), dtype=np.float32)),
        ],
    )

    plan = plan_signal_migration(audit)

    assert plan == {0: ACTION_UNAVAILABLE, 1: ACTION_UNAVAILABLE}


def test_copy_signal_row_is_bit_preserving_and_records_matching_source_dest_hashes():
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    ledger = MigrationLedger()

    copied, updated_ledger = copy_signal_row(0, "height_257", source, ledger)

    np.testing.assert_array_equal(copied, source)
    entry = updated_ledger.entry_for(0, "height_257")
    assert entry is not None
    assert entry.source_hash == entry.destination_hash  # bit-preserving by construction
    assert entry.action == "copied"


def test_copy_signal_row_resumes_instead_of_recomputing_an_already_ledgered_row(tmp_path):
    source = np.ones((4, 4), dtype=np.float32)
    ledger = MigrationLedger()
    _, ledger = copy_signal_row(0, "height_257", source, ledger)
    assert len(ledger.entries) == 1

    # Simulate resuming after an interruption: calling again for the same key must not add a
    # second, duplicate ledger entry.
    _, resumed_ledger = copy_signal_row(0, "height_257", source, ledger)

    assert len(resumed_ledger.entries) == 1
    assert resumed_ledger.entries == ledger.entries


def test_ledger_remaining_reports_only_uncompleted_keys():
    ledger = MigrationLedger()
    _, ledger = copy_signal_row(0, "height_257", np.ones((2, 2), dtype=np.float32), ledger)

    all_keys = frozenset({(0, "height_257"), (1, "height_257")})
    remaining = ledger.remaining(all_keys)

    assert remaining == frozenset({(1, "height_257")})
