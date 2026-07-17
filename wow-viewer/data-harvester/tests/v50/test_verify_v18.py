"""Spec 109 T015: per-signal V18 verification tests, including rejected holes_16 (FR-016/FR-017).
Passing one signal must never promote another signal in the same row -- every test here checks one
signal at a time and confirms the row-level (not whole-signal, not whole-store) verdict."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v50.verify_v18 import V18RowSignalObservation, V18SignalSpec, audit_v18_signal


def test_holes_16_is_always_rejected_regardless_of_its_actual_content():
    # FR-017: known-defective V18 hole masks must never be ported, even if a given row's array
    # looks perfectly fine in isolation.
    spec = V18SignalSpec(
        name="holes_16",
        blacklisted=True,
        blacklist_reason="uncorrected hole mask, known-defective per prior audit",
    )
    observations = [
        V18RowSignalObservation(row_id=0, value=np.zeros((16, 16), dtype=np.float32)),
        V18RowSignalObservation(row_id=1, value=np.ones((16, 16), dtype=np.float32)),
    ]

    result = audit_v18_signal(spec, observations)

    assert result.blacklisted is True
    assert result.migration_policy == "unavailable"
    assert result.passed_row_ids == ()
    assert result.rejected_row_ids == (0, 1)
    assert all(r.reason == spec.blacklist_reason for r in result.row_results)


def test_liquid_mask_is_fresh_only_even_when_every_row_passes_content_checks():
    # FR-017/research.md Decision 2: liquid_mask/liquid_height are fresh-only regardless of audit
    # outcome -- a sound-looking row does not unlock bit-preserving copy for these two signals.
    spec = V18SignalSpec(name="liquid_mask")
    observations = [V18RowSignalObservation(row_id=0, value=np.ones((257, 257), dtype=np.float32))]

    result = audit_v18_signal(spec, observations)

    assert result.migration_policy == "fresh-only"
    assert result.passed_row_ids == (0,)  # content-sound, but still fresh-only, not copy-eligible


def test_a_sound_core_signal_defaults_to_copy_if_verified():
    spec = V18SignalSpec(name="height_257")
    observations = [V18RowSignalObservation(row_id=0, value=np.ones((257, 257), dtype=np.float32))]

    result = audit_v18_signal(spec, observations)

    assert result.migration_policy == "copy-if-verified"
    assert result.passed_row_ids == (0,)


def test_rejects_a_row_with_a_lineage_gap_missing_payload():
    spec = V18SignalSpec(name="height_257")
    observations = [V18RowSignalObservation(row_id=0, value=None)]

    result = audit_v18_signal(spec, observations)

    assert result.row_results[0].passed is False
    assert result.row_results[0].reason == "lineage_gap_missing_payload"


def test_rejects_a_row_with_non_finite_values():
    spec = V18SignalSpec(name="height_257")
    bad_array = np.ones((4, 4), dtype=np.float32)
    bad_array[0, 0] = np.nan
    observations = [V18RowSignalObservation(row_id=0, value=bad_array)]

    result = audit_v18_signal(spec, observations)

    assert result.row_results[0].passed is False
    assert result.row_results[0].reason == "non_finite_values"


def test_rejects_a_row_whose_has_flag_claims_present_but_array_is_actually_empty():
    spec = V18SignalSpec(name="object_mask", has_flag_name="has_object_mask")
    observations = [
        V18RowSignalObservation(
            row_id=0,
            value=np.zeros((16, 16), dtype=np.float32),
            has_flag_value=True,
        )
    ]

    result = audit_v18_signal(spec, observations)

    assert result.row_results[0].passed is False
    assert "claims_present_but_empty" in result.row_results[0].reason


def test_rejects_a_row_whose_has_flag_claims_absent_but_array_is_actually_populated():
    spec = V18SignalSpec(name="object_mask", has_flag_name="has_object_mask")
    observations = [
        V18RowSignalObservation(
            row_id=0,
            value=np.ones((16, 16), dtype=np.float32),
            has_flag_value=False,
        )
    ]

    result = audit_v18_signal(spec, observations)

    assert result.row_results[0].passed is False
    assert "claims_absent_but_populated" in result.row_results[0].reason


def test_mixed_pass_and_fail_rows_are_reported_independently_not_as_one_verdict():
    # The edge case explicitly named in spec.md: "A verified V18 row contains one sound signal and
    # one known-defective signal" -- and, within one signal, some rows sound and some not.
    spec = V18SignalSpec(name="height_257")
    good = np.ones((4, 4), dtype=np.float32)
    bad = np.full((4, 4), np.nan, dtype=np.float32)
    observations = [
        V18RowSignalObservation(row_id=0, value=good),
        V18RowSignalObservation(row_id=1, value=bad),
        V18RowSignalObservation(row_id=2, value=good),
    ]

    result = audit_v18_signal(spec, observations)

    assert result.passed_row_ids == (0, 2)
    assert result.rejected_row_ids == (1,)


def test_blacklisted_signal_spec_requires_a_reason():
    with pytest.raises(ValueError, match="must state a reason"):
        V18SignalSpec(name="holes_16", blacklisted=True)
