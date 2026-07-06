"""Merge-rule unit tests (FR-005): real-agreeing, real-disagreeing,
missing-real, audit-empty."""

import numpy as np
import pytest

from harvester.v24 import lattice, merged_wdl_prior as m


def _grids(height):
    return lattice.sample_lattice_from_height(height)


@pytest.mark.v24
def test_real_agreeing_uses_real(synthetic_height):
    synth = _grids(synthetic_height)
    real = (np.round(synth[0]), np.round(synth[1]))  # int16-style quantization
    merged = m.build_merged_wdl_prior(
        synthetic_height, real, real_wdl_available=True, synth_wdl=synth
    )
    assert (merged.source_outer == m.SOURCE_REAL).all()
    assert (merged.source_inner == m.SOURCE_REAL).all()
    np.testing.assert_array_equal(merged.outer, real[0])
    assert (merged.confidence_outer == m.CONFIDENCE_REAL).all()
    assert merged.disagree_ratio == 0.0
    assert not merged.disagrees_with_real


@pytest.mark.v24
def test_real_disagreeing_falls_back_to_synth(synthetic_height):
    synth = _grids(synthetic_height)
    real_outer = synth[0].copy()
    real_outer[0, 0] += 50.0  # one wildly wrong cell
    merged = m.build_merged_wdl_prior(
        synthetic_height,
        (real_outer, synth[1]),
        real_wdl_available=True,
        synth_wdl=synth,
    )
    assert merged.source_outer[0, 0] == m.SOURCE_SYNTHETIC
    assert merged.outer[0, 0] == synth[0][0, 0]
    assert merged.confidence_outer[0, 0] == pytest.approx(m.CONFIDENCE_DISAGREE)
    assert merged.source_outer[1, 1] == m.SOURCE_REAL
    assert merged.disagrees_with_real
    assert merged.disagree_ratio == pytest.approx(1 / (289 + 256))


@pytest.mark.v24
def test_missing_real_is_all_synthetic(synthetic_height):
    synth = _grids(synthetic_height)
    merged = m.build_merged_wdl_prior(
        synthetic_height, None, real_wdl_available=False, synth_wdl=synth
    )
    assert (merged.source_outer == m.SOURCE_SYNTHETIC).all()
    assert (merged.confidence_inner == np.float32(m.CONFIDENCE_SYNTHETIC)).all()
    np.testing.assert_array_equal(merged.inner, synth[1])


@pytest.mark.v24
def test_audit_empty_learned_fill():
    height = np.zeros((257, 257), dtype=np.float32)
    synth = _grids(height)
    merged = m.build_merged_wdl_prior(
        height, None, real_wdl_available=False, synth_wdl=synth, audit_empty=True
    )
    assert (merged.source_outer == m.SOURCE_LEARNED_FILL).all()
    assert (merged.confidence_outer == 0.0).all()
    np.testing.assert_array_equal(merged.outer, np.zeros((17, 17), np.float32))


@pytest.mark.v24
def test_inclusive_threshold_boundary(synthetic_height):
    # Integer-valued grids mirror the real data path: both the C# synth path and
    # the client WDL are int16-quantized, so threshold-boundary diffs are exact.
    synth = tuple(np.round(g) for g in _grids(synthetic_height))
    real_outer = synth[0] + 1.0  # exactly at the threshold -> counts as agreeing
    merged = m.build_merged_wdl_prior(
        synthetic_height,
        (real_outer, synth[1]),
        real_wdl_available=True,
        synth_wdl=synth,
        disagree_threshold=1.0,
    )
    assert (merged.source_outer == m.SOURCE_REAL).all()
