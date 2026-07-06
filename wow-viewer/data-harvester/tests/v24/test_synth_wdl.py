"""Synthetic-WDL wrapper tests (FR-003/FR-004) — exercise the real C# shim."""

import numpy as np
import pytest

from harvester.v24 import lattice, synth_wdl

from .conftest import requires_shim


@pytest.mark.v24
@requires_shim
def test_synth_single_matches_lattice_rule(synthetic_height):
    outer, inner = synth_wdl.build_synth_wdl(synthetic_height)
    assert outer.shape == (17, 17)
    assert inner.shape == (16, 16)
    assert outer.dtype == np.float32

    # The C# path rounds to int16 at the exact lattice sample points.
    ref_outer, ref_inner = lattice.sample_lattice_from_height(synthetic_height)
    np.testing.assert_allclose(outer, np.round(ref_outer), atol=0.51)
    np.testing.assert_allclose(inner, np.round(ref_inner), atol=0.51)


@pytest.mark.v24
@requires_shim
def test_synth_batch(synthetic_height):
    stack = np.stack([synthetic_height, synthetic_height + 10.0])
    outer, inner = synth_wdl.build_synth_wdl_batch(stack)
    assert outer.shape == (2, 17, 17)
    assert inner.shape == (2, 16, 16)
    np.testing.assert_allclose(outer[1] - outer[0], 10.0, atol=1.01)


@pytest.mark.v24
@requires_shim
def test_synth_liquid_resampled(synthetic_height):
    liquid = np.zeros((256, 256), dtype=np.float32)
    liquid[:64, :64] = 1.0  # flood the top-left corner
    flooded = synthetic_height.copy()
    flooded[:64, :64] = -5000.0  # absurd bed heights strictly under the liquid

    outer_dry, _ = synth_wdl.build_synth_wdl(synthetic_height)
    outer_wet, _ = synth_wdl.build_synth_wdl(flooded, liquid)
    # Liquid lattice points were re-sampled from dry shoreline, not the bed.
    assert outer_wet[:4, :4].min() > -1000.0
    # Dry lattice points are untouched.
    np.testing.assert_allclose(outer_wet[8:, 8:], outer_dry[8:, 8:], atol=0.51)
