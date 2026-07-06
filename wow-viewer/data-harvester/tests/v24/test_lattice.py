"""Lattice geometry tests (spec amendment A6)."""

import numpy as np
import pytest

from harvester.v24 import lattice


@pytest.mark.v24
def test_sample_lattice_positions(synthetic_height):
    outer, inner = lattice.sample_lattice_from_height(synthetic_height)
    assert outer.shape == (17, 17)
    assert inner.shape == (16, 16)
    assert outer[3, 5] == synthetic_height[48, 80]
    assert inner[3, 5] == synthetic_height[56, 88]


@pytest.mark.v24
def test_sample_lattice_batch(synthetic_height):
    stack = np.stack([synthetic_height, synthetic_height * 2.0])
    outer, inner = lattice.sample_lattice_from_height(stack)
    assert outer.shape == (2, 17, 17)
    assert inner.shape == (2, 16, 16)
    np.testing.assert_allclose(outer[1], outer[0] * 2.0)


@pytest.mark.v24
def test_quincunx_carries_lattice_points(synthetic_height):
    outer, inner = lattice.sample_lattice_from_height(synthetic_height)
    q = lattice.quincunx_33(outer, inner)
    assert q.shape == (33, 33)
    np.testing.assert_array_equal(q[::2, ::2], outer)
    np.testing.assert_array_equal(q[1::2, 1::2], inner)
    # Half-step positions are neighbourhood means, so they stay within range.
    assert q.min() >= min(outer.min(), inner.min()) - 1e-4
    assert q.max() <= max(outer.max(), inner.max()) + 1e-4


@pytest.mark.v24
def test_upsample_exact_at_lattice_points(synthetic_height):
    outer, inner = lattice.sample_lattice_from_height(synthetic_height)
    up = lattice.upsample_prior_257(outer, inner)
    assert up.shape == (257, 257)
    np.testing.assert_allclose(up[::16, ::16], outer, atol=1e-4)
    np.testing.assert_allclose(up[8::16, 8::16], inner, atol=1e-4)


@pytest.mark.v24
def test_upsample_flat_grid_is_flat():
    outer = np.full((17, 17), 42.0, dtype=np.float32)
    inner = np.full((16, 16), 42.0, dtype=np.float32)
    up = lattice.upsample_prior_257(outer, inner)
    np.testing.assert_allclose(up, 42.0, atol=1e-4)
