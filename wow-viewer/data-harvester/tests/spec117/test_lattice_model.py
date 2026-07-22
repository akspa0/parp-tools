"""Spec 117 T011: encode/decode round-trip honesty, masked baseline correctness, row selection,
and LatticeNet's output contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.spec117.lattice_model import (
    LatticeNet,
    LatticeTargetError,
    compute_lattice_tile_mean_baseline,
    decode_lattice_target,
    encode_lattice_target,
    lattice_loss,
    select_lattice_rows,
)


def _full_lattice(value_fn):
    outer = np.fromfunction(lambda r, c: value_fn(r, c), (17, 17), dtype=np.float64)
    inner = np.fromfunction(lambda r, c: value_fn(r + 0.5, c + 0.5), (16, 16), dtype=np.float64)
    return outer.astype(np.float32), inner.astype(np.float32)


def test_encode_decode_round_trips_exactly_when_fully_present():
    outer, inner = _full_lattice(lambda r, c: 100.0 + r * 2.0 + c)
    outer_present = np.ones((17, 17), dtype=bool)
    inner_present = np.ones((16, 16), dtype=bool)

    target, mask, tile_min, tile_max = encode_lattice_target(outer, inner, outer_present, inner_present)
    assert mask.shape == (545,)
    assert np.all(mask == 1.0)

    decoded_outer, decoded_inner = decode_lattice_target(target, tile_min, tile_max)
    np.testing.assert_allclose(decoded_outer, outer, atol=1e-3)
    np.testing.assert_allclose(decoded_inner, inner, atol=1e-3)


def test_absent_samples_never_influence_tile_min_max():
    outer, inner = _full_lattice(lambda r, c: 100.0 + r * 2.0 + c)
    outer_present = np.ones((17, 17), dtype=bool)
    inner_present = np.ones((16, 16), dtype=bool)
    # True range is outer[0,0]=100 (global min) .. outer[16,16]=148 (global max). Blank out an
    # INTERIOR sample (neither the min nor the max) and give it an extreme garbage value: if the
    # encoder incorrectly folded it into min/max, tile_max would be corrupted to 10000.
    gap_index = (8, 8)  # value 132, not the global min or max

    outer_with_gap = outer.copy()
    outer_with_gap[gap_index] = 10_000.0
    outer_present_with_gap = outer_present.copy()
    outer_present_with_gap[gap_index] = False

    target_clean, _, min_clean, max_clean = encode_lattice_target(outer, inner, outer_present, inner_present)
    target_gap, mask_gap, min_gap, max_gap = encode_lattice_target(
        outer_with_gap, inner, outer_present_with_gap, inner_present
    )

    assert min_gap == min_clean == 100.0
    assert max_gap == max_clean == 148.0
    flat_gap_index = gap_index[0] * 17 + gap_index[1]
    assert mask_gap[flat_gap_index] == 0.0
    # Every OTHER present sample encodes identically regardless of the absent one's garbage value.
    kept = np.delete(np.arange(545), flat_gap_index)
    np.testing.assert_allclose(target_gap[kept], target_clean[kept], atol=1e-6)


def test_encode_raises_on_zero_present_samples():
    outer = np.zeros((17, 17), dtype=np.float32)
    inner = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(LatticeTargetError):
        encode_lattice_target(outer, inner, np.zeros((17, 17), dtype=bool), np.zeros((16, 16), dtype=bool))


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, index):
        return self._data[index]


class _FakeGroup(dict):
    pass


def test_select_lattice_rows_excludes_all_absent_tiles_and_counts_them():
    outer_present = np.zeros((3, 17, 17), dtype=bool)
    inner_present = np.zeros((3, 16, 16), dtype=bool)
    outer_present[0] = True  # row 0 usable
    outer_present[2, 5, 5] = True  # row 2 usable (single present sample)
    # row 1 stays entirely False -> excluded

    group = _FakeGroup(
        wdl_outer_present=_FakeArray(outer_present),
        wdl_inner_present=_FakeArray(inner_present),
    )
    usable, excluded = select_lattice_rows(group, [0, 1, 2])
    assert usable == [0, 2]
    assert excluded == 1


def test_masked_tile_mean_baseline_matches_hand_computation():
    target = np.array([0.0, 0.5, 1.0, 0.25], dtype=np.float32)
    mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)  # last sample absent
    # present values: 0.0, 0.5, 1.0 -> mean 0.5 -> MAE = mean(|0-0.5|,|0.5-0.5|,|1-0.5|) = mean(0.5,0,0.5) = 1/3
    baseline = compute_lattice_tile_mean_baseline([(target, mask)])
    assert baseline == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_lattice_loss_ignores_absent_samples():
    predicted = torch.tensor([[0.0, 0.9]])
    target = torch.tensor([[0.0, 0.1]])
    mask_all = torch.tensor([[1.0, 1.0]])
    mask_masked = torch.tensor([[1.0, 0.0]])
    assert lattice_loss(predicted, target, mask_all).item() > 0.0
    assert lattice_loss(predicted, target, mask_masked).item() == pytest.approx(0.0, abs=1e-6)


def test_lattice_net_forward_shape_and_range():
    model = LatticeNet(base=8)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 545)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_lattice_net_rejects_wrong_channel_count():
    model = LatticeNet(base=8)
    with pytest.raises(LatticeTargetError):
        model(torch.rand(1, 4, 256, 256))
