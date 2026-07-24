"""Spec 121 T008: tile-level object-mask loss weighting tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from harvester.spec121.object_mask_tile_loss import (
    OBJECT_MASK_ARRAY,
    OBJECT_TOUCHED_COVERAGE,
    ObjectMaskLossError,
    coverage_weights,
    load_tile_coverages,
    touched_untouched_mae,
    weighted_lattice_loss,
)


class FakeGroup:
    def __init__(self, masks: dict[int, np.ndarray] | None) -> None:
        self._masks = masks

    def __contains__(self, name) -> bool:
        return name == OBJECT_MASK_ARRAY and self._masks is not None

    def __getitem__(self, name):
        assert name == OBJECT_MASK_ARRAY
        return _MaskRows(self._masks)


class _MaskRows:
    def __init__(self, masks: dict[int, np.ndarray]) -> None:
        self._masks = masks

    def __getitem__(self, row: int) -> np.ndarray:
        return self._masks[row]


def test_load_tile_coverages_computes_marked_fraction():
    group = FakeGroup({
        0: np.zeros((4, 4), dtype=np.float32),
        1: np.ones((4, 4), dtype=np.float32),
        2: np.concatenate([np.ones((2, 4)), np.zeros((2, 4))]).astype(np.float32),
    })
    cov = load_tile_coverages(group, [0, 1, 2])
    assert cov.dtype == np.float32
    assert list(cov) == [0.0, 1.0, 0.5]


def test_load_tile_coverages_refuses_missing_array():
    with pytest.raises(ObjectMaskLossError):
        load_tile_coverages(FakeGroup(None), [0])


def test_coverage_weights_math_and_bounds():
    weights = coverage_weights(torch.tensor([0.0, 0.5, 1.0]), 1.0)
    assert torch.allclose(weights, torch.tensor([1.0, 0.5, 0.0]))
    weights_half = coverage_weights(torch.tensor([1.0]), 0.5)
    assert torch.allclose(weights_half, torch.tensor([0.5]))
    with pytest.raises(ObjectMaskLossError):
        coverage_weights(torch.tensor([0.5]), 1.5)


def _fixture_batch():
    predicted = torch.zeros(2, 4)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    return predicted, target, mask


def test_weighted_loss_at_weight_zero_equals_plain_per_sample_mean():
    predicted, target, mask = _fixture_batch()
    coverages = torch.tensor([0.0, 1.0])
    got = weighted_lattice_loss(predicted, target, mask, coverages, 0.0)
    per_element = functional.smooth_l1_loss(predicted, target, reduction="none")
    per_sample = (per_element * mask).sum(dim=1) / mask.sum(dim=1)
    assert torch.allclose(got, per_sample.mean())


def test_weighted_loss_downweights_contaminated_tiles():
    predicted, target, mask = _fixture_batch()
    # Tile 1 (fully contaminated, weight 0) contributes nothing at w=1.
    got = weighted_lattice_loss(predicted, target, mask, torch.tensor([0.0, 1.0]), 1.0)
    per_element = functional.smooth_l1_loss(predicted, target, reduction="none")
    per_sample = (per_element * mask).sum(dim=1) / mask.sum(dim=1)
    assert torch.allclose(got, per_sample[0])


def test_all_object_batch_falls_back_to_finite_unweighted_mean():
    predicted, target, mask = _fixture_batch()
    got = weighted_lattice_loss(predicted, target, mask, torch.tensor([1.0, 1.0]), 1.0)
    assert torch.isfinite(got)
    unweighted = weighted_lattice_loss(predicted, target, mask, torch.tensor([1.0, 1.0]), 0.0)
    assert torch.allclose(got, unweighted)


def test_touched_untouched_split():
    mae = np.array([0.1, 0.2, 0.3, 0.4])
    cov = np.array([0.0, 0.01, OBJECT_TOUCHED_COVERAGE, 0.9])
    result = touched_untouched_mae(mae, cov)
    assert result["touched_tiles"] == 2
    assert result["untouched_tiles"] == 2
    assert result["touched_mae"] == pytest.approx(0.35)
    assert result["untouched_mae"] == pytest.approx(0.15)


def test_touched_untouched_all_one_side_yields_none():
    result = touched_untouched_mae(np.array([0.1]), np.array([0.0]))
    assert result["touched_mae"] is None
    assert result["untouched_mae"] == pytest.approx(0.1)


def test_touched_untouched_rejects_shape_mismatch():
    with pytest.raises(ObjectMaskLossError):
        touched_untouched_mae(np.array([0.1, 0.2]), np.array([0.0]))
