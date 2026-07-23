"""Spec 118 T015 (US2): object-masked-loss helper math -- parity at w=0, exact zeroing at w=1,
partial weighting, weight clamping, touched-row selection, and the FR-008 subset summary."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from harvester.spec118.object_loss import (
    OBJECT_MASK_ARRAY,
    clamp_weight,
    mask_touched,
    object_mask_available,
    object_point_weight,
    object_touched_rows,
    subset_metrics,
)


def test_weight_zero_is_exact_parity_ones():
    mask = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    weight = object_point_weight(mask, 0.0)
    assert np.array_equal(weight, np.ones_like(mask))


def test_weight_one_zeroes_exactly_the_visible_pixels():
    mask = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    weight = object_point_weight(mask, 1.0)
    assert np.array_equal(weight, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_partial_weight_scales_linearly():
    mask = np.array([[1.0, 0.5]], dtype=np.float32)
    weight = object_point_weight(mask, 0.4)
    assert weight[0, 0] == pytest.approx(0.6)
    assert weight[0, 1] == pytest.approx(0.8)


def test_weight_is_clamped_into_unit_interval():
    assert clamp_weight(-0.5) == 0.0
    assert clamp_weight(1.5) == 1.0
    mask = np.ones((2, 2), dtype=np.float32)
    assert np.array_equal(object_point_weight(mask, -0.5), np.ones_like(mask))
    assert np.array_equal(object_point_weight(mask, 1.5), np.zeros_like(mask))


def test_mask_touched():
    assert mask_touched(np.zeros((4, 4), dtype=np.float32)) is False
    touched = np.zeros((4, 4), dtype=np.float32)
    touched[3, 3] = 1.0
    assert mask_touched(touched) is True


def test_object_touched_rows_and_availability(tmp_path):
    store = tmp_path / "store.zarr"
    group = zarr.open_group(str(store), mode="w")
    assert object_mask_available(group) is False

    masks = np.zeros((3, 4, 4), dtype=np.float32)
    masks[1, 0, 0] = 1.0
    group.create_array(OBJECT_MASK_ARRAY, data=masks)
    assert object_mask_available(group) is True
    assert object_touched_rows(group, [0, 1, 2]) == [False, True, False]


def test_subset_metrics_shape_and_null_handling():
    metrics = subset_metrics(10.0, 100, 30.0, 200, weight=1.0)
    assert metrics["object_touched_region_mae"] == pytest.approx(0.1)
    assert metrics["object_untouched_region_mae"] == pytest.approx(0.15)
    assert metrics["object_touched_pixels"] == 100
    assert metrics["object_mask_weight"] == 1.0
    assert metrics["source_array"] == OBJECT_MASK_ARRAY

    empty = subset_metrics(0.0, 0, 5.0, 50, weight=0.0)
    assert empty["object_touched_region_mae"] is None
    assert empty["object_untouched_region_mae"] == pytest.approx(0.1)
