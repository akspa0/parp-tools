"""Spec 118 T020 (US3): target derivation, model structure/gradients, base-only reconstruction,
and the IoU/recall metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.spec118.object_contract import CLASS_COUNT, architecture_identity
from harvester.spec118.object_segment_model import (
    ObjectSegmentError,
    ObjectSegmentNet,
    compute_class_weights,
    derive_class_target,
    per_class_iou_recall,
    visible_object_iou,
)


def test_derive_class_target_crops_top_left_and_keeps_values():
    source = np.zeros((257, 257), dtype=np.uint8)
    source[0, 0] = 1
    source[255, 255] = 2
    source[256, 256] = 1  # dropped by the 257->256 crop
    target = derive_class_target(source)
    assert target.shape == (256, 256)
    assert target.dtype == np.int64
    assert target[0, 0] == 1
    assert target[255, 255] == 2
    assert int((target == 1).sum()) == 1  # the (256,256) pixel did not leak in


def test_derive_class_target_refuses_bad_shape_and_vocabulary():
    with pytest.raises(ObjectSegmentError, match="expected"):
        derive_class_target(np.zeros((256, 256), dtype=np.uint8))
    bad = np.zeros((257, 257), dtype=np.uint8)
    bad[10, 10] = 9
    with pytest.raises(ObjectSegmentError, match="vocabulary"):
        derive_class_target(bad)


def test_forward_shape_and_skip_path_gradient():
    model = ObjectSegmentNet(base=8)
    x = torch.rand(2, 3, 256, 256, requires_grad=True)
    logits = model(x)
    assert logits.shape == (2, CLASS_COUNT, 256, 256)
    logits.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # A skip-decoder model must carry gradient into the early encoder blocks.
    assert model.enc1[0][0].weight.grad is not None
    assert model.enc5[0][0].weight.grad is not None


def test_constructable_from_base_alone_and_param_count_recorded():
    small = ObjectSegmentNet(base=8)
    identity = architecture_identity(small, architecture_id="object_segment_net", config={"base": 8})
    assert identity["parameter_count"] == sum(p.numel() for p in small.parameters())
    # Base-only reconstruction: same class, same base -> identical state_dict shapes (bridge path).
    rebuilt = ObjectSegmentNet(base=8)
    assert [p.shape for p in rebuilt.parameters()] == [p.shape for p in small.parameters()]


def test_per_class_iou_recall_hand_computed():
    target = np.array([[0, 0, 1], [1, 2, 2]])
    predicted = np.array([[0, 1, 1], [0, 2, 2]])
    metrics = per_class_iou_recall(predicted, target)
    # none: pred {0,3} wait -- compute: truth none at (0,0),(0,1); pred none at (0,0),(1,0).
    assert metrics["none"]["iou"] == pytest.approx(1 / 3)
    assert metrics["none"]["recall"] == pytest.approx(0.5)
    # doodad: truth (0,2),(1,0); pred (0,1),(0,2) -> intersection 1, union 3, recall 1/2.
    assert metrics["doodad"]["iou"] == pytest.approx(1 / 3)
    assert metrics["doodad"]["recall"] == pytest.approx(0.5)
    # building: perfect.
    assert metrics["building"]["iou"] == pytest.approx(1.0)


def test_per_class_absent_from_both_reports_none():
    metrics = per_class_iou_recall(np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int))
    assert metrics["doodad"]["iou"] is None
    assert metrics["building"]["recall"] is None
    assert metrics["none"]["iou"] == pytest.approx(1.0)


def test_visible_object_iou_union_metric():
    target = np.array([[0, 1, 0], [0, 0, 2]])
    predicted = np.array([[0, 1, 0], [0, 0, 0]])
    # union object pixels: truth (0,1),(2,2); pred (0,1) -> IoU 1/2.
    assert visible_object_iou(predicted, target) == pytest.approx(0.5)
    assert visible_object_iou(np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int)) is None


def test_compute_class_weights_caps_background():
    targets = [np.zeros((4, 4), dtype=np.int64)]
    targets[0][0, 0] = 1
    weights = compute_class_weights(targets)
    assert weights.shape == (CLASS_COUNT,)
    assert weights[0] <= 1.0  # capped background
    assert weights[1] > weights[0]  # rare doodad up-weighted
    assert weights[2] > weights[0]
