"""Spec 119 classifier model tests (T011)."""

from __future__ import annotations

import numpy as np
import torch

from harvester.spec119.classifier_model import (
    ObjectClassifier,
    compute_class_weights,
    majority_class_baseline,
    per_class_precision_recall,
)


def test_forward_output_shape() -> None:
    model = ObjectClassifier(base=16)
    logits = model(torch.zeros(2, 3, 128, 128))
    assert logits.shape == (2, 4)
    embedding = model.embedding(torch.zeros(2, 3, 128, 128))
    assert embedding.shape == (2, 16 * 8)


def test_param_count_below_sc005_cap() -> None:
    model = ObjectClassifier(base=16)
    params = sum(p.numel() for p in model.parameters())
    assert params < 1_000_000  # SC-005: single-digit-millions ceiling, well under
    assert params > 10_000  # but not degenerate


def test_base_only_reconstruction_round_trip() -> None:
    torch.manual_seed(0)
    model = ObjectClassifier(base=8, num_classes=4)
    x = torch.zeros(1, 3, 128, 128)
    rebuilt = ObjectClassifier(base=8, num_classes=4)
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), rebuilt(x))


def test_compute_class_weights_inverse_frequency() -> None:
    labels = [1, 1, 1, 3, 3, 2]  # m2 x3, wmo x2, mdx x1, empty absent
    weights = compute_class_weights(labels, num_classes=4)
    assert weights[0] == 0.0  # absent class gets zero weight
    assert weights[2] > weights[3] > weights[1]  # rarer class -> larger weight
    assert np.isclose(weights[[1, 2, 3]].mean(), 1.0)


def test_majority_class_baseline() -> None:
    assert majority_class_baseline([1, 1, 1, 2]) == 0.75
    assert majority_class_baseline([]) == 0.0


def test_per_class_precision_recall() -> None:
    metrics = per_class_precision_recall([1, 1, 3], [1, 2, 3], num_classes=4)
    assert metrics[1]["precision"] == 0.5
    assert metrics[1]["recall"] == 1.0
    assert metrics[2]["recall"] is None or metrics[2]["recall"] == 0.0
    assert metrics[3]["recall"] == 1.0
