"""Spec 119 segmenter model tests (T017)."""

from __future__ import annotations

import numpy as np
import torch

from harvester.spec119.segmenter_model import (
    ObjectSegmenter,
    binary_iou,
    per_coverage_bucket_iou,
    trivial_iou_baselines,
)


def test_forward_output_shape_and_range() -> None:
    model = ObjectSegmenter(base=16)
    out = model(torch.zeros(1, 3, 128, 128)).detach()
    assert out.shape == (1, 1, 128, 128)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0  # sigmoid output


def test_param_count_below_sc005_cap() -> None:
    model = ObjectSegmenter(base=16)
    params = sum(p.numel() for p in model.parameters())
    assert params < 1_000_000  # SC-005
    assert params > 10_000


def test_base_only_reconstruction_round_trip() -> None:
    torch.manual_seed(0)
    model = ObjectSegmenter(base=8)
    x = torch.zeros(1, 3, 128, 128)
    rebuilt = ObjectSegmenter(base=8)
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), rebuilt(x))


def test_binary_iou_edge_cases() -> None:
    target = np.zeros((4, 4), dtype=np.int64)
    target[0, 0] = 1
    assert binary_iou(target, target) == 1.0
    assert binary_iou(np.zeros((4, 4), dtype=np.int64), target) == 0.0
    # All-background vs all-background is a perfect (degenerate) match.
    assert binary_iou(np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64)) == 1.0


def test_trivial_iou_baselines() -> None:
    half = np.zeros((4, 4), dtype=np.int64)
    half[:2] = 1  # 0.5 coverage
    quarter = np.zeros((4, 4), dtype=np.int64)
    quarter[0, :1] = 1  # 0.0625 coverage
    baselines = trivial_iou_baselines([half, quarter])
    assert baselines["all_foreground"] == (0.5 + 0.0625) / 2
    assert baselines["all_background"] == 0.0
    assert trivial_iou_baselines([]) == {"all_foreground": 0.0, "all_background": 0.0}


def test_per_coverage_bucket_iou() -> None:
    buckets = per_coverage_bucket_iou([0.9, 0.5, 0.1], [0.6, 0.1, 0.01])
    assert buckets["[0.50,1.01)"]["count"] == 1
    assert buckets["[0.50,1.01)"]["mean_iou"] == 0.9
    assert buckets["[0.05,0.20)"]["mean_iou"] == 0.5
    assert buckets["[0.00,0.05)"]["mean_iou"] == 0.1
    assert buckets["[0.20,0.50)"]["count"] == 0
    assert buckets["[0.20,0.50)"]["mean_iou"] is None
