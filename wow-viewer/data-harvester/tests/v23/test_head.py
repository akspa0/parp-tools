from __future__ import annotations

import pytest
import torch

from harvester.v23.encoder import V23FeaturePyramid
from harvester.v23.head import V23HeightHead

pytestmark = pytest.mark.v23


def _synthetic_pyramid(batch_size: int = 2) -> V23FeaturePyramid:
    return V23FeaturePyramid(
        raw_feature_maps=(
            torch.randn(batch_size, 17, 64),
            torch.randn(batch_size, 17, 64),
            torch.randn(batch_size, 17, 64),
            torch.randn(batch_size, 17, 64),
        ),
        neck_features=(
            torch.randn(batch_size, 32, 4, 4),
            torch.randn(batch_size, 32, 8, 8),
            torch.randn(batch_size, 32, 16, 16),
            torch.randn(batch_size, 32, 32, 32),
        ),
        patch_height=4,
        patch_width=4,
        input_height=56,
        input_width=56,
    )


def _schema() -> dict[str, object]:
    return {
        "neck_features": [
            (1, 32, 4, 4),
            (1, 32, 8, 8),
            (1, 32, 16, 16),
            (1, 32, 32, 32),
        ]
    }


def test_head_outputs_disparity_and_affine_anchor() -> None:
    head = V23HeightHead(_schema())
    disparity, affine_anchor = head(_synthetic_pyramid())
    assert tuple(disparity.shape) == (2, 1, 257, 257)
    assert tuple(affine_anchor.shape) == (2, 2)


def test_head_trainable_param_count_stays_below_budget() -> None:
    head = V23HeightHead(_schema())
    trainable = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    assert trainable < 5_000_000


def test_head_disparity_and_metric_anchor_are_finite_and_bounded() -> None:
    head = V23HeightHead(_schema())
    disparity, affine_anchor = head(_synthetic_pyramid())
    scale = affine_anchor[:, 0:1].unsqueeze(-1).unsqueeze(-1)
    shift = affine_anchor[:, 1:2].unsqueeze(-1).unsqueeze(-1)
    metric_height = (disparity * scale) + shift

    assert disparity.dtype == torch.float32
    assert float(disparity.detach().min()) >= 0.0
    assert float(disparity.detach().max()) <= 1.0
    assert torch.isfinite(metric_height).all()
    assert not torch.allclose(metric_height, torch.zeros_like(metric_height))
