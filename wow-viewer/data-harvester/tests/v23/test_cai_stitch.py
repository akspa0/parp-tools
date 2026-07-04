from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from harvester.v23.inference import run_cai_inference

pytestmark = pytest.mark.v23


class _Output:
    def __init__(self, metric_height: torch.Tensor) -> None:
        self.metric_height = metric_height


class _FakeShiftSensitiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> _Output:
        base = F.interpolate(x[:, 0:1], size=(257, 257), mode="bilinear", align_corners=False)
        return _Output(base)


def _edge_l1(stitched: torch.Tensor) -> float:
    left = stitched[:, 256]
    right = stitched[:, 257]
    return float((left - right).abs().mean())


def test_run_cai_inference_reduces_shared_edge_discontinuity() -> None:
    model = _FakeShiftSensitiveModel()
    ramp = torch.linspace(0.0, 1.0, 256, dtype=torch.float32).view(1, 256).repeat(256, 1)
    tile = torch.zeros(15, 256, 256, dtype=torch.float32)
    tile[0] = ramp
    stitched_no_cai = run_cai_inference(model, [tile, tile], cai_r=1)
    stitched_cai = run_cai_inference(model, [tile, tile], cai_r=4)
    assert stitched_no_cai.shape == (257, 513)
    assert stitched_cai.shape == (257, 513)
    assert _edge_l1(stitched_cai) < _edge_l1(stitched_no_cai)
