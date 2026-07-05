from __future__ import annotations

import pytest
import torch

from harvester.v23.losses import (
    affine_invariant_lssi,
    apply_bias_free_masking,
    compute_v23_loss,
    gpct_overlap_consistency,
    gradient_matching_lgm,
    spatial_distance_constraint,
)
from harvester.v23.model import V23ModelOutput

pytestmark = pytest.mark.v23


def _sample_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = torch.randn(2, 1, 257, 257, requires_grad=True)
    target = torch.randn(2, 1, 257, 257)
    mask = torch.ones(2, 1, 257, 257)
    mask[:, :, :16, :16] = 0.0
    return pred, target, mask


def test_affine_invariant_lssi_is_nonnegative_and_backpropagates() -> None:
    pred, target, mask = _sample_tensors()
    loss = affine_invariant_lssi(pred, target, mask)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_gradient_matching_lgm_is_nonnegative_and_backpropagates() -> None:
    pred, target, mask = _sample_tensors()
    loss = gradient_matching_lgm(pred, target, mask)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_spatial_distance_constraint_is_nonnegative_and_backpropagates() -> None:
    pred, target, mask = _sample_tensors()
    loss = spatial_distance_constraint(pred, target, patch_size=16, mask=mask)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_gpct_overlap_consistency_distributes_gradients() -> None:
    preds = [torch.randn(2, 1, 257, 257, requires_grad=True) for _ in range(4)]
    features = [torch.randn(2, 1, 257, 257, requires_grad=True) for _ in range(4)]
    overlaps = [
        (0, 1, 0, 64, 0, 0, 128, 128, 0, 0),
        (0, 2, 64, 0, 0, 0, 128, 128, 0, 0),
        (1, 3, 64, 0, 0, 0, 128, 128, 0, 0),
        (2, 3, 0, 64, 0, 0, 128, 128, 0, 0),
    ]
    loss = gpct_overlap_consistency(preds, features, overlaps, feature_loss=True)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert sum(t.grad.abs().sum().item() for t in preds if t.grad is not None) > 0.0
    assert sum(t.grad.abs().sum().item() for t in features if t.grad is not None) > 0.0


def test_bias_free_masking_returns_same_shape_and_patch_mask() -> None:
    x = torch.rand(2, 15, 256, 256)
    masked, patch_mask = apply_bias_free_masking(x, ratio=0.15, generator=torch.Generator().manual_seed(7))
    assert tuple(masked.shape) == (2, 15, 256, 256)
    assert tuple(patch_mask.shape) == (2, 16, 16)
    assert patch_mask.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_bias_free_masking_accepts_cpu_generator_for_cuda_tensor() -> None:
    x = torch.rand(1, 15, 256, 256, device="cuda")
    masked, patch_mask = apply_bias_free_masking(x, ratio=0.15, generator=torch.Generator().manual_seed(7))
    assert masked.device.type == "cuda"
    assert patch_mask.device.type == "cuda"
    assert tuple(masked.shape) == (1, 15, 256, 256)


def test_compute_v23_loss_bypasses_zero_weight_gpct() -> None:
    pred, target, mask = _sample_tensors()
    outputs = V23ModelOutput(
        disparity=pred,
        affine_anchor=torch.ones(2, 2),
        metric_height=pred.clone(),
        features=None,  # type: ignore[arg-type]
    )
    total, components = compute_v23_loss(
        outputs,
        target,
        {"affine": 1.0, "gradient": 0.5, "sdc": 0.1, "gpct": 0.0},
        valid_mask=mask,
    )
    assert float(total.detach()) >= 0.0
    assert torch.equal(components["gpct"], torch.zeros_like(components["gpct"]))
