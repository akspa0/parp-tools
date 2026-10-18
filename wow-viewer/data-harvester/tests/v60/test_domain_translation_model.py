"""Unit tests for DomainTranslationUNet & Composite Loss (Spec 274)."""

from __future__ import annotations

import pytest
import torch

from harvester.v60.domain_translation_model import (
    DomainTranslationLoss,
    DomainTranslationUNet,
    SobelGradientLoss,
    SSIMLoss,
)


def test_domain_translation_unet_parameter_budget() -> None:
    model = DomainTranslationUNet(in_channels=3, out_channels=3)
    param_count = model.count_parameters()
    assert param_count > 100_000, "Model is suspiciously small"
    assert param_count <= 10_000_000, f"Parameter count {param_count} exceeds 10M budget!"


def test_domain_translation_unet_forward_shape_and_bounds() -> None:
    model = DomainTranslationUNet(in_channels=3, out_channels=3)
    model.eval()

    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 256, 256)
    assert 0.0 <= float(out.min())
    assert float(out.max()) <= 1.0


def test_domain_translation_loss_components() -> None:
    criterion = DomainTranslationLoss(l1_weight=1.0, grad_weight=0.5, ssim_weight=0.3)

    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)

    loss, metrics = criterion(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert float(loss.item()) > 0.0

    assert "loss_l1" in metrics
    assert "loss_grad" in metrics
    assert "loss_ssim" in metrics
    assert metrics["loss_l1"] >= 0.0
    assert metrics["loss_grad"] >= 0.0
    assert metrics["loss_ssim"] >= 0.0


def test_backward_gradient_flow() -> None:
    model = DomainTranslationUNet(in_channels=3, out_channels=3)
    model.train()
    criterion = DomainTranslationLoss()

    x = torch.rand(2, 3, 256, 256, requires_grad=True)
    target = torch.rand(2, 3, 256, 256)

    out = model(x)
    loss, _ = criterion(out, target)
    loss.backward()

    # Verify gradients exist and are finite
    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"
            assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf gradient"
            has_grad = True

    assert has_grad is True
