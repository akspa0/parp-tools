from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from harvester.v60.clean_signal_losses import (
    PARITY_CONFIG,
    STRUCTURAL_CONFIG,
    CleanSignalLossError,
    V7GuidanceConfig,
    clean_signal_loss,
    get_clean_signal_loss_config,
    loss_metrics,
)
from harvester.v60.clean_signal_model import CleanSignalPredictions


def _fields(size: int = 17) -> tuple[CleanSignalPredictions, dict[str, torch.Tensor]]:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, size),
        torch.linspace(0.0, 1.0, size),
        indexing="ij",
    )
    height = (0.35 * xx + 0.65 * yy).unsqueeze(0)
    coarse = height.clone()
    detail = torch.zeros_like(height)
    predictions = CleanSignalPredictions(coarse, detail, height)
    return predictions, {
        "relative_height_257": height,
        "coarse_relief_257": coarse,
        "detail_residual_257": detail,
    }


def test_identity_has_zero_every_component_and_total() -> None:
    predictions, targets = _fields()

    total, components = clean_signal_loss(predictions, targets, profile="v7_structural_v1")

    assert float(total) == pytest.approx(0.0, abs=1e-8)
    assert set(components) == {
        "point",
        "coarse_point",
        "detail_point",
        "gradient",
        "frequency",
        "laplacian",
        "edge",
        "transition",
        "border",
        "low_frequency",
        "high_frequency",
        "total",
    }
    assert all(float(value) == pytest.approx(0.0, abs=1e-8) for value in components.values())


def test_structural_profile_penalizes_smoothing_and_reports_metrics() -> None:
    size = 33
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    target_height = ((xx + yy) % 2).float().unsqueeze(0)
    predictions = CleanSignalPredictions(
        coarse_prediction_257=torch.zeros_like(target_height),
        detail_prediction_257=torch.zeros_like(target_height),
        height_prediction_257=torch.full_like(target_height, 0.5),
    )
    targets = {
        "relative_height_257": target_height,
        "coarse_relief_257": target_height,
        "detail_residual_257": torch.zeros_like(target_height),
    }

    parity_total, parity_components = clean_signal_loss(predictions, targets, profile="parity")
    structural_total, structural_components = clean_signal_loss(
        predictions,
        targets,
        profile="v7_structural_v1",
    )

    assert float(parity_total) > 0.0
    assert float(structural_total) > float(parity_total)
    assert float(structural_components["frequency"]) > 0.0
    assert float(structural_components["laplacian"]) > 0.0
    assert float(structural_components["edge"]) > 0.0
    assert float(structural_components["high_frequency"]) > 0.0
    assert float(parity_components["frequency"]) > 0.0  # disabled terms remain reportable metrics
    assert loss_metrics(structural_components)["total"] == pytest.approx(float(structural_total))


def test_clean_signal_loss_is_differentiable() -> None:
    predictions, targets = _fields()
    height = (predictions.height_prediction_257 + 0.1).clone().requires_grad_(True)
    predictions = CleanSignalPredictions(
        predictions.coarse_prediction_257,
        predictions.detail_prediction_257,
        height,
    )

    total, _ = clean_signal_loss(predictions, targets, profile="v7_structural_v1")
    total.backward()

    assert height.grad is not None
    assert torch.isfinite(height.grad).all()
    assert float(height.grad.abs().sum()) > 0.0


def test_component_isolation_and_profile_validation() -> None:
    predictions, targets = _fields()
    only_edge = replace(
        STRUCTURAL_CONFIG,
        name="only_edge",
        point=0.0,
        gradient=0.0,
        frequency=0.0,
        laplacian=0.0,
        edge=1.0,
        transition=0.0,
        border=0.0,
        low_frequency=0.0,
        high_frequency=0.0,
    )
    total, components = clean_signal_loss(
        predictions,
        targets,
        profile="only_edge",
        config=only_edge,
    )
    assert float(total) == pytest.approx(float(components["edge"]))
    assert get_clean_signal_loss_config("parity") == PARITY_CONFIG
    with pytest.raises(CleanSignalLossError, match="unknown loss profile"):
        get_clean_signal_loss_config("missing")
    with pytest.raises(CleanSignalLossError, match="at least one"):
        V7GuidanceConfig(name="empty", point=0.0, gradient=0.0)
