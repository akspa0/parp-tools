from __future__ import annotations

import json

import pytest
import torch

from harvester.v60.clean_signal_model import (
    CLEAN_SIGNAL_ARCHITECTURES,
    LEGACY_SPATIAL_PADDING_POLICY,
    CleanSignalModelError,
    build_clean_signal_model,
    build_clean_signal_model_from_identity,
    identity_json,
)


@pytest.mark.parametrize("architecture", CLEAN_SIGNAL_ARCHITECTURES)
def test_clean_signal_model_forward_backward_contract(architecture: str) -> None:
    torch.manual_seed(7137)
    model, identity = build_clean_signal_model(architecture, profile="tiny")
    model.train()
    input_tensor = torch.rand((2, 4, 256, 256), requires_grad=False)
    predictions = model(input_tensor)

    assert predictions.coarse_prediction_257.shape == (2, 257, 257)
    assert predictions.detail_prediction_257.shape == (2, 257, 257)
    assert predictions.height_prediction_257.shape == (2, 257, 257)
    assert torch.isfinite(predictions.coarse_prediction_257).all()
    assert torch.isfinite(predictions.detail_prediction_257).all()
    assert torch.isfinite(predictions.height_prediction_257).all()
    assert float(predictions.coarse_prediction_257.detach().min()) >= 0.0
    assert float(predictions.coarse_prediction_257.detach().max()) <= 1.0
    assert float(predictions.detail_prediction_257.detach().min()) >= -0.5
    assert float(predictions.detail_prediction_257.detach().max()) <= 0.5
    assert float(predictions.height_prediction_257.detach().min()) >= 0.0
    assert float(predictions.height_prediction_257.detach().max()) <= 1.0
    torch.testing.assert_close(
        predictions.height_prediction_257,
        torch.clamp(
            predictions.coarse_prediction_257 + predictions.detail_prediction_257,
            0.0,
            1.0,
        ),
    )
    assert identity["parameter_count"] == sum(parameter.numel() for parameter in model.parameters())

    loss = (
        predictions.coarse_prediction_257.mean()
        + predictions.detail_prediction_257.abs().mean()
        + predictions.height_prediction_257.mean()
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_clean_signal_model_rejects_forbidden_or_wrong_inputs() -> None:
    model, _ = build_clean_signal_model("unet_lite_v2", profile="tiny")

    with pytest.raises(CleanSignalModelError, match="only a 4D tensor"):
        model({"luma": torch.zeros((1, 1, 256, 256)), "height": torch.zeros((1, 1, 257, 257))})
    with pytest.raises(CleanSignalModelError, match="4, 256, 256"):
        model(torch.zeros((1, 5, 256, 256)))
    with pytest.raises(CleanSignalModelError, match="floating point"):
        model(torch.zeros((1, 4, 256, 256), dtype=torch.int64))
    with pytest.raises(CleanSignalModelError, match="non-finite"):
        invalid = torch.zeros((1, 4, 256, 256))
        invalid[:, :, 0, 0] = float("nan")
        model(invalid)


@pytest.mark.parametrize("architecture", CLEAN_SIGNAL_ARCHITECTURES)
def test_clean_signal_identity_serializes_and_reconstructs(architecture: str) -> None:
    torch.manual_seed(7137)
    model, identity = build_clean_signal_model(architecture, profile="tiny")
    serialized = identity_json(identity)
    assert json.loads(serialized) == identity

    rebuilt, rebuilt_identity = build_clean_signal_model_from_identity(json.loads(serialized))
    assert rebuilt_identity == identity
    rebuilt.load_state_dict(model.state_dict())
    model.eval()
    rebuilt.eval()
    input_tensor = torch.rand((1, 4, 256, 256))
    with torch.no_grad():
        original = model(input_tensor)
        restored = rebuilt(input_tensor)
    torch.testing.assert_close(original.coarse_prediction_257, restored.coarse_prediction_257)
    torch.testing.assert_close(original.detail_prediction_257, restored.detail_prediction_257)
    torch.testing.assert_close(original.height_prediction_257, restored.height_prediction_257)


def test_clean_signal_identity_rejects_tampered_configuration() -> None:
    _, identity = build_clean_signal_model("pyramid_cnn", profile="tiny")
    tampered = json.loads(identity_json(identity))
    tampered["config"]["fusion_channels"] = 99
    with pytest.raises(CleanSignalModelError, match="config_sha256"):
        build_clean_signal_model_from_identity(tampered)


@pytest.mark.parametrize("architecture", CLEAN_SIGNAL_ARCHITECTURES)
def test_reflective_padding_preserves_constant_input_fields(architecture: str) -> None:
    model, identity = build_clean_signal_model(architecture, profile="tiny")
    assert identity["config"]["spatial_padding"] == "reflect-3x3-v1"
    model.eval()
    input_tensor = torch.zeros((1, 4, 256, 256))
    input_tensor[:, 0] = 0.85
    input_tensor[:, 3] = 1.0
    with torch.no_grad():
        predictions = model(input_tensor)
    for field in (
        predictions.coarse_prediction_257,
        predictions.detail_prediction_257,
        predictions.height_prediction_257,
    ):
        assert float(field.max() - field.min()) < 1e-5


def test_legacy_zero_padding_identity_remains_reconstructable() -> None:
    model, identity = build_clean_signal_model(
        "pyramid_cnn",
        profile="tiny",
        spatial_padding=LEGACY_SPATIAL_PADDING_POLICY,
    )
    assert "spatial_padding" not in identity["config"]
    rebuilt, rebuilt_identity = build_clean_signal_model_from_identity(identity)
    rebuilt.load_state_dict(model.state_dict())
    assert rebuilt_identity == identity
