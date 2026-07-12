from pathlib import Path
import importlib.util

import pytest
import torch

from harvester.v25.h0_offset import H0OffsetModel, OFFSET_SCALE, parameter_count


def _trainer():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_v25_h0_offset.py"
    spec = importlib.util.spec_from_file_location("train_v25_h0_offset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_h0_has_one_scalar_output_and_is_tiny():
    model = H0OffsetModel()
    output = model(torch.rand(4, 3, 64, 64))
    assert output.shape == (4,)
    assert parameter_count(model) < 25_000


def test_h0_gradient_reaches_single_output_model():
    model = H0OffsetModel()
    loss = model(torch.rand(2, 3, 64, 64)).abs().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_h0_zero_initialization_starts_at_rgb_flat_residual_zero():
    model = H0OffsetModel()
    assert torch.equal(model(torch.rand(2, 3, 64, 64)), torch.zeros(2))
    assert OFFSET_SCALE == 256.0


def test_h0_contract_caps_epochs_and_refuses_cpu(monkeypatch):
    trainer = _trainer()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer.validate_run_contract(3, "cuda")
    with pytest.raises(ValueError, match="hard-capped"):
        trainer.validate_run_contract(4, "cuda")
    with pytest.raises(ValueError, match="CUDA-only"):
        trainer.validate_run_contract(3, "cpu")


def test_h0_input_manifest_allows_rgb_only():
    trainer = _trainer()
    manifest = trainer.validate_model_inputs(H0OffsetModel())
    assert manifest["deployment_inputs"] == ["minimap_rgb"]
    assert manifest["output_signal"] == "tile_offset_residual"
    assert "height_257" in manifest["target_only"]
