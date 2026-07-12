from pathlib import Path
import importlib.util

import pytest
import torch

from harvester.v25.h1_coarse import H1CoarseReliefModel, parameter_count


def _trainer():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_v25_h1_coarse.py"
    spec = importlib.util.spec_from_file_location("train_v25_h1_coarse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_h1_has_one_coarse_residual_output_and_is_tiny():
    model = H1CoarseReliefModel()
    output = model(torch.rand(2, 3, 64, 64), torch.tensor([10.0, 20.0]))
    assert output.shape == (2, 33, 33)
    assert torch.equal(output, torch.zeros_like(output))
    assert parameter_count(model) < 50_000


def test_h1_gradient_reaches_model():
    model = H1CoarseReliefModel()
    output = model(torch.rand(2, 3, 64, 64), torch.tensor([10.0, 20.0]))
    (output - torch.ones_like(output)).abs().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_h1_requires_passed_h0_and_cuda(monkeypatch):
    trainer = _trainer()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer.validate_contract(3, "cuda", {"gate_pass": True})
    with pytest.raises(RuntimeError, match="H0 gate"):
        trainer.validate_contract(3, "cuda", {"gate_pass": False})
    with pytest.raises(ValueError, match="hard-capped"):
        trainer.validate_contract(4, "cuda", {"gate_pass": True})
    with pytest.raises(ValueError, match="CUDA-only"):
        trainer.validate_contract(3, "cpu", {"gate_pass": True})
