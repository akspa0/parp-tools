from __future__ import annotations

import torch

from harvester.spec102.w1 import W1WdlResidual, WDL_SAMPLE_COUNT, masked_residual_l1


def test_w1_outputs_one_numeric_545_sample_vector() -> None:
    model = W1WdlResidual()
    output = model(torch.zeros(2, 3, 64, 64), torch.zeros(2))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (2, WDL_SAMPLE_COUNT)
    assert 3_000_000 <= parameters <= 12_000_000


def test_w1_masked_loss_ignores_invalid_samples() -> None:
    prediction = torch.tensor([[0.0, 100.0]])
    target = torch.tensor([[1.0, 0.0]])
    valid = torch.tensor([[True, False]])
    assert masked_residual_l1(prediction, target, valid).item() == 1.0
