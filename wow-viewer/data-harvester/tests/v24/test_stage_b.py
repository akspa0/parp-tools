"""Stage B model tests (FR-016/FR-019): shape, param cap, gating, upsample."""

import numpy as np
import pytest
import torch

from harvester.v24 import lattice, stage_b


@pytest.mark.v24
def test_stage_b_forward_shape_and_params():
    model = stage_b.StageBModel()
    params = stage_b.parameter_count(model)
    assert params <= 2_000_000, f"Stage B has {params} params (> 2M)"

    x = torch.zeros(1, stage_b.IN_CHANNELS, 257, 257)
    residual = model(x)
    assert residual.shape == (1, 257, 257)


@pytest.mark.v24
def test_stage_b_deterministic_eval():
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(2)
        model = stage_b.StageBModel().eval()
        x = torch.randn(1, stage_b.IN_CHANNELS, 257, 257, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            r1 = model(x)
            torch.manual_seed(31337)
            r2 = model(x)
        assert torch.equal(r1, r2)
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.v24
def test_gated_l1_ignores_invalid():
    pred = torch.zeros(1, 257, 257)
    target = torch.ones(1, 257, 257)
    valid = torch.zeros(1, 257, 257)
    valid[:, :10, :10] = 1.0
    loss = stage_b.gated_l1(pred, target, valid)
    assert loss.item() == pytest.approx(1.0)


@pytest.mark.v24
def test_upsample_prior_exact(synthetic_height):
    outer, inner = lattice.sample_lattice_from_height(synthetic_height)
    up = stage_b.upsample_prior(outer, inner)
    np.testing.assert_allclose(up[::16, ::16], outer, atol=1e-4)
    np.testing.assert_allclose(up[8::16, 8::16], inner, atol=1e-4)
