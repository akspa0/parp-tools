"""Stage A model tests (FR-011/FR-014): shape, param cap, determinism."""

import numpy as np
import pytest
import torch

from harvester.v24 import stage_a


@pytest.mark.v24
def test_stage_a_forward_shape_and_params():
    model = stage_a.StageAModel()
    params = stage_a.parameter_count(model)
    assert params <= 1_000_000, f"Stage A has {params} params (> 1M)"

    x = torch.zeros(2, stage_a.IN_CHANNELS, 64, 64)
    q = torch.zeros(2, 33, 33)
    outer, inner = model(x, q)
    assert outer.shape == (2, 17, 17)
    assert inner.shape == (2, 16, 16)


@pytest.mark.v24
def test_stage_a_residual_anchor():
    """Zero-init head means the fresh model reproduces the synth prior exactly."""
    model = stage_a.StageAModel().eval()
    x = torch.randn(1, stage_a.IN_CHANNELS, 64, 64, generator=torch.Generator().manual_seed(5))
    q = torch.randn(1, 33, 33, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        outer, inner = model(x, q)
    assert torch.allclose(outer, q[:, ::2, ::2], atol=1e-6)
    assert torch.allclose(inner, q[:, 1::2, 1::2], atol=1e-6)


@pytest.mark.v24
def test_stage_a_deterministic_eval():
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(1)
        model = stage_a.StageAModel().eval()
        x = torch.randn(1, stage_a.IN_CHANNELS, 64, 64, generator=torch.Generator().manual_seed(7))
        q = torch.randn(1, 33, 33, generator=torch.Generator().manual_seed(8))
        with torch.no_grad():
            a1 = model(x, q)
            torch.manual_seed(999)  # different seed must not change eval output
            a2 = model(x, q)
        assert torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1])
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.v24
def test_weighted_l1_excludes_zero_weight():
    pred_o = torch.zeros(1, 17, 17)
    pred_i = torch.zeros(1, 16, 16)
    tgt_o = torch.ones(1, 17, 17) * 5.0
    tgt_i = torch.ones(1, 16, 16) * 3.0
    w_o = torch.zeros(1, 17, 17)
    w_i = torch.ones(1, 16, 16)
    loss = stage_a.weighted_l1(pred_o, pred_i, tgt_o, tgt_i, w_o, w_i)
    assert loss.item() == pytest.approx(3.0)


@pytest.mark.v24
def test_build_input_channels(synthetic_height):
    from harvester.v24 import lattice
    from harvester.v24.tiles import TileRecord

    outer, inner = lattice.sample_lattice_from_height(synthetic_height)
    record = TileRecord(
        row=0, v18_row=0, map_name="t", tile_x=0, tile_y=0,
        audit_empty=False, real_available=False,
        cleaned_minimap=np.zeros((256, 256, 3), np.float32),
        alpha=np.zeros((256, 256, 4), np.float32),
        normal=np.zeros((257, 257, 3), np.float32),
        mcnr_mask=np.ones((257, 257), np.float32),
        object_mask=np.zeros((257, 257), np.float32),
        liquid_mask=np.zeros((256, 256), np.float32),
        holes=np.zeros((16, 16), bool),
        height=synthetic_height,
        prior_outer=outer, prior_inner=inner,
        source_outer=np.zeros((17, 17), np.uint8),
        source_inner=np.zeros((16, 16), np.uint8),
        confidence_outer=np.ones((17, 17), np.float32),
        confidence_inner=np.ones((16, 16), np.float32),
        synth_outer=outer, synth_inner=inner,
    )
    x, q = stage_a.build_input(record, include_synth=True)
    assert x.shape == (13, 64, 64)
    assert q.shape == (33, 33)
    assert x[12].max() == 1.0  # presence flag
    assert np.abs(q).max() > 0.0

    x_dropped, q_dropped = stage_a.build_input(record, include_synth=False)
    assert x_dropped[11].max() == 0.0
    assert x_dropped[12].max() == 0.0
    assert np.abs(q_dropped).max() == 0.0
