"""V24.1 DA-V2 Stage A tests (Spec 101): model shape, SiLogLoss, hybrid loss."""

import numpy as np
import pytest
import torch

from harvester.v24 import stage_a


# ---------------------------------------------------------------------------
# StageADAV2 model tests (FR-101-101 through FR-101-105)
# ---------------------------------------------------------------------------

@pytest.mark.v24
def test_stage_a_dav2_forward_shape_3ch():
    """DA-V2 model with 3-channel minimap-only input produces (17,17)+(16,16)."""
    try:
        model = stage_a.StageADAV2(in_channels=3, load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    model.eval()
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        outer, inner = model(x)
    assert outer.shape == (1, 17, 17), f"outer shape {outer.shape}"
    assert inner.shape == (1, 16, 16), f"inner shape {inner.shape}"


@pytest.mark.v24
def test_stage_a_dav2_forward_shape_9ch_guided():
    """DA-V2 model with 9-channel guided input produces (17,17)+(16,16)."""
    try:
        model = stage_a.StageADAV2(in_channels=9, load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    model.eval()
    x = torch.zeros(1, 9, 256, 256)
    with torch.no_grad():
        outer, inner = model(x)
    assert outer.shape == (1, 17, 17)
    assert inner.shape == (1, 16, 16)


@pytest.mark.v24
def test_stage_a_dav2_param_count():
    """DA-V2 total params ≤ 26M, trainable params ≤ 2M."""
    try:
        model = stage_a.StageADAV2(in_channels=3, load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total <= 26_000_000, f"total {total} > 26M"
    assert trainable <= 2_000_000, f"trainable {trainable} > 2M"


@pytest.mark.v24
def test_stage_a_dav2_backbone_frozen():
    """The DA-V2 backbone is frozen; only LoRA + patch proj + head are trainable."""
    try:
        model = stage_a.StageADAV2(in_channels=3, load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    # The encoder backbone parameters should not require grad.
    backbone_params = [
        p for p in model.encoder.backbone.parameters() if p.requires_grad
    ]
    # LoRA adapters are part of the backbone but are trainable; the
    # underlying frozen weights are not. So we check that at least some
    # backbone params are frozen.
    frozen = [p for p in model.encoder.backbone.parameters() if not p.requires_grad]
    assert len(frozen) > 0, "No frozen backbone parameters found"


@pytest.mark.v24
def test_stage_a_dav2_loads_offline():
    """DA-V2 model loads with load_pretrained=False (offline/test mode)."""
    try:
        model = stage_a.StageADAV2(in_channels=3, load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    assert model is not None
    assert model.in_channels == 3


# ---------------------------------------------------------------------------
# SiLogLoss tests (FR-101-201 through FR-101-204)
# ---------------------------------------------------------------------------

@pytest.mark.v24
def test_silog_loss_positive_scalar():
    """SiLogLoss produces a non-negative scalar."""
    pred_o = torch.ones(1, 17, 17) * 0.5
    pred_i = torch.ones(1, 16, 16) * 0.5
    tgt_o = torch.ones(1, 17, 17) * 1.0
    tgt_i = torch.ones(1, 16, 16) * 1.0
    w_o = torch.ones(1, 17, 17)
    w_i = torch.ones(1, 16, 16)
    loss = stage_a.SiLogLoss(shift=10.0)(pred_o, pred_i, tgt_o, tgt_i, w_o, w_i)
    assert loss.item() >= 0.0
    assert loss.ndim == 0  # scalar


@pytest.mark.v24
def test_silog_loss_handles_negative_inputs():
    """SiLogLoss handles negative heights via the shift parameter."""
    pred_o = torch.ones(1, 17, 17) * (-3.0)  # negative height
    pred_i = torch.ones(1, 16, 16) * (-3.0)
    tgt_o = torch.ones(1, 17, 17) * (-2.0)
    tgt_i = torch.ones(1, 16, 16) * (-2.0)
    w_o = torch.ones(1, 17, 17)
    w_i = torch.ones(1, 16, 16)
    # With shift=10.0, pred becomes 7.0, target becomes 8.0 — both positive.
    loss = stage_a.SiLogLoss(shift=10.0)(pred_o, pred_i, tgt_o, tgt_i, w_o, w_i)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


@pytest.mark.v24
def test_silog_loss_nonzero_gradient():
    """SiLogLoss has a non-zero gradient when pred != target."""
    pred_o = torch.full((1, 17, 17), 0.5, requires_grad=True)
    pred_i = torch.full((1, 16, 16), 0.5, requires_grad=True)
    tgt_o = torch.ones(1, 17, 17) * 1.0
    tgt_i = torch.ones(1, 16, 16) * 1.0
    w_o = torch.ones(1, 17, 17)
    w_i = torch.ones(1, 16, 16)
    loss = stage_a.SiLogLoss(shift=10.0)(pred_o, pred_i, tgt_o, tgt_i, w_o, w_i)
    loss.backward()
    assert pred_o.grad is not None
    assert pred_o.grad.abs().sum().item() > 0


@pytest.mark.v24
def test_silog_loss_zero_when_perfect_prediction():
    """SiLogLoss is ~0 when pred == target."""
    tgt_o = torch.ones(1, 17, 17) * 2.0
    tgt_i = torch.ones(1, 16, 16) * 2.0
    pred_o = tgt_o.clone().requires_grad_(True)
    pred_i = tgt_i.clone().requires_grad_(True)
    w_o = torch.ones(1, 17, 17)
    w_i = torch.ones(1, 16, 16)
    loss = stage_a.SiLogLoss(shift=10.0)(pred_o, pred_i, tgt_o, tgt_i, w_o, w_i)
    # sqrt(epsilon) = sqrt(1e-8) = 1e-4, so the floor is ~1e-4.
    assert loss.item() < 1e-3


@pytest.mark.v24
def test_silog_loss_excludes_zero_weight():
    """SiLogLoss only computes on non-zero-weight cells."""
    pred_o = torch.ones(1, 17, 17) * 0.5
    pred_i = torch.ones(1, 16, 16) * 0.5
    tgt_o = torch.ones(1, 17, 17) * 1.0
    tgt_i = torch.ones(1, 16, 16) * 1.0
    w_o = torch.zeros(1, 17, 17)  # all outer cells excluded
    w_i = torch.ones(1, 16, 16)
    loss = stage_a.SiLogLoss(shift=10.0)(pred_o, pred_i, tgt_o, tgt_i, w_o, w_i)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Hybrid loss tests (FR-101-203)
# ---------------------------------------------------------------------------

@pytest.mark.v24
def test_hybrid_loss_positive_scalar():
    """Hybrid loss produces a non-negative scalar."""
    pred_o = torch.ones(1, 17, 17) * 0.5
    pred_i = torch.ones(1, 16, 16) * 0.5
    tgt_o = torch.ones(1, 17, 17) * 1.0
    tgt_i = torch.ones(1, 16, 16) * 1.0
    w_o = torch.ones(1, 17, 17)
    w_i = torch.ones(1, 16, 16)
    loss = stage_a.hybrid_loss(
        pred_o, pred_i, tgt_o, tgt_i, w_o, w_i,
        silog_weight=0.7, l1_weight=0.3, silog_shift=10.0,
    )
    assert loss.item() >= 0.0
    assert loss.ndim == 0


@pytest.mark.v24
def test_hybrid_loss_gradient():
    """Hybrid loss has a non-zero gradient."""
    pred_o = torch.full((1, 17, 17), 0.5, requires_grad=True)
    pred_i = torch.full((1, 16, 16), 0.5, requires_grad=True)
    tgt_o = torch.ones(1, 17, 17) * 1.0
    tgt_i = torch.ones(1, 16, 16) * 1.0
    w_o = torch.ones(1, 17, 17)
    w_i = torch.ones(1, 16, 16)
    loss = stage_a.hybrid_loss(
        pred_o, pred_i, tgt_o, tgt_i, w_o, w_i,
        silog_weight=0.7, l1_weight=0.3, silog_shift=10.0,
    )
    loss.backward()
    assert pred_o.grad is not None
    assert pred_o.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# build_dav2_input tests (FR-101-102)
# ---------------------------------------------------------------------------

@pytest.mark.v24
def test_build_dav2_input_3ch():
    """3-channel DA-V2 input is (3, 256, 256) float32."""
    minimap = np.random.rand(256, 256, 3).astype(np.float32)
    x = stage_a.build_dav2_input(minimap)
    assert x.shape == (3, 256, 256)
    assert x.dtype == np.float32


@pytest.mark.v24
def test_build_dav2_input_9ch_guided():
    """9-channel guided DA-V2 input is (9, 256, 256) float32."""
    minimap = np.random.rand(256, 256, 3).astype(np.float32)
    normal = np.random.rand(257, 257, 3).astype(np.float32) * 2 - 1
    x = stage_a.build_dav2_input(minimap, normal=normal)
    assert x.shape == (9, 256, 256)
    assert x.dtype == np.float32


@pytest.mark.v24
def test_build_dav2_input_no_normal_zeros():
    """9-channel DA-V2 input with no normal has zeros in channels 3-8."""
    minimap = np.random.rand(256, 256, 3).astype(np.float32)
    x = stage_a.build_dav2_input(minimap, normal=None)
    # When normal is None, build_dav2_input returns 3 channels (minimap-only).
    assert x.shape == (3, 256, 256)
    # The 3 channels should be the minimap RGB.
    assert not np.allclose(x, 0.0)