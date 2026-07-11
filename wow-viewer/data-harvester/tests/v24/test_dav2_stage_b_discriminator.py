"""V24.1 Stage B PromptDA + PatchGAN discriminator tests (Spec 101 Slices 6-7)."""

import numpy as np
import pytest
import torch

from harvester.v24 import stage_b
from harvester.v24 import discriminator
from harvester.v24.tiles import HEIGHT_SCALE


# ---------------------------------------------------------------------------
# StageBPromptDA tests (FR-101-601 through FR-101-604)
# ---------------------------------------------------------------------------

@pytest.mark.v24
def test_stage_b_promptda_forward_shape():
    """PromptDA Stage B model produces (B, 257, 257) heightmap."""
    try:
        model = stage_b.StageBPromptDA(load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    model.eval()
    x = torch.zeros(1, 4, 256, 256)
    with torch.no_grad():
        height = model(x)
    assert height.shape == (1, 257, 257), f"height shape {height.shape}"


@pytest.mark.v24
def test_stage_b_promptda_param_count():
    """PromptDA Stage B total params ≤ 26M, trainable ≤ 2M."""
    try:
        model = stage_b.StageBPromptDA(load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total <= 26_000_000, f"total {total} > 26M"
    assert trainable <= 2_000_000, f"trainable {trainable} > 2M"


@pytest.mark.v24
def test_stage_b_promptda_backbone_frozen():
    """The DA-V2 backbone is frozen in the PromptDA Stage B model."""
    try:
        model = stage_b.StageBPromptDA(load_pretrained=False)
    except ImportError:
        pytest.skip("transformers/peft not installed")
    frozen = [p for p in model.encoder.backbone.parameters() if not p.requires_grad]
    assert len(frozen) > 0, "No frozen backbone parameters found"


@pytest.mark.v24
def test_build_promptda_input():
    """build_promptda_input produces (4, 256, 256) float32."""
    minimap = np.random.rand(256, 256, 3).astype(np.float32)
    prior_up = np.random.rand(256, 256).astype(np.float32) * 500
    x = stage_b.build_promptda_input(minimap, prior_up)
    assert x.shape == (4, 256, 256)
    assert x.dtype == np.float32
    # Channel 3 should be the normalized depth prompt.
    assert not np.allclose(x[3], 0.0)


# ---------------------------------------------------------------------------
# WDLDiscriminator tests (FR-101-701 through FR-101-704)
# ---------------------------------------------------------------------------

@pytest.mark.v24
def test_wdl_discriminator_forward_shape_33():
    """Discriminator on 33×33 quincunx produces patch logits."""
    model = discriminator.WDLDiscriminator(in_channels=1, base=64, n_layers=3)
    x = torch.randn(2, 1, 33, 33)
    out = model(x)
    assert out.ndim == 4, f"output ndim {out.ndim}"
    assert out.shape[0] == 2
    assert out.shape[1] == 1


@pytest.mark.v24
def test_wdl_discriminator_forward_shape_257():
    """Discriminator on 257×257 upsampled prior produces patch logits."""
    model = discriminator.WDLDiscriminator(in_channels=1, base=64, n_layers=3)
    x = torch.randn(1, 1, 257, 257)
    out = model(x)
    assert out.ndim == 4
    assert out.shape[0] == 1
    assert out.shape[1] == 1


@pytest.mark.v24
def test_wdl_discriminator_param_count():
    """Discriminator has ~250K params (±50%)."""
    model = discriminator.WDLDiscriminator(in_channels=1, base=32, n_layers=3)
    params = discriminator.parameter_count(model)
    # With base=32, ~693K params (including BatchNorm). Accept 50K to 1M.
    assert 50_000 <= params <= 1_000_000, f"discriminator has {params} params"


@pytest.mark.v24
def test_wdl_discriminator_gradient():
    """Discriminator has non-zero gradients."""
    model = discriminator.WDLDiscriminator(in_channels=1, base=64, n_layers=3)
    x = torch.randn(1, 1, 33, 33, requires_grad=False)
    out = model(x)
    loss = out.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No non-zero gradients in discriminator"


@pytest.mark.v24
def test_render_quincunx_33():
    """_render_quincunx_33 correctly interleaves outer + inner into 33×33."""
    outer = torch.ones(1, 17, 17) * 1.0
    inner = torch.ones(1, 16, 16) * 2.0
    q = discriminator._render_quincunx_33(outer, inner)
    assert q.shape == (1, 33, 33)
    # Even rows/cols should be 1.0 (outer).
    assert torch.allclose(q[:, ::2, ::2], torch.full_like(q[:, ::2, ::2], 1.0))
    # Odd rows/cols should be 2.0 (inner).
    assert torch.allclose(q[:, 1::2, 1::2], torch.full_like(q[:, 1::2, 1::2], 2.0))
    # Off-diagonal should be 0.0.
    assert torch.allclose(q[:, ::2, 1::2], torch.zeros_like(q[:, ::2, 1::2]))
    assert torch.allclose(q[:, 1::2, ::2], torch.zeros_like(q[:, 1::2, ::2]))


@pytest.mark.v24
def test_gan_step_updates_both_models():
    """GAN step updates both discriminator and generator."""
    # Use a tiny generator (StageAMinimapOnly) for the test.
    from harvester.v24 import stage_a
    model_G = stage_a.StageAMinimapOnly()
    model_D = discriminator.WDLDiscriminator(in_channels=1, base=32, n_layers=3)
    opt_G = torch.optim.Adam(model_G.parameters(), lr=1e-3)
    opt_D = torch.optim.Adam(model_D.parameters(), lr=1e-3)

    x = torch.randn(2, 3, 64, 64)
    real_prior = torch.randn(2, 1, 33, 33)
    target_outer = torch.randn(2, 17, 17)
    target_inner = torch.randn(2, 16, 16)
    weight_outer = torch.ones(2, 17, 17)
    weight_inner = torch.ones(2, 16, 16)

    # Snapshot params before the step.
    g_params_before = [p.clone() for p in model_G.parameters() if p.requires_grad]
    d_params_before = [p.clone() for p in model_D.parameters() if p.requires_grad]

    result = discriminator.gan_step(
        model_D=model_D,
        model_G=model_G,
        real_prior=real_prior,
        generator_input=x,
        opt_D=opt_D,
        opt_G=opt_G,
        lambda_adv=0.1,
        l1_loss_fn=stage_a.weighted_l1,
        l1_targets=(target_outer, target_inner),
        l1_weights=(weight_outer, weight_inner),
    )

    assert "d_loss" in result
    assert "g_adv_loss" in result
    assert "g_l1_loss" in result
    assert result["d_loss"] >= 0.0
    assert result["g_adv_loss"] >= 0.0

    # Check that at least some params changed.
    g_changed = any(not torch.equal(p_before, p_after)
                    for p_before, p_after in zip(
                        g_params_before,
                        [p for p in model_G.parameters() if p.requires_grad],
                    ))
    d_changed = any(not torch.equal(p_before, p_after)
                    for p_before, p_after in zip(
                        d_params_before,
                        [p for p in model_D.parameters() if p.requires_grad],
                    ))
    assert g_changed, "Generator params did not change"
    assert d_changed, "Discriminator params did not change"