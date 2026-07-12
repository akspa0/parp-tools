import torch
import pytest
from harvester.v25.losses import V25UnifiedLoss, compute_texture_density, frequency_split_loss

def test_compute_texture_density():
    """Verify that compute_texture_density returns weights in the correct range [0.5, 1.5]."""
    minimap = torch.rand(2, 3, 256, 256)
    density = compute_texture_density(minimap)
    assert density.shape == (2, 1, 256, 256)
    assert density.min() >= 0.5
    assert density.max() <= 1.5

def test_frequency_split_loss():
    """Verify frequency_split_loss splits height loss into lf and hf parts correctly."""
    pred = torch.randn(2, 257, 257)
    target = torch.randn(2, 257, 257)
    lf_loss, hf_loss = frequency_split_loss(pred, target, cutoff=0.1)
    assert lf_loss.dim() == 0
    assert hf_loss.dim() == 0
    assert lf_loss >= 0
    assert hf_loss >= 0

def test_unified_loss_forward():
    """Verify that V25UnifiedLoss computes correct dict terms and gradients flow to predictions."""
    loss_fn = V25UnifiedLoss()
    
    B = 2
    max_objects = 8
    num_classes = 16
    vocab_size = 64
    
    # 1. Mock predictions
    pred_outputs = {
        "mask_logits": torch.randn(B, 1, 256, 256, requires_grad=True),
        "h_257": torch.randn(B, 257, 257, requires_grad=True),
        "placements": {
            "class_logits": torch.randn(B, max_objects, num_classes, requires_grad=True),
            "coords": torch.randn(B, max_objects, 3, requires_grad=True),
            "rotations": torch.randn(B, max_objects, 3, requires_grad=True),
            "exist_logits": torch.randn(B, max_objects, 1, requires_grad=True)
        },
        "mtex_logits": torch.randn(B, vocab_size, requires_grad=True),
        "mcly_logits": torch.randn(B, 4, 16, 16, 4, requires_grad=True),
        "alpha_256": torch.rand(B, 4, 256, 256, requires_grad=True)
    }
    
    # 2. Mock target outputs
    target_outputs = {
        "mask": torch.rand(B, 1, 256, 256),
        "h_257": torch.randn(B, 257, 257),
        "placements": {
            "class_ids": torch.randint(0, num_classes, (B, max_objects)),
            "coords": torch.randn(B, max_objects, 3),
            "rotations": torch.randn(B, max_objects, 3),
            "exist": torch.randint(0, 2, (B, max_objects))
        },
        "mtex_labels": torch.randint(0, 2, (B, vocab_size)),
        "mcly_labels": torch.randint(0, 4, (B, 16, 16, 4)),
        "alpha_256": torch.rand(B, 4, 256, 256)
    }
    
    minimap = torch.randn(B, 3, 256, 256)

    # Run forward pass with minimap
    losses = loss_fn(pred_outputs, target_outputs, minimap=minimap)
    
    # Assert loss keys
    assert "loss" in losses
    assert "mask" in losses
    assert "height" in losses
    assert "coords" in losses
    assert "rotations" in losses
    assert "exist" in losses
    assert "class" in losses
    assert "mtex" in losses
    assert "mcly" in losses
    assert "alpha" in losses
    assert "height_lf" in losses
    assert "height_hf" in losses
    assert "texture_density_mean" in losses
    
    # Assert gradients backprop
    loss_val = losses["loss"]
    assert loss_val.dim() == 0 # Scalar
    loss_val.backward()
    
    assert pred_outputs["mask_logits"].grad is not None
    assert pred_outputs["h_257"].grad is not None
    assert pred_outputs["placements"]["coords"].grad is not None
    assert pred_outputs["placements"]["rotations"].grad is not None
    assert pred_outputs["placements"]["exist_logits"].grad is not None
    assert pred_outputs["placements"]["class_logits"].grad is not None
    assert pred_outputs["mtex_logits"].grad is not None
    assert pred_outputs["mcly_logits"].grad is not None
    assert pred_outputs["alpha_256"].grad is not None


def test_unified_loss_optional_clean_and_prior_terms():
    """clean_rgb (inpaint) and h_33 (Stage A prior) terms activate when both sides provide them."""
    loss_fn = V25UnifiedLoss(use_freq_split=False)

    B, max_objects, num_classes, vocab_size = 2, 4, 2, 16
    pred_outputs = {
        "mask_logits": torch.randn(B, 1, 256, 256),
        "h_257": torch.randn(B, 257, 257),
        "placements": {
            "class_logits": torch.randn(B, max_objects, num_classes),
            "coords": torch.randn(B, max_objects, 3),
            "rotations": torch.randn(B, max_objects, 3),
            "exist_logits": torch.randn(B, max_objects, 1),
        },
        "mtex_logits": torch.randn(B, vocab_size),
        "mcly_logits": torch.randn(B, 4, 16, 16, 4),
        "alpha_256": torch.rand(B, 4, 256, 256),
    }
    target_outputs = {
        "mask": torch.rand(B, 1, 256, 256),
        "h_257": torch.randn(B, 257, 257),
        "placements": {
            "class_ids": torch.randint(0, num_classes, (B, max_objects)),
            "coords": torch.randn(B, max_objects, 3),
            "rotations": torch.randn(B, max_objects, 3),
            "exist": torch.randint(0, 2, (B, max_objects)),
        },
        "mtex_labels": torch.randint(0, 2, (B, vocab_size)),
        "mcly_labels": torch.randint(0, 4, (B, 16, 16, 4)),
        "alpha_256": torch.rand(B, 4, 256, 256),
    }

    # Without the optional keys the terms are absent.
    base = loss_fn(pred_outputs, target_outputs)
    assert "clean_rgb" not in base
    assert "h_33" not in base

    clean_pred = torch.rand(B, 3, 256, 256, requires_grad=True)
    h33_pred = torch.randn(B, 33, 33, requires_grad=True)
    pred_outputs["clean_rgb"] = clean_pred
    pred_outputs["h_33"] = h33_pred
    target_outputs["clean_rgb"] = torch.rand(B, 3, 256, 256)
    target_outputs["h_33"] = torch.randn(B, 33, 33)

    full = loss_fn(pred_outputs, target_outputs)
    assert full["clean_rgb"] >= 0
    assert full["h_33"] >= 0
    assert full["loss"].item() > base["loss"].item() - 1e-6  # terms add weight

    full["loss"].backward()
    assert clean_pred.grad is not None
    assert h33_pred.grad is not None


@pytest.mark.parametrize("use_freq_split", [True, False])
def test_height_loss_liquid_mask(use_freq_split):
    """height_mask zeroes supervision where liquid sits — errors there stop counting."""
    loss_fn = V25UnifiedLoss(use_freq_split=use_freq_split)

    B, max_objects, num_classes, vocab_size = 1, 2, 2, 8
    target_h = torch.randn(B, 257, 257)
    pred_h = target_h.clone()
    pred_h[:, 100:150, 100:150] += 50.0  # large error inside the "liquid" region
    pred_h.requires_grad_(True)

    mask = torch.ones(B, 257, 257)
    mask[:, 100:150, 100:150] = 0.0

    def run(with_mask: bool):
        pred_outputs = {
            "mask_logits": torch.zeros(B, 1, 256, 256),
            "h_257": pred_h,
            "placements": {
                "class_logits": torch.zeros(B, max_objects, num_classes),
                "coords": torch.zeros(B, max_objects, 3),
                "rotations": torch.zeros(B, max_objects, 3),
                "exist_logits": torch.zeros(B, max_objects, 1),
            },
            "mtex_logits": torch.zeros(B, vocab_size),
            "mcly_logits": torch.zeros(B, 4, 16, 16, 4),
            "alpha_256": torch.zeros(B, 4, 256, 256),
            "h_33": torch.zeros(B, 33, 33),
        }
        target_outputs = {
            "mask": torch.zeros(B, 1, 256, 256),
            "h_257": target_h,
            "placements": {
                "class_ids": torch.zeros(B, max_objects, dtype=torch.long),
                "coords": torch.zeros(B, max_objects, 3),
                "rotations": torch.zeros(B, max_objects, 3),
                "exist": torch.zeros(B, max_objects, dtype=torch.long),
            },
            "mtex_labels": torch.zeros(B, vocab_size, dtype=torch.long),
            "mcly_labels": torch.zeros(B, 16, 16, 4, dtype=torch.long),
            "alpha_256": torch.zeros(B, 4, 256, 256),
            "h_33": torch.zeros(B, 33, 33),
        }
        if with_mask:
            target_outputs["height_mask"] = mask
            target_outputs["h_33_mask"] = mask[:, ::8, ::8]
        return loss_fn(pred_outputs, target_outputs)

    unmasked = run(with_mask=False)
    masked = run(with_mask=True)
    assert masked["height"].item() < 1e-4          # error lives only under the mask
    assert unmasked["height"].item() > masked["height"].item() + 1.0

    masked["loss"].backward()
    assert pred_h.grad is not None
    # No height gradient flows into the masked (liquid) region.
    assert pred_h.grad[:, 100:150, 100:150].abs().max().item() == pytest.approx(0.0)

