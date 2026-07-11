import torch
import pytest
from harvester.v25.losses import V25UnifiedLoss

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
    
    # Run forward pass
    losses = loss_fn(pred_outputs, target_outputs)
    
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
