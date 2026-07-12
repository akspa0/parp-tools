import torch
import pytest
from harvester.v25.segformer import V25SegformerDecompiler, TerrainInpaintHead

def test_segformer_forward():
    """Assert output shape formats, projections, and linear classification boundaries of the SegFormer decompiler."""
    model = V25SegformerDecompiler(num_classes=32, max_objects=16)
    x = torch.rand(2, 3, 256, 256)
    out = model(x)

    assert out["mask_logits"].shape == (2, 1, 256, 256)
    assert out["final_feats"].shape == (2, 256, 8, 8)

    # FR-102-102: the unified decompiler emits the clean terrain-shadow map
    assert out["clean_rgb"].shape == (2, 3, 256, 256)
    assert out["clean_rgb"].min() >= 0.0
    assert out["clean_rgb"].max() <= 1.0

    placements = out["placements"]
    assert "class_logits" in placements
    assert placements["class_logits"].shape == (2, 16, 32)
    assert placements["coords"].shape == (2, 16, 3)
    assert placements["rotations"].shape == (2, 16, 3)
    assert placements["exist_logits"].shape == (2, 16, 1)

def test_inpaint_head():
    """Assert output range boundaries, U-net concatenation connections, and channels of TerrainInpaintHead."""
    inpaint = TerrainInpaintHead()
    x_rgb = torch.rand(2, 3, 256, 256)
    mask = torch.rand(2, 1, 256, 256)
    out = inpaint(x_rgb, mask)
    
    assert out.shape == (2, 3, 256, 256)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    
    # Assert gradients flow back to inputs
    loss = out.sum()
    loss.backward()
    assert x_rgb.grad is None  # Since x_rgb requires_grad was False by default
    
    x_rgb_grad = torch.rand(2, 3, 256, 256, requires_grad=True)
    mask_grad = torch.rand(2, 1, 256, 256, requires_grad=True)
    out_grad = inpaint(x_rgb_grad, mask_grad)
    loss_grad = out_grad.sum()
    loss_grad.backward()
    assert x_rgb_grad.grad is not None
    assert mask_grad.grad is not None
