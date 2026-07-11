import torch
import pytest
from harvester.v25.prior import V25StageAPredictor, WdlDownsampler

def test_stage_a_predictor():
    """Verify that the visual height prior predictor yields a 33x33 height map."""
    model = V25StageAPredictor(in_channels=256)
    feats = torch.randn(2, 256, 8, 8)
    h_33 = model(feats)
    
    assert h_33.shape == (2, 33, 33)

def test_wdl_downsampler_alignment():
    """Verify that WdlDownsampler correctly performs node-stride index mapping (X[i] = Y[i * 8])."""
    downsampler = WdlDownsampler()
    h_257 = torch.randn(2, 257, 257)
    wdl_prior = downsampler(h_257)
    
    assert wdl_prior.shape == (2, 33, 33)
    
    # Assert coordinate matching
    for b in range(2):
        for y in range(33):
            for x in range(33):
                assert wdl_prior[b, y, x] == h_257[b, y * 8, x * 8]

def test_downsampler_dimensions_and_gradients():
    """Verify backpropagation gradients flow stably through the downsampler to height tensors."""
    downsampler = WdlDownsampler()
    h_257 = torch.randn(1, 1, 257, 257, requires_grad=True)
    wdl_prior = downsampler(h_257)
    
    assert wdl_prior.shape == (1, 1, 33, 33)
    loss = wdl_prior.sum()
    loss.backward()
    
    assert h_257.grad is not None
    # Check that gradient is non-zero only at the strided coordinate nodes
    grad_mask = (h_257.grad != 0.0).squeeze(0).squeeze(0)
    for y in range(257):
        for x in range(257):
            if y % 8 == 0 and x % 8 == 0:
                assert grad_mask[y, x] == True
            else:
                assert grad_mask[y, x] == False
