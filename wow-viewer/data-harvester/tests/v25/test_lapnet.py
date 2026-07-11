import torch
import pytest
from harvester.v25.lapnet import V25StageBPredictor

def test_stage_b_predictor():
    """Verify that the progressive Sylvester predictor yields 257x257 edge-aligned output meshes."""
    model = V25StageBPredictor()
    h_33 = torch.randn(2, 33, 33)
    clean_map = torch.randn(2, 3, 256, 256)
    
    h_257 = model(h_33, clean_map)
    assert h_257.shape == (2, 257, 257)

def test_stage_b_gradients():
    """Verify that backpropagation gradients flow stably through the interpolation and Sylvester solver modules."""
    model = V25StageBPredictor()
    h_33 = torch.randn(1, 33, 33, requires_grad=True)
    clean_map = torch.randn(1, 3, 256, 256, requires_grad=True)
    
    h_257 = model(h_33, clean_map)
    loss = h_257.sum()
    loss.backward()
    
    assert h_33.grad is not None
    assert clean_map.grad is not None
