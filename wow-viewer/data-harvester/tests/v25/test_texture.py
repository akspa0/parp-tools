import torch
import pytest
from harvester.v25.texture import MtexPredictor, MclyDecoder

def test_mtex_predictor():
    """Verify that MTEX predicts over global vocabulary sizes and backprops correctly."""
    predictor = MtexPredictor(in_channels=256, vocab_size=128)
    feats = torch.randn(2, 256, 8, 8, requires_grad=True)
    
    logits = predictor(feats)
    assert logits.shape == (2, 128)
    
    loss = logits.sum()
    loss.backward()
    assert feats.grad is not None

def test_mcly_decoder():
    """Verify that MclyDecoder outputs 4-layer classification logit grids of shape (B, 4, 16, 16, 4)."""
    decoder = MclyDecoder(in_channels=256, num_layers=4)
    feats = torch.randn(2, 256, 8, 8, requires_grad=True)
    
    logits = decoder(feats)
    assert logits.shape == (2, 4, 16, 16, 4)
    
    loss = logits.sum()
    loss.backward()
    assert feats.grad is not None
