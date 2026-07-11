import torch
import pytest
from harvester.v25.fractal import DifferentiableFractalGenerator, FractalParameterHead

def test_fractal_generator_shape():
    """Verify that the differentiable noise generator yields the correct 256x256 shape and value bounds."""
    generator = DifferentiableFractalGenerator()
    B = 2
    offsets = torch.randn(B, 2)
    frequency = torch.rand(B, 1) + 0.5
    persistence = torch.rand(B, 1)
    amplitude = torch.rand(B, 1)
    
    noise = generator(offsets, frequency, persistence, amplitude)
    
    assert noise.shape == (B, 256, 256)
    assert noise.min() >= 0.0
    assert noise.max() <= 1.0

def test_fractal_gradients():
    """Verify that backpropagation gradients flow stably back to translation offset and frequency inputs."""
    generator = DifferentiableFractalGenerator()
    offsets = torch.randn(1, 2, requires_grad=True)
    frequency = torch.rand(1, 1, requires_grad=True)
    persistence = torch.rand(1, 1)
    amplitude = torch.rand(1, 1)
    
    noise = generator(offsets, frequency, persistence, amplitude)
    loss = noise.sum()
    loss.backward()
    
    assert offsets.grad is not None
    assert frequency.grad is not None

def test_parameter_head_forward():
    """Verify that the parameter head outputs boundary masks and scalar lists matching the blend layers."""
    head = FractalParameterHead(in_channels=256, num_layers=4)
    x = torch.randn(2, 256, 8, 8)
    outputs = head(x)
    
    assert "boundaries" in outputs
    assert outputs["boundaries"].shape == (2, 4, 256, 256)
    assert outputs["offsets"].shape == (2, 4, 2)
    assert outputs["frequency"].shape == (2, 4)
    assert outputs["persistence"].shape == (2, 4)
    assert outputs["amplitude"].shape == (2, 4)
