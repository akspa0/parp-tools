"""Spec 114 spectral guidance: fractal statistics are supervised, DC/mean is not."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v50.spectral_guidance import (
    multiscale_gradient_loss,
    radial_log_power,
    radial_spectral_loss,
)


def _pink_noise(shape: tuple[int, int], *, beta: float, seed: int) -> torch.Tensor:
    """Synthesize a 1/f^beta field (fractal-like terrain statistics) for discrimination tests."""
    rng = np.random.default_rng(seed)
    h, w = shape
    white = rng.standard_normal((h, w))
    spectrum = np.fft.rfft2(white)
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.rfftfreq(w).reshape(1, -1)
    radius = np.sqrt(fy**2 + fx**2)
    radius[0, 0] = 1.0
    spectrum = spectrum / radius ** (beta / 2.0)
    spectrum[0, 0] = 0.0
    field = np.fft.irfft2(spectrum, s=(h, w))
    field = (field - field.min()) / (field.max() - field.min())
    return torch.from_numpy(field.astype(np.float32))


def test_exact_prediction_has_zero_losses() -> None:
    field = _pink_noise((64, 64), beta=2.6, seed=1).unsqueeze(0)
    assert radial_spectral_loss(field, field.clone()).item() == pytest.approx(0.0, abs=1e-7)
    assert multiscale_gradient_loss(field, field.clone()).item() == pytest.approx(0.0, abs=1e-7)


def test_smoothed_prediction_is_penalized() -> None:
    truth = _pink_noise((64, 64), beta=2.6, seed=2).unsqueeze(0)
    smoothed = torch.nn.functional.avg_pool2d(
        truth.unsqueeze(1), 5, stride=1, padding=2
    ).squeeze(1)[..., :64, :64]
    loss = radial_spectral_loss(smoothed, truth)
    assert loss.item() > 0.01  # high-frequency fractal energy is missing


def test_fractal_and_white_noise_spectra_discriminate() -> None:
    pink = _pink_noise((64, 64), beta=2.6, seed=3).unsqueeze(0)
    rng = np.random.default_rng(3)
    white = torch.from_numpy(rng.random((1, 64, 64), dtype=np.float32))
    pink_curve = radial_log_power(pink, bins=33)[0]
    white_curve = radial_log_power(white, bins=33)[0]
    # Pink (fractal) spectrum decays strongly with frequency; white noise stays flat.
    pink_decay = pink_curve[2].item() - pink_curve[-1].item()
    white_decay = abs(white_curve[2].item() - white_curve[-1].item())
    assert pink_decay > 0.2
    assert white_decay < pink_decay * 0.5
    assert radial_spectral_loss(white, pink).item() > 0.01


def test_dc_offset_does_not_change_spectral_curve() -> None:
    field = _pink_noise((64, 64), beta=2.6, seed=4).unsqueeze(0)
    shifted = field + 0.37  # pure offset: no clamping, which would alter structure
    base = radial_log_power(field, bins=33)
    moved = radial_log_power(shifted, bins=33)
    assert torch.allclose(base[:, 1:], moved[:, 1:], atol=1e-4)


def test_multiscale_loss_supervises_coarse_octaves() -> None:
    truth = _pink_noise((64, 64), beta=2.6, seed=5).unsqueeze(0)
    fine_only = truth + torch.from_numpy(
        np.random.default_rng(6).standard_normal((1, 64, 64)).astype(np.float32) * 0.01
    )
    loss = multiscale_gradient_loss(fine_only.clamp(0, 1), truth, levels=3)
    assert loss.item() > 0.0
    with pytest.raises(ValueError, match="levels"):
        multiscale_gradient_loss(truth, truth, levels=0)


def test_losses_are_differentiable() -> None:
    truth = _pink_noise((32, 32), beta=2.6, seed=7).unsqueeze(0)
    predicted = (truth * 0.5).requires_grad_(True)
    loss = radial_spectral_loss(predicted, truth) + multiscale_gradient_loss(predicted, truth)
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0


def test_shape_mismatch_fails_loudly() -> None:
    a = torch.zeros(1, 32, 32)
    b = torch.zeros(1, 16, 16)
    with pytest.raises(ValueError, match="shapes must match"):
        radial_spectral_loss(a, b)
    with pytest.raises(ValueError, match="shapes must match"):
        multiscale_gradient_loss(a, b)
    with pytest.raises(ValueError, match="expected"):
        radial_log_power(torch.zeros(4, 4), bins=3)
