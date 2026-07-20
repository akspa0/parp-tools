"""Spec 114 spectral guidance: fractal statistics are supervised, DC/mean is not."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v50.spectral_guidance import (
    frequency_loss_2d,
    frequency_split_loss,
    laplacian_loss,
    multiscale_gradient_loss,
    radial_log_power,
    radial_spectral_loss,
    sobel_edge_loss,
    transition_focus_loss,
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


# ---------------------------------------------------------------------------
# T063: V7 multi-frequency structural prior + V25 LF/HF band split.
# ---------------------------------------------------------------------------


def test_t063_exact_prediction_has_zero_losses() -> None:
    field = _pink_noise((64, 64), beta=2.6, seed=10).unsqueeze(0)
    assert frequency_loss_2d(field, field.clone()).item() == pytest.approx(0.0, abs=1e-7)
    assert laplacian_loss(field, field.clone()).item() == pytest.approx(0.0, abs=1e-7)
    assert sobel_edge_loss(field, field.clone()).item() == pytest.approx(0.0, abs=1e-7)
    assert transition_focus_loss(field, field.clone()).item() == pytest.approx(0.0, abs=1e-7)
    lf, hf = frequency_split_loss(field, field.clone())
    assert lf.item() == pytest.approx(0.0, abs=1e-7)
    assert hf.item() == pytest.approx(0.0, abs=1e-7)


def test_t063_frequency_2d_penalizes_smoothed_prediction() -> None:
    truth = _pink_noise((64, 64), beta=2.6, seed=11).unsqueeze(0)
    smoothed = torch.nn.functional.avg_pool2d(
        truth.unsqueeze(1), 5, stride=1, padding=2
    ).squeeze(1)[..., :64, :64]
    loss = frequency_loss_2d(smoothed, truth)
    assert loss.item() > 0.01  # high-frequency fractal energy is missing


def test_t063_frequency_2d_differs_from_radial_average() -> None:
    """Full 2D FFT loss and radial average are different functions with different values."""
    truth = _pink_noise((64, 64), beta=2.6, seed=12).unsqueeze(0)
    other = _pink_noise((64, 64), beta=2.6, seed=99).unsqueeze(0)
    radial_val = radial_spectral_loss(other, truth).item()
    full_2d_val = frequency_loss_2d(other, truth).item()
    # Both should be positive (different fields), but they measure different things
    assert radial_val > 0.0
    assert full_2d_val > 0.0
    # They should not be exactly equal (radial averages over angles; 2D does not)
    assert radial_val != full_2d_val


def test_t063_laplacian_penalizes_curvature_mismatch() -> None:
    truth = _pink_noise((64, 64), beta=2.6, seed=13).unsqueeze(0)
    # A blurred field has different curvature (second-derivative) structure
    blurred = torch.nn.functional.avg_pool2d(
        truth.unsqueeze(1), 5, stride=1, padding=2
    ).squeeze(1)[..., :64, :64]
    loss_blurred = laplacian_loss(blurred, truth)
    assert loss_blurred.item() > 0.005  # curvature is smoothed away
    # Exact match gives zero
    assert laplacian_loss(truth, truth.clone()).item() == pytest.approx(0.0, abs=1e-7)


def test_t063_sobel_edge_penalizes_edge_mismatch() -> None:
    truth = _pink_noise((64, 64), beta=2.6, seed=14).unsqueeze(0)
    smoothed = torch.nn.functional.avg_pool2d(
        truth.unsqueeze(1), 5, stride=1, padding=2
    ).squeeze(1)[..., :64, :64]
    loss = sobel_edge_loss(smoothed, truth)
    assert loss.item() > 0.01  # edges are smoothed away


def test_t063_transition_focus_upweights_transition_regions() -> None:
    """Transition-focus loss should penalize transition-region errors more than flat-region errors."""
    truth = _pink_noise((64, 64), beta=2.6, seed=15).unsqueeze(0)
    # Error only in flat regions (low gradient)
    flat_error = truth.clone()
    # Find flat regions: where gradient magnitude is below median
    grad_x = truth[0, :, 1:] - truth[0, :, :-1]
    grad_y = truth[0, 1:, :] - truth[0, :-1, :]
    grad_mag = torch.zeros_like(truth[0])
    grad_mag[:, 1:] += grad_x.abs()
    grad_mag[1:, :] += grad_y.abs()
    flat_mask = grad_mag < grad_mag.median()
    flat_error[0][flat_mask] += 0.1  # add error only in flat regions

    # Error only in transition regions (high gradient)
    trans_error = truth.clone()
    trans_error[0][~flat_mask] += 0.1  # add error only in transition regions

    flat_loss = transition_focus_loss(flat_error, truth).item()
    trans_loss = transition_focus_loss(trans_error, truth).item()
    # Transition-region errors should be penalized more heavily than flat-region errors
    assert trans_loss > flat_loss * 1.1


def test_t063_frequency_split_returns_lf_and_hf() -> None:
    truth = _pink_noise((64, 64), beta=2.6, seed=16).unsqueeze(0)
    lf, hf = frequency_split_loss(truth, truth.clone())
    assert isinstance(lf, torch.Tensor)
    assert isinstance(hf, torch.Tensor)
    assert lf.item() == pytest.approx(0.0, abs=1e-7)
    assert hf.item() == pytest.approx(0.0, abs=1e-7)


def test_t063_frequency_split_lf_captures_structure_hf_captures_detail() -> None:
    truth = _pink_noise((64, 64), beta=2.6, seed=17).unsqueeze(0)
    # Smoothed prediction: LF (structure) should match better than HF (detail)
    smoothed = torch.nn.functional.avg_pool2d(
        truth.unsqueeze(1), 5, stride=1, padding=2
    ).squeeze(1)[..., :64, :64]
    lf_smooth, hf_smooth = frequency_split_loss(smoothed, truth)
    lf_exact, hf_exact = frequency_split_loss(truth, truth)
    # HF loss for smoothed should be much larger than for exact
    assert hf_smooth.item() > hf_exact.item() * 10
    # LF loss for smoothed should be smaller than HF loss (structure preserved)
    assert lf_smooth.item() < hf_smooth.item()


def test_t063_all_new_losses_are_differentiable() -> None:
    truth = _pink_noise((32, 32), beta=2.6, seed=18).unsqueeze(0)
    predicted = (truth * 0.5).requires_grad_(True)
    loss = (
        frequency_loss_2d(predicted, truth)
        + laplacian_loss(predicted, truth)
        + sobel_edge_loss(predicted, truth)
        + transition_focus_loss(predicted, truth)
    )
    lf, hf = frequency_split_loss(predicted, truth)
    loss = loss + lf + hf
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0


def test_t063_shape_mismatch_fails_loudly() -> None:
    a = torch.zeros(1, 32, 32)
    b = torch.zeros(1, 16, 16)
    with pytest.raises(ValueError, match="shapes must match"):
        frequency_loss_2d(a, b)
    with pytest.raises(ValueError, match="shapes must match"):
        laplacian_loss(a, b)
    with pytest.raises(ValueError, match="shapes must match"):
        sobel_edge_loss(a, b)
    with pytest.raises(ValueError, match="shapes must match"):
        transition_focus_loss(a, b)
    with pytest.raises(ValueError, match="shapes must match"):
        frequency_split_loss(a, b)
