"""Spec 114 loss guidance: fractal/spectral structure of terrain relief (Spec 068 US1, revived).

Evidence (2026-07-19, `mit_b0-authored-v1` epoch-65 fixed preview): Smooth-L1 + a single-scale
gradient term teaches coarse layout but regresses self-similar ridge/drainage detail to smooth
blobs and systematically under-predicts relief amplitude — classic spectral bias. WoW terrain is
the output of fractal processes, so its height fields carry power-law spectra (P(f) ∝ 1/f^β) and
multi-scale self-similarity that pixel-space losses do not supervise.

Two training-only, head-free guidance terms (one-output constitution preserved; no deployment
input changes):

- ``radial_spectral_loss``: L1 between radially-averaged log-power spectra of prediction and
  target, per tile, DC removed. Matching the log-power curve forces the same 1/f^β fractal
  statistics (the spectral slope IS the fractal dimension proxy for fBm-like terrain).
- ``multiscale_gradient_loss``: first-derivative L1 at each level of an average-pool pyramid, so
  ridge/coastline structure is supervised at the octave it actually lives on instead of only at
  full resolution.

Both are O(N log N)/O(N), differentiable, and CPU-testable. Weights default to 0 in the trainer
(bootstrap parity); the documented ablation enables them explicitly.
"""

from __future__ import annotations

import torch
from torch import nn


def radial_log_power(field: torch.Tensor, *, bins: int) -> torch.Tensor:
    """Radially-averaged log power spectrum of (B, H, W) fields -> (B, bins).

    DC is removed per tile first, so the curve describes STRUCTURE (relief texture), not each
    tile's mean level — consistent with the offset-invariant v112.1 target.
    """
    if field.ndim != 3:
        raise ValueError(f"expected (B, H, W) fields, got {tuple(field.shape)}")
    b, h, w = field.shape
    centered = field - field.mean(dim=(1, 2), keepdim=True)
    spectrum = torch.fft.rfft2(centered, norm="ortho")
    power = spectrum.real**2 + spectrum.imag**2

    fy = torch.fft.fftfreq(h, device=field.device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=field.device).view(1, -1)
    radius = torch.sqrt(fy**2 + fx**2) * h  # cycles per tile, 0..~h/2
    bin_index = torch.clamp(radius.round().long(), 0, bins - 1).reshape(-1)

    flat = power.reshape(b, -1)
    summed = torch.zeros(b, bins, device=field.device, dtype=power.dtype)
    summed.index_add_(1, bin_index, flat)
    counts = torch.zeros(bins, device=field.device, dtype=power.dtype)
    counts.index_add_(0, bin_index, torch.ones_like(bin_index, dtype=power.dtype))
    mean_power = summed / counts.clamp_min(1.0).unsqueeze(0)
    return torch.log1p(mean_power)


def radial_spectral_loss(
    predicted: torch.Tensor, target: torch.Tensor, *, skip_bins: int = 1
) -> torch.Tensor:
    """L1 between prediction/target radial log-power curves.

    ``skip_bins=1`` drops the DC bin (already zeroed by centering, kept only for alignment).
    Exact predictions give exactly zero loss; a smoothed prediction loses high-frequency power and
    is penalized in direct proportion to the missing fractal energy.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/target shapes must match, got {tuple(predicted.shape)} vs {tuple(target.shape)}"
        )
    bins = predicted.shape[-2] // 2 + 1
    pred_curve = radial_log_power(predicted, bins=bins)[:, skip_bins:]
    target_curve = radial_log_power(target, bins=bins)[:, skip_bins:]
    return nn.functional.l1_loss(pred_curve, target_curve)


def multiscale_gradient_loss(
    predicted: torch.Tensor, target: torch.Tensor, *, levels: int = 3
) -> torch.Tensor:
    """Mean over pyramid octaves of first-derivative L1 (both axes), full resolution first.

    At 257x257, levels 0-2 supervise structure at 257 / 128 / 64 px — the scales where the
    epoch-65 predictions went smooth. Each level is mean-normalized, so fine scales cannot be
    drowned out by coarse amplitude.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/target shapes must match, got {tuple(predicted.shape)} vs {tuple(target.shape)}"
        )
    if levels < 1:
        raise ValueError("levels must be >= 1")
    pred = predicted.unsqueeze(1)
    truth = target.unsqueeze(1)
    total = pred.new_zeros(())
    for level in range(levels):
        if level > 0:
            pred = nn.functional.avg_pool2d(pred, 2, ceil_mode=True)
            truth = nn.functional.avg_pool2d(truth, 2, ceil_mode=True)
        dx = nn.functional.l1_loss(pred[..., :, 1:] - pred[..., :, :-1],
                                   truth[..., :, 1:] - truth[..., :, :-1])
        dy = nn.functional.l1_loss(pred[..., 1:, :] - pred[..., :-1, :],
                                   truth[..., 1:, :] - truth[..., :-1, :])
        total = total + dx + dy
    return total / levels
