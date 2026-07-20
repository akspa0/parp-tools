"""Spec 114 loss guidance: fractal/spectral structure of terrain relief (Spec 068 US1, revived).

Evidence (2026-07-19, `mit_b0-authored-v1` epoch-65 fixed preview): Smooth-L1 + a single-scale
gradient term teaches coarse layout but regresses self-similar ridge/drainage detail to smooth
blobs and systematically under-predicts relief amplitude — classic spectral bias. WoW terrain is
the output of fractal processes, so its height fields carry power-law spectra (P(f) ∝ 1/f^β) and
multi-scale self-similarity that pixel-space losses do not supervise.

T063 (2026-07-20) extends the stack with V7's multi-frequency structural prior and V25's LF/HF
band split, ported to the v50 single-channel (B, H, W) API. V7 (April 2026) was the most
successful terrain model to date despite a dirty half-broken dataset; its four structural loss
terms — full 2D frequency, Laplacian curvature, Sobel edge, transition-focus weighting —
supervised the multi-band structure that pixel-space L1 cannot see. V25's frequency_split_loss
adds an explicit LF/HF separation via radial FFT cutoff, so structure (LF) and detail (HF) are
supervised independently rather than averaged together.

All seven training-only, head-free guidance terms (one-output constitution preserved; no
deployment input changes):

- ``radial_spectral_loss``: L1 between radially-averaged log-power spectra of prediction and
  target, per tile, DC removed. Matching the log-power curve forces the same 1/f^β fractal
  statistics (the spectral slope IS the fractal dimension proxy for fBm-like terrain).
- ``multiscale_gradient_loss``: first-derivative L1 at each level of an average-pool pyramid, so
  ridge/coastline structure is supervised at the octave it actually lives on instead of only at
  full resolution.
- ``frequency_loss_2d``: L1 of the full 2D log-magnitude FFT (V7 ``frequency_loss``). Unlike the
  radial average, this preserves directional structure — ridge orientation, coastline bearing —
  not just the isotropic power curve.
- ``laplacian_loss``: L1 of the 5-point discrete Laplacian (V7 ``laplacian_loss``). Supervises
  curvature (second-derivative) structure, complementing the first-derivative gradient term.
- ``sobel_edge_loss``: L1 of Sobel-filtered edge magnitude (V7 ``edge_loss``). Supervises
  ridge/cliff edge structure at the scale the Sobel kernel responds to.
- ``transition_focus_loss``: L1 weighted by target gradient magnitude (V7
  ``transition_focus_loss``). Up-weights L1 at terrain transitions (cliff edges, coastlines) so
  the model cannot trade away transition accuracy for flat-region smoothness.
- ``frequency_split_loss``: LF/HF band split via radial FFT cutoff (V25
  ``frequency_split_loss``). Returns ``(lf_loss, hf_loss)`` so the trainer can weight structure
  and detail independently.

All are O(N log N)/O(N), differentiable, and CPU-testable. Weights default to 0 in the trainer
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


# ---------------------------------------------------------------------------
# T063: V7 multi-frequency structural prior + V25 LF/HF band split.
#
# V7 (April 2026, spec103/v7_losses.py) was the most successful terrain model
# to date. Its four structural loss terms supervised multi-band structure that
# pixel-space L1 cannot see. V25 (v25/losses.py) added an explicit LF/HF band
# split via radial FFT cutoff. Both are ported here to the v50 single-channel
# (B, H, W) API — V7 operated on 2-channel (global + local) heightmaps; we
# adapt to one output (the v112.1 relative-height field). All use
# ``norm="ortho"`` for FFT scale-independence (V25 lesson: unnormalised FFT
# made the frequency loss thousands of times larger than every other head).
# ---------------------------------------------------------------------------


def _check_3d_match(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/target shapes must match, got {tuple(predicted.shape)} vs {tuple(target.shape)}"
        )


def frequency_loss_2d(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 of the full 2D log-magnitude FFT (V7 ``frequency_loss``, single-channel).

    Unlike :func:`radial_spectral_loss` (which averages the 2D spectrum into 1D
    radial bins), this preserves the full 2D frequency map — so directional
    structure (ridge orientation, coastline bearing) is supervised, not just
    the isotropic power curve. ``norm="ortho"`` keeps the loss scale-independent
    of grid size.
    """
    _check_3d_match(predicted, target)
    pred_fft = torch.fft.rfft2(predicted.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    pred_mag = torch.log1p(pred_fft.abs())
    target_mag = torch.log1p(target_fft.abs())
    return nn.functional.l1_loss(pred_mag, target_mag)


def laplacian_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 of the 5-point discrete Laplacian (V7 ``laplacian_loss``, single-channel).

    Supervises curvature (second-derivative) structure — peaks, valleys, and
    saddle points — complementing the first-derivative gradient term. The
    kernel ``[[0,1,0],[1,-4,1],[0,1,0]]`` is the standard 5-point stencil.
    """
    _check_3d_match(predicted, target)
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32,
        device=predicted.device,
    ).view(1, 1, 3, 3)
    pred = predicted.float().unsqueeze(1)
    truth = target.float().unsqueeze(1)
    pred_lap = nn.functional.conv2d(pred, kernel, padding=1)
    truth_lap = nn.functional.conv2d(truth, kernel, padding=1)
    return nn.functional.l1_loss(pred_lap, truth_lap)


def sobel_edge_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 of Sobel-filtered edge magnitude (V7 ``edge_loss``, single-channel).

    Supervises ridge/cliff edge structure at the 3×3 scale the Sobel kernel
    responds to. The Sobel operator is a first-derivative approximation with
    smoothing along the orthogonal axis, making it more robust to noise than a
    simple finite difference.
    """
    _check_3d_match(predicted, target)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=predicted.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    pred = predicted.float().unsqueeze(1)
    truth = target.float().unsqueeze(1)
    pred_edge = (
        nn.functional.conv2d(pred, sobel_x, padding=1).abs()
        + nn.functional.conv2d(pred, sobel_y, padding=1).abs()
    )
    truth_edge = (
        nn.functional.conv2d(truth, sobel_x, padding=1).abs()
        + nn.functional.conv2d(truth, sobel_y, padding=1).abs()
    )
    return nn.functional.l1_loss(pred_edge, truth_edge)


def transition_focus_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    gain: float = 3.0,
) -> torch.Tensor:
    """L1 weighted by target gradient magnitude (V7 ``transition_focus_loss``).

    Up-weights L1 at terrain transitions (cliff edges, coastlines, ridge lines)
    so the model cannot trade away transition accuracy for flat-region
    smoothness. The weight map is ``1 + gain * clamp(|∇target| / mean(|∇target|), 0, 1)``,
    so flat regions get weight 1 (unchanged) and transition regions get up to
    ``1 + gain`` (default 4× with ``gain=3``).
    """
    _check_3d_match(predicted, target)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=target.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    target_f = target.float().unsqueeze(1)
    grad_x = nn.functional.conv2d(target_f, sobel_x, padding=1)
    grad_y = nn.functional.conv2d(target_f, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    normalized = magnitude / (magnitude.mean(dim=(2, 3), keepdim=True) + 1e-6)
    weights = 1.0 + gain * torch.clamp(normalized, min=0.0, max=1.0)
    diff = (predicted.float().unsqueeze(1) - target.float().unsqueeze(1)).abs()
    return (diff * weights).mean()


def frequency_split_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    cutoff: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LF/HF band split via radial FFT cutoff (V25 ``frequency_split_loss``).

    Splits the frequency-domain L1 into low-frequency (structure) and
    high-frequency (detail) components using a radial mask centred at DC. The
    cutoff is a fraction of the spatial extent (``0.1`` = keep the lowest 10%
    of frequencies as LF). Returns ``(lf_loss, hf_loss)`` so the trainer can
    weight structure and detail independently.

    ``norm="ortho"`` prevents the loss from scaling with the 257×257 grid size
    (V25 lesson: unnormalised FFT made this loss thousands of times larger than
    every other head and silently starved them of signal).
    """
    _check_3d_match(predicted, target)
    pred_fft = torch.fft.rfft2(predicted.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    h, w_half = pred_fft.shape[-2], pred_fft.shape[-1]
    y = torch.arange(h, device=predicted.device, dtype=torch.float32)
    x = torch.arange(w_half, device=predicted.device, dtype=torch.float32)
    # DC is at (0, 0) for rfft2 — wrap y around H/2
    y = torch.where(y > h / 2, y - h, y)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = cutoff * min(h, w_half * 2)
    lf_mask = ((yy**2 + xx**2) <= radius**2).float()
    diff_fft = (pred_fft - target_fft).abs()
    lf_loss = (diff_fft * lf_mask).mean()
    hf_loss = (diff_fft * (1.0 - lf_mask)).mean()
    return lf_loss, hf_loss
