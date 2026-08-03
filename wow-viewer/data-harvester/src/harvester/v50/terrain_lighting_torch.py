"""Differentiable terrain lighting forward model (PyTorch) for Spec 126 US7 refinement.

Composes: height → surface normals → Lambert hillshade → affine shading fit,
all fully differentiable so the refinement loop can backprop through the
entire chain to update the height pixels.

The synthesizer's lighting model is the forward model: Lambert N·L with a
fixed NW solar direction, ambient term, and a per-tile affine gain. Because
we know the exact forward model, the refinement is constrained by real
physics, not learned approximations.
"""

from __future__ import annotations

import torch


def sun_vector_torch(
    azimuth_deg: float = 45.0,
    elevation_deg: float = 90.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Unit vector pointing toward the sun.

    Default: 45° azimuth (NW), 90° elevation (noon zenith). Matches the
    traced 1.0.0 ``SetDirection`` bearing. The convention is:
    - azimuth: 0° = north, 90° = east (standard math convention)
    - elevation: 0° = horizon, 90° = zenith

    Returns a 3-element vector on the specified device.
    """
    az = torch.tensor(azimuth_deg, dtype=torch.float64, device=device).deg2rad()
    el = torch.tensor(elevation_deg, dtype=torch.float64, device=device).deg2rad()
    horizontal = torch.cos(el)
    return torch.stack(
        [horizontal * torch.cos(az), horizontal * torch.sin(az), torch.sin(el)]
    ).to(dtype=torch.float32)


def lambert_shading_torch(
    normals: torch.Tensor,
    light_dir: torch.Tensor,
) -> torch.Tensor:
    """``clamp(N·L, 0, 1)`` over a batch of unit normals.

    Args:
        normals: ``(B, 3, H, W)`` unit surface normals (PyTorch channel-first).
        light_dir: ``(3,)`` unit light direction vector.

    Returns:
        ``(B, 1, H, W)`` Lambert shading in [0, 1].
    """
    # N·L: (B, 3, H, W) · (3,) -> (B, H, W)
    dot = (normals * light_dir.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return torch.clamp(dot, 0.0, 1.0)


def render_hillshade_torch(
    height: torch.Tensor,
    light_dir: torch.Tensor | None = None,
    *,
    spacing: float = 533.333 / 256.0,
    azimuth_deg: float = 45.0,
    elevation_deg: float = 90.0,
) -> torch.Tensor:
    """Differentiable forward model: height → normal → Lambert shading.

    Args:
        height: ``(B, 1, H, W)`` height field. Must be at least 2×2.
        light_dir: Optional precomputed ``(3,)`` light direction. If None,
            computed from ``azimuth_deg`` / ``elevation_deg``.
        spacing: World-space distance between adjacent height samples.
            Default: 533.333 world units / 256 pixels ≈ 2.083.
        azimuth_deg: Solar azimuth in degrees (0=N, 90=E).
        elevation_deg: Solar elevation in degrees (0=horizon, 90=zenith).

    Returns:
        ``(B, 1, H, W)`` Lambert hillshade in [0, 1].
    """
    from harvester.height_to_normal import analytic_normals_from_height

    if light_dir is None:
        light_dir = sun_vector_torch(azimuth_deg, elevation_deg, device=height.device)

    # height: (B, 1, H, W) -> normals: (B, 3, H, W)
    normals = analytic_normals_from_height(height, spacing=spacing)
    return lambert_shading_torch(normals, light_dir)


def fit_affine_shading_torch(
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit ``target ≈ gain * rendered + ambient``, closed-form.

    Both tensors are ``(B, 1, H, W)``. Returns ``(gain, ambient)`` per batch
    element as ``(B, 1, 1, 1)`` scalars.

    This is the differentiable equivalent of the numpy ``_affine_r2`` in
    ``v50_measure_residual_shading_law.py``. The affine fit absorbs the
    per-tile gain/ambient that the residual already carries, so the
    refinement can focus on the *shape* mismatch rather than the overall
    brightness offset.
    """
    B = rendered.shape[0]
    x = rendered.reshape(B, -1).to(dtype=torch.float64)
    y = target.reshape(B, -1).to(dtype=torch.float64)
    # design = [x, 1]
    ones = torch.ones_like(x)
    x_sum = x.sum(dim=1)
    x2_sum = (x * x).sum(dim=1)
    xy_sum = (x * y).sum(dim=1)
    y_sum = y.sum(dim=1)
    n = float(x.shape[1])
    denom = n * x2_sum - x_sum * x_sum
    # Avoid division by zero for flat tiles.
    safe = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
    gain = (n * xy_sum - x_sum * y_sum) / safe
    ambient = (y_sum - gain * x_sum) / n
    gain = torch.where(denom.abs() > 1e-12, gain, torch.zeros_like(gain))
    ambient = torch.where(denom.abs() > 1e-12, ambient, y.mean(dim=1))
    return (
        gain.to(dtype=torch.float32).view(B, 1, 1, 1),
        ambient.to(dtype=torch.float32).view(B, 1, 1, 1),
    )


def total_variation_loss(height: torch.Tensor) -> torch.Tensor:
    """Isotropic TV smoothness prior: L1 of spatial gradients.

    Penalizes pixel-level noise in the height estimate without over-smoothing
    real edges. Standard for optimization-based inverse problems.
    """
    dy = torch.abs(height[:, :, 1:, :] - height[:, :, :-1, :])
    dx = torch.abs(height[:, :, :, 1:] - height[:, :, :, :-1])
    return dy.mean() + dx.mean()


def shape_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Scale-and-shift invariant L1 loss: operates on mean-centered, variance-normalised signals.

    ``rendered`` and ``target`` are ``(B, 1, H, W)``.  The loss is::

        L1( (rendered - mean(rendered)) / std(rendered),
            (target - mean(target)) / std(target) )

    This is the optimisation equivalent of minimising ``1 - Pearson_r``.  It
    forces the optimiser to match the *shape* of the shading field, not its
    absolute brightness, which is exactly what the forward-model-as-referee
    needs when starting from a flat height estimate.
    """
    B = rendered.shape[0]
    r = rendered.reshape(B, -1).to(dtype=torch.float64)
    t = target.reshape(B, -1).to(dtype=torch.float64)
    r = r - r.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    r_std = r.std(dim=1, keepdim=True).clamp(min=1e-8)
    t_std = t.std(dim=1, keepdim=True).clamp(min=1e-8)
    r_norm = (r / r_std).to(dtype=torch.float32)
    t_norm = (t / t_std).to(dtype=torch.float32)
    return torch.nn.functional.l1_loss(r_norm, t_norm)