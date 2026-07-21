"""Normal-derived gradient supervision for the direct-geometry lane.

MCNR gives a real, authored surface orientation at every terrain vertex, and nothing in the geometry
lane was using it. Point-wise height loss (L1/smooth-L1) is famously blind to local *slope*: many
different surfaces share nearly the same per-pixel error while looking completely different, and the
minimiser among them is the smooth one. That is the spectral-bias blur recorded in Spec 114.

This module turns normals into a direct constraint on the predicted height's gradient field. It is
the non-adversarial equivalent of what a PatchGAN discriminator provides — local structural realism
— except the supervision is exact ground truth rather than a discriminator's opinion, so it cannot
hallucinate plausible-but-wrong relief.

Geometry, and why the scale factor matters
------------------------------------------
For a heightfield ``z = h(x, y)`` the unit normal is proportional to ``(-dh/dx, -dh/dy, 1)``, so::

    dh/dx = -nx / nz        dh/dy = -ny / nz     (world height per world distance)

The model does not predict world height: it predicts the ``v112.1`` per-tile normalised target
``p = (h - tile_min) / denom``. Converting, the expected gradient of ``p`` **per grid step** is::

    dp/dx = (-nx / nz) * GRID_SPACING / denom

Dropping either ``GRID_SPACING`` or ``denom`` silently changes this loss by orders of magnitude
between tiles (``denom`` is per-tile), which would make the term meaningless.

Axis/sign convention was verified empirically against real ``height_257`` gradients rather than
assumed: channel 0 (nx) tracks the axis-1 (x) gradient and channel 1 (ny) the axis-0 (y) gradient,
correlating up to +0.97 on clean tiles, while the swapped pairings correlate far worse.

Only vertices where ``mcnr_mask_257`` is set carry real MCNR data (the 145-per-chunk quincunx
lattice, 50% of the 257x257 grid); the rest is format-level interpolation and is excluded.
"""

from __future__ import annotations

import torch

# One ADT tile spans 533.33333 world units across 256 grid intervals (257 vertices).
TILE_WORLD_SIZE = 533.33333
GRID_SPACING = TILE_WORLD_SIZE / 256.0

# nz -> 0 on a vertical face, where dh/dx diverges. Clamp so a near-vertical vertex contributes a
# large-but-finite target instead of an inf that would destroy the batch.
MIN_NZ = 1e-2
# Cap the expected per-step gradient; beyond this the surface is effectively a cliff and the exact
# value is not a meaningful regression target.
MAX_EXPECTED_GRADIENT = 4.0


def _central_gradient(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, H, W) -> (d/dy, d/dx) per grid step, central differences with one-sided edges.

    Matches ``numpy.gradient`` semantics, which is what the convention above was verified against.
    """
    gy = torch.empty_like(field)
    gx = torch.empty_like(field)
    gy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) * 0.5
    gy[:, 0, :] = field[:, 1, :] - field[:, 0, :]
    gy[:, -1, :] = field[:, -1, :] - field[:, -2, :]
    gx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) * 0.5
    gx[:, :, 0] = field[:, :, 1] - field[:, :, 0]
    gx[:, :, -1] = field[:, :, -1] - field[:, :, -2]
    return gy, gx


def expected_normalized_gradients(
    normal_xyz: torch.Tensor, height_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-grid-step gradients of the NORMALISED height implied by the authored normals.

    ``normal_xyz`` is ``(B, H, W, 3)`` in ADT grid axes (channel 2 is up).
    ``height_scale`` is ``(B,)`` — the per-tile ``v112.1`` denominator.
    """
    nx = normal_xyz[..., 0]
    ny = normal_xyz[..., 1]
    nz = torch.clamp(normal_xyz[..., 2], min=MIN_NZ)
    scale = height_scale.view(-1, 1, 1).clamp(min=1e-6)
    factor = GRID_SPACING / scale
    expected_x = torch.clamp(
        (-nx / nz) * factor, -MAX_EXPECTED_GRADIENT, MAX_EXPECTED_GRADIENT
    )
    expected_y = torch.clamp(
        (-ny / nz) * factor, -MAX_EXPECTED_GRADIENT, MAX_EXPECTED_GRADIENT
    )
    return expected_y, expected_x


def normal_gradient_loss(
    predicted: torch.Tensor,
    normal_xyz: torch.Tensor,
    normal_mask: torch.Tensor,
    height_scale: torch.Tensor,
) -> torch.Tensor:
    """L1 between the predicted height's gradient field and the one the authored normals imply.

    Evaluated only at real MCNR vertices. Returns a zero scalar (still connected to the graph) when
    a batch has no valid vertices, so an unlucky batch cannot produce NaN.
    """
    if predicted.ndim != 3:
        raise ValueError(f"predicted must be (B, H, W), got {tuple(predicted.shape)}")
    if normal_xyz.shape[:3] != predicted.shape or normal_xyz.shape[-1] != 3:
        raise ValueError(
            f"normal_xyz must be (B, H, W, 3) matching predicted, got {tuple(normal_xyz.shape)}"
        )

    predicted_gy, predicted_gx = _central_gradient(predicted)
    expected_gy, expected_gx = expected_normalized_gradients(normal_xyz, height_scale)

    mask = normal_mask.to(predicted.dtype)
    total = mask.sum()
    if float(total) <= 0:
        return predicted.sum() * 0.0
    error = (predicted_gx - expected_gx).abs() + (predicted_gy - expected_gy).abs()
    return (error * mask).sum() / total
