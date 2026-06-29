"""Spec 077 Phase 6 (US5) analytic normals from predicted height.

The first spec 077 normal surface is derived deterministically from the
predicted ``height_257`` via the cross product of height gradients. No
separate normal model is trained in the first pass — the analytic
normals provide a free, high-quality baseline. A later slice can train
a separate normal-refinement model on top of the analytic baseline if
it is needed; that lane is intentionally out of scope for the MVP.

Spec 077 data-model.md §5 says the baseline comes from deterministic
height-to-normal derivation rather than a joint terrain model. This
module is that baseline.
"""

from __future__ import annotations

import numpy as np
import torch


def analytic_normals_from_height(
    height: np.ndarray | torch.Tensor,
    *,
    spacing: float = 1.0,
) -> np.ndarray | torch.Tensor:
    """Compute per-vertex normals from a height field via cross product.

    Parameters
    ----------
    height:
        ``(H, W)`` or ``(B, 1, H, H)`` (or ``(B, H, W)``) height field.
        H and W must both be >= 2; otherwise the function returns
        unit-z normals.
    spacing:
        World-space distance between adjacent samples. Default 1.0
        (callers can rescale to the ADT world units per-tile).

    Returns
    -------
    Same shape as ``height`` with a trailing XYZ channel added:
    ``(H, W, 3)`` for 2-D input, ``(B, 3, H, W)`` for 4-D input.
    Every normal is unit-length and points "up" out of the height
    surface (positive Z component).
    """
    if isinstance(height, np.ndarray):
        if height.ndim == 2:
            dh_dx, dh_dy = _np_gradients(height, spacing=spacing)
            return _np_normalize(np.stack([-dh_dx, -dh_dy, np.ones_like(height)], axis=-1))
        if height.ndim == 3:
            # (B, H, W) — treat as a batch
            dh_dx, dh_dy = _np_gradients(height, spacing=spacing)
            stacked = np.stack([-dh_dx, -dh_dy, np.ones_like(height)], axis=1)
            return _np_normalize(stacked, axis=1)
        raise ValueError(f"Expected 2-D or 3-D numpy height; got {height.shape}")
    if isinstance(height, torch.Tensor):
        if height.ndim == 2:
            dh_dx, dh_dy = _torch_gradients(height, spacing=spacing)
            stacked = torch.stack([-dh_dx, -dh_dy, torch.ones_like(height)], dim=-1)
            return _torch_normalize(stacked)
        if height.ndim == 3:
            dh_dx, dh_dy = _torch_gradients(height, spacing=spacing)
            stacked = torch.stack([-dh_dx, -dh_dy, torch.ones_like(height)], dim=1)
            return _torch_normalize(stacked, dim=1)
        if height.ndim == 4 and height.shape[1] == 1:
            squeezed = height[:, 0]
            dh_dx, dh_dy = _torch_gradients(squeezed, spacing=spacing)
            stacked = torch.stack([-dh_dx, -dh_dy, torch.ones_like(squeezed)], dim=1)
            return _torch_normalize(stacked, dim=1)
        raise ValueError(f"Expected 2-D / 3-D / (B,1,H,W) torch height; got {tuple(height.shape)}")
    raise TypeError(f"Unsupported type: {type(height).__name__}")


# --- numpy helpers ---------------------------------------------------------

def _np_gradients(height: np.ndarray, *, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    if height.shape[-2] < 2 or height.shape[-1] < 2:
        out = np.zeros_like(height)
        return out, out
    # central differences with one-sided fallbacks at the borders
    dh_dx = np.zeros_like(height)
    dh_dy = np.zeros_like(height)
    dh_dx[..., :, 1:-1] = (height[..., :, 2:] - height[..., :, :-2]) * 0.5 / spacing
    dh_dx[..., :, 0] = (height[..., :, 1] - height[..., :, 0]) / spacing
    dh_dx[..., :, -1] = (height[..., :, -1] - height[..., :, -2]) / spacing
    dh_dy[..., 1:-1, :] = (height[..., 2:, :] - height[..., :-2, :]) * 0.5 / spacing
    dh_dy[..., 0, :] = (height[..., 1, :] - height[..., 0, :]) / spacing
    dh_dy[..., -1, :] = (height[..., -1, :] - height[..., -2, :]) / spacing
    return dh_dx, dh_dy


def _np_normalize(stacked: np.ndarray, *, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(stacked, axis=axis, keepdims=True)
    return stacked / np.clip(norm, 1e-8, None)


# --- torch helpers ---------------------------------------------------------

def _torch_gradients(height: torch.Tensor, *, spacing: float) -> tuple[torch.Tensor, torch.Tensor]:
    if height.shape[-1] < 2 or height.shape[-2] < 2:
        out = torch.zeros_like(height)
        return out, out
    dh_dx = torch.zeros_like(height)
    dh_dy = torch.zeros_like(height)
    dh_dx[..., 1:-1] = (height[..., 2:] - height[..., :-2]) * 0.5 / spacing
    dh_dx[..., 0] = (height[..., 1] - height[..., 0]) / spacing
    dh_dx[..., -1] = (height[..., -1] - height[..., -2]) / spacing
    dh_dy[..., 1:-1, :] = (height[..., 2:, :] - height[..., :-2, :]) * 0.5 / spacing
    dh_dy[..., 0, :] = (height[..., 1, :] - height[..., 0, :]) / spacing
    dh_dy[..., -1, :] = (height[..., -1, :] - height[..., -2, :]) / spacing
    return dh_dx, dh_dy


def _torch_normalize(stacked: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    norm = torch.linalg.norm(stacked, dim=dim, keepdim=True).clamp_min(1e-8)
    return stacked / norm


def _normal_channel_axis(normals: np.ndarray | torch.Tensor) -> int:
    if normals.ndim >= 3 and normals.shape[-1] == 3:
        return -1
    if normals.ndim >= 3 and normals.shape[1] == 3:
        return 1
    raise ValueError(f"Could not identify XYZ normal channel in shape {tuple(normals.shape)}")


def analytic_normal_difference(
    predicted_height: np.ndarray | torch.Tensor,
    reference_height: np.ndarray | torch.Tensor,
    *,
    spacing: float = 1.0,
) -> float:
    """Mean angular error (radians) between normals derived from two height fields.

    Useful for sanity-checking a height prediction: if two heights
    differ by a small per-vertex delta, the corresponding normals
    should also be close. Returns 0.0 for empty inputs.
    """
    if isinstance(predicted_height, np.ndarray) and isinstance(reference_height, np.ndarray):
        if np.array_equal(predicted_height, reference_height):
            return 0.0
    if isinstance(predicted_height, torch.Tensor) and isinstance(reference_height, torch.Tensor):
        if torch.equal(predicted_height, reference_height):
            return 0.0
    pred_n = analytic_normals_from_height(predicted_height, spacing=spacing)
    ref_n = analytic_normals_from_height(reference_height, spacing=spacing)
    if isinstance(pred_n, torch.Tensor):
        cos = (pred_n * ref_n).sum(dim=_normal_channel_axis(pred_n)).clamp(-1.0, 1.0)
        angles = torch.arccos(cos)
        return float(angles.mean().item())
    cos = np.clip((pred_n * ref_n).sum(axis=_normal_channel_axis(pred_n)), -1.0, 1.0)
    angles = np.arccos(cos)
    return float(angles.mean())
