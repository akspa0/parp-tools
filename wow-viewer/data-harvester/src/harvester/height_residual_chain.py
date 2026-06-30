"""Shared helpers for the Spec 077 H0/H1 residual height chain."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_SOBEL_X: torch.Tensor | None = None
_SOBEL_Y: torch.Tensor | None = None


def height_chain_input_channels(*, use_albedo: bool = False, use_density: bool = False, include_base: bool = False) -> int:
    channels = 3
    if use_albedo:
        channels += 3
    if use_density:
        channels += 3
    if include_base:
        channels += 1
    return channels


def _get_sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    global _SOBEL_X, _SOBEL_Y
    if _SOBEL_X is None or _SOBEL_X.device != device or _SOBEL_X.dtype != dtype:
        _SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
        _SOBEL_Y = _SOBEL_X.transpose(2, 3).contiguous()
    return _SOBEL_X, _SOBEL_Y


def compute_density_channels(rgb: torch.Tensor) -> torch.Tensor:
    """Derive high/medium/low edge-density channels from RGB in ``[0, 1]``."""
    rgb = rgb.float()
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    sx, sy = _get_sobel_kernels(rgb.device, rgb.dtype)
    gx = F.conv2d(gray, sx, padding=1)
    gy = F.conv2d(gray, sy, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    flat = grad_mag.reshape(rgb.shape[0], -1)
    p25 = flat.quantile(0.25, dim=1)
    p75 = flat.quantile(0.75, dim=1)
    hi = p75[:, None, None, None]
    lo = p25[:, None, None, None]
    high = (grad_mag > hi).to(rgb.dtype)
    low = (grad_mag <= lo).to(rgb.dtype)
    medium = ((grad_mag > lo) & (grad_mag <= hi)).to(rgb.dtype)
    return torch.cat([high, medium, low], dim=1)


def build_height_chain_input(
    batch: dict,
    *,
    device: torch.device,
    use_albedo: bool = False,
    use_density: bool = False,
    base_height_257: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the documented H0/H1 input tensor from a dataset batch."""
    prior = batch["input_prior"].to(device, non_blocking=True)[:, :3, :, :].float()
    parts = [prior]
    if use_albedo:
        if "albedo_rgb" not in batch:
            raise KeyError("--albedo was requested but batch has no albedo_rgb tensor")
        parts.append(batch["albedo_rgb"].to(device, non_blocking=True).float())
    if use_density:
        parts.append(compute_density_channels(prior))
    if base_height_257 is not None:
        base = base_height_257.to(device, non_blocking=True).float()
        target_size = base.shape[-2:]
        parts = [
            F.interpolate(part, size=target_size, mode="bilinear", align_corners=True)
            if part.shape[-2:] != target_size
            else part
            for part in parts
        ]
        parts.append(base)
    return torch.cat(parts, dim=1)


def downsample_height_target(
    height_257: torch.Tensor,
    weight_257: torch.Tensor,
    *,
    coarse_size: int = 65,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = F.interpolate(height_257.float(), size=(int(coarse_size), int(coarse_size)), mode="area")
    weight = F.interpolate(weight_257.float(), size=(int(coarse_size), int(coarse_size)), mode="area").clamp(0.0, 1.0)
    return target, weight


def upsample_coarse_height(coarse_height: torch.Tensor, *, size: int = 257) -> torch.Tensor:
    return F.interpolate(coarse_height.float(), size=(int(size), int(size)), mode="bilinear", align_corners=True)


def residual_target(height_257: torch.Tensor, base_height_257: torch.Tensor) -> torch.Tensor:
    return height_257.float() - base_height_257.float()


def compose_refined_height(base_height_257: torch.Tensor, height_delta_257: torch.Tensor) -> torch.Tensor:
    return base_height_257.float() + height_delta_257.float()


def masked_charbonnier(diff: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-3) -> torch.Tensor:
    loss_map = torch.sqrt(diff.float() * diff.float() + float(eps) * float(eps))
    return (loss_map * weight.float()).sum() / weight.float().sum().clamp_min(1e-8)
