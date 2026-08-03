"""Relative-height target contract and lean CNN (Spec 112 US3, T017).

The rejected spec103 lane's structural flaw was its target: absolute global elevation, which
minimap pixels cannot determine (proven 2026-07-18 — best val loss at epoch 1, degrading after,
against a held-out map at a different altitude). This module's target is per-tile min-max
normalized height: adding any constant offset to a tile's heights leaves the target unchanged by
construction, so cross-tile altitude cannot leak into supervision at all
(``contracts/relative-height-target-contract.md``).

The model is deliberately lean (time-to-signal): a from-scratch U-Net-lite, minimap RGB in, one
normalized height field out. Single output, no auxiliary heads, no shared weights (constitution
IV); growth to other targets is a future spec, not an extension here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

TARGET_CONTRACT_VERSION = "v112.1"
# Denominator floor in world units. For a sub-floor tile the target occupies only the corresponding
# fraction of [0, 1], so sub-unit noise is not amplified into a full-range target. Part of the
# versioned contract.
RANGE_FLOOR = 1.0

HEIGHT_GRID = 257  # height_257 world grid; the model upsamples its final feature map to this


def encode_relative_height(height: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return ``(normalized, tile_min, tile_max)`` under contract ``v112.1``.

    The published contract is literal: ``(h - tile_min) / max(tile_range, RANGE_FLOOR)``. A flat
    tile therefore maps to zero and a near-flat tile retains its small relief without magnifying
    it. float64 intermediates keep the altitude-offset-invariance property stable; storage remains
    float32.
    """
    h = np.asarray(height, dtype=np.float64)
    if h.ndim != 2 or h.size == 0 or not np.isfinite(h).all():
        raise ValueError("height must be a non-empty finite 2D array")
    tile_min = float(h.min())
    tile_max = float(h.max())
    denominator = max(tile_max - tile_min, RANGE_FLOOR)
    normalized = np.clip((h - tile_min) / denominator, 0.0, 1.0).astype(np.float32)
    return normalized, tile_min, tile_max


def decode_relative_height(normalized: np.ndarray, tile_min: float, tile_max: float) -> np.ndarray:
    """Invert ``encode_relative_height`` with the same versioned denominator floor."""
    denominator = max(tile_max - tile_min, RANGE_FLOOR)
    return (np.asarray(normalized, dtype=np.float64) * denominator + tile_min).astype(np.float32)


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class HeightRelativeNet(nn.Module):
    """U-Net-lite: 256x256x3 minimap -> 257x257 normalized relative height in [0, 1].

    The default ``base=32`` network has 1,561,537 trainable parameters. Spec 114 records it as
    ``direct_cnn_v112`` so tonight's authored-only bootstrap and the later corrected dual-view
    bakeoff use the exact same baseline architecture.
    """

    def __init__(self, base: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = _block(in_channels, base)             # 256
        self.enc2 = _block(base, base * 2, stride=2)     # 128
        self.enc3 = _block(base * 2, base * 4, stride=2)  # 64
        self.enc4 = _block(base * 4, base * 8, stride=2)  # 32
        self.mid = _block(base * 8, base * 8)
        self.up3 = _block(base * 8 + base * 4, base * 4)
        self.up2 = _block(base * 4 + base * 2, base * 2)
        self.up1 = _block(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        m = self.mid(e4)
        u3 = self.up3(torch.cat([nn.functional.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False), e3], dim=1))
        u2 = self.up2(torch.cat([nn.functional.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False), e2], dim=1))
        u1 = self.up1(torch.cat([nn.functional.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False), e1], dim=1))
        out = torch.sigmoid(self.head(u1))
        # World grid is 257x257; align the prediction to it at the very end.
        return nn.functional.interpolate(out, size=(HEIGHT_GRID, HEIGHT_GRID), mode="bilinear", align_corners=True).squeeze(1)


def height_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 point loss plus a light gradient term so flat-but-wrong plateaus and noisy relief
    are both penalized (same topology rationale as the established wdl_loss)."""
    point = nn.functional.smooth_l1_loss(predicted, target)
    gradient = (
        nn.functional.l1_loss(predicted[:, 1:, :] - predicted[:, :-1, :], target[:, 1:, :] - target[:, :-1, :])
        + nn.functional.l1_loss(predicted[:, :, 1:] - predicted[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
    )
    return point + 0.25 * gradient
