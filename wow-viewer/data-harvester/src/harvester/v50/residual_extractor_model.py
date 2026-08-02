"""Residual-extractor target contract and lean CNN (Spec 125 US7).

The step-1 decomposition (user, 2026-08-02): train a model to extract the textureless terrain-shadow
residual from any minimap RGB tile. Once the residual is known, subtracting it from the minimap strips
away the shading layer, revealing the albedo/texturing (MCAL/MCLY/MTEX) underneath.

The scale test refuted the direct heightmap-transform hypothesis (correlation ~0.20, best-fit scale
~-0.0003), so the residual is a learned (nonlinear) shading signal — which means the way to get it
from an arbitrary minimap is to LEARN the extraction. The synthesizer produces both the full minimap
RGB and the textureless residual for the same tile, so the training pairs are exact and free.

The model is a lean U-Net-lite: minimap RGB in (3 channels), one residual field out (256x256). Single
output, no auxiliary heads, no shared weights (constitution IV).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

TARGET_CONTRACT_VERSION = "v125.2"
RESIDUAL_GRID = 256  # the textureless residual is 256x256


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class ResidualExtractorNet(nn.Module):
    """U-Net-lite: 256x256x3 minimap RGB -> 256x256 residual in [0, 1].

    The default ``base=32`` network mirrors the Spec 112/114 architecture family so the extractor is
    directly comparable to the other minimap lanes.
    """

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = _block(3, base)                       # 256
        self.enc2 = _block(base, base * 2, stride=2)      # 128
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
        return nn.functional.interpolate(out, size=(RESIDUAL_GRID, RESIDUAL_GRID), mode="bilinear", align_corners=True).squeeze(1)


def residual_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 point loss plus a light gradient term so flat-but-wrong plateaus and noisy relief
    are both penalized (same topology rationale as the established height_loss)."""
    point = nn.functional.smooth_l1_loss(predicted, target)
    gradient = (
        nn.functional.l1_loss(predicted[:, 1:, :] - predicted[:, :-1, :], target[:, 1:, :] - target[:, :-1, :])
        + nn.functional.l1_loss(predicted[:, :, 1:] - predicted[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
    )
    return point + 0.25 * gradient
