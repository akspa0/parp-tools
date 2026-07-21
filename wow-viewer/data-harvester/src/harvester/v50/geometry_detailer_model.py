"""Spec 114 T059: residual detailer architecture (RGB + generated coarse -> one residual field).

Constitution IV residual chain, made literal: the frozen coarse geometry stage owns coarse relief;
this stage owns exactly one residual signal ``truth − coarse`` (in ``v112.1`` normalized space).
Final relief is ``coarse + residual`` — clamped to [0, 1] only for metrics and artifacts, never
inside the training graph (a hard clamp would zero gradients exactly where the residual is large).

Input layout: minimap RGB (3x256x256) concatenated with the GENERATED coarse field resized to
256x256 (1 channel). The trunk mirrors the proven ``HeightRelativeNet`` U-Net-lite so the two
stages share one well-understood capacity class; the head is LINEAR (no sigmoid) because a
residual is signed. The final 257x257 world-grid alignment matches the coarse stage exactly, so
``coarse_257 + residual_257`` is pixel-consistent.

No ground-truth height, normals, or any other truth signal enters the input. The coarse field is
always the upstream model's generated output, including its errors — never teacher-forced truth.
"""

from __future__ import annotations

import torch
from torch import nn

from harvester.v50.height_relative_model import HEIGHT_GRID
from harvester.v50.model_stage_contract import sha256_json

DETAILER_ARCHITECTURE_ID = "detailer_unet_v1"
INPUT_SIZE = 256


class DetailerContractError(ValueError):
    """Raised when the detailer contract is violated."""


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class GeometryDetailerNet(nn.Module):
    """U-Net-lite residual refiner: (RGB[+features] Cx256x256, coarse 1x257x257) -> residual 257x257.

    Default ``base=32``, ``in_channels=3`` gives 1,561,857 trainable parameters — the coarse
    baseline's capacity class plus one input channel. A zero-initialized head makes the initial
    composition exactly the coarse prediction, so training starts FROM the strong baseline instead
    of below it. ``in_channels`` > 3 admits a Spec 115/116 GENERATED feature map concatenated onto
    RGB (never ground truth), mirroring the coarse stage's own ``--feature-store`` contract.
    """

    def __init__(self, base: int = 32, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.enc1 = _block(in_channels + 1, base)         # 256
        self.enc2 = _block(base, base * 2, stride=2)      # 128
        self.enc3 = _block(base * 2, base * 4, stride=2)  # 64
        self.enc4 = _block(base * 4, base * 8, stride=2)  # 32
        self.mid = _block(base * 8, base * 8)
        self.up3 = _block(base * 8 + base * 4, base * 4)
        self.up2 = _block(base * 4 + base * 2, base * 2)
        self.up1 = _block(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != self.in_channels:
            raise DetailerContractError(
                f"rgb must be (B, {self.in_channels}, {INPUT_SIZE}, {INPUT_SIZE}), got {tuple(rgb.shape)}"
            )
        if coarse.ndim != 3 or coarse.shape[-2:] != (HEIGHT_GRID, HEIGHT_GRID):
            raise DetailerContractError(
                f"coarse must be (B, {HEIGHT_GRID}, {HEIGHT_GRID}), got {tuple(coarse.shape)}"
            )
        coarse_256 = nn.functional.interpolate(
            coarse.unsqueeze(1), size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear", align_corners=True,
        )
        x = torch.cat([rgb, coarse_256], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        m = self.mid(e4)
        u3 = self.up3(torch.cat([
            nn.functional.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False), e3
        ], dim=1))
        u2 = self.up2(torch.cat([
            nn.functional.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False), e2
        ], dim=1))
        u1 = self.up1(torch.cat([
            nn.functional.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False), e1
        ], dim=1))
        residual = self.head(u1)
        return nn.functional.interpolate(
            residual, size=(HEIGHT_GRID, HEIGHT_GRID), mode="bilinear", align_corners=True
        ).squeeze(1)


def compose_final(coarse: torch.Tensor, residual: torch.Tensor, *, clamp: bool) -> torch.Tensor:
    """Final relief = coarse + residual; clamped only for metrics/artifacts, never for training."""
    if coarse.shape != residual.shape:
        raise DetailerContractError(
            f"coarse/residual shapes must match, got {tuple(coarse.shape)} vs {tuple(residual.shape)}"
        )
    final = coarse + residual
    return torch.clamp(final, 0.0, 1.0) if clamp else final


def detailer_identity(model: nn.Module, *, base: int = 32, in_channels: int = 3) -> dict:
    """Schema-conformant architecture block for the detailer stage."""
    config = {
        "class": "GeometryDetailerNet",
        "base": base,
        "in_channels": in_channels,
        "input": f"rgb[+features] {in_channels}x{INPUT_SIZE}x{INPUT_SIZE} + generated coarse 1x{HEIGHT_GRID}x{HEIGHT_GRID}",
        "output": f"residual 1x{HEIGHT_GRID}x{HEIGHT_GRID} (final = coarse + residual)",
        "head": "linear, zero-initialized",
        "target_contract": "v112.1 residual",
    }
    return {
        "id": DETAILER_ARCHITECTURE_ID,
        "config_sha256": sha256_json(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
