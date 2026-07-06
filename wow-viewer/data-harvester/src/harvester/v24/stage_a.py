"""FR-011: Stage A — minimap -> WDL prior correlation model.

A small U-Net (<= 1M params) on 64x64 inputs with two interpolation heads that
emit the C# reader's grid shapes (17x17 outer + 16x16 inner) via the 33x33
quincunx lattice (spec amendments A6/A7).

Input channels (13):
  0-2  cleaned minimap RGB (256 -> 64 mean pool)
  3-6  alpha_256 (256 -> 64)
  7-9  normal_xyz (257 -> 256 crop -> 64)
  10   mcnr_mask_257 (257 -> 256 crop -> 64)
  11   synthetic-WDL "cheat" channel (quincunx 33 -> 64, heights / HEIGHT_SCALE)
  12   synth presence flag (1 when channel 11 is populated)

During training the synth channel is dropped with probability 0.5 so the model
also learns the minimap-only deployment regime (User Story 3, scenario 5).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from harvester.v24 import lattice
from harvester.v24.tiles import HEIGHT_SCALE, TileRecord, downsample_mean

IN_CHANNELS = 13
GRID = 64


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(4, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(4, cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StageAModel(nn.Module):
    """Small U-Net predicting a residual over the synth quincunx (RULE 7).

    When the synthetic prior is present (cheat regime) the model starts at the
    trivial baseline thanks to the zero-initialized head; when it is dropped
    (minimap-only deployment regime) the model predicts the full field.
    """

    def __init__(self, base: int = 28):
        super().__init__()
        self.enc1 = _ConvBlock(IN_CHANNELS, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, x: torch.Tensor, synth_quincunx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (outer (B,17,17), inner (B,16,16)) in normalized height space.

        ``synth_quincunx`` is the (B, 33, 33) normalized synthetic prior with
        dropped samples zeroed; it is added to the predicted residual field.
        """
        e1 = self.enc1(x)  # (B, base, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, 2b, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, 4b, 16, 16)

        d2 = F.interpolate(e3, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))  # (B, 2b, 32, 32)
        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1))  # (B, b, 64, 64)

        field = self.head(d1)  # (B, 1, 64, 64)
        quincunx = F.interpolate(
            field, size=(33, 33), mode="bilinear", align_corners=True
        ).squeeze(1)
        if synth_quincunx is not None:
            quincunx = quincunx + synth_quincunx
        outer = quincunx[:, ::2, ::2]
        inner = quincunx[:, 1::2, 1::2]
        return outer, inner


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_input(record: TileRecord, include_synth: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (13, 64, 64) input tensor + the (33, 33) synth quincunx.

    The quincunx (normalized heights; zeros when ``include_synth`` is False) is
    the residual anchor the model adds its prediction to.
    """
    minimap = downsample_mean(record.cleaned_minimap, 4)  # (64, 64, 3)
    alpha = downsample_mean(record.alpha, 4)  # (64, 64, 4)
    normal = downsample_mean(record.normal[:256, :256], 4)  # (64, 64, 3)
    mcnr = downsample_mean(record.mcnr_mask[:256, :256].astype(np.float32), 4)  # (64, 64)

    channels = [
        minimap[..., 0], minimap[..., 1], minimap[..., 2],
        alpha[..., 0], alpha[..., 1], alpha[..., 2], alpha[..., 3],
        normal[..., 0], normal[..., 1], normal[..., 2],
        mcnr,
    ]

    if include_synth:
        quincunx = lattice.quincunx_33(record.synth_outer, record.synth_inner) / HEIGHT_SCALE
        synth64 = _resize_bilinear(quincunx, GRID)
        channels.append(synth64)
        channels.append(np.ones((GRID, GRID), dtype=np.float32))
    else:
        quincunx = np.zeros((33, 33), dtype=np.float32)
        channels.append(np.zeros((GRID, GRID), dtype=np.float32))
        channels.append(np.zeros((GRID, GRID), dtype=np.float32))

    return np.stack(channels).astype(np.float32), quincunx.astype(np.float32)


def build_target(record: TileRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Targets + per-cell loss weights (confidence, learned-fill excluded)."""
    outer = record.prior_outer / HEIGHT_SCALE
    inner = record.prior_inner / HEIGHT_SCALE
    weight_outer = record.confidence_outer * (record.source_outer != 2)
    weight_inner = record.confidence_inner * (record.source_inner != 2)
    return (
        outer.astype(np.float32),
        inner.astype(np.float32),
        weight_outer.astype(np.float32),
        weight_inner.astype(np.float32),
    )


def weighted_l1(
    pred_outer: torch.Tensor,
    pred_inner: torch.Tensor,
    target_outer: torch.Tensor,
    target_inner: torch.Tensor,
    weight_outer: torch.Tensor,
    weight_inner: torch.Tensor,
) -> torch.Tensor:
    num = (weight_outer * (pred_outer - target_outer).abs()).sum() + (
        weight_inner * (pred_inner - target_inner).abs()
    ).sum()
    den = weight_outer.sum() + weight_inner.sum()
    return num / den.clamp_min(1e-6)


def _resize_bilinear(array: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    out = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=True)
    return out[0, 0].numpy()
