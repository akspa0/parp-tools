"""R1 U-Net: Terrain Reconstruction from residual.

Architecture per the V14 plan:
    Input:  residual (3, 256, 256)  — 3-channel RGB residual image
    Output: height     (1, 257, 257) — world-space Z heightmap
            hole_mask  (1, 16, 16)   — binary hole mask
            liquid     (1, 256, 256) — liquid coverage mask

Encoder: 32 → 64 → 128 → 256 → 512  (double-conv + MaxPool)
Bottleneck: 512 → 512
Decoder (with skip from shallower encoder level):
    dec4: bottleneck + e3 → 256@32
    dec3: 256 + e2 → 128@64
    dec2: 128 + e1 → 64@128
    dec1: 64 + e0 → 32@256
    dec0: final refinement, no skip

Heads: 3 independent conv heads from final d0 features.

~5.2M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_conv = nn.Conv2d(in_ch, out_ch, 2, bias=False)
        self.conv = _ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.up_conv(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                   diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class R1UNet(nn.Module):
    """V14 Model R1: Terrain Reconstruction.

    Input:  (B, 3, 256, 256)
    Output: height (B, 1, 257, 257), holes (B, 1, 16, 16), liquid (B, 1, 256, 256)
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder (5 levels)
        self.enc0 = _ConvBlock(3, 32)
        self.enc1 = _DownBlock(32, 64)
        self.enc2 = _DownBlock(64, 128)
        self.enc3 = _DownBlock(128, 256)
        self.enc4 = _DownBlock(256, 512)

        self.bottleneck = _ConvBlock(512, 512)

        # Decoder — skips from one encoder level ABOVE (shallower)
        self.dec4 = _UpBlock(512, 256, 256)   # skip e3
        self.dec3 = _UpBlock(256, 128, 128)   # skip e2
        self.dec2 = _UpBlock(128, 64, 64)     # skip e1
        self.dec1 = _UpBlock(64, 32, 32)      # skip e0
        self.dec0 = _ConvBlock(32, 32)        # final refinement

        # Heads
        self.head_height = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
        )
        self.head_holes = nn.Sequential(
            nn.Conv2d(32, 8, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(8, 1, 1),
        )
        self.head_liquid = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = self.enc0(x)          # 32×256×256
        e1 = self.enc1(e0)         # 64×128×128
        e2 = self.enc2(e1)         # 128×64×64
        e3 = self.enc3(e2)         # 256×32×32
        e4 = self.enc4(e3)         # 512×16×16

        b = self.bottleneck(e4)    # 512×16×16

        # Skip from shallower level: b(+e3), d4(+e2), d3(+e1), d2(+e0)
        d4 = self.dec4(b, e3)     # 256×32×32
        d3 = self.dec3(d4, e2)    # 128×64×64
        d2 = self.dec2(d3, e1)    # 64×128×128
        d1 = self.dec1(d2, e0)    # 32×256×256
        d0 = self.dec0(d1)        # 32×256×256

        height = self.head_height(d0)
        holes = self.head_holes(d0)
        liquid = torch.sigmoid(self.head_liquid(d0))
        return height, holes, liquid

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
