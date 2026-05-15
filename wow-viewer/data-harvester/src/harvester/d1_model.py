"""D1 U-Net: small encoder-decoder with 4 output heads.

Architecture:
    Input:  (3, 256, 256)  — minimap_rgb_256
    Output: (3, 256, 256)  — tileset_layer_1
            (3, 256, 256)  — tileset_layer_2
            (1, 256, 256)  — alpha_mask_1
            (1, 256, 256)  — alpha_mask_2

Encoder: 64 → 96 → 160 → 224  (double-conv blocks with stride-2 down)
Decoder: 224 → 160 → 96 → 64  (upsample + convolution blocks with skip)
Heads:   4 independent 1×1 conv heads from final 32-channel feature map.

~3.0M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Double convolution: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None) -> None:
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    """MaxPool → ConvBlock (stride-2 downsample then double conv)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    """Upsample → Conv → concat skip → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, padding=0, bias=False)
        self.conv = _ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.up_conv(x)
        # Handle odd-sized tensors from pooling
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class D1UNet(nn.Module):
    """V14 Model D1: Tileset Decomposition U-Net.

    Input:  (B, 3, 256, 256)
    Output: (tileset_layer_1, tileset_layer_2, alpha_mask_1, alpha_mask_2)
            each tileset: (B, 3, 256, 256), each alpha: (B, 1, 256, 256)
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.enc0 = _ConvBlock(3, 64)  # 256×256 → 256×256
        self.enc1 = _DownBlock(64, 96)  # 256×256 → 128×128
        self.enc2 = _DownBlock(96, 160)  # 128×128 → 64×64
        self.enc3 = _DownBlock(160, 224)  # 64×64   → 32×32

        # Bottleneck
        self.bottleneck = _ConvBlock(224, 224)  # 32×32   → 32×32

        # Decoder
        self.dec3 = _UpBlock(224, 224, 160)  # 32×32   → 64×64
        self.dec2 = _UpBlock(160, 160, 96)  # 64×64   → 128×128
        self.dec1 = _UpBlock(96, 96, 64)  # 128×128 → 256×256
        self.dec0 = _UpBlock(64, 64, 32)  # 256×256 → 256×256

        # Output heads
        self.head_tileset_1 = nn.Conv2d(32, 3, kernel_size=1)
        self.head_tileset_2 = nn.Conv2d(32, 3, kernel_size=1)
        self.head_alpha_1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head_alpha_2 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        # Heads
        tileset_1 = torch.sigmoid(self.head_tileset_1(d0))
        tileset_2 = torch.sigmoid(self.head_tileset_2(d0))
        alpha_1 = self.head_alpha_1(d0)
        alpha_2 = self.head_alpha_2(d0)

        return tileset_1, tileset_2, alpha_1, alpha_2

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
