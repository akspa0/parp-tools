"""V21 scar-mask segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as torch_f


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class V21ScarMaskModel(nn.Module):
    """Small U-Net producing one scar-mask logits tensor."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc0 = _ConvBlock(3, c)
        self.enc1 = _ConvBlock(c, c * 2)
        self.enc2 = _ConvBlock(c * 2, c * 4)
        self.bottleneck = _ConvBlock(c * 4, c * 4)
        self.dec2 = _ConvBlock(c * 4 + c * 4, c * 2)
        self.dec1 = _ConvBlock(c * 2 + c * 2, c)
        self.dec0 = _ConvBlock(c + c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(torch_f.max_pool2d(e0, 2))
        e2 = self.enc2(torch_f.max_pool2d(e1, 2))
        b = self.bottleneck(torch_f.max_pool2d(e2, 2))
        d2 = torch_f.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = torch_f.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = torch_f.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([d0, e0], dim=1))
        return self.head(d0)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
