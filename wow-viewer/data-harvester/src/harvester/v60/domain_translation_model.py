"""Lightweight Domain Translation Model for Minimap Object Stripping & Hillshade Synthesis (Spec 274).

Translates real textured minimap tiles (with alpha splats, doodads, roofs, roads)
into object-free synthetic DXT1 hillshade representations.

Parameter Budget: <= 10M parameters (Actual ~4.4M).
Loss: Composite L1 + Sobel Edge Gradient + SSIM Structural Loss.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual Convolutional Block with GroupNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act2 = nn.GELU()

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + res)


class DomainTranslationUNet(nn.Module):
    """Lightweight U-Net translating real textured minimap RGB -> synthetic DXT1 hillshade.

    Architecture:
        - 4 encoder stages (32 -> 64 -> 128 -> 256 channels)
        - Bottleneck with dilated residual blocks (256 channels)
        - 4 decoder stages with skip connections
        - Sigmoid output head in [0, 1]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ConvBlock(32, 32),
        )  # (B, 32, 256, 256)

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ConvBlock(64, 64),
        )  # (B, 64, 128, 128)

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ConvBlock(128, 128),
        )  # (B, 128, 64, 64)

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ConvBlock(256, 256),
        )  # (B, 256, 32, 32)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(256, 256, dilation=2),
            ConvBlock(256, 256, dilation=1),
        )  # (B, 256, 32, 32)

        # Decoder
        self.up3 = nn.Sequential(
            ConvBlock(256 + 128, 128),
            ConvBlock(128, 128),
        )

        self.up2 = nn.Sequential(
            ConvBlock(128 + 64, 64),
            ConvBlock(64, 64),
        )

        self.up1 = nn.Sequential(
            ConvBlock(64 + 32, 32),
            ConvBlock(32, 32),
        )

        self.out_head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.init_conv(x)      # (B, 32, 256, 256)
        e1 = self.down1(e0)         # (B, 64, 128, 128)
        e2 = self.down2(e1)         # (B, 128, 64, 64)
        e3 = self.down3(e2)         # (B, 256, 32, 32)

        # Bottleneck
        b = self.bottleneck(e3)     # (B, 256, 32, 32)

        # Decoder with skip connections
        u3 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)  # (B, 256, 64, 64)
        d2 = self.up3(torch.cat([u3, e2], dim=1))                                     # (B, 128, 64, 64)

        u2 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False) # (B, 128, 128, 128)
        d1 = self.up2(torch.cat([u2, e1], dim=1))                                     # (B, 64, 128, 128)

        u1 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False) # (B, 64, 256, 256)
        d0 = self.up1(torch.cat([u1, e0], dim=1))                                     # (B, 32, 256, 256)

        return self.out_head(d0)


class SobelGradientLoss(nn.Module):
    """Computes L1 loss on horizontal and vertical Sobel gradients to preserve ridge lines."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)

        weight_x = sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        weight_y = sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)

        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)
        self.channels = in_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_x_pred = F.conv2d(pred, self.weight_x, padding=1, groups=self.channels)
        grad_y_pred = F.conv2d(pred, self.weight_y, padding=1, groups=self.channels)

        grad_x_tgt = F.conv2d(target, self.weight_x, padding=1, groups=self.channels)
        grad_y_tgt = F.conv2d(target, self.weight_y, padding=1, groups=self.channels)

        return F.l1_loss(grad_x_pred, grad_x_tgt) + F.l1_loss(grad_y_pred, grad_y_tgt)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for structural and lighting gradient preservation."""

    def __init__(self, window_size: int = 11, in_channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = in_channels

        # Gaussian window
        sigma = 1.5
        gauss = torch.tensor(
            [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)],
            dtype=torch.float32,
        )
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        self.register_buffer("window", window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        p = self.window_size // 2
        mu1 = F.conv2d(img1, self.window, padding=p, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=p, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=p, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=p, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=p, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()


class DomainTranslationLoss(nn.Module):
    """Composite loss balancing L1 photometric fidelity, Sobel edges, and SSIM structure."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        grad_weight: float = 0.5,
        ssim_weight: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.ssim_weight = ssim_weight

        self.grad_loss = SobelGradientLoss(in_channels=in_channels)
        self.ssim_loss = SSIMLoss(in_channels=in_channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = F.l1_loss(pred, target)
        grad = self.grad_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        total = self.l1_weight * l1 + self.grad_weight * grad + self.ssim_weight * ssim
        metrics = {
            "loss_total": float(total.item()),
            "loss_l1": float(l1.item()),
            "loss_grad": float(grad.item()),
            "loss_ssim": float(ssim.item()),
        }
        return total, metrics
