"""Supervised Neural Elevation Model for WoW Minimap-to-3D Terrain Reconstruction (Spec 264).

Directly maps authored 256x256 minimap RGB images to true 257x257 world-space
elevation grids in physical yards, learning both macroscopic topography and
high-frequency ridge contours without classical Shape-from-Shading integration drift.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Residual convolution block with GroupNorm and GELU activations."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + residual)


class SupervisedElevationUNet(nn.Module):
    """End-to-end multi-scale U-Net mapping 256x256 minimap RGB to 257x257 elevation in yards.

    Inputs:
        x: (B, in_channels, 256, 256) float32 in [0, 1] (minimap RGB or RGB + optional mask)
    Outputs:
        z: (B, 257, 257) float32 in physical world yards
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        global_elevation_mean: float = 75.0,
        global_elevation_std: float = 75.0,
    ):
        super().__init__()
        self.global_elevation_mean = global_elevation_mean
        self.global_elevation_std = global_elevation_std

        c1 = base_channels       # 48
        c2 = base_channels * 2   # 96
        c3 = base_channels * 4   # 192
        c4 = base_channels * 8   # 384

        # Encoder Stage 1: 256x256
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.GELU(),
            ResidualBlock(c1),
        )

        # Encoder Stage 2: 128x128
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.GELU(),
            ResidualBlock(c2),
        )

        # Encoder Stage 3: 64x64
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.GELU(),
            ResidualBlock(c3),
        )

        # Encoder Stage 4: 32x32 (Bottleneck with dilated receptive field)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c4),
            nn.GELU(),
            ResidualBlock(c4),
            nn.Conv2d(c4, c4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, c4),
            nn.GELU(),
            ResidualBlock(c4),
        )

        # Decoder Stage 3: 64x64
        self.up1_conv = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.GELU(),
            ResidualBlock(c3),
        )

        # Decoder Stage 2: 128x128
        self.up2_conv = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.GELU(),
            ResidualBlock(c2),
        )

        # Decoder Stage 1: 256x256
        self.up3_conv = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.GELU(),
            ResidualBlock(c1),
        )

        # Final head to 1 channel normalized elevation
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, c1 // 2),
            nn.GELU(),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass emitting elevation in physical yards."""
        # Encoder
        x1 = self.inc(x)         # (B, c1, 256, 256)
        x2 = self.down1(x1)      # (B, c2, 128, 128)
        x3 = self.down2(x2)      # (B, c3, 64, 64)
        x4 = self.down3(x3)      # (B, c4, 32, 32)

        # Decoder
        u1 = self.up1_conv(x4)
        d1 = self.dec1(torch.cat([u1, x3], dim=1))

        u2 = self.up2_conv(d1)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u3 = self.up3_conv(d2)
        d3 = self.dec3(torch.cat([u3, x1], dim=1))

        # Predict normalized relative height
        norm_elev_256 = self.head(d3)  # (B, 1, 256, 256)

        # Interpolate to exact 257x257 vertex grid
        norm_elev_257 = F.interpolate(
            norm_elev_256,
            size=(257, 257),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)  # (B, 257, 257)

        # Denormalize to true physical yards
        elevation_yards = norm_elev_257 * self.global_elevation_std + self.global_elevation_mean
        return elevation_yards


def compute_elevation_normals(z: torch.Tensor, yard_per_sample: float = 533.3333 / 256.0) -> torch.Tensor:
    """Compute surface unit normals from elevation grid in yards.

    Args:
        z: (B, 257, 257) tensor of heights in yards
        yard_per_sample: physical distance between samples (2.083 yards)
    Returns:
        normals: (B, 3, 257, 257) unit normal vectors
    """
    dz_dy = (z[:, 2:, 1:-1] - z[:, :-2, 1:-1]) / (2.0 * yard_per_sample)
    dz_dx = (z[:, 1:-1, 2:] - z[:, 1:-1, :-2]) / (2.0 * yard_per_sample)

    normals = torch.zeros((z.shape[0], 3, z.shape[1] - 2, z.shape[2] - 2), device=z.device, dtype=z.dtype)
    normals[:, 0] = -dz_dx
    normals[:, 1] = -dz_dy
    normals[:, 2] = 1.0
    norm = torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-6)
    return normals / norm


def elevation_composite_loss(
    pred_z: torch.Tensor,
    gt_z: torch.Tensor,
    alpha_grad: float = 0.5,
    beta_normal: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-scale composite loss: L1 elevation error + gradient consistency + surface normal alignment.

    Args:
        pred_z: (B, 257, 257) predicted height in yards
        gt_z: (B, 257, 257) ground truth height in yards
    Returns:
        loss: scalar tensor
        metrics: breakdown dict for logging
    """
    # 1. Direct elevation L1 error (in physical yards)
    l1_loss = F.l1_loss(pred_z, gt_z)

    # 2. Gradient / slope error (prevents over-smoothed rolling hills)
    pred_grad_y = pred_z[:, 1:, :] - pred_z[:, :-1, :]
    gt_grad_y = gt_z[:, 1:, :] - gt_z[:, :-1, :]
    pred_grad_x = pred_z[:, :, 1:] - pred_z[:, :, :-1]
    gt_grad_x = gt_z[:, :, 1:] - gt_z[:, :, :-1]
    grad_loss = F.l1_loss(pred_grad_y, gt_grad_y) + F.l1_loss(pred_grad_x, gt_grad_x)

    # 3. Surface normal cosine loss (enforces authentic cliff/ridge angles)
    pred_n = compute_elevation_normals(pred_z)
    gt_n = compute_elevation_normals(gt_z)
    cos_sim = torch.sum(pred_n * gt_n, dim=1)  # (B, 255, 255)
    normal_loss = torch.mean(1.0 - cos_sim)

    total_loss = l1_loss + alpha_grad * grad_loss + beta_normal * normal_loss

    metrics = {
        "l1_yards": float(l1_loss.item()),
        "grad_loss": float(grad_loss.item()),
        "normal_loss": float(normal_loss.item()),
        "total_loss": float(total_loss.item()),
    }
    return total_loss, metrics
