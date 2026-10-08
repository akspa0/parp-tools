"""Trestle Elevation Neural Network (Spec 266).

Modern lean successor to the February 2026 v7 WDL trestle model.
Consumes 6-channel input (Minimap RGB + Upsampled WDL Trestle + Photometric Normals),
predicting high-frequency residual elevation carving ΔZ and global height bounds [Z_min, Z_max].
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from harvester.v60.supervised_elevation_model import (
    ResidualBlock,
    compute_elevation_normals,
)


class TrestleElevationUNet(nn.Module):
    """Lean 6-channel U-Net with WDL trestle injection and dual residual/bounds heads."""

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 32,
        delta_std: float = 60.0,
        delta_mean: float = 0.0,
    ):
        super().__init__()
        self.delta_std = delta_std
        self.delta_mean = delta_mean

        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256

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

        # Auxiliary Global Height Bounds Head: predicts [Z_min, Z_max] in yards
        self.bounds_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c4, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # Decoder Stage 3: 64x64
        self.up1 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.GELU(),
            ResidualBlock(c3),
        )

        # Decoder Stage 2: 128x128
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.GELU(),
            ResidualBlock(c2),
        )

        # Decoder Stage 1: 256x256
        self.up3 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.GELU(),
            ResidualBlock(c1),
        )

        # Dense Residual Elevation Head
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, c1 // 2),
            nn.GELU(),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: (B, 6, 256, 256) tensor:
                - ch 0..2: Minimap RGB [0, 1]
                - ch 3: Normalized WDL trestle
                - ch 4..5: Surface normals (Nx, Ny)

        Returns:
            delta_z: (B, 257, 257) predicted residual height in yards
            pred_bounds: (B, 2) predicted [Z_min, Z_max] in yards
        """
        x1 = self.inc(features)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bounds head from bottleneck
        pred_bounds = self.bounds_head(x4)

        # Decoder
        u1 = self.up1(x4)
        d1 = self.dec1(torch.cat([u1, x3], dim=1))

        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, x1], dim=1))

        pred_256 = self.head(d3)
        pred_257 = F.interpolate(
            pred_256,
            size=(257, 257),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

        delta_z = pred_257 * self.delta_std + self.delta_mean
        return delta_z, pred_bounds

    def reconstruct(
        self,
        features: torch.Tensor,
        z_trestle: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct absolute terrain surface Z_final = Z_trestle + ΔZ."""
        delta_z, _ = self.forward(features)
        return z_trestle + delta_z

    @property
    def parameter_count(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def trestle_composite_loss(
    pred_delta_z: torch.Tensor,
    pred_bounds: torch.Tensor,
    z_trestle: torch.Tensor,
    gt_z: torch.Tensor,
    gt_bounds: torch.Tensor,
    alpha_final_l1: float = 1.0,
    alpha_grad: float = 0.5,
    beta_normal: float = 0.25,
    gamma_bounds: float = 0.1,
    lambda_delta_reg: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Composite loss supervising residual carving and global elevation fidelity."""
    # 1. Final surface height loss: Z_final = Z_trestle + pred_delta_z
    z_final = z_trestle + pred_delta_z
    final_l1 = F.l1_loss(z_final, gt_z)

    # 2. Gradient consistency on final refined surface
    final_grad_y = z_final[:, 1:, :] - z_final[:, :-1, :]
    gt_grad_y = gt_z[:, 1:, :] - gt_z[:, :-1, :]
    final_grad_x = z_final[:, :, 1:] - z_final[:, :, :-1]
    gt_grad_x = gt_z[:, :, 1:] - gt_z[:, :, :-1]
    grad_loss = F.l1_loss(final_grad_y, gt_grad_y) + F.l1_loss(final_grad_x, gt_grad_x)

    # 3. Surface normal cosine loss
    pred_n = compute_elevation_normals(z_final)
    gt_n = compute_elevation_normals(gt_z)
    cos_sim = torch.sum(pred_n * gt_n, dim=1)
    normal_loss = torch.mean(1.0 - cos_sim)

    # 4. Bounds loss on [Z_min, Z_max]
    bounds_loss = F.l1_loss(pred_bounds, gt_bounds)

    # 5. Delta smoothness / magnitude regularization (prevents spiky high-frequency noise)
    delta_grad_y = pred_delta_z[:, 1:, :] - pred_delta_z[:, :-1, :]
    delta_grad_x = pred_delta_z[:, :, 1:] - pred_delta_z[:, :, :-1]
    delta_smooth = torch.mean(torch.abs(delta_grad_y)) + torch.mean(torch.abs(delta_grad_x))

    total_loss = (
        alpha_final_l1 * final_l1
        + alpha_grad * grad_loss
        + beta_normal * normal_loss
        + gamma_bounds * bounds_loss
        + lambda_delta_reg * delta_smooth
    )

    metrics = {
        "final_l1": float(final_l1.item()),
        "grad_loss": float(grad_loss.item()),
        "normal_loss": float(normal_loss.item()),
        "bounds_loss": float(bounds_loss.item()),
        "delta_smooth": float(delta_smooth.item()),
        "total_loss": float(total_loss.item()),
    }
    return total_loss, metrics
