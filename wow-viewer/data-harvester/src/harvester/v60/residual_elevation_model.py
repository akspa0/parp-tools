"""Stage 2 Cascaded Residual Elevation Refinement Model (Spec 265).

Consumes 6-channel feature maps (Minimap RGB + Initial Elevation + Surface Normals)
and directly predicts the residual elevation error field ΔZ in physical world yards,
restoring compressed mountain amplitudes and correcting texture splatting artifacts.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from harvester.v60.supervised_elevation_model import (
    ResidualBlock,
    compute_elevation_normals,
)


class ResidualElevationRefiner(nn.Module):
    """Stage 2 U-Net predicting ΔZ = Z_gt - Z_initial in physical yards."""

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 40,
        delta_std: float = 80.0,
        delta_mean: float = 0.0,
    ):
        super().__init__()
        self.delta_std = delta_std
        self.delta_mean = delta_mean

        c1 = base_channels       # 40
        c2 = base_channels * 2   # 80
        c3 = base_channels * 4   # 160
        c4 = base_channels * 8   # 320

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

        # Encoder Stage 4: 32x32 (Receptive Field across 533-yard tile)
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

        # Residual Prediction Head
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, c1 // 2),
            nn.GELU(),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Emits predicted residual elevation field ΔZ in physical yards.

        Args:
            features: (B, 6, 256, 256) tensor
        Returns:
            delta_z: (B, 257, 257) predicted residual height in yards
        """
        x1 = self.inc(features)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

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
        return delta_z


def residual_composite_loss(
    pred_delta_z: torch.Tensor,
    gt_delta_z: torch.Tensor,
    z_init: torch.Tensor,
    gt_z: torch.Tensor,
    alpha_final_l1: float = 1.0,
    alpha_grad: float = 0.5,
    beta_normal: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite loss evaluating both the residual error and final reconstructed terrain surface."""
    # 1. Direct L1 loss on the residual field
    res_l1 = F.l1_loss(pred_delta_z, gt_delta_z)

    # 2. End-to-end composite height loss: Z_final = Z_init + pred_delta_z
    z_final = z_init + pred_delta_z
    final_l1 = F.l1_loss(z_final, gt_z)

    # 3. Gradient consistency on final refined surface
    final_grad_y = z_final[:, 1:, :] - z_final[:, :-1, :]
    gt_grad_y = gt_z[:, 1:, :] - gt_z[:, :-1, :]
    final_grad_x = z_final[:, :, 1:] - z_final[:, :, :-1]
    gt_grad_x = gt_z[:, :, 1:] - gt_z[:, :, :-1]
    grad_loss = F.l1_loss(final_grad_y, gt_grad_y) + F.l1_loss(final_grad_x, gt_grad_x)

    # 4. Surface normal cosine loss
    pred_n = compute_elevation_normals(z_final)
    gt_n = compute_elevation_normals(gt_z)
    cos_sim = torch.sum(pred_n * gt_n, dim=1)
    normal_loss = torch.mean(1.0 - cos_sim)

    total_loss = res_l1 + alpha_final_l1 * final_l1 + alpha_grad * grad_loss + beta_normal * normal_loss

    metrics = {
        "res_l1": float(res_l1.item()),
        "final_l1": float(final_l1.item()),
        "grad_loss": float(grad_loss.item()),
        "normal_loss": float(normal_loss.item()),
        "total_loss": float(total_loss.item()),
    }
    return total_loss, metrics
