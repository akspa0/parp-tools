"""W1: cleaned minimap + frozen H0 datum -> one 545-sample WDL residual."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .m0 import ConvBlock


WDL_SAMPLE_COUNT = (17 * 17) + (16 * 16)
RELIEF_SCALE = 128.0


def wdl_query_coordinates(device: torch.device | None = None) -> torch.Tensor:
    outer_axis = torch.linspace(-1.0, 1.0, 17, device=device)
    outer_y, outer_x = torch.meshgrid(outer_axis, outer_axis, indexing="ij")
    inner_axis = ((torch.arange(16, device=device, dtype=torch.float32) * 16.0 + 8.0) / 256.0) * 2.0 - 1.0
    inner_y, inner_x = torch.meshgrid(inner_axis, inner_axis, indexing="ij")
    return torch.cat(
        [
            torch.stack([outer_x.reshape(-1), outer_y.reshape(-1)], dim=1),
            torch.stack([inner_x.reshape(-1), inner_y.reshape(-1)], dim=1),
        ],
        dim=0,
    )


class W1WdlResidual(nn.Module):
    """Shared coordinate-query decoder; no separate outer/inner prediction heads."""

    def __init__(self, base_channels: int = 50) -> None:
        super().__init__()
        self.encoder0 = ConvBlock(3, base_channels)
        self.encoder1 = ConvBlock(base_channels, base_channels * 2)
        self.encoder2 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder3 = ConvBlock(base_channels * 4, base_channels * 8)
        feature_channels = base_channels * 8
        self.query_decoder = nn.Sequential(
            nn.Conv1d(feature_channels + 3, feature_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv1d(feature_channels, feature_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv1d(feature_channels // 2, 1, 1),
        )
        self.register_buffer("query_coordinates", wdl_query_coordinates(), persistent=True)

    def forward(self, cleaned_rgb: torch.Tensor, h0_datum: torch.Tensor) -> torch.Tensor:
        if cleaned_rgb.ndim != 4 or cleaned_rgb.shape[1] != 3:
            raise ValueError(f"W1 expects cleaned RGB [batch,3,H,W], got {tuple(cleaned_rgb.shape)}")
        if h0_datum.ndim not in (1, 2):
            raise ValueError(f"W1 expects H0 datum [batch] or [batch,1], got {tuple(h0_datum.shape)}")
        features = self.encoder0(cleaned_rgb)
        features = self.encoder1(F.max_pool2d(features, 2))
        features = self.encoder2(F.max_pool2d(features, 2))
        features = self.encoder3(F.max_pool2d(features, 2))
        batch = cleaned_rgb.shape[0]
        grid = self.query_coordinates.view(1, WDL_SAMPLE_COUNT, 1, 2).expand(batch, -1, -1, -1)
        sampled = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled[:, :, :, 0]
        coordinates = self.query_coordinates.T.unsqueeze(0).expand(batch, -1, -1)
        datum = h0_datum.reshape(batch, 1, 1).expand(-1, 1, WDL_SAMPLE_COUNT) / 500.0
        decoded = self.query_decoder(torch.cat([sampled, coordinates, datum], dim=1))
        return decoded[:, 0]


def masked_residual_l1(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(dtype=prediction.dtype)
    return (torch.abs(prediction - target) * weights).sum() / weights.sum().clamp(min=1.0)
