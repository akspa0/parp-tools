"""V19 height regressor model.

Minimal-input terrain height prediction from minimap RGB (+ optional normals).
No WDL prior. No pretrained backbone. Trained from scratch on V18 Zarr data.

Architecture: ResConvBlock + GroupNorm + BilinearUp U-Net with skip connections.
~20M parameters. Outputs heightmap (257x257) + bounds (4 values).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ----------------------------------------------------------------------
# Residual ConvBlock (from V7)
# ----------------------------------------------------------------------
class ResConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "group", groupnorm_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.bn1 = self._build_norm_layer(out_channels, norm_type, groupnorm_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.bn2 = self._build_norm_layer(out_channels, norm_type, groupnorm_groups)
        self.use_residual = in_channels == out_channels

    def _build_norm_layer(self, channels: int, norm_type: str, groups: int):
        norm_key = str(norm_type).strip().lower()
        if norm_key == "batch":
            return nn.BatchNorm2d(channels)
        # GroupNorm fallback
        groups = max(1, int(groups))
        if channels % groups != 0:
            groups = 1
        return nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + identity
        return F.relu(out, inplace=True)


# ----------------------------------------------------------------------
# Bilinear UpSampling (from V7)
# ----------------------------------------------------------------------
class BilinearUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


# ----------------------------------------------------------------------
# UNet Backbone (V19)
# ----------------------------------------------------------------------
class _UNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, norm_type: str = "group", groupnorm_groups: int = 8):
        super().__init__()
        # Encoder (256 -> 128 -> 64)
        self.enc0 = ResConvBlock(in_channels, 24, norm_type, groupnorm_groups)
        self.enc1 = ResConvBlock(24, 48, norm_type, groupnorm_groups)
        self.enc2 = ResConvBlock(48, 96, norm_type, groupnorm_groups)

        # Bottleneck (32x32)
        self.bottleneck = ResConvBlock(96, 96, norm_type, groupnorm_groups)

        # Decoder
        self.up2 = BilinearUp(96, 48)
        self.dec2 = ResConvBlock(48 + 96, 48, norm_type, groupnorm_groups)
        self.up1 = BilinearUp(48, 24)
        self.dec1 = ResConvBlock(24 + 48, 24, norm_type, groupnorm_groups)
        self.dec0 = ResConvBlock(24, 24, norm_type, groupnorm_groups)

        # Global pooling for height bounds
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.height_bounds_fc = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder path (256 -> 128 -> 64)
        enc0 = self.enc0(x)
        enc1 = self.enc1(F.max_pool2d(enc0, 2))
        enc2 = self.enc2(F.max_pool2d(enc1, 2))

        # Bottleneck (32x32)
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))

        # Decoder path
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        dec0 = self.dec0(dec1)

        # Global context for bounds
        pooled = self.global_pool(bottleneck).view(bottleneck.size(0), -1)
        bounds = self.height_bounds_fc(pooled)

        return dec0, bounds


# ----------------------------------------------------------------------
# V19 Model
# ----------------------------------------------------------------------
class V19HeightModel(nn.Module):
    """
    Minimal terrain reconstruction model.
    - Input: minimap RGB (3 channels) + optional normal map (3 channels)
    - Output: global heightmap, local heightmap, height bounds
    - No WDL prior required; works with just minimap tiles.
    """

    def __init__(
        self,
        in_channels: int = 3,  # minimap RGB; add extra channels for normals if desired
        out_channels: int = 2,  # global + local height
        norm_type: str = "group",
        groupnorm_groups: int = 16,
    ):
        super().__init__()
        self.backbone = _UNetBackbone(in_channels, norm_type, groupnorm_groups)
        self.out_conv = nn.Conv2d(24, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        refined, bounds = self.backbone(x)
        raw_outputs = self.out_conv(refined)
        # Upsample from 128x128 to 257x257
        raw_outputs = F.interpolate(raw_outputs, size=(257, 257), mode="bilinear", align_corners=False)
        global_output = torch.clamp(raw_outputs[:, 0:1], 0.0, 1.0)
        local_output = torch.clamp(raw_outputs[:, 1:2], 0.0, 1.0)
        return global_output, local_output, bounds