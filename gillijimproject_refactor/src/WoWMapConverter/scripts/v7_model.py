from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_SIZE = 512
MODEL_INPUT_CHANNELS = 13
MODEL_OUTPUT_CHANNELS = 2
MODEL_OUTPUT_CHANNELS_V77 = 3
DEFAULT_GLOBAL_RESIDUAL_SCALE = 0.20
DEFAULT_DETAIL_RESIDUAL_SCALE = 1.0
DEFAULT_NORM_TYPE = "group"
DEFAULT_GROUPNORM_GROUPS = 16
MODEL_VARIANT_WDL_TRESTLE_REFLECT = "wdl-trestle-reflect-v1"
MODEL_VARIANT_LEGACY = "legacy-absolute-v1"
MODEL_VARIANT_WDL_TRESTLE_REFLECT_V77 = "wdl-trestle-reflect-v77"


def _resolve_group_count(channels: int, preferred_groups: int) -> int:
    preferred_groups = max(1, int(preferred_groups))
    for group_count in range(min(preferred_groups, channels), 0, -1):
        if channels % group_count == 0:
            return group_count
    return 1


def _build_norm_layer(channels: int, norm_type: str, groupnorm_groups: int) -> nn.Module:
    norm_key = str(norm_type).strip().lower()
    if norm_key == "batch":
        return nn.BatchNorm2d(channels)

    groups = _resolve_group_count(channels, groupnorm_groups)
    return nn.GroupNorm(groups, channels)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "batch", groupnorm_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.bn1 = _build_norm_layer(out_channels, norm_type, groupnorm_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.bn2 = _build_norm_layer(out_channels, norm_type, groupnorm_groups)
        self.use_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + identity
        return F.relu(out, inplace=True)


class BilinearUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class MultiChannelUNetV7(nn.Module):
    def __init__(
        self,
        in_channels: int = MODEL_INPUT_CHANNELS,
        out_channels: int = MODEL_OUTPUT_CHANNELS,
        use_wdl_global_trestle: bool = False,
        global_residual_scale: float = DEFAULT_GLOBAL_RESIDUAL_SCALE,
        use_detail_head: bool = False,
        detail_residual_scale: float = DEFAULT_DETAIL_RESIDUAL_SCALE,
        norm_type: str = DEFAULT_NORM_TYPE,
        groupnorm_groups: int = DEFAULT_GROUPNORM_GROUPS,
    ):
        super().__init__()
        self.use_wdl_global_trestle = use_wdl_global_trestle
        self.global_residual_scale = float(global_residual_scale)
        self.use_detail_head = bool(use_detail_head)
        self.detail_residual_scale = float(detail_residual_scale)
        self.norm_type = str(norm_type).strip().lower()
        self.groupnorm_groups = max(1, int(groupnorm_groups))

        self.enc1 = ResConvBlock(in_channels, 64, self.norm_type, self.groupnorm_groups)
        self.enc2 = ResConvBlock(64, 128, self.norm_type, self.groupnorm_groups)
        self.enc3 = ResConvBlock(128, 256, self.norm_type, self.groupnorm_groups)
        self.enc4 = ResConvBlock(256, 512, self.norm_type, self.groupnorm_groups)
        self.enc5 = ResConvBlock(512, 1024, self.norm_type, self.groupnorm_groups)
        self.bottleneck = ResConvBlock(1024, 2048, self.norm_type, self.groupnorm_groups)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.height_bounds_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

        self.up5 = BilinearUp(2048, 1024)
        self.dec5 = ResConvBlock(2048, 1024, self.norm_type, self.groupnorm_groups)
        self.up4 = BilinearUp(1024, 512)
        self.dec4 = ResConvBlock(1024, 512, self.norm_type, self.groupnorm_groups)
        self.up3 = BilinearUp(512, 256)
        self.dec3 = ResConvBlock(512, 256, self.norm_type, self.groupnorm_groups)
        self.up2 = BilinearUp(256, 128)
        self.dec2 = ResConvBlock(256, 128, self.norm_type, self.groupnorm_groups)
        self.up1 = BilinearUp(128, 64)
        self.dec1 = ResConvBlock(128, 64, self.norm_type, self.groupnorm_groups)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        bottleneck = self.bottleneck(self.pool(enc5))

        pooled = self.global_pool(bottleneck).view(bottleneck.size(0), -1)
        bounds = self.height_bounds_fc(pooled)

        dec5 = self.up5(bottleneck)
        dec5 = torch.cat([dec5, enc5], dim=1)
        dec5 = self.dec5(dec5)

        dec4 = self.up4(dec5)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        raw_outputs = self.out_conv(dec1)
        global_output = raw_outputs[:, 0:1]
        if self.use_wdl_global_trestle and inputs.shape[1] > 6:
            wdl_base = inputs[:, 6:7]
            global_delta = torch.tanh(global_output) * self.global_residual_scale
            global_output = torch.clamp(wdl_base + global_delta, 0.0, 1.0)
        else:
            global_output = torch.clamp(global_output, 0.0, 1.0)

        local_output = torch.clamp(raw_outputs[:, 1:2], 0.0, 1.0)
        outputs = torch.cat([global_output, local_output], dim=1)
        if self.use_detail_head and raw_outputs.shape[1] > 2:
            detail_output = torch.tanh(raw_outputs[:, 2:3]) * self.detail_residual_scale
            outputs = torch.cat([outputs, detail_output], dim=1)
        if outputs.shape[-2:] != (OUTPUT_SIZE, OUTPUT_SIZE):
            outputs = F.interpolate(outputs, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False)

        return outputs, bounds


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = MODEL_OUTPUT_CHANNELS):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def resolve_model_architecture_from_metadata(metadata: Optional[Dict[str, object]]) -> Tuple[bool, float]:
    if not metadata:
        return DEFAULT_GLOBAL_RESIDUAL_SCALE != 0.0, DEFAULT_GLOBAL_RESIDUAL_SCALE

    variant = str(metadata.get("model_variant", "")).strip().lower()
    use_wdl_global_trestle = bool(metadata.get("use_wdl_global_trestle", False))
    if variant == MODEL_VARIANT_WDL_TRESTLE_REFLECT:
        use_wdl_global_trestle = True
    elif variant == MODEL_VARIANT_WDL_TRESTLE_REFLECT_V77:
        use_wdl_global_trestle = True
    elif variant == MODEL_VARIANT_LEGACY:
        use_wdl_global_trestle = False

    global_residual_scale = float(metadata.get("global_residual_scale", DEFAULT_GLOBAL_RESIDUAL_SCALE))
    return use_wdl_global_trestle, global_residual_scale


def resolve_model_detail_head_from_metadata(metadata: Optional[Dict[str, object]]) -> Tuple[bool, float]:
    if not metadata:
        return False, DEFAULT_DETAIL_RESIDUAL_SCALE

    variant = str(metadata.get("model_variant", "")).strip().lower()
    use_detail_head = bool(metadata.get("use_detail_head", False))
    if variant == MODEL_VARIANT_WDL_TRESTLE_REFLECT_V77:
        use_detail_head = True
    detail_residual_scale = float(metadata.get("detail_residual_scale", DEFAULT_DETAIL_RESIDUAL_SCALE))
    return use_detail_head, detail_residual_scale