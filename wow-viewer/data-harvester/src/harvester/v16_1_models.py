"""Independent V16.1 model definitions.

These classes share code, but each trained checkpoint owns its own weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_TEXTURE_PALETTE_16 = torch.tensor(
    [
        [0.62, 0.58, 0.47],
        [0.52, 0.60, 0.42],
        [0.43, 0.53, 0.34],
        [0.65, 0.61, 0.54],
        [0.51, 0.47, 0.40],
        [0.36, 0.46, 0.54],
        [0.42, 0.55, 0.63],
        [0.50, 0.38, 0.28],
        [0.71, 0.66, 0.53],
        [0.45, 0.35, 0.27],
        [0.59, 0.44, 0.34],
        [0.34, 0.41, 0.26],
        [0.58, 0.56, 0.62],
        [0.73, 0.71, 0.65],
        [0.27, 0.33, 0.41],
        [0.83, 0.79, 0.68],
    ],
    dtype=torch.float32,
)


class _ConvBlock(nn.Module):
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
        self.up_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.conv = _ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.up_conv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class _UNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.enc0 = _ConvBlock(in_channels, 64)
        self.enc1 = _DownBlock(64, 96)
        self.enc2 = _DownBlock(96, 160)
        self.enc3 = _DownBlock(160, 224)
        self.bottleneck = _ConvBlock(224, 224)
        self.dec3 = _UpBlock(224, 224, 160)
        self.dec2 = _UpBlock(160, 160, 96)
        self.dec1 = _UpBlock(96, 96, 64)
        self.dec0 = _UpBlock(64, 64, 32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        pooled16 = F.adaptive_avg_pool2d(d0, (16, 16))
        return d0, pooled16

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class V161HeightModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UNetBackbone()
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d0, _ = self.backbone(x)
        return self.head(d0)


class V161NormalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UNetBackbone(3)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d0, _ = self.backbone(x)
        return self.head(d0)


class V161NormalHeightModel(nn.Module):
    """Normal model with height as an input channel: cat(minimap_rgb, height_norm) → normals."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UNetBackbone(4)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d0, _ = self.backbone(x)
        return self.head(d0)


class V161NormalHeightCombinedModel(nn.Module):
    """Combined model: cat(minimap_rgb, height_norm) -> (normals_3ch, height_1ch).

    Shared backbone, two heads. Single checkpoint, both signals.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UNetBackbone(4)
        self.normal_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d0, _ = self.backbone(x)
        return self.normal_head(d0), self.height_head(d0)


class V161NormalRefiner(nn.Module):
    """Small conv refiner: pred_normals(3ch) + height(1ch) → refined_normals(3ch).

    No masking. No object/liquid gating. Pure geometric refinement using
    height as a structural prior. Skip connection from input normals to
    output so the refiner starts as identity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(4, 32, 3, padding=1)
        self.bn_proj = nn.BatchNorm2d(32)

        self.res1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_normals = x[:, :3]
        h = self.proj(x)
        h = self.bn_proj(h)
        h = F.relu(h, inplace=True)
        h = h + self.res1(h)
        h = h + self.res2(h)
        delta = self.res3(h)
        return torch.tanh(input_normals + delta)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class V161HolesModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UNetBackbone()
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d0, _ = self.backbone(x)
        return self.head(d0)


class V161LiquidModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UNetBackbone()
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.type_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 5, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d0, pooled16 = self.backbone(x)
        return self.mask_head(d0), self.type_head(pooled16)


def compute_compositor_weights_torch(alpha_pack: torch.Tensor) -> torch.Tensor:
    """Compute compositor weights from raw MCAL alpha pack.

    alpha_pack: (B, 4, H, W)
    returns: (B, 4, H, W)
    """

    a1 = alpha_pack[:, 0:1]
    a2 = alpha_pack[:, 1:2]
    a3 = alpha_pack[:, 2:3]
    a4 = alpha_pack[:, 3:4]
    w0 = 1.0 - a1
    w1 = a1 * (1.0 - a2)
    w2 = a1 * a2 * (1.0 - a3)
    w3 = a1 * a2 * a3 * (1.0 - a4)
    weights = torch.cat([w0, w1, w2, w3], dim=1)
    total = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights / total


def recompose_from_mcly_alpha(
    alpha_pack: torch.Tensor,
    mcly_logits: torch.Tensor,
    mcly_mask: torch.Tensor,
) -> torch.Tensor:
    """Build a terrain-only RGB proxy from predicted IDs/masks + alpha pack.

    alpha_pack: (B, 4, 256, 256)
    mcly_logits: (B, 4, 16, 16, 16)
    mcly_mask: (B, 4, 16, 16)
    returns: (B, 3, 256, 256)
    """

    palette = _TEXTURE_PALETTE_16.to(alpha_pack.device)
    probs = mcly_logits.softmax(dim=2)
    layer_colors_16 = torch.einsum("blchw,cd->bldhw", probs, palette)
    layer_colors_256 = layer_colors_16.repeat_interleave(16, dim=3).repeat_interleave(16, dim=4)
    layer_mask_256 = mcly_mask.unsqueeze(2).repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)
    layer_mask_256 = layer_mask_256.unsqueeze(2)
    layer_colors_256 = layer_colors_256 * layer_mask_256

    blend = compute_compositor_weights_torch(alpha_pack).unsqueeze(2)
    blend = blend * layer_mask_256
    blend = blend / blend.sum(dim=1, keepdim=True).clamp_min(1e-6)
    rgb = (blend * layer_colors_256).sum(dim=1)
    return rgb.clamp(0.0, 1.0)


class V161TexcompModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _UNetBackbone()
        self.alpha_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
            nn.Sigmoid(),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
            nn.Sigmoid(),
        )
        self.ids_head = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4 * 16, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d0, pooled16 = self.backbone(x)
        alpha = self.alpha_head(d0)
        mask = self.mask_head(pooled16)
        ids_logits = self.ids_head(pooled16).view(-1, 4, 16, 16, 16)
        return alpha, mask, ids_logits
