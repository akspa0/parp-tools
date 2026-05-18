"""V15 Terrain Model — minimap image → full terrain mesh patch.

Encoder: ConvNeXt V2 Nano (15.6M) from timm
Decoder: U-Net with skip fusion at 4 resolutions
Heads:  height (257×257), normals (257×257×3),
        alpha (256×256×4), holes (16×16),
        liquid mask (256×256), liquid height (256×256)

Inference requires only a minimap image — no priors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpFuse(nn.Module):
    """Upsample + fuse with skip connection."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.fuse = _ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align sizes
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.fuse(x)


class V15Model(nn.Module):
    """V15 terrain model — minimap → terrain mesh.

    Input:  (B, 3, 256, 256)  minimap RGB
    Output: height    (B, 1, 257, 257)
            normals   (B, 3, 257, 257)
            alpha     (B, 4, 256, 256)
            holes     (B, 1, 16, 16)
            liquid_mask   (B, 1, 256, 256)
            liquid_height (B, 1, 256, 256)
    """

    def __init__(self) -> None:
        super().__init__()

        # ConvNeXt V2 Nano encoder (15.6M pretrained backbone)
        import timm
        self.encoder = timm.create_model(
            "convnextv2_nano", pretrained=True, features_only=True
        )
        # Encoder outputs at strides: 4, 8, 16, 32
        # Channels: 80, 160, 320, 640
        # Sizes:  64×64, 32×32, 16×16, 8×8

        # Bottleneck
        self.bottleneck = _ConvBlock(640, 640)

        # Decoder
        self.dec3 = _UpFuse(640, 320, 320)   # 8→16
        self.dec2 = _UpFuse(320, 160, 160)   # 16→32
        self.dec1 = _UpFuse(160, 80, 80)     # 32→64
        self.dec0 = _ConvBlock(80, 64)       # 64→64 refine

        # Height head — upsample 64→257
        self.head_height = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 1),
        )

        # Normals head
        self.head_normals = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(257, 257), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )

        # Alpha head (256×256)
        self.head_alpha = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 4, 1),
            nn.Sigmoid(),
        )

        # Holes head (16×16)
        self.head_holes = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        # Liquid mask head (256×256)
        self.head_liquid = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Liquid height head (256×256), supervised where liquid is present.
        self.head_liquid_height = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 1),
        )

        self.head_mcly = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16 * 4, 1),  # 4 layers × 16-class vocab
        )

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        feats = self.encoder(x)
        e0, e1, e2, e3 = feats[0], feats[1], feats[2], feats[3]

        b = self.bottleneck(e3)

        d3 = self.dec3(b, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1)

        height = self.head_height(d0)
        normals = self.head_normals(d0)
        alpha = self.head_alpha(d0)
        holes = self.head_holes(d0)
        liquid = self.head_liquid(d0)
        liquid_height = self.head_liquid_height(d0)
        mcly_logits = self.head_mcly(d0)  # (B, 64, 16, 16) → reshape to (B, 4, 16, 16)

        return height, normals, alpha, holes, liquid, liquid_height, mcly_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
