"""M0: raw minimap RGB -> one strict terrain-visible object-mask signal."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.nn import functional as F  # noqa: N812

STRICT_OBJECT_TARGET_KEY = "object_geometry_visible_mask_257"

# Compatibility alias for the active Spec 102 modules. It deliberately points
# at the new strict target, never at the historical object_precise_mask_257
# array whose fallback provenance cannot be recovered.
PRECISE_MASK_KEY = STRICT_OBJECT_TARGET_KEY


def strict_object_target_256(strict_mask_257: np.ndarray) -> np.ndarray:
    """Project the strict 257x257 terrain-visible footprint onto minimap cells."""
    precise = np.asarray(strict_mask_257, dtype=np.float32)
    if precise.shape != (257, 257):
        raise ValueError(f"{STRICT_OBJECT_TARGET_KEY} must be [257,257], got {precise.shape}")
    corners = np.stack(
        [precise[:-1, :-1], precise[1:, :-1], precise[:-1, 1:], precise[1:, 1:]],
        axis=0,
    )
    return corners.max(axis=0)


def precise_object_target_256(precise_mask_257: np.ndarray) -> np.ndarray:
    """Backward-compatible spelling for the strict Spec 102 target projection."""
    return strict_object_target_256(precise_mask_257)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class M0ObjectMask(nn.Module):
    """A compact U-Net with exactly one output: object-mask logits."""

    def __init__(self, base_channels: int = 40) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoder0 = ConvBlock(3, widths[0])
        self.encoder1 = ConvBlock(widths[0], widths[1])
        self.encoder2 = ConvBlock(widths[1], widths[2])
        self.bottleneck = ConvBlock(widths[2], widths[3])
        self.decoder2 = ConvBlock(widths[3] + widths[2], widths[2])
        self.decoder1 = ConvBlock(widths[2] + widths[1], widths[1])
        self.decoder0 = ConvBlock(widths[1] + widths[0], widths[0])
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, minimap_rgb: torch.Tensor) -> torch.Tensor:
        if minimap_rgb.ndim != 4 or minimap_rgb.shape[1] != 3:
            raise ValueError(f"M0 expects [batch,3,height,width], got {tuple(minimap_rgb.shape)}")
        e0 = self.encoder0(minimap_rgb)
        e1 = self.encoder1(F.max_pool2d(e0, 2))
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        latent = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.decoder2(torch.cat([F.interpolate(latent, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        d0 = self.decoder0(torch.cat([F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0], dim=1))
        return self.output(d0)


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_weight: float = 4.0,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One segmentation loss family: weighted BCE plus Dice."""
    target = target.to(dtype=logits.dtype)
    pos_weight = torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = bce + (dice_weight * dice)
    return total, {"bce": bce.detach(), "dice_loss": dice.detach()}


def clean_minimap_with_mask(minimap_rgb: np.ndarray, predicted_mask_256: np.ndarray) -> np.ndarray:
    """Deterministically replace masked pixels with their nearest unmasked terrain pixel."""
    rgb = np.asarray(minimap_rgb)
    mask = np.asarray(predicted_mask_256, dtype=bool)
    if rgb.shape != (256, 256, 3):
        raise ValueError(f"minimap_rgb must be [256,256,3], got {rgb.shape}")
    if mask.shape != (256, 256):
        raise ValueError(f"predicted_mask_256 must be [256,256], got {mask.shape}")
    output = rgb.copy()
    if not mask.any():
        return output
    if mask.all():
        output[...] = np.asarray(rgb, dtype=np.float32).reshape(-1, 3).mean(axis=0).astype(rgb.dtype)
        return output
    _, indices = distance_transform_edt(mask, return_indices=True)
    output[mask] = rgb[indices[0][mask], indices[1][mask]]
    return output
