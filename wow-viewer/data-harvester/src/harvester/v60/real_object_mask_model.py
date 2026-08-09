"""Model and target contract for real v50 object-mask supervision."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

REAL_OBJECT_TARGETS = ("object_precise_mask", "object_mask")


def project_mask_257_to_256(mask_257: np.ndarray) -> np.ndarray:
    """Project a native 257x257 vertex mask onto 256x256 minimap cells by corner max."""
    mask = np.asarray(mask_257, dtype=np.float32)
    if mask.shape != (257, 257):
        raise ValueError(f"real object mask must be [257,257], got {mask.shape}")
    corners = np.stack(
        [mask[:-1, :-1], mask[1:, :-1], mask[:-1, 1:], mask[1:, 1:]], axis=0
    )
    return np.clip(corners.max(axis=0), 0.0, 1.0).astype(np.float32, copy=False)


def normalize_target_names(targets: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(target) for target in targets)
    if not names or any(name not in REAL_OBJECT_TARGETS for name in names):
        raise ValueError(f"targets must be a non-empty subset of {REAL_OBJECT_TARGETS}")
    if len(set(names)) != len(names):
        raise ValueError("target names must be unique")
    return names


class _ConvBlock(nn.Module):
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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class RealObjectMaskNet(nn.Module):
    """Compact U-Net with independently selectable precise/coarse mask heads."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        target_names: Iterable[str] = ("object_precise_mask", "object_mask"),
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels not in {3, 4}:
            raise ValueError("real object model supports RGB (3) or RGB+edge (4) inputs")
        self.in_channels = in_channels
        self.target_names = normalize_target_names(target_names)
        widths = [base_channels, base_channels * 2, base_channels * 3, base_channels * 4]
        self.encoder0 = _ConvBlock(in_channels, widths[0])
        self.encoder1 = _ConvBlock(widths[0], widths[1])
        self.encoder2 = _ConvBlock(widths[1], widths[2])
        self.bottleneck = _ConvBlock(widths[2], widths[3])
        self.decoder2 = _ConvBlock(widths[3] + widths[2], widths[2])
        self.decoder1 = _ConvBlock(widths[2] + widths[1], widths[1])
        self.decoder0 = _ConvBlock(widths[1] + widths[0], widths[0])
        self.output = nn.Conv2d(widths[0], len(self.target_names), 1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"real object model expects [batch,{self.in_channels},height,width], got {tuple(inputs.shape)}"
            )
        e0 = self.encoder0(inputs)
        e1 = self.encoder1(F.max_pool2d(e0, 2))
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        latent = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.decoder2(torch.cat((F.interpolate(latent, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), dim=1))
        d1 = self.decoder1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1))
        d0 = self.decoder0(torch.cat((F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0), dim=1))
        return self.output(d0)


def real_object_mask_loss(
    logits: Tensor,
    targets: Tensor,
    target_names: Iterable[str],
    *,
    positive_weight: float = 4.0,
    dice_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return total loss plus detached per-target loss components."""
    names = normalize_target_names(target_names)
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != len(names):
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if positive_weight <= 0.0 or dice_weight < 0.0:
        raise ValueError("positive_weight must be positive and dice_weight must be non-negative")
    losses: dict[str, Tensor] = {}
    total = logits.new_zeros(())
    pos_weight = torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    for index, name in enumerate(names):
        prediction = logits[:, index:index + 1]
        target = targets[:, index:index + 1].to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(prediction, target, pos_weight=pos_weight)
        probability = prediction.sigmoid()
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        component = bce + dice_weight * dice_loss
        total = total + component
        losses[f"{name}_loss"] = component.detach()
    return total / len(names), losses


__all__ = [
    "REAL_OBJECT_TARGETS",
    "RealObjectMaskNet",
    "normalize_target_names",
    "project_mask_257_to_256",
    "real_object_mask_loss",
]
