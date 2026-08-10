"""Small v60 object-sieve model and loss variants.

The model is deliberately a bounded control experiment. It consumes the canonical one-channel
textureless/objectified terrain signal and predicts clean terrain plus an object-contamination mask.
The clean head is an identity-preserving residual: it starts as ``clean = input`` and learns only a
correction. It never accepts a ground-truth mask as an input channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

ObjectSieveVariant = Literal["clean_only", "auxiliary_mask_loss", "predicted_mask_guided"]


@dataclass(frozen=True)
class ObjectSievePredictions:
    clean_terrain: Tensor
    contamination_logits: Tensor


class ObjectSieveNet(nn.Module):
    """Compact shared-trunk/two-head network for the sieve ablation."""

    def __init__(self, variant: ObjectSieveVariant = "auxiliary_mask_loss") -> None:
        super().__init__()
        if variant not in {"clean_only", "auxiliary_mask_loss", "predicted_mask_guided"}:
            raise ValueError(f"unknown object-sieve variant: {variant}")
        self.variant = variant
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(48, 1, kernel_size=1)
        clean_channels = 49 if variant == "predicted_mask_guided" else 48
        self.clean_head = nn.Sequential(
            nn.Conv2d(clean_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # The contaminated input is already a strong clean-terrain baseline because object pixels
        # occupy a small fraction of a tile. Start exactly at that baseline and let training learn
        # a bounded correction instead of rewriting uncontaminated terrain everywhere.
        nn.init.zeros_(self.clean_head[-1].weight)
        nn.init.zeros_(self.clean_head[-1].bias)

    def forward(self, objectified_terrain: Tensor) -> ObjectSievePredictions:
        if objectified_terrain.ndim != 4 or objectified_terrain.shape[1] != 1:
            raise ValueError(
                f"expected objectified terrain (N,1,H,W), got {tuple(objectified_terrain.shape)}"
            )
        features = self.encoder(objectified_terrain)
        contamination_logits = self.mask_head(features)
        clean_features = features
        if self.variant == "predicted_mask_guided":
            clean_features = torch.cat((features, torch.sigmoid(contamination_logits)), dim=1)
        clean_delta = self.clean_head(clean_features)
        clean_terrain = torch.clamp(objectified_terrain + clean_delta, 0.0, 1.0)
        return ObjectSievePredictions(clean_terrain=clean_terrain, contamination_logits=contamination_logits)


def _dice_loss(logits: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    probabilities = logits.sigmoid()
    probabilities = probabilities.reshape(probabilities.shape[0], -1)
    target = target.reshape(target.shape[0], -1).float()
    intersection = (probabilities * target).sum(dim=1)
    denominator = probabilities.sum(dim=1) + target.sum(dim=1)
    return (1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))).mean()


def object_sieve_loss(
    predictions: ObjectSievePredictions,
    clean_target: Tensor,
    contamination_target: Tensor,
    variant: ObjectSieveVariant,
    *,
    mask_weight: float = 0.5,
) -> dict[str, Tensor]:
    """Return separate clean and mask losses plus the weighted total."""
    if clean_target.shape != predictions.clean_terrain.shape:
        raise ValueError("clean target and clean prediction shapes differ")
    if contamination_target.shape != predictions.contamination_logits.shape:
        raise ValueError("contamination target and mask prediction shapes differ")
    if mask_weight < 0.0:
        raise ValueError("mask_weight must be non-negative")
    clean_loss = functional.smooth_l1_loss(predictions.clean_terrain, clean_target.float())
    if variant == "clean_only":
        mask_loss = clean_loss.new_zeros(())
    else:
        mask_loss = functional.binary_cross_entropy_with_logits(
            predictions.contamination_logits, contamination_target.float()
        ) + _dice_loss(predictions.contamination_logits, contamination_target)
    return {
        "clean_loss": clean_loss,
        "mask_loss": mask_loss,
        "total_loss": clean_loss + (mask_weight * mask_loss),
    }


__all__ = [
    "ObjectSieveNet",
    "ObjectSievePredictions",
    "ObjectSieveVariant",
    "object_sieve_loss",
]
