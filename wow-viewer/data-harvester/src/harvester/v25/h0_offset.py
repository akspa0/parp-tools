"""Spec 102 H0: one tiny model predicting one tile-offset residual."""

from __future__ import annotations

import torch
from torch import nn

OFFSET_SCALE = 256.0


class H0OffsetModel(nn.Module):
    """RGB minimap -> scalar residual over the frozen train-global height mean."""

    def __init__(self, base: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 3, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, base * 3),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.offset = nn.Linear(base * 3, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, minimap_rgb: torch.Tensor) -> torch.Tensor:
        # Optimize in normalized units while exposing a world-unit residual.
        return self.offset(self.features(minimap_rgb).flatten(1)).squeeze(1) * OFFSET_SCALE


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
