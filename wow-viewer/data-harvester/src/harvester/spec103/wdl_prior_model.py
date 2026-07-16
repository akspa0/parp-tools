"""Spec 108: RGB-only predictor for the verified paired WDL lattice."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from harvester.spec103.v7_inputs import (
    HEIGHT_GLOBAL_MAX,
    HEIGHT_GLOBAL_MIN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    normalize_height,
    wdl_lattice_from_height257,
)

MODEL_VARIANT_WDL_PRIOR = "spec108-rgb-wdl-prior-spatial-v2"
WDL_OUTER_SIZE = 17
WDL_INNER_SIZE = 16
WDL_VALUE_COUNT = WDL_OUTER_SIZE * WDL_OUTER_SIZE + WDL_INNER_SIZE * WDL_INNER_SIZE
INPUT_CONTRACT = "minimap_rgb_only_imagenet_normalized"
TARGET_CONTRACT = "outer=height_257[::16,::16];inner=height_257[8::16,8::16]"


def normalize_minimap_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Convert one RGB minimap to the model's only inference input."""
    values = np.asarray(rgb, dtype=np.float32)
    if values.shape[-1] != 3:
        raise ValueError(f"minimap_rgb must end in 3 channels, got {values.shape}")
    if values.dtype == np.uint8 or values.max(initial=0.0) > 1.0:
        values = values / 255.0
    values = np.clip(values, 0.0, 1.0)
    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def build_wdl_target(height_257: np.ndarray) -> np.ndarray:
    """Normalized paired WDL target; the layout is part of the checkpoint contract."""
    outer, inner = wdl_lattice_from_height257(height_257)
    return np.concatenate((normalize_height(outer).reshape(-1), normalize_height(inner).reshape(-1))).astype(np.float32)


def decode_wdl_target(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world-unit outer and inner grids from a normalized model output."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size != WDL_VALUE_COUNT:
        raise ValueError(f"WDL prediction must contain {WDL_VALUE_COUNT} values, got {flat.size}")
    world = np.clip(flat, 0.0, 1.0) * (HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN) + HEIGHT_GLOBAL_MIN
    split = WDL_OUTER_SIZE * WDL_OUTER_SIZE
    return world[:split].reshape(WDL_OUTER_SIZE, WDL_OUTER_SIZE), world[split:].reshape(WDL_INNER_SIZE, WDL_INNER_SIZE)


class WdlPriorNet(nn.Module):
    """Spatial RGB front-end that emits coherent outer/inner WDL lattices.

    The old model globally pooled to 4x4 and independently emitted 545 numbers;
    it could minimize normalized L1 with spatial noise.  This keeps a 16x16
    feature plane, predicts two local lattice shapes, and composes each with
    learned per-tile absolute elevation bounds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.GELU(),
        )
        self.shape_head = nn.Sequential(nn.Conv2d(192, 96, 3, padding=1), nn.GELU(), nn.Conv2d(96, 2, 1))
        self.bounds_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, minimap_rgb: torch.Tensor) -> torch.Tensor:
        features = self.encoder(minimap_rgb)
        local = torch.sigmoid(self.shape_head(features))
        outer_local = torch.nn.functional.interpolate(local[:, :1], size=(17, 17), mode="bilinear", align_corners=True)
        inner_local = local[:, 1:2]
        bounds = torch.sigmoid(self.bounds_head(features))
        low = torch.minimum(bounds[:, :1], bounds[:, 1:2]).view(-1, 1, 1, 1)
        high = torch.maximum(bounds[:, :1], bounds[:, 1:2]).view(-1, 1, 1, 1)
        # Keep a nonzero span so the shape heads always receive a useful gradient.
        high = torch.maximum(high, low + 0.02)
        outer = low + (high - low) * outer_local
        inner = low + (high - low) * inner_local
        return torch.cat((outer.flatten(1), inner.flatten(1)), dim=1)
