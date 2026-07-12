"""Direct minimap -> terrain-shape model (Spec 102, honest reframing).

The height-signal diagnostic showed the heightmap splits into two signals a
top-down minimap treats very differently:

  * absolute per-tile elevation: std ~355-410 across tiles, ~322 units off
    distribution on held-out maps -> physically invisible top-down (arbitrary
    map Z origins) -> not learnable from a minimap.
  * terrain SHAPE (heightmap minus its own per-tile mean): relief MAE ~49-91,
    consistent across train/val/era -> visible as minimap texture/shading ->
    learnable and map-transferable.

So this model predicts only the shape. A plain, small, deterministic conv
U-Net (no attention, no pretrained depth model, no stochastic anything):
cleaned minimap RGB -> detrended 256x256 height field. The absolute datum is
supplied separately (it is one scalar and is not an image problem).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(),
    )


class DirectShapeUNet(nn.Module):
    """Cleaned minimap (B, 3, 256, 256) -> detrended height (B, 256, 256).

    Small symmetric U-Net. Output is zero-mean by construction (its own
    spatial mean is subtracted in forward), so the model spends all capacity
    on shape and can never accidentally chase the unlearnable absolute datum.
    """

    def __init__(self, in_ch: int = 3, base: int = 32, height_scale: float = 256.0):
        super().__init__()
        self.height_scale = height_scale
        self.enc1 = _block(in_ch, base)
        self.enc2 = _block(base, base * 2)
        self.enc3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, minimap_rgb: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(minimap_rgb)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        field = self.out(d1).squeeze(1) * self.height_scale
        # Zero-mean the prediction: this is a SHAPE model, the datum lives
        # elsewhere. Subtracting the spatial mean makes that explicit and
        # keeps training focused on relief.
        return field - field.mean(dim=(1, 2), keepdim=True)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
