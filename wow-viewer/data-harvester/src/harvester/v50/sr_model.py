"""Spec 113 US3 SR model (T014) — RealPLKSR, ComfyUI-native by construction.

ComfyUI's ``Load Upscale Model`` node loads checkpoints via **spandrel**, and spandrel ships the
reference ``RealPLKSR`` implementation itself. So instead of vendoring a copy that could drift from
spandrel's key-detection, we train *spandrel's own class* and save its bare ``state_dict()`` —
the resulting ``.pth`` is loadable by ComfyUI with zero custom nodes, proven by a
``spandrel.ModelLoader`` round-trip test rather than assumed.

Single-purpose SR generator: one output, no multi-task heads, no shared weights (FR-006). The GAN
stage adds a compact PatchGAN discriminator + LPIPS perceptual term, entered only after a user
reviews the PSNR stage (contract §6); the discriminator is a training-time artifact and is never
part of the deliverable checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from spandrel.architectures.PLKSR import RealPLKSR

SR_SCALE = 4


def build_generator(*, dim: int = 64, n_blocks: int = 28, kernel_size: int = 17) -> nn.Module:
    """The deliverable: spandrel's RealPLKSR at its standard x4 configuration."""
    return RealPLKSR(
        dim=dim,
        n_blocks=n_blocks,
        upscaling_factor=SR_SCALE,
        kernel_size=kernel_size,
        split_ratio=0.25,
        use_ea=True,
        norm_groups=4,
        dysample=False,
        layer_norm=False,
    )


def save_comfyui_checkpoint(model: nn.Module, path: Path) -> None:
    """Save the bare state dict — exactly what ComfyUI/spandrel's loader expects. Run metadata
    lives in the sibling run summary, never inside this file (a wrapped dict breaks detection)."""
    torch.save(model.state_dict(), path)


def verify_comfyui_loadable(path: Path):
    """Round-trip the saved checkpoint through spandrel's real loader (the exact code ComfyUI
    runs). Returns the descriptor; raises if the checkpoint is not a recognized upscale model."""
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_file(str(path))
    if descriptor.scale != SR_SCALE:
        raise ValueError(f"checkpoint loads but with scale {descriptor.scale}, expected {SR_SCALE}")
    return descriptor


class PatchDiscriminator(nn.Module):
    """Compact PatchGAN for the optional GAN stage (training-time only, never shipped)."""

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        def block(cin: int, cout: int, stride: int) -> list[nn.Module]:
            return [
                nn.utils.spectral_norm(nn.Conv2d(cin, cout, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.net = nn.Sequential(
            *block(3, base, 2),
            *block(base, base * 2, 2),
            *block(base * 2, base * 4, 2),
            nn.utils.spectral_norm(nn.Conv2d(base * 4, 1, 4, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
