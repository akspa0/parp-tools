"""Spec 115: terrain-feature classifier architecture (RGB -> per-pixel family logits).

One independently trained, independently promoted model (constitution IV / Spec 114 FR-011): no
weights are shared with the geometry or detailer stages, and there is no multi-task head. Its only
deployment product is a generated feature map, which downstream geometry consumes as an input
channel stack.

Deployment input is exactly the minimap RGB tile -- the same contract the geometry model already
honours -- so the classifier runs unchanged on arbitrary images that have no client-derived ground
truth at all. That is the entire point: ground-truth texture IDs supervise training and never appear
at inference.

The trunk mirrors ``HeightRelativeNet``/``GeometryDetailerNet`` so all three stages sit in one
well-understood capacity class; only the stem width and the head differ.
"""

from __future__ import annotations

import torch
from torch import nn

from harvester.v50.model_stage_contract import sha256_json
from harvester.v50.terrain_feature_labels import CLASS_COUNT, FAMILY_NAMES, TAXONOMY_REVISION

TERRAIN_FEATURE_ARCHITECTURE_ID = "terrain_feature_unet_v1"
INPUT_SIZE = 256


class TerrainFeatureModelError(ValueError):
    """Raised when the classifier contract is violated."""


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class TerrainFeatureNet(nn.Module):
    """U-Net-lite: RGB 3x256x256 -> per-pixel family logits CLASS_COUNTx256x256.

    Output stays at the input's 256x256 resolution (not the 257x257 height grid): labels are defined
    per minimap pixel, aligned pixel-for-pixel with the RGB the model actually sees. Consumers that
    need the height grid resample the generated map at their own boundary, exactly as the geometry
    stages already do.
    """

    def __init__(self, base: int = 32, num_classes: int = CLASS_COUNT) -> None:
        super().__init__()
        if num_classes < 2:
            raise TerrainFeatureModelError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes
        self.enc1 = _block(3, base)                       # 256
        self.enc2 = _block(base, base * 2, stride=2)      # 128
        self.enc3 = _block(base * 2, base * 4, stride=2)  # 64
        self.enc4 = _block(base * 4, base * 8, stride=2)  # 32
        self.mid = _block(base * 8, base * 8)
        self.up3 = _block(base * 8 + base * 4, base * 4)
        self.up2 = _block(base * 4 + base * 2, base * 2)
        self.up1 = _block(base * 2 + base, base)
        self.head = nn.Conv2d(base, num_classes, 3, padding=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise TerrainFeatureModelError(
                f"rgb must be (B, 3, H, W), got {tuple(rgb.shape)}"
            )
        e1 = self.enc1(rgb)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        m = self.mid(e4)
        u3 = self.up3(torch.cat([
            nn.functional.interpolate(m, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3
        ], dim=1))
        u2 = self.up2(torch.cat([
            nn.functional.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2
        ], dim=1))
        u1 = self.up1(torch.cat([
            nn.functional.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1
        ], dim=1))
        return self.head(u1)


def terrain_feature_identity(model: nn.Module, *, base: int = 32, num_classes: int = CLASS_COUNT) -> dict:
    """Schema-conformant architecture block for the terrain-feature stage.

    The taxonomy revision is part of the config hash: the same weights against a different family
    taxonomy are a different model, and must not present the same identity.
    """
    config = {
        "class": "TerrainFeatureNet",
        "base": base,
        "input": f"rgb 3x{INPUT_SIZE}x{INPUT_SIZE}",
        "output": f"logits {num_classes}x{INPUT_SIZE}x{INPUT_SIZE}",
        "num_classes": num_classes,
        "families": list(FAMILY_NAMES),
        "taxonomy_revision": TAXONOMY_REVISION,
    }
    return {
        "id": TERRAIN_FEATURE_ARCHITECTURE_ID,
        "config_sha256": sha256_json(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_terrain_feature_model(
    *, base: int = 32, num_classes: int = CLASS_COUNT
) -> tuple[nn.Module, dict]:
    """Build the classifier plus its full schema identity block."""
    model = TerrainFeatureNet(base=base, num_classes=num_classes)
    return model, {
        "architecture": terrain_feature_identity(model, base=base, num_classes=num_classes),
        "pretrained_source": None,
    }
