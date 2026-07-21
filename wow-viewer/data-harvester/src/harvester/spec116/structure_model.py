"""Spec 116 US3: per-slot structure classifier architecture (RGB -> per-chunk family logits).

One independently trained, independently promoted model per detail slot (constitution IV /
Spec 116 D-04): no weights are shared between slots, and there is no multi-task head. Each
``StructureSlotNet`` predicts exactly one detail slot's per-chunk surface family over the
``v115.1`` taxonomy's ``CLASS_COUNT`` classes. The base slot (0) is never predicted (FR-008):
it is the opaque terrain under everything else and is materialised by subtraction downstream.

The trunk mirrors ``TerrainFeatureNet`` so the structure stage sits in the same well-understood
capacity class; only the head differs. Where ``TerrainFeatureNet`` emits per-pixel logits at the
input resolution (256x256), ``StructureSlotNet`` emits per-chunk logits at the chunk grid
resolution (16x16) via an adaptive-pool head. Labels are defined per chunk, aligned chunk-for-
chunk with the 16x16 MCLY grid the model actually predicts.

Deployment input is exactly the minimap RGB tile -- the same contract the geometry and feature
stages already honour -- so the classifier runs unchanged on arbitrary images that have no
client-derived ground truth at all. Ground-truth texture IDs supervise training and never appear
at inference.
"""

from __future__ import annotations

import torch
from torch import nn

from harvester.v50.model_stage_contract import sha256_json
from harvester.v50.terrain_feature_labels import CLASS_COUNT, FAMILY_NAMES, TAXONOMY_REVISION

STRUCTURE_ARCHITECTURE_ID = "structure_slot_unet_v1"
INPUT_SIZE = 256
CHUNK_GRID = 16
DETAIL_SLOTS: tuple[int, ...] = (1, 2, 3)
MAX_DETAIL_SLOT = 3


class StructureModelError(ValueError):
    """Raised when the structure classifier contract is violated."""


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class StructureSlotNet(nn.Module):
    """U-Net-lite: RGB 3x256x256 -> per-chunk family logits CLASS_COUNT x 16 x 16.

    The encoder/decoder trunk is identical to ``TerrainFeatureNet`` (same base width, same
    depth, same skip connections). The head differs: an adaptive average pool collapses the
    256x256 feature map to the 16x16 chunk grid, then a 1x1 conv produces ``num_classes``
    logits per chunk. This keeps the capacity class identical while matching the label
    resolution (one family prediction per 16x16 MCLY chunk, not per pixel).

    Exactly one detail slot is predicted per instance. Constructing a multi-slot or multi-head
    variant is refused (constitution IV: no multi-task, no shared weights between slots).
    """

    def __init__(self, *, slot: int, base: int = 32, num_classes: int = CLASS_COUNT) -> None:
        super().__init__()
        if slot not in DETAIL_SLOTS:
            raise StructureModelError(
                f"slot must be one of {DETAIL_SLOTS} (base slot 0 is never predicted, FR-008); "
                f"got {slot}"
            )
        if num_classes < 2:
            raise StructureModelError(f"num_classes must be >= 2, got {num_classes}")
        self.slot = slot
        self.num_classes = num_classes
        self.base = base

        # Trunk: identical to TerrainFeatureNet.
        self.enc1 = _block(3, base)                       # 256
        self.enc2 = _block(base, base * 2, stride=2)      # 128
        self.enc3 = _block(base * 2, base * 4, stride=2)  # 64
        self.enc4 = _block(base * 4, base * 8, stride=2)  # 32
        self.mid = _block(base * 8, base * 8)
        self.up3 = _block(base * 8 + base * 4, base * 4)
        self.up2 = _block(base * 4 + base * 2, base * 2)
        self.up1 = _block(base * 2 + base, base)

        # Head: collapse to chunk grid, then 1x1 classifier.
        self.pool = nn.AdaptiveAvgPool2d(CHUNK_GRID)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise StructureModelError(
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
        pooled = self.pool(u1)
        return self.head(pooled)


def structure_model_identity(
    model: nn.Module, *, slot: int, base: int = 32, num_classes: int = CLASS_COUNT
) -> dict:
    """Architecture block for the ``v50-structure-run-v1`` record.

    The structure-run schema's ``architecture`` object requires ``class``, ``base``, ``slot``,
    ``num_classes``, ``param_count``. The taxonomy revision and family list are folded into a
    config hash that the trainer records separately (in the ``inputs`` block), so the same
    weights under a different taxonomy are a different model.
    """
    config = {
        "class": "StructureSlotNet",
        "base": base,
        "slot": slot,
        "input": f"rgb 3x{INPUT_SIZE}x{INPUT_SIZE}",
        "output": f"logits {num_classes}x{CHUNK_GRID}x{CHUNK_GRID}",
        "num_classes": num_classes,
        "families": list(FAMILY_NAMES),
        "taxonomy_revision": TAXONOMY_REVISION,
    }
    return {
        "class": "StructureSlotNet",
        "base": base,
        "slot": slot,
        "num_classes": num_classes,
        "param_count": sum(parameter.numel() for parameter in model.parameters()),
        "config_sha256": sha256_json(config),
    }


def build_structure_model(
    *, slot: int, base: int = 32, num_classes: int = CLASS_COUNT
) -> tuple[nn.Module, dict]:
    """Build one per-slot structure classifier plus its full identity block.

    Returns ``(model, identity)`` where ``identity`` carries the architecture block and a
    ``pretrained_source`` of ``None`` (structure models are trained from scratch; they never
    inherit weights from another slot or stage).
    """
    model = StructureSlotNet(slot=slot, base=base, num_classes=num_classes)
    return model, {
        "architecture": structure_model_identity(model, slot=slot, base=base, num_classes=num_classes),
        "pretrained_source": None,
    }
