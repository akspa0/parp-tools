"""Spec 115 follow-on: per-cell liquid classifier (minimap RGB -> 16x16 liquid class logits).

An independently trained, independently promoted model (constitution IV): no shared weights, no
multi-task head. Its deployment product is a generated per-cell liquid map, consumable downstream
exactly like the terrain-feature map.

Architecture note: the trunk is a plain 4x stride-2 encoder, 256 -> 16, so the output grid IS the
MCNK chunk grid. There is deliberately no decoder/upsample path -- the prediction target is
chunk-resolution, and reconstructing 256x256 only to pool it back down would invent detail the
labels never had.
"""

from __future__ import annotations

import torch
from torch import nn

from harvester.v50.liquid_cell_labels import CLASS_COUNT, CLASS_NAMES, TAXONOMY_REVISION
from harvester.v50.model_stage_contract import sha256_json

LIQUID_CELL_ARCHITECTURE_ID = "liquid_cell_cnn_v1"
INPUT_SIZE = 256
CELL_GRID = 16


class LiquidCellModelError(ValueError):
    """Raised when the liquid-cell classifier contract is violated."""


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class LiquidCellNet(nn.Module):
    """RGB 3x256x256 -> per-cell logits CLASS_COUNT x cell_grid x cell_grid.

    The encoder always descends to 16x16 so every prediction sees a wide receptive field (water
    identity depends on regional context: a river reads as a river partly because of what surrounds
    it). When ``cell_grid`` is finer than 16 the decoder walks back up with skip connections, so
    quad-resolution predictions keep both the global context and the local boundary detail.

    ``cell_grid=16``  = MCNK chunk grid (matches ``mcnk_flags_16``).
    ``cell_grid=128`` = the real quad grid (129 outer vertices per tile axis -> 128 quads).
    """

    def __init__(
        self, base: int = 24, num_classes: int = CLASS_COUNT, cell_grid: int = CELL_GRID
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise LiquidCellModelError(f"num_classes must be >= 2, got {num_classes}")
        if cell_grid < 1 or (cell_grid & (cell_grid - 1)) != 0 or cell_grid > INPUT_SIZE:
            raise LiquidCellModelError(
                f"cell_grid must be a power of two in 1..{INPUT_SIZE}, got {cell_grid}"
            )
        self.num_classes = num_classes
        self.cell_grid = cell_grid
        self.stem = _block(3, base)                        # 256
        self.down1 = _block(base, base * 2, stride=2)      # 128
        self.down2 = _block(base * 2, base * 4, stride=2)  # 64
        self.down3 = _block(base * 4, base * 8, stride=2)  # 32
        self.down4 = _block(base * 8, base * 8, stride=2)  # 16
        self.mix = _block(base * 8, base * 8)
        # Decoder stages only exist when the target grid is finer than the 16x16 bottleneck.
        self.up3 = _block(base * 8 + base * 8, base * 4) if cell_grid > 16 else None   # 32
        self.up2 = _block(base * 4 + base * 4, base * 2) if cell_grid > 32 else None   # 64
        self.up1 = _block(base * 2 + base * 2, base * 2) if cell_grid > 64 else None   # 128
        head_ch = base * 8
        if cell_grid > 64:
            head_ch = base * 2
        elif cell_grid > 32:
            head_ch = base * 2
        elif cell_grid > 16:
            head_ch = base * 4
        self.head = nn.Conv2d(head_ch, num_classes, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise LiquidCellModelError(f"rgb must be (B, 3, H, W), got {tuple(rgb.shape)}")
        e0 = self.stem(rgb)      # 256
        e1 = self.down1(e0)      # 128
        e2 = self.down2(e1)      # 64
        e3 = self.down3(e2)      # 32
        x = self.mix(self.down4(e3))  # 16

        def _up(feat, skip, block):
            feat = nn.functional.interpolate(
                feat, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            return block(torch.cat([feat, skip], dim=1))

        if self.up3 is not None:
            x = _up(x, e3, self.up3)   # 32
        if self.up2 is not None:
            x = _up(x, e2, self.up2)   # 64
        if self.up1 is not None:
            x = _up(x, e1, self.up1)   # 128

        logits = self.head(x)
        if logits.shape[-2:] != (self.cell_grid, self.cell_grid):
            logits = nn.functional.adaptive_avg_pool2d(logits, (self.cell_grid, self.cell_grid))
        return logits


def liquid_cell_identity(
    model: nn.Module,
    *,
    base: int = 24,
    num_classes: int = CLASS_COUNT,
    cell_grid: int = CELL_GRID,
) -> dict:
    """Schema-conformant architecture block.

    Both the taxonomy revision AND the cell grid are hashed: a chunk-resolution and a
    quad-resolution model predict different things and must never share an identity.
    """
    config = {
        "class": "LiquidCellNet",
        "base": base,
        "input": f"rgb 3x{INPUT_SIZE}x{INPUT_SIZE}",
        "output": f"logits {num_classes}x{cell_grid}x{cell_grid}",
        "num_classes": num_classes,
        "cell_grid": cell_grid,
        "classes": list(CLASS_NAMES),
        "taxonomy_revision": TAXONOMY_REVISION,
    }
    return {
        "id": LIQUID_CELL_ARCHITECTURE_ID,
        "config_sha256": sha256_json(config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def build_liquid_cell_model(
    *, base: int = 24, num_classes: int = CLASS_COUNT, cell_grid: int = CELL_GRID
) -> tuple[nn.Module, dict]:
    model = LiquidCellNet(base=base, num_classes=num_classes, cell_grid=cell_grid)
    return model, {
        "architecture": liquid_cell_identity(
            model, base=base, num_classes=num_classes, cell_grid=cell_grid
        ),
        "pretrained_source": None,
    }
