"""FR-016: Stage B — lattice detailer (prior + minimap -> 257x257 residual).

A small conv-deconv (<= 2M params) on 257x257 inputs. The Stage A prior is
upsampled to 257 via the exact quincunx bilinear (amendment A6); the model
predicts the residual ``height_257 - upsampled_prior`` in RESIDUAL_SCALE space.

Input channels (13):
  0    upsampled prior (heights / HEIGHT_SCALE)
  1-3  cleaned minimap RGB (256 -> 257 edge pad)
  4-7  alpha_256 (256 -> 257 edge pad)
  8-10 normal_xyz
  11   mcnr_mask_257
  12   object_precise_mask

Loss is L1 on the residual, gated to non-liquid, non-object, non-hole pixels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from harvester.v24 import lattice
from harvester.v24.tiles import (
    HEIGHT_SCALE,
    RESIDUAL_SCALE,
    TileRecord,
    holes_to_257,
    pad_256_to_257,
)

IN_CHANNELS = 13
FULL = 257


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.GroupNorm(4, cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StageBModel(nn.Module):
    """Conv-deconv residual predictor in RESIDUAL_SCALE space."""

    def __init__(self, base: int = 40):
        super().__init__()
        self.stem = _ConvBlock(IN_CHANNELS, base)  # 257
        self.down1 = _ConvBlock(base, base * 2, stride=2)  # 129
        self.down2 = _ConvBlock(base * 2, base * 4, stride=2)  # 65
        self.mid = nn.Sequential(
            _ConvBlock(base * 4, base * 4),
            _ConvBlock(base * 4, base * 4),
        )
        self.up1 = _ConvBlock(base * 4 + base * 2, base * 2)  # 129
        self.up2 = _ConvBlock(base * 2 + base, base)  # 257
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the residual (B, 257, 257) in normalized residual space."""
        s = self.stem(x)
        d1 = self.down1(s)
        d2 = self.down2(d1)
        m = self.mid(d2)
        u1 = F.interpolate(m, size=d1.shape[-2:], mode="bilinear", align_corners=True)
        u1 = self.up1(torch.cat([u1, d1], dim=1))
        u2 = F.interpolate(u1, size=s.shape[-2:], mode="bilinear", align_corners=True)
        u2 = self.up2(torch.cat([u2, s], dim=1))
        return self.head(u2).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def upsample_prior(prior_outer: np.ndarray, prior_inner: np.ndarray) -> np.ndarray:
    """Exact quincunx bilinear upsample to (257, 257) world-space heights."""
    return lattice.upsample_prior_257(prior_outer, prior_inner)


def build_input(
    record: TileRecord,
    prior_outer: np.ndarray,
    prior_inner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (13, 257, 257) input and return (input, upsampled_prior)."""
    prior_up = upsample_prior(prior_outer, prior_inner)  # (257, 257) world space
    minimap = pad_256_to_257(record.cleaned_minimap)  # (257, 257, 3)
    alpha = pad_256_to_257(record.alpha)  # (257, 257, 4)

    channels = [
        prior_up / HEIGHT_SCALE,
        minimap[..., 0], minimap[..., 1], minimap[..., 2],
        alpha[..., 0], alpha[..., 1], alpha[..., 2], alpha[..., 3],
        record.normal[..., 0], record.normal[..., 1], record.normal[..., 2],
        record.mcnr_mask.astype(np.float32),
        record.object_mask,
    ]
    return np.stack(channels).astype(np.float32), prior_up


def build_target(record: TileRecord, prior_up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residual target (normalized) + validity mask for the gated loss."""
    residual = (record.height - prior_up) / RESIDUAL_SCALE

    liquid_257 = pad_256_to_257(record.liquid_mask) > 0.5
    holes_257 = holes_to_257(record.holes)
    objects_257 = record.object_mask > 0.5
    valid = ~(liquid_257 | holes_257 | objects_257)
    return residual.astype(np.float32), valid.astype(np.float32)


def gated_l1(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    num = (valid * (pred_residual - target_residual).abs()).sum()
    return num / valid.sum().clamp_min(1e-6)


# ---------------------------------------------------------------------------
# V24.1 — PromptDA-style Stage B (Spec 101 Slice 6)
# ---------------------------------------------------------------------------

class _PromptDARefineBlock(nn.Module):
    """Small residual refine block for the PromptDA DPT head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StageBPromptDA(nn.Module):
    """DA-V2-Small encoder + depth-prompt fusion + DPT head for Stage B (Spec 101).

    Takes minimap RGB (3ch) + WDL prior as a depth prompt (1ch, upsampled to
    256×256) and outputs a 257×257 heightmap. This is the "depth completion"
    formulation: the WDL prior is the coarse depth signal, the minimap provides
    visual context, and the model fills in the detail.

    Reuses the V23 ``DepthAnythingV2SmallEncoder`` with 4 input channels
    (3 RGB + 1 depth prompt). The backbone is frozen; only LoRA + patch proj
    + DPT head are trained.

    If the ``promptda`` package is available, this can be swapped for the real
    PromptDA-Small model. The fallback uses the same DA-V2 encoder with a
    depth-prompt input channel.

    Total params: ~25M. Trainable: ~1-2M (LoRA + patch proj + head).
    """

    def __init__(
        self,
        *,
        load_pretrained: bool = False,
        local_files_only: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        fusion_channels: int = 64,
    ):
        super().__init__()
        from harvester.v23.encoder import DepthAnythingV2SmallEncoder

        # 4 input channels: 3 RGB minimap + 1 depth prompt (WDL prior).
        self.encoder = DepthAnythingV2SmallEncoder(
            in_channels=4,
            load_pretrained=load_pretrained,
            local_files_only=local_files_only,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        from harvester.v23.model import infer_encoder_feature_schema

        schema_input_size = int(self.encoder.config.backbone_config.image_size)
        self._feature_schema = infer_encoder_feature_schema(
            self.encoder, input_size=schema_input_size,
        )
        neck_shapes = self._feature_schema["neck_features"]
        self._neck_channels = [int(shape[1]) for shape in neck_shapes]

        self.projections = nn.ModuleList(
            nn.Conv2d(c, fusion_channels, kernel_size=1, bias=False)
            for c in self._neck_channels
        )
        self.refine_blocks = nn.ModuleList(
            _PromptDARefineBlock(fusion_channels) for _ in self._neck_channels
        )
        # Head: fusion_channels → 1 → 257×257 heightmap (normalized).
        self.head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the predicted heightmap (B, 257, 257) in normalized height space.

        ``x`` is (B, 4, 256, 256) — 3 RGB minimap + 1 depth prompt (WDL prior
        upsampled to 256×256, normalized by HEIGHT_SCALE).
        """
        features = self.encoder(x)
        pyramid = list(features.neck_features)

        projected = [
            proj(level) for proj, level in zip(self.projections, pyramid, strict=True)
        ]
        fused = self.refine_blocks[0](projected[0])
        for idx in range(1, len(projected)):
            fused = F.interpolate(
                fused,
                size=projected[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.refine_blocks[idx](fused + projected[idx])

        field = self.head(fused)  # (B, 1, H, W)
        height = F.interpolate(
            field, size=(257, 257), mode="bilinear", align_corners=True
        ).squeeze(1)  # (B, 257, 257)
        return height

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable parameters (LoRA + patch proj + head)."""
        return [p for p in self.parameters() if p.requires_grad]


def build_promptda_input(
    cleaned_minimap: np.ndarray,
    prior_up_256: np.ndarray,
) -> np.ndarray:
    """Assemble the (4, 256, 256) input for the PromptDA Stage B model.

    Args:
        cleaned_minimap: (256, 256, 3) float32 [0, 1] — minimap RGB.
        prior_up_256: (256, 256) float32 — WDL prior upsampled to 256×256,
            in world units. Will be normalized by HEIGHT_SCALE.

    Returns:
        (4, 256, 256) float32 — 3 RGB + 1 normalized depth prompt.
    """
    channels = [
        cleaned_minimap[..., 0],
        cleaned_minimap[..., 1],
        cleaned_minimap[..., 2],
        prior_up_256 / HEIGHT_SCALE,
    ]
    return np.stack(channels).astype(np.float32)
