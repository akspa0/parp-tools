"""Spec 102 H1: one tiny model predicting one 33x33 coarse-relief residual.

v4 adds an optional FROZEN pretrained low-level feature branch
(`timm` `mobilenetv3_small_050.lamb_in1k`, ImageNet-1k, stage-1 stride-4
features only -- 272K frozen params, zero of them trainable). This is
explicitly not a Depth Anything model and is not used the way DA was: it
never predicts height itself, it only supplies generic deterministic
edge/texture features as an extra input to the still-tiny, still fully
custom, still separately-gated relief head. v1-v3 diagnosed the actual
problem -- train loss improved cleanly in every attempt while held-out-map
validation stayed pinned near the trivial flat-plane baseline, i.e. the
from-scratch conv net was overfitting to map-specific pixel patterns
instead of learning transferable texture cues from ~2.4k training tiles in
three epochs. A frozen, well-understood, deterministic ImageNet feature
extractor (eval-mode always, no dropout, no fine-tuning, no generation) is
architecturally nothing like DA's non-deterministic depth generation; it
is the same kind of "give a tiny head useful primitives it couldn't learn
itself" move as using Sobel filters, just with richer, pretrained ones.

v5 adds neighboring-tile context. v1-v4 all shared one unquestioned
structural choice: the model saw exactly one isolated 256x256 tile with a
hard boundary, nothing beyond it. Terrain relief (ridgelines, valleys) is
not confined to a single 533-yard tile -- a model with zero visibility past
its own tile edge is structurally starved of exactly the low-frequency
context (is this tile part of a mountain range, a valley floor, a
coastline?) that a coarse *relief* residual most needs. This is not another
small-model swap: it is a fixed, identified information gap. Nothing in the
spec's Input Invariant prohibits it -- "adjacent minimap tiles may be used
... when supplied together as RGB pixels at deployment" is explicitly
allowed, and adjacent minimap tiles are exactly as available as the center
one. H1 still predicts exactly one residual signal for exactly one tile;
it just gets to look past that tile's own edge first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

RELIEF_SCALE = 256.0

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenTextureBackbone(nn.Module):
    """Frozen ImageNet-pretrained low-level texture/edge feature extractor.

    Stage-1 (stride-4) features of `mobilenetv3_small_050` -- shallow enough
    to stay generic (edges, gradients, colour-texture blobs) rather than
    ImageNet-class-specific semantics. Always runs in eval mode regardless of
    the parent model's train()/eval() state, so its BatchNorm running stats
    never drift from the pretrained values on our (very different, very
    small-batch) minimap distribution -- true freeze, not just no-grad.
    """

    def __init__(self):
        super().__init__()
        import timm

        try:
            backbone = timm.create_model(
                "mobilenetv3_small_050.lamb_in1k",
                pretrained=True,
                features_only=True,
                out_indices=(1,),
            )
            self.pretrained = True
        except Exception:
            # Offline/no-network fallback: architecture only, random weights.
            # Still deterministic (fixed seed), just not usefully pretrained
            # -- recorded honestly in config so this is never miscredited.
            backbone = timm.create_model(
                "mobilenetv3_small_050.lamb_in1k",
                pretrained=False,
                features_only=True,
                out_indices=(1,),
            )
            self.pretrained = False

        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone = backbone
        self.out_channels = backbone.feature_info.channels()[0]
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Always stay in eval mode: frozen means frozen, including BatchNorm
        # running statistics, not just gradient flow.
        return super().train(False)

    @torch.no_grad()
    def forward(self, minimap_rgb_01: torch.Tensor) -> torch.Tensor:
        normalized = (minimap_rgb_01 - self.mean) / self.std
        return self.backbone(normalized)[0]


NEIGHBOR_SLOTS = ("x_minus", "x_plus", "y_minus", "y_plus")


class NeighborhoodContextEncoder(nn.Module):
    """Small conv+global-pool read of the four adjacent tiles' minimap RGB.

    Input is (B, 12, ctx, ctx): four neighbor images (x_minus, x_plus,
    y_minus, y_plus -- see ``NEIGHBOR_SLOTS``), each already coarsely
    block-mean-pooled by the trainer, stacked along channels. Output is a
    per-sample vector, broadcast into the main head as extra constant
    planes -- the same "compact global fact" pattern H0's tile-mean plane
    already uses, just carrying coarse neighboring-relief context instead
    of a self-height guess. Missing neighbors (map edge, or curated out of
    the corpus) are replicated from the center tile's own image upstream in
    the trainer, never a black/zero tile the model would have to
    special-case as an artificial edge that doesn't exist in-world.
    """

    def __init__(self, out_channels: int = 8, base: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, base, 3, padding=1, bias=False), nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, base * 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(base * 2, out_channels)
        self.out_channels = out_channels

    def forward(self, neighbor_context: torch.Tensor) -> torch.Tensor:
        return self.project(self.net(neighbor_context).flatten(1))


class H1CoarseReliefModel(nn.Module):
    """RGB plus frozen H0 tile mean plus neighboring-tile context -> one
    coarse-relief residual field for the center tile.

    ``use_pretrained_texture=False`` / ``use_neighbor_context=False``
    reproduce earlier ablations exactly for direct comparison.
    """

    def __init__(
        self,
        base: int = 12,
        use_pretrained_texture: bool = True,
        use_neighbor_context: bool = True,
        neighbor_context_channels: int = 8,
    ):
        super().__init__()
        self.texture = FrozenTextureBackbone() if use_pretrained_texture else None
        texture_channels = self.texture.out_channels if self.texture is not None else 0
        self.neighbor_encoder = (
            NeighborhoodContextEncoder(out_channels=neighbor_context_channels)
            if use_neighbor_context else None
        )
        neighbor_channels = neighbor_context_channels if use_neighbor_context else 0

        self.stem = nn.Sequential(
            nn.Conv2d(4 + neighbor_channels, base, 3, padding=1, bias=False), nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, base * 2), nn.SiLU(),
        )
        self.fuse = (
            nn.Conv2d(base * 2 + texture_channels, base * 2, 1, bias=False)
            if texture_channels else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1, bias=False), nn.GroupNorm(4, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base, 3, padding=1), nn.SiLU(),
        )
        self.relief = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.relief.weight)
        nn.init.zeros_(self.relief.bias)

    def forward(
        self,
        minimap_rgb: torch.Tensor,
        h0_tile_mean: torch.Tensor,
        neighbor_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        size = minimap_rgb.shape[-1]
        planes = [(h0_tile_mean / RELIEF_SCALE).view(-1, 1, 1, 1).expand(-1, 1, size, size)]
        if self.neighbor_encoder is not None:
            if neighbor_context is None:
                raise ValueError("use_neighbor_context=True requires a neighbor_context tensor")
            context_vec = self.neighbor_encoder(neighbor_context)
            planes.append(context_vec.view(*context_vec.shape, 1, 1).expand(-1, -1, size, size))
        x = self.stem(torch.cat([minimap_rgb, *planes], dim=1))
        if self.texture is not None:
            texture_feats = self.texture(minimap_rgb)
            if texture_feats.shape[-2:] != x.shape[-2:]:
                texture_feats = F.interpolate(
                    texture_feats, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = self.fuse(torch.cat([x, texture_feats], dim=1))
        field = self.relief(self.head(x))
        return F.interpolate(field, size=(33, 33), mode="bilinear", align_corners=True).squeeze(1) * RELIEF_SCALE


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
