"""V23 DPT-style height head for Spec 089 Phase 3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from harvester.v23.encoder import V23FeaturePyramid


class _RefineBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class V23HeightHead(nn.Module):
    """Top-down height decoder that predicts disparity plus an affine anchor."""

    def __init__(
        self,
        feature_dict_schema: Mapping[str, object],
        *,
        fusion_channels: int = 128,
        output_size: tuple[int, int] = (257, 257),
    ) -> None:
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)
        self.feature_channels = tuple(self._parse_feature_channels(feature_dict_schema))
        if len(self.feature_channels) != 4:
            raise ValueError("V23HeightHead expects exactly four neck feature levels")

        self.projections = nn.ModuleList(
            nn.Conv2d(channels, fusion_channels, kernel_size=1, bias=False)
            for channels in self.feature_channels
        )
        self.refine_blocks = nn.ModuleList(_RefineBlock(fusion_channels) for _ in self.feature_channels)
        self.disparity_head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, fusion_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 4, 1, kernel_size=1),
        )
        anchor_input_channels = sum(self.feature_channels)
        self.anchor_head = nn.Sequential(
            nn.Linear(anchor_input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    @staticmethod
    def _parse_feature_channels(feature_dict_schema: Mapping[str, object]) -> Sequence[int]:
        neck_shapes = feature_dict_schema.get("neck_features")
        if not isinstance(neck_shapes, Sequence):
            raise TypeError("feature_dict_schema['neck_features'] must be a sequence")
        channels: list[int] = []
        for shape in neck_shapes:
            if not isinstance(shape, Sequence) or len(shape) != 4:
                raise TypeError("Each neck feature shape must be a 4D shape tuple")
            channels.append(int(shape[1]))
        return channels

    def forward(self, features: V23FeaturePyramid) -> tuple[torch.Tensor, torch.Tensor]:
        pyramid = list(features.neck_features)
        if len(pyramid) != len(self.projections):
            raise ValueError("Unexpected number of neck features for V23HeightHead")

        projected = [proj(level) for proj, level in zip(self.projections, pyramid, strict=True)]
        fused = self.refine_blocks[0](projected[0])
        for idx in range(1, len(projected)):
            fused = F.interpolate(
                fused,
                size=projected[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.refine_blocks[idx](fused + projected[idx])

        disparity = torch.sigmoid(self.disparity_head(fused))
        disparity = F.interpolate(
            disparity,
            size=self.output_size,
            mode="bicubic",
            align_corners=False,
        )

        pooled = [F.adaptive_avg_pool2d(level, output_size=1).flatten(1) for level in pyramid]
        anchor_raw = self.anchor_head(torch.cat(pooled, dim=1))
        scale = F.softplus(anchor_raw[:, 0:1]) + 1e-3
        shift = anchor_raw[:, 1:2]
        affine_anchor = torch.cat([scale, shift], dim=1)
        return disparity.float(), affine_anchor.float()
