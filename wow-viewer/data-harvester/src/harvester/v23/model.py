"""Combined V23 height predictor for Spec 089 Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import DepthAnythingConfig

from harvester.v23.encoder import DepthAnythingV2SmallEncoder, V23FeaturePyramid
from harvester.v23.head import V23HeightHead


@dataclass(frozen=True)
class V23ModelOutput:
    """Typed output surface for the combined V23 model."""

    disparity: torch.Tensor
    affine_anchor: torch.Tensor
    metric_height: torch.Tensor
    features: V23FeaturePyramid


def infer_encoder_feature_schema(
    encoder: DepthAnythingV2SmallEncoder,
    *,
    input_size: int | None = None,
) -> dict[str, Any]:
    """Infer the encoder feature schema using a synthetic forward pass."""
    size = int(input_size or encoder.config.backbone_config.image_size)
    cache = getattr(encoder, "_feature_schema_cache", {})
    if size in cache:
        return cache[size]

    try:
        parameter = next(encoder.parameters())
        device = parameter.device
    except StopIteration:
        device = torch.device("cpu")

    was_training = encoder.training
    encoder.eval()
    with torch.inference_mode():
        sample = torch.zeros((1, encoder.in_channels, size, size), dtype=torch.float32, device=device)
        features = encoder(sample)
    if was_training:
        encoder.train()

    schema = {
        "raw_feature_maps": [tuple(level.shape) for level in features.raw_feature_maps],
        "neck_features": [tuple(level.shape) for level in features.neck_features],
        "patch_height": int(features.patch_height),
        "patch_width": int(features.patch_width),
        "input_height": int(features.input_height),
        "input_width": int(features.input_width),
    }
    cache[size] = schema
    encoder._feature_schema_cache = cache
    return schema


class V23HeightPredictor(nn.Module):
    """Combined encoder + head height predictor with affine anchoring."""

    def __init__(
        self,
        in_channels: int = 15,
        *,
        encoder: DepthAnythingV2SmallEncoder | None = None,
        config: DepthAnythingConfig | None = None,
        load_pretrained: bool = False,
        feature_dict_schema: dict[str, Any] | None = None,
        schema_input_size: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or DepthAnythingV2SmallEncoder(
            in_channels=in_channels,
            config=config,
            load_pretrained=load_pretrained,
        )
        self.feature_dict_schema = feature_dict_schema or infer_encoder_feature_schema(
            self.encoder,
            input_size=schema_input_size,
        )
        self.head = V23HeightHead(self.feature_dict_schema)

    def forward(self, x: torch.Tensor) -> V23ModelOutput:
        features = self.encoder(x)
        disparity, affine_anchor = self.head(features)
        scale = affine_anchor[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        shift = affine_anchor[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        metric_height = (disparity * scale) + shift
        return V23ModelOutput(
            disparity=disparity.float(),
            affine_anchor=affine_anchor.float(),
            metric_height=metric_height.float(),
            features=features,
        )
