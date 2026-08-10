"""Terrain-only model registry for the v60 architecture bakeoff.

Every candidate consumes one textureless ``terrain_shadow_256`` channel and emits one
sigmoid-bounded ``height_257`` field.  The registry deliberately creates all encoders from
scratch.  Hugging Face and timm provide architecture implementations, but no Hub checkpoint or
other external weight source is loaded by this module.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional

from harvester.v50.direct_geometry_model import (
    MitB0RegressionNet,
    default_mit_config,
    tiny_mit_config,
)
from harvester.v50.height_relative_model import HEIGHT_GRID, HeightRelativeNet
from harvester.v50.model_stage_contract import sha256_json

INPUT_SIZE = 256
INPUT_CHANNELS = 1

UNET_LITE_ID = "unet_lite_v2"
PYRAMID_CNN_ID = "pyramid_cnn"
DPT_SMALL_ID = "dpt_small"
SEGFORMER_B0_ID = "segformer_b0"
TERRAIN_ARCHITECTURES = (
    UNET_LITE_ID,
    PYRAMID_CNN_ID,
    DPT_SMALL_ID,
    SEGFORMER_B0_ID,
)


class TerrainModelError(ValueError):
    """Raised when a terrain model cannot satisfy the v60 tensor contract."""


def _check_input(x: torch.Tensor, *, architecture: str) -> None:
    if x.ndim != 4 or tuple(x.shape[1:]) != (INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE):
        raise TerrainModelError(
            f"{architecture} consumes (B, 1, 256, 256); got shape {tuple(x.shape)}"
        )


def _finish_prediction(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    output = torch.sigmoid(
        functional.interpolate(
            logits, size=(HEIGHT_GRID, HEIGHT_GRID), mode="bilinear", align_corners=True
        )
    )
    return output.squeeze(1)


class _LocalPyramidEncoder(nn.Module):
    """Small local hierarchical CNN used by CPU contract tests."""

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                _ConvStage(INPUT_CHANNELS, 32),
                _ConvStage(32, 64),
                _ConvStage(64, 128),
                _ConvStage(128, 256),
            ]
        )
        self.channels = (32, 64, 128, 256)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class _ConvStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _PyramidDecoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], fusion_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel_count, fusion_channels, kernel_size=1)
            for channel_count in channels
        )
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(min(8, fusion_channels), fusion_channels),
                    nn.SiLU(inplace=True),
                )
                for _ in channels
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        projected = [layer(feature) for layer, feature in zip(self.lateral, features, strict=True)]
        fused = self.refine[-1](projected[-1])
        for index in range(len(projected) - 2, -1, -1):
            fused = functional.interpolate(
                fused, size=projected[index].shape[-2:], mode="bilinear", align_corners=False
            )
            fused = self.refine[index](fused + projected[index])
        return fused


class PyramidCNNNet(nn.Module):
    """Hierarchical CNN encoder with a top-down FPN-style terrain decoder.

    The full profile uses timm's ConvNeXtV2-Nano feature pyramid with ``pretrained=False``.  The
    tiny profile is a local four-stage CNN so tests remain cheap and do not depend on timm model
    registration details.
    """

    def __init__(self, *, profile: str = "full") -> None:
        super().__init__()
        if profile == "tiny":
            self.backbone = _LocalPyramidEncoder()
            channels = self.backbone.channels
            backbone_name = "local_hierarchical_cnn"
            fusion_channels = 32
        elif profile == "full":
            import timm

            self.backbone = timm.create_model(
                "convnextv2_nano",
                pretrained=False,
                features_only=True,
                in_chans=INPUT_CHANNELS,
                drop_path_rate=0.0,
            )
            channels = tuple(int(value) for value in self.backbone.feature_info.channels())
            backbone_name = "convnextv2_nano"
            fusion_channels = 96
        else:
            raise TerrainModelError(f"unknown pyramid profile {profile!r}")
        self.profile = profile
        self.backbone_name = backbone_name
        self.feature_channels = channels
        self.decoder = _PyramidDecoder(channels, fusion_channels)
        self.head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, fusion_channels // 2), fusion_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, architecture=PYRAMID_CNN_ID)
        features = list(self.backbone(x))
        fused = self.decoder(features)
        logits = self.head(
            functional.interpolate(
                fused, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
            )
        )
        return _finish_prediction(logits)


class DptSmallNet(nn.Module):
    """Locally initialized compact DPT multi-scale encoder/decoder.

    This is the generic DPT architecture exposed by Transformers, not Depth Anything.  No
    ``from_pretrained`` path exists here; all weights are initialized by the supplied seed.
    """

    def __init__(self, *, profile: str = "full") -> None:
        super().__init__()
        from transformers import DPTConfig, DPTForDepthEstimation

        if profile == "tiny":
            config = DPTConfig(
                image_size=INPUT_SIZE,
                num_channels=INPUT_CHANNELS,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=128,
                patch_size=16,
                backbone_out_indices=[0, 1],
                reassemble_factors=[4, 2],
                neck_hidden_sizes=[32, 64],
                fusion_hidden_size=32,
                use_auxiliary_head=False,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        elif profile == "full":
            config = DPTConfig(
                image_size=INPUT_SIZE,
                num_channels=INPUT_CHANNELS,
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                patch_size=16,
                backbone_out_indices=[0, 1, 2, 3],
                reassemble_factors=[4, 2, 1, 0.5],
                neck_hidden_sizes=[32, 64, 128, 256],
                fusion_hidden_size=96,
                use_auxiliary_head=False,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        else:
            raise TerrainModelError(f"unknown DPT profile {profile!r}")
        self.profile = profile
        self.config = config
        self.dpt = DPTForDepthEstimation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, architecture=DPT_SMALL_ID)
        result = self.dpt(pixel_values=x)
        return _finish_prediction(result.predicted_depth)


def _set_segformer_deterministic(config: Any) -> Any:
    for name in ("hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout_prob"):
        if hasattr(config, name):
            setattr(config, name, 0.0)
    if hasattr(config, "drop_path_rate"):
        config.drop_path_rate = 0.0
    return config


class SegformerB0Net(MitB0RegressionNet):
    """From-scratch HF SegFormer/MiT-B0 continuous regression candidate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, architecture=SEGFORMER_B0_ID)
        return super().forward(x)


def _config_payload(config: Any) -> dict[str, Any]:
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return {str(key): raw[key] for key in sorted(raw)}


def _identity(architecture: str, model: nn.Module, config: Any, *, profile: str) -> dict[str, Any]:
    payload = {
        "architecture": architecture,
        "profile": profile,
        "input": {"channels": INPUT_CHANNELS, "height": INPUT_SIZE, "width": INPUT_SIZE},
        "output": {"channels": 1, "height": HEIGHT_GRID, "width": HEIGHT_GRID},
        "weights": "random_init",
        "pretrained": False,
        "config": _config_payload(config),
    }
    return {
        "id": architecture,
        "config_sha256": sha256_json(payload),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_contract": "terrain_shadow_256:1x256x256",
        "output_contract": "height_257:1x257x257",
        "weights": "random_init",
        "pretrained": False,
        "profile": profile,
    }


def build_terrain_model(
    architecture: str,
    *,
    profile: str = "full",
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a v60 terrain candidate and its reproducibility identity."""

    if architecture == UNET_LITE_ID:
        base = 8 if profile == "tiny" else 32
        model: nn.Module = HeightRelativeNet(base=base, in_channels=INPUT_CHANNELS)
        config: Any = {
            "class": "HeightRelativeNet",
            "base": base,
            "in_channels": INPUT_CHANNELS,
        }
    elif architecture == PYRAMID_CNN_ID:
        model = PyramidCNNNet(profile=profile)
        config = {
            "class": "PyramidCNNNet",
            "backbone": model.backbone_name,
            "feature_channels": list(model.feature_channels),
            "pretrained": False,
        }
    elif architecture == DPT_SMALL_ID:
        model = DptSmallNet(profile=profile)
        config = model.config
    elif architecture == SEGFORMER_B0_ID:
        config = tiny_mit_config(in_channels=INPUT_CHANNELS) if profile == "tiny" else default_mit_config(in_channels=INPUT_CHANNELS)
        _set_segformer_deterministic(config)
        model = SegformerB0Net(config)
    else:
        raise TerrainModelError(
            f"architecture must be one of {list(TERRAIN_ARCHITECTURES)}, got {architecture!r}"
        )
    return model, _identity(architecture, model, config, profile=profile)
