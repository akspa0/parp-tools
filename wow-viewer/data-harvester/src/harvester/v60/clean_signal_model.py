"""Image-only v7-style coarse/detail model contract for Spec 139.

The model consumes exactly four channels from ``clean_signal_inputs`` and returns two independently
named training heads plus their recomposed published height.  The architecture candidates are
small local implementations so CPU contract tests do not download or initialize external model
weights.  The models are random-initialized; this module does not provide a training entrypoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from harvester.v50.model_stage_contract import sha256_json

CLEAN_INPUT_SCHEMA = "v7-clean-signal-input-v1"
CLEAN_OUTPUT_SCHEMA = "v7-clean-signal-model-output-v1"
MODEL_IDENTITY_SCHEMA = "v7-clean-signal-model-identity-v1"
INPUT_SIZE = 256
INPUT_CHANNELS = 4
OUTPUT_SIZE = 257
DETAIL_RESIDUAL_SCALE = 0.5

UNET_LITE_ID = "unet_lite_v2"
PYRAMID_CNN_ID = "pyramid_cnn"
SEGFORMER_B0_ID = "segformer_b0"
CLEAN_SIGNAL_ARCHITECTURES = (PYRAMID_CNN_ID, SEGFORMER_B0_ID, UNET_LITE_ID)
CleanSignalArchitecture = Literal["pyramid_cnn", "segformer_b0", "unet_lite_v2"]


class CleanSignalModelError(ValueError):
    """Raised when a clean-signal model violates its tensor or identity contract."""


@dataclass(frozen=True)
class CleanSignalPredictions:
    """Coarse/detail model outputs at the published 257x257 height resolution."""

    coarse_prediction_257: Tensor
    detail_prediction_257: Tensor
    height_prediction_257: Tensor

    @property
    def coarse_relief_257(self) -> Tensor:
        return self.coarse_prediction_257

    @property
    def detail_residual_257(self) -> Tensor:
        return self.detail_prediction_257

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "coarse_relief_257": self.coarse_prediction_257,
            "detail_residual_257": self.detail_prediction_257,
            "height_prediction_257": self.height_prediction_257,
        }


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.body(x))


class _PyramidStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, residual: bool) -> None:
        super().__init__()
        self.downsample = _ConvNormAct(in_channels, out_channels, stride=2)
        self.refine = _ResidualConvBlock(out_channels) if residual else _ConvNormAct(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.refine(self.downsample(x))


class _PyramidEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_channels = INPUT_CHANNELS
        for out_channels in channels:
            stages.append(_PyramidStage(in_channels, out_channels, residual=True))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.channels = channels

    def forward(self, x: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class _UnetEncoder(_PyramidEncoder):
    def __init__(self, channels: tuple[int, ...]) -> None:
        nn.Module.__init__(self)
        stages: list[nn.Module] = []
        in_channels = INPUT_CHANNELS
        for out_channels in channels:
            stages.append(_PyramidStage(in_channels, out_channels, residual=False))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.channels = channels


class _MixingBlock(nn.Module):
    """Compact SegFormer-like overlap patch plus channel-mixing block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.depthwise(self.norm(x))
        return x + self.mlp(mixed)


class _SegformerEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_channels = INPUT_CHANNELS
        for out_channels in channels:
            stages.append(
                nn.Sequential(
                    _ConvNormAct(in_channels, out_channels, stride=2),
                    _MixingBlock(out_channels),
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.channels = channels

    def forward(self, x: Tensor) -> list[Tensor]:
        features: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class CleanSignalFeatureAdapter(nn.Module):
    """Project arbitrary candidate encoder features into one shared decoder width."""

    def __init__(self, feature_channels: tuple[int, ...], fusion_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, fusion_channels, kernel_size=1)
            for channels in feature_channels
        )
        self.refine = nn.ModuleList(
            _ConvNormAct(fusion_channels, fusion_channels) for _ in feature_channels
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != len(self.projections):
            raise CleanSignalModelError(
                f"feature count {len(features)} does not match adapter count {len(self.projections)}"
            )
        projected = [
            refine(projection(feature))
            for projection, refine, feature in zip(self.projections, self.refine, features, strict=True)
        ]
        fused = projected[-1]
        for feature in reversed(projected[:-1]):
            fused = functional.interpolate(fused, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            fused = fused + feature
        return fused


class CleanSignalTwoHeadDecoder(nn.Module):
    """Shared decoder with independent coarse and signed-detail heads."""

    def __init__(self, feature_channels: tuple[int, ...], fusion_channels: int) -> None:
        super().__init__()
        self.adapter = CleanSignalFeatureAdapter(feature_channels, fusion_channels)
        self.refine = _ConvNormAct(fusion_channels, fusion_channels)
        self.coarse_head = nn.Conv2d(fusion_channels, 1, kernel_size=1)
        self.detail_head = nn.Conv2d(fusion_channels, 1, kernel_size=1)

    def forward(self, features: list[Tensor]) -> CleanSignalPredictions:
        fused = self.refine(self.adapter(features))
        fused = functional.interpolate(fused, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        coarse = torch.sigmoid(self.coarse_head(fused))
        detail = torch.tanh(self.detail_head(fused)) * DETAIL_RESIDUAL_SCALE
        coarse = functional.interpolate(coarse, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=True).squeeze(1)
        detail = functional.interpolate(detail, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=True).squeeze(1)
        height = torch.clamp(coarse + detail, 0.0, 1.0)
        return CleanSignalPredictions(coarse, detail, height)


class CleanSignalModel(nn.Module):
    """Candidate architecture under the shared four-channel coarse/detail contract."""

    def __init__(self, architecture: CleanSignalArchitecture, *, profile: str = "tiny") -> None:
        super().__init__()
        if architecture not in CLEAN_SIGNAL_ARCHITECTURES:
            raise CleanSignalModelError(
                f"architecture must be one of {list(CLEAN_SIGNAL_ARCHITECTURES)}, got {architecture!r}"
            )
        if profile not in {"tiny", "full"}:
            raise CleanSignalModelError(f"unknown clean-signal profile {profile!r}")
        channels = (16, 24, 32, 48) if profile == "tiny" else (32, 64, 128, 192)
        fusion_channels = 24 if profile == "tiny" else 64
        if architecture == UNET_LITE_ID:
            self.encoder = _UnetEncoder(channels)
            encoder_kind = "local_unet_encoder"
        elif architecture == PYRAMID_CNN_ID:
            self.encoder = _PyramidEncoder(channels)
            encoder_kind = "local_pyramid_encoder"
        else:
            self.encoder = _SegformerEncoder(channels)
            encoder_kind = "local_segformer_encoder"
        self.decoder = CleanSignalTwoHeadDecoder(channels, fusion_channels)
        self.architecture = architecture
        self.profile = profile
        self.encoder_kind = encoder_kind
        self.feature_channels = channels
        self.fusion_channels = fusion_channels
        # Start the detail branch at zero so the initial prediction is the coarse branch only.
        nn.init.zeros_(self.decoder.detail_head.weight)
        nn.init.zeros_(self.decoder.detail_head.bias)

    def forward(self, x: Tensor) -> CleanSignalPredictions:
        if not isinstance(x, Tensor):
            raise CleanSignalModelError("clean-signal inference accepts only a 4D tensor")
        if x.ndim != 4 or tuple(x.shape[1:]) != (INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE):
            raise CleanSignalModelError(
                f"{self.architecture} consumes (B, 4, 256, 256); got shape {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise CleanSignalModelError("clean-signal input tensor must be floating point")
        if not torch.isfinite(x).all():
            raise CleanSignalModelError("clean-signal input tensor contains non-finite values")
        return self.decoder(self.encoder(x))


def _model_config(model: CleanSignalModel) -> dict[str, Any]:
    return {
        "architecture": model.architecture,
        "profile": model.profile,
        "encoder": model.encoder_kind,
        "feature_channels": list(model.feature_channels),
        "fusion_channels": model.fusion_channels,
        "input_schema": CLEAN_INPUT_SCHEMA,
        "input_shape": [INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE],
        "output_schema": CLEAN_OUTPUT_SCHEMA,
        "output_shape": [3, OUTPUT_SIZE, OUTPUT_SIZE],
        "detail_residual_scale": DETAIL_RESIDUAL_SCALE,
        "weights": "random_init",
        "pretrained": False,
    }


def model_identity(model: CleanSignalModel) -> dict[str, Any]:
    """Return JSON-serializable identity sufficient to reconstruct the model."""

    config = _model_config(model)
    return {
        "schema": MODEL_IDENTITY_SCHEMA,
        "id": model.architecture,
        "architecture": model.architecture,
        "profile": model.profile,
        "config": config,
        "config_sha256": sha256_json(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_contract": f"{CLEAN_INPUT_SCHEMA}:{INPUT_CHANNELS}x{INPUT_SIZE}x{INPUT_SIZE}",
        "output_contract": f"{CLEAN_OUTPUT_SCHEMA}:coarse+detail+height@{OUTPUT_SIZE}",
        "forbidden_inference_inputs": ["wdl", "height", "normal", "liquid", "object", "alpha"],
        "weights": "random_init",
        "pretrained": False,
    }


def build_clean_signal_model(
    architecture: CleanSignalArchitecture,
    *,
    profile: str = "tiny",
) -> tuple[CleanSignalModel, dict[str, Any]]:
    """Build one random-initialized clean-signal candidate and its identity."""

    model = CleanSignalModel(architecture, profile=profile)
    return model, model_identity(model)


def build_clean_signal_model_from_identity(identity: dict[str, Any]) -> tuple[CleanSignalModel, dict[str, Any]]:
    """Rebuild a candidate and refuse identities whose config hash or schema is inconsistent."""

    if identity.get("schema") != MODEL_IDENTITY_SCHEMA:
        raise CleanSignalModelError(f"expected identity schema {MODEL_IDENTITY_SCHEMA!r}")
    config = identity.get("config")
    if not isinstance(config, dict):
        raise CleanSignalModelError("model identity config must be an object")
    if identity.get("config_sha256") != sha256_json(config):
        raise CleanSignalModelError("model identity config_sha256 does not match config")
    model, rebuilt = build_clean_signal_model(
        str(identity.get("architecture", "")),
        profile=str(identity.get("profile", "")),
    )
    if rebuilt["config_sha256"] != identity["config_sha256"]:
        raise CleanSignalModelError("model identity configuration does not reconstruct the same model")
    if int(identity.get("parameter_count", -1)) != rebuilt["parameter_count"]:
        raise CleanSignalModelError("model identity parameter_count does not match reconstructed model")
    return model, rebuilt


def identity_json(identity: dict[str, Any]) -> str:
    """Serialize an identity deterministically for reports and checkpoint sidecars."""

    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CLEAN_INPUT_SCHEMA",
    "CLEAN_OUTPUT_SCHEMA",
    "CLEAN_SIGNAL_ARCHITECTURES",
    "CleanSignalFeatureAdapter",
    "CleanSignalModel",
    "CleanSignalModelError",
    "CleanSignalPredictions",
    "CleanSignalTwoHeadDecoder",
    "DETAIL_RESIDUAL_SCALE",
    "INPUT_CHANNELS",
    "MODEL_IDENTITY_SCHEMA",
    "build_clean_signal_model",
    "build_clean_signal_model_from_identity",
    "identity_json",
    "model_identity",
]
