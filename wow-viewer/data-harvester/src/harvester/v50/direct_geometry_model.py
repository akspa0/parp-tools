"""Spec 114 T014: direct-geometry architecture registry and the MiT-B0 regression candidate.

Two architectures share one contract: authored/synthetic minimap RGB (3x256x256) in, exactly one
sigmoid-bounded ``relative_height_257`` field (257x257) out. No WDL input, no auxiliary heads, no
shared weights with any other stage (constitution IV; FR-001/FR-011).

- ``direct_cnn_v112``: the frozen Spec 112 from-scratch U-Net-lite (1,561,537 parameters). Mandatory
  baseline; its authored-only run is recorded as rejected negative evidence in research.md.
- ``mit_b0_regression``: a SegFormer/MiT-B0 hierarchical encoder with a one-channel continuous
  regression decoder, per the original plan's Candidate A. Pretrained encoder weights are OPTIONAL
  (FR-013): license-recorded, revision/hash-pinned, and always compared against the from-scratch
  baseline on the same frozen split. DepthAnything-family sources are refused outright by the
  standing project decision.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from harvester.v50.height_relative_model import HEIGHT_GRID, HeightRelativeNet
from harvester.v50.model_stage_contract import sha256_json

DIRECT_CNN_ID = "direct_cnn_v112"
MIT_B0_ID = "mit_b0_regression"
ARCHITECTURE_IDS = frozenset({DIRECT_CNN_ID, MIT_B0_ID})

MIT_B0_HUB_ID = "nvidia/mit-b0"
MIT_B0_LICENSE = "Apache-2.0"

EXPECTED_DIRECT_CNN_PARAMS = 1_561_537
INPUT_SIZE = 256

_DEPTH_ANYTHING_MARKER = "depth-anything"


class GeometryModelError(ValueError):
    """Raised when a geometry architecture or weight source violates the Spec 114 contract."""


DEFAULT_INPUT_CHANNELS = 3


def default_mit_config(*, in_channels: int = DEFAULT_INPUT_CHANNELS) -> Any:
    """MiT-B0/SegFormer-B0 sizes with a single continuous output channel (from-scratch default).

    ``in_channels`` defaults to 3 (RGB only), keeping every existing checkpoint bit-identical. A
    deconfounded run (Spec 115) passes ``3 + K`` to concatenate a generated feature-map; because
    ``num_channels`` is part of the hashed config, a wider-input model can never present the same
    architecture identity as the RGB-only baseline.
    """
    from transformers import SegformerConfig

    return SegformerConfig(num_labels=1, num_channels=in_channels)


def tiny_mit_config(*, in_channels: int = DEFAULT_INPUT_CHANNELS) -> Any:
    """CPU-fixture config: identical topology and output contract at a fraction of the cost."""
    from transformers import SegformerConfig

    return SegformerConfig(
        num_labels=1,
        num_channels=in_channels,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        decoder_hidden_size=16,
        num_attention_heads=[1, 2, 4, 8],
    )


class MitB0RegressionNet(nn.Module):
    """SegFormer encoder + one-channel regression decoder, 256x256x3 -> 257x257 in [0, 1].

    The decode head emits at 1/4 input resolution; the prediction is upsampled to the world grid
    exactly once, mirroring the ``HeightRelativeNet`` boundary so both architectures are directly
    comparable on the same target and metrics.
    """

    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.config = config if config is not None else default_mit_config()
        if int(self.config.num_labels) != 1:
            raise GeometryModelError(
                f"mit_b0_regression must have exactly one output channel, got {self.config.num_labels}"
            )
        self.in_channels = int(self.config.num_channels)
        self.segformer = SegformerForSemanticSegmentation(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise GeometryModelError(
                f"mit_b0_regression consumes (B, {self.in_channels}, H, W); got shape {tuple(x.shape)}"
            )
        logits = self.segformer(pixel_values=x).logits
        out = torch.sigmoid(
            nn.functional.interpolate(
                logits, size=(HEIGHT_GRID, HEIGHT_GRID), mode="bilinear", align_corners=True
            )
        )
        return out.squeeze(1)


def validate_pretrained_source(record: dict | None) -> dict | None:
    """FR-013: pretrained weights are optional, license-recorded, hash-pinned — never DA-family."""
    if record is None:
        return None
    hub_id = str(record.get("hub_id", ""))
    if _DEPTH_ANYTHING_MARKER in hub_id.lower():
        raise GeometryModelError(
            f"DepthAnything-family sources are forbidden in this lane, got {hub_id!r}"
        )
    for key in ("hub_id", "revision", "license"):
        if not str(record.get(key, "")):
            raise GeometryModelError(f"pretrained_source requires a non-empty {key!r}")
    sha256 = str(record.get("sha256", ""))
    if len(sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha256):
        raise GeometryModelError("pretrained_source requires a pinned 64-hex sha256")
    return {
        "hub_id": hub_id,
        "revision": str(record["revision"]),
        "sha256": sha256,
        "license": str(record["license"]),
        "optional": True,
    }


def _config_dict(config: Any) -> dict:
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return {key: raw[key] for key in sorted(raw)}


def architecture_identity(architecture: str, model: nn.Module, *, config: Any | None) -> dict:
    """Schema-conformant architecture block: local id, content-hashed config, parameter count."""
    if architecture not in ARCHITECTURE_IDS:
        raise GeometryModelError(
            f"architecture must be one of {sorted(ARCHITECTURE_IDS)}, got {architecture!r}"
        )
    config_payload: dict[str, Any]
    if architecture == DIRECT_CNN_ID:
        config_payload = {"class": "HeightRelativeNet", "base": 32, "input": "3x256x256",
                          "output": f"1x{HEIGHT_GRID}x{HEIGHT_GRID}", "target_contract": "v112.1"}
    else:
        config_payload = _config_dict(config if config is not None else default_mit_config())
    return {
        "id": architecture,
        "config_sha256": sha256_json(config_payload),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_geometry_model(
    architecture: str,
    *,
    mit_config: Any | None = None,
    in_channels: int = DEFAULT_INPUT_CHANNELS,
    pretrained_source: dict | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build one geometry model plus its full schema identity block.

    Returns ``(model, identity)`` where identity carries ``architecture`` and ``pretrained_source``
    exactly as the ``v50-model-stage-run-v1`` contract records them. ``in_channels`` defaults to 3
    (RGB); a Spec 115 deconfounded run passes ``3 + K`` and the wider config hashes to a distinct
    architecture identity. Only ``mit_b0_regression`` supports extra channels; the frozen
    ``direct_cnn_v112`` baseline is RGB-only by contract.
    """
    if architecture == DIRECT_CNN_ID:
        if pretrained_source is not None:
            raise GeometryModelError("direct_cnn_v112 is the from-scratch baseline; no pretrained source")
        if in_channels != DEFAULT_INPUT_CHANNELS:
            raise GeometryModelError(
                f"direct_cnn_v112 is RGB-only (3 channels); got in_channels={in_channels}"
            )
        model: nn.Module = HeightRelativeNet()
        config = None
    elif architecture == MIT_B0_ID:
        config = mit_config if mit_config is not None else default_mit_config(in_channels=in_channels)
        model = MitB0RegressionNet(config)
    else:
        raise GeometryModelError(
            f"architecture must be one of {sorted(ARCHITECTURE_IDS)}, got {architecture!r}"
        )
    identity = {
        "architecture": architecture_identity(architecture, model, config=config),
        "pretrained_source": validate_pretrained_source(pretrained_source),
    }
    return model, identity


def load_pretrained_encoder(model: MitB0RegressionNet, *, hub_id: str, revision: str) -> None:
    """USER-RUN path: load pinned HF encoder weights into the regression model's backbone.

    Kept separate from ``build_geometry_model`` so CPU fixture tests never touch the Hub. The
    recorded ``pretrained_source`` (with its pinned sha256) is the run identity; this call performs
    the actual download only inside a user-confirmed training run.
    """
    from transformers import SegformerModel

    if _DEPTH_ANYTHING_MARKER in hub_id.lower():
        raise GeometryModelError(f"DepthAnything-family sources are forbidden, got {hub_id!r}")
    if not isinstance(model, MitB0RegressionNet):
        raise GeometryModelError("pretrained encoder weights only apply to mit_b0_regression")
    encoder = SegformerModel.from_pretrained(hub_id, revision=revision)
    model.segformer.segformer.load_state_dict(encoder.state_dict())
