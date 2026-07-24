"""Spec 121 US1: SegFormer-B0 backbone lattice predictor (Stage A).

Spec 117's from-scratch ``LatticeNet`` (675K params at base 24) plateaued above the tile-mean
baseline — a capacity/inductive-bias failure, not a data failure (research.md D-01). This module
keeps Spec 117's native-direct output philosophy (predict the 17x17 outer + 16x16 inner grids at
their own resolution; NO interpolation in the model's output path) but swaps the encoder for the
SegFormer-B0 hierarchical transformer that Spec 114 already validated in this repo
(``MitB0RegressionNet``), optionally initialized from pretrained ``nvidia/mit-b0`` weights.

Head placement mirrors ``LatticeNet`` v5 exactly: the inner 16x16 grid reads the encoder's native
16x16 stage; the outer 17x17 grid reads the native 32x32 stage through a learned k2/s2/p1 conv
(floor((32+2-2)/2)+1 = 17) — a localized learned downsample, never an interpolation.

Param band (FR-003, SC-004): the default B0 config lands at ~3.4M params, inside the user's
3–30M band; ``parameter_band_ok`` enforces/flags this for backbone architectures.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from harvester.spec117.lattice_contract import INNER_DIM, OUTER_DIM, SAMPLE_COUNT
from harvester.spec117.lattice_model import LatticeTargetError
from harvester.v50.direct_geometry_model import _DEPTH_ANYTHING_MARKER

LATTICE_NET_ID = "lattice_net"
MIT_B0_LATTICE_ID = "mit_b0_lattice"
ARCHITECTURE_IDS = frozenset({LATTICE_NET_ID, MIT_B0_LATTICE_ID})

MIT_B0_HUB_ID = "nvidia/mit-b0"
MIT_B0_LICENSE = "Apache-2.0"

PARAM_BAND_MIN = 3_000_000
PARAM_BAND_MAX = 30_000_000
INPUT_SIZE = 256


class LatticeBackboneError(ValueError):
    """Raised when a Stage A architecture or weight source violates the Spec 121 contract."""


def default_lattice_mit_config() -> Any:
    """MiT-B0/SegFormer-B0 encoder sizes (RGB input). From-scratch default; pretrained optional."""
    from transformers import SegformerConfig

    return SegformerConfig(num_labels=1, num_channels=3)


def tiny_lattice_mit_config() -> Any:
    """CPU-fixture config: identical topology and output contract at a fraction of the cost."""
    from transformers import SegformerConfig

    return SegformerConfig(
        num_labels=1,
        num_channels=3,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        decoder_hidden_size=16,
        num_attention_heads=[1, 2, 4, 8],
    )


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv3x3 + GroupNorm + SiLU — the same primitive the v50 U-Nets and LatticeNet use."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class MitB0LatticeNet(nn.Module):
    """SegFormer-B0 encoder + native lattice heads: 256x256x3 -> (B, 545) in [0, 1].

    The encoder's four stages emit at strides 4/8/16/32; for a 256 input that is 64/32/16/8.
    Heads read the native 32x32 stage (outer) and the native 16x16 stage (inner), exactly the
    resolutions ``LatticeNet`` v5 proved the heads want. Output ordering matches every other
    lattice consumer: outer 289 values first, then inner 256.
    """

    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        from transformers import SegformerModel

        self.config = config if config is not None else default_lattice_mit_config()
        hidden = list(self.config.hidden_sizes)
        if len(hidden) != 4:
            raise LatticeBackboneError(
                f"MitB0LatticeNet expects a 4-stage SegFormer config, got hidden_sizes={hidden}"
            )
        self.in_channels = int(self.config.num_channels)
        self.encoder = SegformerModel(self.config)
        stage32_ch = hidden[1]  # native 32x32 map for the 256 input
        stage16_ch = hidden[2]  # native 16x16 map
        # Outer 17x17 head: learned k2/s2/p1 downsample 32 -> 17, then refine (LatticeNet v5 rule).
        self.outer_reduce = nn.Conv2d(stage32_ch, stage32_ch, kernel_size=2, stride=2, padding=1)
        self.outer_head = nn.Sequential(
            _block(stage32_ch, max(stage32_ch // 2, 8)),
            nn.Conv2d(max(stage32_ch // 2, 8), 1, 1),
        )
        # Inner 16x16 head: native, straight off the 16x16 stage.
        self.inner_head = nn.Sequential(
            _block(stage16_ch, max(stage16_ch // 2, 8)),
            nn.Conv2d(max(stage16_ch // 2, 8), 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise LatticeTargetError(
                f"MitB0LatticeNet consumes (B, {self.in_channels}, H, W); got shape {tuple(x.shape)}"
            )
        hidden_states = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
        stage32 = hidden_states[1]  # (B, C, 32, 32) for a 256 input
        stage16 = hidden_states[2]  # (B, C, 16, 16)
        outer = torch.sigmoid(self.outer_head(self.outer_reduce(stage32))).flatten(1)  # (B, 289)
        inner = torch.sigmoid(self.inner_head(stage16)).flatten(1)                     # (B, 256)
        out = torch.cat([outer, inner], dim=1)
        if out.shape[1] != SAMPLE_COUNT:
            raise LatticeBackboneError(
                f"lattice head emitted {out.shape[1]} values, expected {SAMPLE_COUNT} "
                f"({OUTER_DIM}x{OUTER_DIM} + {INNER_DIM}x{INNER_DIM}); input size must be {INPUT_SIZE}"
            )
        return out


def backbone_config_payload(config: Any) -> dict[str, Any]:
    """Plain-dict config payload: hashed into the architecture identity AND stored verbatim in the
    checkpoint so the exact encoder shape is reconstructable without the Hub (Spec 117's
    ``lattice_config.base`` lesson — a sha256 does not carry the config)."""
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return {key: raw[key] for key in sorted(raw)}


def config_from_payload(payload: dict[str, Any]) -> Any:
    """Rebuild a SegformerConfig from ``backbone_config_payload`` output."""
    from transformers import SegformerConfig

    return SegformerConfig(**payload)


def load_pretrained_lattice_encoder(model: MitB0LatticeNet, *, hub_id: str, revision: str) -> None:
    """USER-RUN path: load HF encoder weights into the backbone. Never called in dry-runs or
    CPU fixture tests (mirrors ``direct_geometry_model.load_pretrained_encoder``)."""
    from transformers import SegformerModel

    if _DEPTH_ANYTHING_MARKER in hub_id.lower():
        raise LatticeBackboneError(f"DepthAnything-family sources are forbidden, got {hub_id!r}")
    if not isinstance(model, MitB0LatticeNet):
        raise LatticeBackboneError("pretrained encoder weights only apply to mit_b0_lattice")
    encoder = SegformerModel.from_pretrained(hub_id, revision=revision)
    model.encoder.load_state_dict(encoder.state_dict())


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_band_ok(model: nn.Module) -> bool:
    """SC-004 band check for backbone architectures (3–30M params)."""
    return PARAM_BAND_MIN <= parameter_count(model) <= PARAM_BAND_MAX


def build_stage_a_model(
    architecture: str,
    *,
    base: int = 64,
    mit_config: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build one Stage A model plus the identity config payload for ``architecture_identity``.

    Returns ``(model, config_payload)``. ``lattice_net`` is the Spec 117 from-scratch fallback
    (constructable from ``base`` alone); ``mit_b0_lattice`` is the backbone lane (config payload
    is the full SegformerConfig dict).
    """
    if architecture == LATTICE_NET_ID:
        from harvester.spec117.lattice_model import LatticeNet

        model: nn.Module = LatticeNet(base=base)
        payload: dict[str, Any] = {
            "class": "LatticeNet", "arch": "lattice_net_v5",
            "base": base, "input": "3x256x256", "output": str(SAMPLE_COUNT),
        }
    elif architecture == MIT_B0_LATTICE_ID:
        config = mit_config if mit_config is not None else default_lattice_mit_config()
        model = MitB0LatticeNet(config)
        payload = {"class": "MitB0LatticeNet", "arch": MIT_B0_LATTICE_ID,
                   **backbone_config_payload(config)}
    else:
        raise LatticeBackboneError(
            f"architecture must be one of {sorted(ARCHITECTURE_IDS)}, got {architecture!r}"
        )
    return model, payload


__all__ = [
    "ARCHITECTURE_IDS",
    "INPUT_SIZE",
    "LATTICE_NET_ID",
    "MIT_B0_HUB_ID",
    "MIT_B0_LATTICE_ID",
    "MIT_B0_LICENSE",
    "PARAM_BAND_MAX",
    "PARAM_BAND_MIN",
    "LatticeBackboneError",
    "MitB0LatticeNet",
    "backbone_config_payload",
    "build_stage_a_model",
    "config_from_payload",
    "default_lattice_mit_config",
    "load_pretrained_lattice_encoder",
    "parameter_band_ok",
    "parameter_count",
    "tiny_lattice_mit_config",
]
