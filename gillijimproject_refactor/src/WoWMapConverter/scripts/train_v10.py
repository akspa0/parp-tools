from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import train_v9_optimized as v9


DEFAULT_OUTPUT_DIR = v9.WORKSPACE_ROOT / "output" / "ml-training" / "v10_no_wdl"

DEFAULT_PRIOR_NO_WDL_PROB = 0.50
DEFAULT_PRIOR_REAL_WDL_PROB = 0.30
DEFAULT_PRIOR_CORRUPT_WDL_PROB = 0.20
DEFAULT_CORRUPT_SHIFT_MAX = 96.0
DEFAULT_CORRUPT_NOISE_STD = 24.0
DEFAULT_GATE_SUPPRESSION_WEIGHT = 0.05
DEFAULT_DETAIL_RESIDUAL_WEIGHT = 0.08
DEFAULT_GRADIENT_WEIGHT = 0.35
DEFAULT_MID_L1_WEIGHT = 0.55
DEFAULT_COARSE_L1_WEIGHT = 0.35
DEFAULT_QUALITY_REWARD = 0.35
DEFAULT_LOW_SIGNAL_PENALTY = 0.25
DEFAULT_BLANK_TILE_PENALTY = 0.45

VISUAL_PREFIX_CHANNELS = 8
OPTIONAL_PRIOR_CHANNELS = 3
V10_VISUAL_CHANNELS = 20

V10_ACTIVE_INPUT_SIGNALS = [
    "terrain_only_or_no_liquid_or_no_object_or_no_mccv_or_image_minimap_rgb",
    "normal_rgb",
    "minimap_luma",
    "minimap_detail_gradient",
    "optional_prior_height_257",
    "optional_prior_present_mask",
    "optional_prior_quality_mask",
    "height_min_mask",
    "height_max_mask",
    "height_range_context",
    "detail_energy_context",
    "minimap_variance_context",
    "liquid_mask",
    "liquid_height_prior",
    "object_footprint_mask",
    "object_precise_mask",
    "pm4_footprint_mask",
    "brush_imprint_mask",
    "hole_mask_16x16",
]

V10_NATIVE_TARGET_SIGNALS = [
    "coarse_height_17_absolute",
    "mid_height_65_absolute",
    "full_height_257_absolute",
]


def resolve_amp_dtype(amp_dtype: str, device: torch.device) -> torch.dtype:
    return v9.resolve_amp_dtype(amp_dtype, device)


def _sized_mask(value: float, size: tuple[int, int]) -> torch.Tensor:
    return torch.full((1, size[0], size[1]), float(value), dtype=torch.float32)


def _interpolate_height(height_17: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(height_17.unsqueeze(0), size=size, mode="bilinear", align_corners=True).squeeze(0)


def build_v10_feature_contract(args: argparse.Namespace | None = None) -> dict[str, list[str] | str | dict[str, float]]:
    zeroed_input_signals: list[str] = []
    if args is not None and getattr(args, "disable_brush_mask", False):
        zeroed_input_signals.append("brush_imprint_mask")

    return {
        "contract_version": "v10-no-wdl-inputs.v1",
        "active_input_signals": list(V10_ACTIVE_INPUT_SIGNALS),
        "native_target_signals": list(V10_NATIVE_TARGET_SIGNALS),
        "zeroed_input_signals": zeroed_input_signals,
        "prior_mode_defaults": {
            "no_wdl": float(getattr(args, "prior_no_wdl_prob", DEFAULT_PRIOR_NO_WDL_PROB)) if args is not None else DEFAULT_PRIOR_NO_WDL_PROB,
            "real_wdl": float(getattr(args, "prior_real_wdl_prob", DEFAULT_PRIOR_REAL_WDL_PROB)) if args is not None else DEFAULT_PRIOR_REAL_WDL_PROB,
            "corrupt_wdl": float(getattr(args, "prior_corrupt_wdl_prob", DEFAULT_PRIOR_CORRUPT_WDL_PROB)) if args is not None else DEFAULT_PRIOR_CORRUPT_WDL_PROB,
        },
        "summary": (
            "V10 trains a no-WDL-first terrain predictor. The coarse terrain branch must work without a prior, "
            "while WDL becomes an optional conditioning signal carried through an explicit prior-present and prior-quality stack."
        ),
    }


def describe_v10_input_stack(args: argparse.Namespace) -> str:
    brush_state = "disabled" if getattr(args, "disable_brush_mask", False) else "enabled"
    return (
        "Active v10 inputs | minimap RGB + normal RGB + minimap luma/edge detail priors + optional WDL prior + prior-present/prior-quality masks + "
        f"height hints/range/detail context + liquid/object/precise-object/PM4 masks + brush mask {brush_state} + liquid height + hole mask"
    )


def choose_prior_mode(
    *,
    has_wdl: bool,
    requested_mode: str,
    seed: int,
    epoch: int,
    index: int,
    no_wdl_prob: float,
    real_wdl_prob: float,
    corrupt_wdl_prob: float,
) -> str:
    if not has_wdl:
        return "no_wdl"
    if requested_mode != "mixed":
        return requested_mode

    weights = [max(0.0, no_wdl_prob), max(0.0, real_wdl_prob), max(0.0, corrupt_wdl_prob)]
    total = sum(weights)
    if total <= 0.0:
        return "no_wdl"

    normalized = [weight / total for weight in weights]
    rng = random.Random((seed * 1000003) + (epoch * 1009) + index)
    sample = rng.random()
    if sample < normalized[0]:
        return "no_wdl"
    if sample < normalized[0] + normalized[1]:
        return "real_wdl"
    return "corrupt_wdl"


def corrupt_prior_17(prior_17: torch.Tensor, rng: random.Random, shift_max: float, noise_std: float) -> torch.Tensor:
    corrupted = prior_17.clone()
    corruption_mode = rng.choice(["flatten", "shift", "noise", "blur", "mixed"])

    if corruption_mode in {"flatten", "mixed"}:
        flattened = torch.full_like(corrupted, float(corrupted.mean().item()))
        blend = 0.65 if corruption_mode == "mixed" else 1.0
        corrupted = torch.lerp(corrupted, flattened, blend)

    if corruption_mode in {"shift", "mixed"}:
        corrupted = corrupted + rng.uniform(-shift_max, shift_max)

    if corruption_mode in {"noise", "mixed"} and noise_std > 0.0:
        corrupted = corrupted + (torch.randn_like(corrupted) * float(noise_std))

    if corruption_mode == "blur":
        blurred = F.avg_pool2d(corrupted.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        corrupted = torch.lerp(corrupted, blurred, 0.75)

    return corrupted


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_entry_signal_strength(entry: v9.V9SampleEntry, args: argparse.Namespace) -> float:
    height_strength = _clamp01(entry.height_range / max(float(args.min_height_range) * 8.0, 1.0))
    variance_strength = _clamp01(entry.minimap_variance / max(float(args.min_minimap_variance) * 8.0, 1.0e-8))
    gradient_strength = _clamp01(entry.minimap_gradient / max(float(args.min_minimap_gradient) * 4.0, 1.0e-8))
    return float((height_strength * 0.45) + (variance_strength * 0.25) + (gradient_strength * 0.30))


def is_blank_like_entry(entry: v9.V9SampleEntry, args: argparse.Namespace) -> bool:
    return (
        entry.height_range < float(args.min_height_range)
        and entry.minimap_variance < float(args.min_minimap_variance)
        and entry.minimap_gradient < float(args.min_minimap_gradient)
    )


def compute_entry_sample_weight(entry: v9.V9SampleEntry, args: argparse.Namespace) -> float:
    signal_strength = compute_entry_signal_strength(entry, args)
    reward = float(args.quality_reward) * signal_strength
    low_signal_penalty = 0.0
    if signal_strength < 0.5:
        low_signal_penalty = float(args.low_signal_penalty) * ((0.5 - signal_strength) / 0.5)
    blank_penalty = float(args.blank_tile_penalty) if is_blank_like_entry(entry, args) else 0.0
    return float(max(0.1, min(2.0, 1.0 + reward - low_signal_penalty - blank_penalty)))


def summarize_entry_weighting(entries: Sequence[v9.V9SampleEntry], args: argparse.Namespace) -> dict[str, float | int]:
    if not entries:
        return {
            "count": 0,
            "blank_like_count": 0,
            "blank_like_fraction": 0.0,
            "low_signal_count": 0,
            "low_signal_fraction": 0.0,
            "min_sample_weight": 0.0,
            "mean_sample_weight": 0.0,
            "max_sample_weight": 0.0,
        }

    weights = [compute_entry_sample_weight(entry, args) for entry in entries]
    signal_strengths = [compute_entry_signal_strength(entry, args) for entry in entries]
    blank_like_count = sum(1 for entry in entries if is_blank_like_entry(entry, args))
    low_signal_count = sum(1 for strength in signal_strengths if strength < 0.5)
    return {
        "count": len(entries),
        "blank_like_count": blank_like_count,
        "blank_like_fraction": blank_like_count / len(entries),
        "low_signal_count": low_signal_count,
        "low_signal_fraction": low_signal_count / len(entries),
        "min_sample_weight": float(min(weights)),
        "mean_sample_weight": float(sum(weights) / len(weights)),
        "max_sample_weight": float(max(weights)),
    }


@dataclass(frozen=True)
class V10SampleState:
    sample_key: str
    tile_name: str
    visual_prefix: torch.Tensor
    visual_suffix: torch.Tensor
    preview_minimap_rgb: torch.Tensor
    target_height_257: torch.Tensor
    target_height_65: torch.Tensor
    target_height_17: torch.Tensor
    target_detail_residual_257: torch.Tensor
    wdl_17: torch.Tensor | None
    has_wdl: bool
    sample_weight: float


class V10NativeDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[v9.V9SampleEntry],
        arrays_cache: dict[str, dict[str, np.ndarray]],
        *,
        height_scale: float,
        residual_scale: float,
        include_brush_mask: bool,
        training: bool,
        prior_mode: str,
        seed: int,
        prior_no_wdl_prob: float,
        prior_real_wdl_prob: float,
        prior_corrupt_wdl_prob: float,
        corrupt_shift_max: float,
        corrupt_noise_std: float,
        sample_weight_args: argparse.Namespace,
    ):
        self.entries = list(entries)
        self.arrays_cache = arrays_cache
        self.height_scale = float(height_scale)
        self.residual_scale = float(residual_scale)
        self.include_brush_mask = bool(include_brush_mask)
        self.training = bool(training)
        self.prior_mode = str(prior_mode)
        self.seed = int(seed)
        self.prior_no_wdl_prob = float(prior_no_wdl_prob)
        self.prior_real_wdl_prob = float(prior_real_wdl_prob)
        self.prior_corrupt_wdl_prob = float(prior_corrupt_wdl_prob)
        self.corrupt_shift_max = float(corrupt_shift_max)
        self.corrupt_noise_std = float(corrupt_noise_std)
        self.sample_weight_args = sample_weight_args
        self.current_epoch = 0

        self._precomputed: list[V10SampleState] = [self._precompute(entry) for entry in self.entries]

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _precompute(self, entry: v9.V9SampleEntry) -> V10SampleState:
        arrays = self.arrays_cache[str(entry.shard_path)]

        height_17 = torch.from_numpy(arrays["height_17"].astype(np.float32)).unsqueeze(0) / self.height_scale
        height_65 = torch.from_numpy(arrays["height_65"].astype(np.float32)).unsqueeze(0) / self.height_scale
        height_257 = torch.from_numpy(arrays["height_257"].astype(np.float32)).unsqueeze(0) / self.height_scale
        target_detail_residual_257 = height_257 - _interpolate_height(height_65, (257, 257))

        zero_base_257 = torch.zeros((1, 257, 257), dtype=torch.float32)
        v9_inputs, preview_minimap_rgb = v9._build_v9_input_channels(
            entry=entry,
            arrays=arrays,
            base_257_scaled=zero_base_257,
            include_brush_mask=self.include_brush_mask,
        )
        visual_prefix = v9_inputs[:VISUAL_PREFIX_CHANNELS].clone()
        visual_suffix = v9_inputs[VISUAL_PREFIX_CHANNELS + 1 :].clone()

        wdl_17 = None
        if "wdl_17" in arrays:
            wdl_17 = torch.from_numpy(arrays["wdl_17"].astype(np.float32)).unsqueeze(0)

        return V10SampleState(
            sample_key=entry.sample_key,
            tile_name=entry.tile_name,
            visual_prefix=visual_prefix,
            visual_suffix=visual_suffix,
            preview_minimap_rgb=preview_minimap_rgb,
            target_height_257=height_257,
            target_height_65=height_65,
            target_height_17=height_17,
            target_detail_residual_257=target_detail_residual_257,
            wdl_17=wdl_17,
            has_wdl=wdl_17 is not None,
            sample_weight=compute_entry_sample_weight(entry, self.sample_weight_args),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_prior(self, state: V10SampleState, index: int) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = choose_prior_mode(
            has_wdl=state.has_wdl,
            requested_mode=self.prior_mode,
            seed=self.seed,
            epoch=self.current_epoch if self.training else 0,
            index=index,
            no_wdl_prob=self.prior_no_wdl_prob,
            real_wdl_prob=self.prior_real_wdl_prob,
            corrupt_wdl_prob=self.prior_corrupt_wdl_prob,
        )

        if mode == "real_wdl" and state.wdl_17 is not None:
            prior_17 = state.wdl_17.clone()
            present_value = 1.0
            quality_value = 1.0
        elif mode == "corrupt_wdl" and state.wdl_17 is not None:
            rng = random.Random((self.seed * 2000003) + (self.current_epoch * 9176) + index)
            prior_17 = corrupt_prior_17(state.wdl_17, rng, self.corrupt_shift_max, self.corrupt_noise_std)
            present_value = 1.0
            quality_value = 0.35
        else:
            prior_17 = torch.zeros((1, 17, 17), dtype=torch.float32)
            present_value = 0.0
            quality_value = 0.0
            mode = "no_wdl"

        prior_65 = _interpolate_height(prior_17, (65, 65)) / self.height_scale
        prior_257 = _interpolate_height(prior_17, (257, 257)) / self.height_scale
        prior_present_257 = _sized_mask(present_value, (257, 257))
        prior_quality_257 = _sized_mask(quality_value, (257, 257))

        return mode, prior_17 / self.height_scale, prior_65, prior_257, prior_present_257, prior_quality_257

    def __getitem__(self, index: int) -> dict[str, Any]:
        state = self._precomputed[index]
        prior_mode, prior_17, prior_65, prior_257, prior_present_257, prior_quality_257 = self._resolve_prior(state, index)

        inputs = torch.cat(
            [
                state.visual_prefix,
                prior_257,
                prior_present_257,
                prior_quality_257,
                state.visual_suffix,
            ],
            dim=0,
        )

        return {
            "sample_key": state.sample_key,
            "tile_name": state.tile_name,
            "inputs": inputs,
            "preview_minimap_rgb": state.preview_minimap_rgb,
            "target_height_257": state.target_height_257,
            "target_height_65": state.target_height_65,
            "target_height_17": state.target_height_17,
            "target_detail_residual_257": state.target_detail_residual_257,
            "prior_height_17": prior_17,
            "prior_height_65": prior_65,
            "prior_height_257": prior_257,
            "prior_present_257": prior_present_257,
            "prior_quality_257": prior_quality_257,
            "prior_mode": prior_mode,
            "sample_weight": torch.tensor(state.sample_weight, dtype=torch.float32),
        }


class V10TerrainModel(nn.Module):
    def __init__(self, hidden_channels: int = v9.DEFAULT_HIDDEN_CHANNELS, blocks_per_stage: int = v9.DEFAULT_BLOCKS_PER_STAGE):
        super().__init__()

        prior_hidden = max(hidden_channels // 2, 16)

        self.visual_stem = nn.Sequential(
            nn.Conv2d(V10_VISUAL_CHANNELS, hidden_channels, kernel_size=5, padding=2, padding_mode="reflect"),
            nn.GroupNorm(v9._resolve_group_count(hidden_channels, 8), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.visual_enc1 = nn.Sequential(*[v9.ResidualConvBlock(hidden_channels, dilation=1 + (i % 2)) for i in range(blocks_per_stage)])
        self.visual_down1 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.visual_enc2 = nn.Sequential(*[v9.ResidualConvBlock(hidden_channels * 2, dilation=1 + (i % 3)) for i in range(blocks_per_stage)])
        self.visual_down2 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.visual_enc3 = nn.Sequential(*[v9.ResidualConvBlock(hidden_channels * 4, dilation=1 + (i % 4)) for i in range(blocks_per_stage)])

        self.prior_stem = nn.Sequential(
            nn.Conv2d(OPTIONAL_PRIOR_CHANNELS, prior_hidden, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(v9._resolve_group_count(prior_hidden, 8), prior_hidden),
            nn.SiLU(inplace=True),
        )
        self.prior_enc1 = nn.Sequential(*[v9.ResidualConvBlock(prior_hidden, dilation=1) for _ in range(max(1, blocks_per_stage - 1))])
        self.prior_down1 = nn.Conv2d(prior_hidden, prior_hidden * 2, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.prior_enc2 = nn.Sequential(*[v9.ResidualConvBlock(prior_hidden * 2, dilation=1 + (i % 2)) for i in range(max(1, blocks_per_stage - 1))])
        self.prior_down2 = nn.Conv2d(prior_hidden * 2, prior_hidden * 4, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
        self.prior_enc3 = nn.Sequential(*[v9.ResidualConvBlock(prior_hidden * 4, dilation=1 + (i % 2)) for i in range(max(1, blocks_per_stage - 1))])

        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d((hidden_channels * 4) + (prior_hidden * 4), hidden_channels * 4, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(v9._resolve_group_count(hidden_channels * 4, 8), hidden_channels * 4),
            nn.SiLU(inplace=True),
        )

        self.coarse_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((17, 17)),
            nn.Conv2d(hidden_channels * 2, 1, kernel_size=1),
        )
        self.mid_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, 1, kernel_size=1),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d((hidden_channels * 4) + (hidden_channels * 2) + (prior_hidden * 2), hidden_channels * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(v9._resolve_group_count(hidden_channels * 2, 8), hidden_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.dec2 = v9.ResidualConvBlock(hidden_channels * 2, dilation=1)

        self.up1 = nn.Sequential(
            nn.Conv2d((hidden_channels * 2) + hidden_channels + prior_hidden, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(v9._resolve_group_count(hidden_channels, 8), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.dec1 = v9.ResidualConvBlock(hidden_channels, dilation=1)

        gate_channels = hidden_channels + prior_hidden + 1
        self.prior_gate = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.detail_head = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_inputs = torch.cat([inputs[:, :VISUAL_PREFIX_CHANNELS], inputs[:, VISUAL_PREFIX_CHANNELS + OPTIONAL_PRIOR_CHANNELS :]], dim=1)
        prior_inputs = inputs[:, VISUAL_PREFIX_CHANNELS : VISUAL_PREFIX_CHANNELS + OPTIONAL_PRIOR_CHANNELS]

        visual_stem = self.visual_stem(visual_inputs)
        visual_enc1 = self.visual_enc1(visual_stem)
        visual_enc2 = self.visual_enc2(F.silu(self.visual_down1(visual_enc1), inplace=True))
        visual_enc3 = self.visual_enc3(F.silu(self.visual_down2(visual_enc2), inplace=True))

        prior_stem = self.prior_stem(prior_inputs)
        prior_enc1 = self.prior_enc1(prior_stem)
        prior_enc2 = self.prior_enc2(F.silu(self.prior_down1(prior_enc1), inplace=True))
        prior_enc3 = self.prior_enc3(F.silu(self.prior_down2(prior_enc2), inplace=True))

        bottleneck = self.bottleneck_fuse(torch.cat([visual_enc3, prior_enc3], dim=1))

        coarse_height_17 = self.coarse_head(bottleneck)
        mid_delta_65 = self.mid_head(bottleneck)

        up2 = F.interpolate(bottleneck, size=visual_enc2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(torch.cat([up2, visual_enc2, prior_enc2], dim=1))
        up2 = self.dec2(up2)

        up1 = F.interpolate(up2, size=visual_enc1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat([up1, visual_enc1, prior_enc1], dim=1))
        up1 = self.dec1(up1)

        prior_height_257 = prior_inputs[:, :1]
        gate_features = torch.cat([up1, prior_stem, prior_height_257], dim=1)
        prior_gate = torch.sigmoid(self.prior_gate(gate_features))
        detail_delta_257 = self.detail_head(gate_features)
        return coarse_height_17, mid_delta_65, detail_delta_257, prior_gate


def build_v10_predictions(
    *,
    coarse_height_17: torch.Tensor,
    mid_delta_65: torch.Tensor,
    detail_delta_257: torch.Tensor,
    prior_height_257: torch.Tensor,
    prior_present_257: torch.Tensor,
    prior_quality_257: torch.Tensor,
    prior_gate: torch.Tensor,
    residual_scale: float,
    height_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coarse_65 = F.interpolate(coarse_height_17, size=(65, 65), mode="bilinear", align_corners=True)
    mid_height_65 = coarse_65 + (mid_delta_65 * (residual_scale / height_scale))
    mid_257 = F.interpolate(mid_height_65, size=(257, 257), mode="bilinear", align_corners=True)

    effective_gate = prior_gate * prior_present_257 * prior_quality_257
    fused_mid_257 = mid_257 + (effective_gate * (prior_height_257 - mid_257))
    full_height_257 = fused_mid_257 + (detail_delta_257 * (residual_scale / height_scale))
    return coarse_height_17, mid_height_65, fused_mid_257, full_height_257, effective_gate


def reduce_samplewise_mean(value: torch.Tensor) -> torch.Tensor:
    return v9.reduce_samplewise_mean(value)


def compute_v10_loss(
    *,
    coarse_height_17: torch.Tensor,
    mid_delta_65: torch.Tensor,
    detail_delta_257: torch.Tensor,
    prior_gate: torch.Tensor,
    batch: dict[str, torch.Tensor],
    residual_scale: float,
    height_scale: float,
    mid_l1_weight: float,
    coarse_l1_weight: float,
    gradient_weight: float,
    detail_residual_weight: float,
    gate_suppression_weight: float,
    apply_sample_weight: bool,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, dict[str, torch.Tensor]]:
    coarse_pred_17, mid_pred_65, fused_mid_257, full_pred_257, effective_gate = build_v10_predictions(
        coarse_height_17=coarse_height_17,
        mid_delta_65=mid_delta_65,
        detail_delta_257=detail_delta_257,
        prior_height_257=batch["prior_height_257"],
        prior_present_257=batch["prior_present_257"],
        prior_quality_257=batch["prior_quality_257"],
        prior_gate=prior_gate,
        residual_scale=residual_scale,
        height_scale=height_scale,
    )

    full_l1_per_sample = reduce_samplewise_mean(full_pred_257 - batch["target_height_257"])
    mid_l1_per_sample = reduce_samplewise_mean(mid_pred_65 - batch["target_height_65"])
    coarse_l1_per_sample = reduce_samplewise_mean(coarse_pred_17 - batch["target_height_17"])

    pred_dx = full_pred_257[:, :, :, 1:] - full_pred_257[:, :, :, :-1]
    pred_dy = full_pred_257[:, :, 1:, :] - full_pred_257[:, :, :-1, :]
    target_dx = batch["target_height_257"][:, :, :, 1:] - batch["target_height_257"][:, :, :, :-1]
    target_dy = batch["target_height_257"][:, :, 1:, :] - batch["target_height_257"][:, :, :-1, :]
    gradient_loss_x_per_sample = reduce_samplewise_mean(pred_dx - target_dx)
    gradient_loss_y_per_sample = reduce_samplewise_mean(pred_dy - target_dy)
    gradient_loss_per_sample = gradient_loss_x_per_sample + gradient_loss_y_per_sample

    detail_pred_residual = full_pred_257 - fused_mid_257
    detail_residual_per_sample = reduce_samplewise_mean(detail_pred_residual - batch["target_detail_residual_257"])

    gate_penalty_per_sample = reduce_samplewise_mean(effective_gate * (1.0 - batch["prior_quality_257"]))

    total_loss_per_sample = (
        full_l1_per_sample
        + (mid_l1_weight * mid_l1_per_sample)
        + (coarse_l1_weight * coarse_l1_per_sample)
        + (gradient_weight * gradient_loss_per_sample)
        + (detail_residual_weight * detail_residual_per_sample)
        + (gate_suppression_weight * gate_penalty_per_sample)
    )
    sample_weight = batch.get("sample_weight") if apply_sample_weight else None
    if sample_weight is not None:
        total_loss = (total_loss_per_sample * sample_weight.reshape(-1)).mean()
    else:
        total_loss = total_loss_per_sample.mean()

    components = {
        "full_l1": float(full_l1_per_sample.mean().item()),
        "mid_l1": float(mid_l1_per_sample.mean().item()),
        "coarse_l1": float(coarse_l1_per_sample.mean().item()),
        "gradient": float(gradient_loss_per_sample.mean().item()),
        "detail_residual": float(detail_residual_per_sample.mean().item()),
        "gate_penalty": float(gate_penalty_per_sample.mean().item()),
    }
    predictions = {
        "coarse_height_17": coarse_pred_17,
        "mid_height_65": mid_pred_65,
        "full_height_257": full_pred_257,
        "effective_gate": effective_gate,
    }
    return total_loss, components, total_loss_per_sample.detach(), predictions


def run_epoch(
    *,
    model: V10TerrainModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    residual_scale: float,
    height_scale: float,
    channels_last: bool,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float], float, dict[str, float]]:
    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    sample_count = 0
    aggregate_components = {
        "full_l1": 0.0,
        "mid_l1": 0.0,
        "coarse_l1": 0.0,
        "gradient": 0.0,
        "detail_residual": 0.0,
        "gate_penalty": 0.0,
    }
    observed_sample_losses: dict[str, float] = {}
    epoch_start = time.perf_counter()

    for batch in loader:
        sample_keys = list(batch["sample_key"])
        device_batch = v9.move_batch_to_device(batch, device, channels_last)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else nullcontext()):
            coarse_height_17, mid_delta_65, detail_delta_257, prior_gate = model(device_batch["inputs"])
            loss, components, sample_losses, _ = compute_v10_loss(
                coarse_height_17=coarse_height_17,
                mid_delta_65=mid_delta_65,
                detail_delta_257=detail_delta_257,
                prior_gate=prior_gate,
                batch=device_batch,
                residual_scale=residual_scale,
                height_scale=height_scale,
                mid_l1_weight=args.mid_l1_weight,
                coarse_l1_weight=args.coarse_l1_weight,
                gradient_weight=args.gradient_weight,
                detail_residual_weight=args.detail_residual_weight,
                gate_suppression_weight=args.gate_suppression_weight,
                apply_sample_weight=is_training,
            )

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = len(sample_keys)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        for key, value in components.items():
            aggregate_components[key] += float(value) * batch_size

        for sample_key, sample_loss in zip(sample_keys, sample_losses.tolist()):
            observed_sample_losses[str(sample_key)] = float(sample_loss)

    elapsed = max(time.perf_counter() - epoch_start, 1e-6)
    mean_loss = total_loss / max(sample_count, 1)
    mean_components = {key: value / max(sample_count, 1) for key, value in aggregate_components.items()}
    samples_per_second = sample_count / elapsed
    return mean_loss, mean_components, samples_per_second, observed_sample_losses


def evaluate_model_on_entries(
    *,
    model: V10TerrainModel,
    dataset: V10NativeDataset,
    device: torch.device,
    amp_dtype: torch.dtype,
    residual_scale: float,
    height_scale: float,
    channels_last: bool,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    model_was_training = model.training
    model.eval()

    abs_sum = 0.0
    sq_sum = 0.0
    total_pixels = 0
    tile_mae_sum = 0.0
    tile_count = 0

    with torch.no_grad():
        for batch in loader:
            device_batch = v9.move_batch_to_device(batch, device, channels_last)
            with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else nullcontext()):
                coarse_height_17, mid_delta_65, detail_delta_257, prior_gate = model(device_batch["inputs"])
                _, _, _, full_height_257, _ = build_v10_predictions(
                    coarse_height_17=coarse_height_17,
                    mid_delta_65=mid_delta_65,
                    detail_delta_257=detail_delta_257,
                    prior_height_257=device_batch["prior_height_257"],
                    prior_present_257=device_batch["prior_present_257"],
                    prior_quality_257=device_batch["prior_quality_257"],
                    prior_gate=prior_gate,
                    residual_scale=residual_scale,
                    height_scale=height_scale,
                )

            prediction = (full_height_257.detach().cpu().numpy() * height_scale).astype(np.float32)
            target = (device_batch["target_height_257"].detach().cpu().numpy() * height_scale).astype(np.float32)
            error = prediction - target
            abs_error = np.abs(error)

            abs_sum += float(abs_error.sum())
            sq_sum += float((error ** 2).sum())
            total_pixels += int(target.size)
            tile_mae_sum += float(abs_error.reshape(abs_error.shape[0], -1).mean(axis=1).sum())
            tile_count += int(abs_error.shape[0])

    model.train(model_was_training)
    return {
        "tile_count": float(tile_count),
        "global_mae": abs_sum / max(total_pixels, 1),
        "global_rmse": math.sqrt(sq_sum / max(total_pixels, 1)),
        "mean_tile_mae": tile_mae_sum / max(tile_count, 1),
    }


def _height_to_rgb(height: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return v9._height_to_rgb(height, min_value, max_value)


def _error_to_rgb(predicted: np.ndarray, target: np.ndarray, height_scale: float) -> np.ndarray:
    return v9._error_to_rgb(predicted, target, height_scale)


def _single_channel_to_rgb(channel: np.ndarray, min_value: float | None = None, max_value: float | None = None) -> np.ndarray:
    return v9._single_channel_to_rgb(channel, min_value, max_value)


def export_preview_images(
    *,
    model: V10TerrainModel,
    dataset: V10NativeDataset,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    height_scale: float,
    residual_scale: float,
    preview_count: int,
    channels_last: bool,
    preview_seed: int,
    epoch: int,
) -> None:
    if len(dataset) == 0 or preview_count <= 0:
        return

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for existing in preview_dir.glob("*.png"):
        existing.unlink()

    ordered = list(range(len(dataset)))
    rng = random.Random(preview_seed)
    rng.shuffle(ordered)
    selected_indices = sorted(ordered[: min(preview_count, len(dataset))])

    readme = (
        "Preview layout\n"
        "1. Minimap RGB input\n"
        "2. Optional prior height input for this preview mode\n"
        "3. Predicted full 257x257 terrain\n"
        "4. Ground-truth full 257x257 terrain\n"
        "5. Absolute error heatmap\n"
        "6. Learned prior gate heatmap\n"
        "These previews are generated from the no-WDL validation path so success here means the model is learning without depending on WDL.\n"
        f"Source epoch: {epoch}.\n"
    )
    v9.write_text(preview_dir / "README.txt", readme)

    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in selected_indices:
            batch = dataset[index]
            preview_batch: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    preview_batch[key] = value.unsqueeze(0)
                else:
                    preview_batch[key] = [value]

            device_batch = v9.move_batch_to_device(preview_batch, device, channels_last)
            with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else nullcontext()):
                coarse_height_17, mid_delta_65, detail_delta_257, prior_gate = model(device_batch["inputs"])
                _, _, _, full_height_257, effective_gate = build_v10_predictions(
                    coarse_height_17=coarse_height_17,
                    mid_delta_65=mid_delta_65,
                    detail_delta_257=detail_delta_257,
                    prior_height_257=device_batch["prior_height_257"],
                    prior_present_257=device_batch["prior_present_257"],
                    prior_quality_257=device_batch["prior_quality_257"],
                    prior_gate=prior_gate,
                    residual_scale=residual_scale,
                    height_scale=height_scale,
                )

            minimap_rgb_np = (batch["preview_minimap_rgb"].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            prior_np = batch["prior_height_257"].squeeze(0).cpu().numpy()
            full_target_np = batch["target_height_257"].squeeze(0).cpu().numpy()
            full_pred_np = full_height_257.squeeze(0).squeeze(0).detach().cpu().numpy()
            gate_np = effective_gate.squeeze(0).squeeze(0).detach().cpu().numpy()

            h_min = float(min(full_target_np.min(), full_pred_np.min(), prior_np.min(initial=0.0)))
            h_max = float(max(full_target_np.max(), full_pred_np.max(), prior_np.max(initial=0.0)))
            panel = np.concatenate(
                [
                    minimap_rgb_np,
                    _height_to_rgb(prior_np, h_min, h_max),
                    _height_to_rgb(full_pred_np, h_min, h_max),
                    _height_to_rgb(full_target_np, h_min, h_max),
                    _error_to_rgb(full_pred_np, full_target_np, height_scale),
                    _single_channel_to_rgb(gate_np, 0.0, 1.0),
                ],
                axis=1,
            )
            tile_name = str(batch["tile_name"])
            Image.fromarray(panel).save(preview_dir / f"preview_{tile_name}_{epoch:04d}.png")
    model.train(model_was_training)


def build_train_loader(
    dataset: V10NativeDataset,
    entries: Sequence[v9.V9SampleEntry],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    sample_loss_ema: dict[str, float],
) -> DataLoader:
    dataset.set_epoch(epoch)
    sampler = v9.OrderedIndexSampler(
        v9.build_epoch_training_order(
            entries,
            epoch=epoch,
            seed=args.seed,
            sampler_mode=args.train_sampler,
            hard_replay_fraction=args.hard_replay_fraction,
            hard_replay_warmup_epochs=args.hard_replay_warmup_epochs,
            sample_loss_ema=sample_loss_ema,
            detail_focus_active=v9.is_detail_focus_epoch(epoch, args, getattr(args, "current_stall", 0)),
            detail_focus_top_fraction=args.detail_focus_top_fraction,
        )
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.train_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor if args.train_workers > 0 else None,
    )


def build_eval_loader(dataset: V10NativeDataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor if args.val_workers > 0 else None,
    )


def build_checkpoint(
    *,
    args: argparse.Namespace,
    epoch: int,
    model: V10TerrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    history: list[dict[str, Any]],
    best_val_loss: float,
    best_epoch: int,
    epochs_since_best: int,
) -> dict[str, Any]:
    return {
        "schema_version": "v10-train-run.v1",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": list(history),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_since_best": int(epochs_since_best),
        "config": vars(args),
    }


def load_checkpoint(
    *,
    checkpoint_path: Path,
    model: V10TerrainModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return {
        "history": list(checkpoint.get("history", [])),
        "start_epoch": int(checkpoint.get("epoch", 0)),
        "best_val_loss": float(checkpoint.get("best_val_loss", math.inf)),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "epochs_since_best": int(checkpoint.get("epochs_since_best", 0)),
    }


def train_single_run(
    *,
    selected_entries: Sequence[v9.V9SampleEntry],
    dev_eval_entries: Sequence[v9.V9SampleEntry],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    v9.seed_everything(args.seed)

    if len(selected_entries) < 2:
        raise SystemExit("Need at least 2 accepted samples to train and validate.")

    train_indices, val_indices = v9.split_grouped_indices(selected_entries, args.val_fraction, args.seed, args.group_block_size)
    train_entries = [selected_entries[index] for index in train_indices]
    val_entries = [selected_entries[index] for index in val_indices]

    print("\n=== V10 Training ===")
    print(f"  train_workers={args.train_workers} | val_workers={args.val_workers} | channels_last={args.channels_last} | compile={args.use_compile}")
    print(f"  prior mix: no_wdl={args.prior_no_wdl_prob:.2f} real={args.prior_real_wdl_prob:.2f} corrupt={args.prior_corrupt_wdl_prob:.2f}")
    train_weighting_summary = summarize_entry_weighting(train_entries, args)
    print(
        "  reward weighting: "
        f"blank_like={train_weighting_summary['blank_like_count']}/{train_weighting_summary['count']} "
        f"low_signal={train_weighting_summary['low_signal_count']}/{train_weighting_summary['count']} "
        f"weight_range={train_weighting_summary['min_sample_weight']:.2f}-{train_weighting_summary['max_sample_weight']:.2f}"
    )

    preload_entries = list(selected_entries)
    preload_ids = {entry.sample_key for entry in preload_entries}
    for entry in dev_eval_entries:
        if entry.sample_key not in preload_ids:
            preload_entries.append(entry)
            preload_ids.add(entry.sample_key)

    arrays_cache = v9.preload_arrays_cache(preload_entries)

    train_dataset = V10NativeDataset(
        train_entries,
        arrays_cache,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
        training=True,
        prior_mode="mixed",
        seed=args.seed,
        prior_no_wdl_prob=args.prior_no_wdl_prob,
        prior_real_wdl_prob=args.prior_real_wdl_prob,
        prior_corrupt_wdl_prob=args.prior_corrupt_wdl_prob,
        corrupt_shift_max=args.corrupt_shift_max,
        corrupt_noise_std=args.corrupt_noise_std,
        sample_weight_args=args,
    )
    val_dataset_no_wdl = V10NativeDataset(
        val_entries,
        arrays_cache,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
        training=False,
        prior_mode="no_wdl",
        seed=args.seed,
        prior_no_wdl_prob=args.prior_no_wdl_prob,
        prior_real_wdl_prob=args.prior_real_wdl_prob,
        prior_corrupt_wdl_prob=args.prior_corrupt_wdl_prob,
        corrupt_shift_max=args.corrupt_shift_max,
        corrupt_noise_std=args.corrupt_noise_std,
        sample_weight_args=args,
    )
    val_dataset_real_wdl = V10NativeDataset(
        val_entries,
        arrays_cache,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
        training=False,
        prior_mode="real_wdl",
        seed=args.seed,
        prior_no_wdl_prob=args.prior_no_wdl_prob,
        prior_real_wdl_prob=args.prior_real_wdl_prob,
        prior_corrupt_wdl_prob=args.prior_corrupt_wdl_prob,
        corrupt_shift_max=args.corrupt_shift_max,
        corrupt_noise_std=args.corrupt_noise_std,
        sample_weight_args=args,
    )
    dev_dataset_no_wdl = V10NativeDataset(
        dev_eval_entries,
        arrays_cache,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
        training=False,
        prior_mode="no_wdl",
        seed=args.seed,
        prior_no_wdl_prob=args.prior_no_wdl_prob,
        prior_real_wdl_prob=args.prior_real_wdl_prob,
        prior_corrupt_wdl_prob=args.prior_corrupt_wdl_prob,
        corrupt_shift_max=args.corrupt_shift_max,
        corrupt_noise_std=args.corrupt_noise_std,
        sample_weight_args=args,
    )
    dev_dataset_real_wdl = V10NativeDataset(
        dev_eval_entries,
        arrays_cache,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
        training=False,
        prior_mode="real_wdl",
        seed=args.seed,
        prior_no_wdl_prob=args.prior_no_wdl_prob,
        prior_real_wdl_prob=args.prior_real_wdl_prob,
        prior_corrupt_wdl_prob=args.prior_corrupt_wdl_prob,
        corrupt_shift_max=args.corrupt_shift_max,
        corrupt_noise_std=args.corrupt_noise_std,
        sample_weight_args=args,
    )
    dev_dataset_corrupt_wdl = V10NativeDataset(
        dev_eval_entries,
        arrays_cache,
        height_scale=args.height_scale,
        residual_scale=args.residual_scale,
        include_brush_mask=not args.disable_brush_mask,
        training=False,
        prior_mode="corrupt_wdl",
        seed=args.seed,
        prior_no_wdl_prob=args.prior_no_wdl_prob,
        prior_real_wdl_prob=args.prior_real_wdl_prob,
        prior_corrupt_wdl_prob=args.prior_corrupt_wdl_prob,
        corrupt_shift_max=args.corrupt_shift_max,
        corrupt_noise_std=args.corrupt_noise_std,
        sample_weight_args=args,
    )

    val_loader_no_wdl = build_eval_loader(val_dataset_no_wdl, args, device)
    val_loader_real_wdl = build_eval_loader(val_dataset_real_wdl, args, device)

    model = V10TerrainModel(hidden_channels=args.hidden_channels, blocks_per_stage=args.blocks_per_stage).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    compile_active = False
    if args.use_compile and hasattr(torch, "compile"):
        compile_disable_reason = v9.get_torch_compile_disable_reason(device)
        if compile_disable_reason is not None:
            print(f"  {compile_disable_reason}")
        else:
            print("  Compiling model with torch.compile...")
            compile_start = time.perf_counter()
            try:
                model = torch.compile(model)
                compile_active = True
                print(f"  Compile done in {time.perf_counter() - compile_start:.1f}s")
            except Exception as ex:
                print(f"  torch.compile setup failed ({ex.__class__.__name__}: {ex}); continuing without compile.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        min_lr=args.min_learning_rate,
    )

    history: list[dict[str, Any]] = []
    best_val_loss = math.inf
    best_epoch = 0
    epochs_since_best = 0
    start_epoch = 0
    resumed_from: str | None = None
    sample_loss_ema: dict[str, float] = {}
    stop_reason = "completed_requested_epochs"

    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint does not exist: {resume_path}")
    elif last_checkpoint_path.exists():
        resume_path = last_checkpoint_path

    if resume_path is not None:
        resume_state = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        history = resume_state["history"]
        start_epoch = resume_state["start_epoch"]
        best_val_loss = resume_state["best_val_loss"]
        best_epoch = resume_state["best_epoch"]
        epochs_since_best = resume_state["epochs_since_best"]
        resumed_from = str(resume_path)

    print(
        f"Training V10 | epochs={args.epochs} | batch={args.batch_size} | lr={args.learning_rate:.2e} | "
        f"channels_last={args.channels_last} | compile={compile_active}"
    )
    print(f"Dataset split | train_samples={len(train_entries)} | val_samples={len(val_entries)}")
    print(describe_v10_input_stack(args))
    if resumed_from is not None:
        print(f"Resuming from checkpoint | path={resumed_from} | start_epoch={start_epoch + 1} | best {best_val_loss:.6f}@{best_epoch}")

    if start_epoch >= args.epochs:
        print(
            f"Resume checkpoint is already at epoch {start_epoch}, which meets or exceeds requested epochs={args.epochs}; skipping training."
        )
        stop_reason = "resume_checkpoint_already_complete"

    for epoch in range(start_epoch + 1, args.epochs + 1):
        args.current_stall = epochs_since_best
        train_loader = build_train_loader(train_dataset, train_entries, args, device, epoch, sample_loss_ema)

        train_loss, train_components, train_sps, train_sample_losses = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_dtype=amp_dtype,
            residual_scale=args.residual_scale,
            height_scale=args.height_scale,
            channels_last=args.channels_last,
            args=args,
        )
        v9.update_sample_loss_ema(sample_loss_ema, train_sample_losses, args.hard_replay_ema_decay)

        val_no_wdl_loss, val_no_wdl_components, val_no_wdl_sps, _ = run_epoch(
            model=model,
            loader=val_loader_no_wdl,
            optimizer=None,
            device=device,
            amp_dtype=amp_dtype,
            residual_scale=args.residual_scale,
            height_scale=args.height_scale,
            channels_last=args.channels_last,
            args=args,
        )
        val_real_wdl_loss, val_real_wdl_components, val_real_wdl_sps, _ = run_epoch(
            model=model,
            loader=val_loader_real_wdl,
            optimizer=None,
            device=device,
            amp_dtype=amp_dtype,
            residual_scale=args.residual_scale,
            height_scale=args.height_scale,
            channels_last=args.channels_last,
            args=args,
        )

        dev_eval_metrics: dict[str, dict[str, float]] | None = None
        if dev_eval_entries and args.dev_eval_every > 0 and epoch % args.dev_eval_every == 0:
            dev_eval_metrics = {
                "no_wdl": evaluate_model_on_entries(
                    model=model,
                    dataset=dev_dataset_no_wdl,
                    device=device,
                    amp_dtype=amp_dtype,
                    residual_scale=args.residual_scale,
                    height_scale=args.height_scale,
                    channels_last=args.channels_last,
                    batch_size=args.batch_size,
                    num_workers=args.val_workers,
                ),
                "real_wdl": evaluate_model_on_entries(
                    model=model,
                    dataset=dev_dataset_real_wdl,
                    device=device,
                    amp_dtype=amp_dtype,
                    residual_scale=args.residual_scale,
                    height_scale=args.height_scale,
                    channels_last=args.channels_last,
                    batch_size=args.batch_size,
                    num_workers=args.val_workers,
                ),
                "corrupt_wdl": evaluate_model_on_entries(
                    model=model,
                    dataset=dev_dataset_corrupt_wdl,
                    device=device,
                    amp_dtype=amp_dtype,
                    residual_scale=args.residual_scale,
                    height_scale=args.height_scale,
                    channels_last=args.channels_last,
                    batch_size=args.batch_size,
                    num_workers=args.val_workers,
                ),
            }
            dev_eval_metrics["prior_lift"] = {
                "real_minus_no_wdl_mae": dev_eval_metrics["no_wdl"]["global_mae"] - dev_eval_metrics["real_wdl"]["global_mae"],
                "corrupt_minus_no_wdl_mae": dev_eval_metrics["no_wdl"]["global_mae"] - dev_eval_metrics["corrupt_wdl"]["global_mae"],
            }

        history_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_no_wdl_loss,
            "val_real_wdl_loss": val_real_wdl_loss,
            "train_components": train_components,
            "val_components": val_no_wdl_components,
            "val_real_wdl_components": val_real_wdl_components,
            "train_samples_per_second": train_sps,
            "val_samples_per_second": val_no_wdl_sps,
            "val_real_wdl_samples_per_second": val_real_wdl_sps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if dev_eval_metrics is not None:
            history_record["dev_eval"] = dev_eval_metrics
        history.append(history_record)

        previous_best = best_val_loss
        is_best = val_no_wdl_loss < best_val_loss
        if is_best:
            best_val_loss = val_no_wdl_loss
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        status = "BEST" if is_best else f"stall={epochs_since_best}"
        print(
            f"epoch {epoch:03d}/{args.epochs} | {status} | train {train_loss:.6f} | val(no_wdl) {val_no_wdl_loss:.6f} | "
            f"val(real_wdl) {val_real_wdl_loss:.6f} | delta {val_no_wdl_loss - previous_best:+.6f} | best {best_val_loss:.6f}@{best_epoch} | lr {current_lr:.2e}"
        )
        print(
            f"  full {val_no_wdl_components['full_l1']:.6f} | mid {val_no_wdl_components['mid_l1']:.6f} | coarse {val_no_wdl_components['coarse_l1']:.6f} | "
            f"grad {val_no_wdl_components['gradient']:.6f} | gate_penalty {val_no_wdl_components['gate_penalty']:.6f} | train_sps {train_sps:.1f}"
        )
        if dev_eval_metrics is not None:
            print(
                f"  dev_eval | no_wdl_mae {dev_eval_metrics['no_wdl']['global_mae']:.6f} | real_wdl_mae {dev_eval_metrics['real_wdl']['global_mae']:.6f} | "
                f"corrupt_wdl_mae {dev_eval_metrics['corrupt_wdl']['global_mae']:.6f} | real_lift {dev_eval_metrics['prior_lift']['real_minus_no_wdl_mae']:.6f}"
            )

        if is_best or (args.preview_every_epochs > 0 and epoch % args.preview_every_epochs == 0):
            export_preview_images(
                model=model,
                dataset=val_dataset_no_wdl,
                output_dir=output_dir,
                device=device,
                amp_dtype=amp_dtype,
                height_scale=args.height_scale,
                residual_scale=args.residual_scale,
                preview_count=args.preview_count,
                channels_last=args.channels_last,
                preview_seed=args.seed + epoch,
                epoch=epoch,
            )
            print(f"  refreshed previews in {output_dir / 'previews'}")

        if is_best:
            torch.save(build_checkpoint(
                args=args,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_since_best=epochs_since_best,
            ), output_dir / "best_model.pt")
            print(f"  saved best checkpoint in {output_dir / 'best_model.pt'}")

        previous_lr = current_lr
        scheduler.step(val_no_wdl_loss)
        new_lr = float(optimizer.param_groups[0]["lr"])
        if new_lr < previous_lr:
            print(f"  learning rate reduced: {previous_lr:.2e} -> {new_lr:.2e}")

        torch.save(
            build_checkpoint(
                args=args,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_since_best=epochs_since_best,
            ),
            last_checkpoint_path,
        )

        if epoch >= args.early_stop_min_epochs and epochs_since_best >= args.early_stop_patience:
            print(
                f"  early stop: no new best no-WDL val loss for {epochs_since_best} epoch(s) after epoch {best_epoch}; best val remained {best_val_loss:.6f}"
            )
            stop_reason = "early_stop_no_new_best"
            break

    run_summary = {
        "schema_version": "v10-train-run.v1",
        "created_at_utc": v9.utc_now_iso(),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "samples": len(selected_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "history": history,
        "final_epoch": history[-1]["epoch"] if history else 0,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "config": vars(args),
        "reward_weighting": train_weighting_summary,
        "preview_count": args.preview_count,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "feature_contract": build_v10_feature_contract(args),
        "resumed_from": resumed_from,
    }
    v9.write_json(output_dir / "run_summary.json", run_summary)
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the v10 no-WDL terrain model.")
    parser.add_argument("cache_manifest", help="Path to v9_tensor_cache_manifest.json produced by build_v9_native_tensor_cache.py.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for reports, logs, and checkpoints.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on accepted samples before splitting.")
    parser.add_argument("--subset", type=int, default=None, help="Optional random subset of sane samples before curation.")
    parser.add_argument("--subset-seed", type=int, default=None, help="Subset selection seed. Defaults to --seed.")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=v9.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=v9.DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=v9.DEFAULT_SEED)
    parser.add_argument("--height-scale", type=float, default=v9.DEFAULT_HEIGHT_SCALE)
    parser.add_argument("--residual-scale", type=float, default=v9.DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--val-fraction", type=float, default=v9.DEFAULT_VAL_FRACTION)
    parser.add_argument("--hidden-channels", type=int, default=v9.DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument("--blocks-per-stage", type=int, default=v9.DEFAULT_BLOCKS_PER_STAGE)
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default=v9.DEFAULT_AMP_DTYPE)
    parser.add_argument("--train-workers", type=int, default=v9.DEFAULT_TRAIN_WORKERS)
    parser.add_argument("--val-workers", type=int, default=v9.DEFAULT_VAL_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=v9.DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--channels-last", type=lambda x: x.lower() in ("true", "1", "yes"), default=v9.DEFAULT_CHANNELS_LAST)
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--use-compile", type=lambda x: x.lower() in ("true", "1", "yes"), default=v9.DEFAULT_USE_COMPILE)
    parser.add_argument("--preview-count", type=int, default=v9.DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--preview-every-epochs", type=int, default=v9.DEFAULT_PREVIEW_EVERY_EPOCHS)
    parser.add_argument("--disable-brush-mask", action="store_true")
    parser.add_argument("--lr-plateau-patience", type=int, default=v9.DEFAULT_LR_PLATEAU_PATIENCE)
    parser.add_argument("--lr-plateau-factor", type=float, default=v9.DEFAULT_LR_PLATEAU_FACTOR)
    parser.add_argument("--min-learning-rate", type=float, default=v9.DEFAULT_MIN_LEARNING_RATE)
    parser.add_argument("--early-stop-patience", type=int, default=72)
    parser.add_argument("--early-stop-min-epochs", type=int, default=72)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dev-eval-cache-manifest", type=str, default=None)
    parser.add_argument("--dev-eval-limit", type=int, default=None)
    parser.add_argument("--dev-eval-every", type=int, default=v9.DEFAULT_DEV_EVAL_EVERY)
    parser.add_argument("--train-sampler", default=v9.DEFAULT_TRAIN_SAMPLER)
    parser.add_argument("--hard-replay-fraction", type=float, default=v9.DEFAULT_HARD_REPLAY_FRACTION)
    parser.add_argument("--hard-replay-warmup-epochs", type=int, default=v9.DEFAULT_HARD_REPLAY_WARMUP_EPOCHS)
    parser.add_argument("--hard-replay-ema-decay", type=float, default=v9.DEFAULT_HARD_REPLAY_EMA_DECAY)
    parser.add_argument("--detail-focus-every-epochs", type=int, default=v9.DEFAULT_DETAIL_FOCUS_EVERY_EPOCHS)
    parser.add_argument("--detail-focus-min-epoch", type=int, default=v9.DEFAULT_DETAIL_FOCUS_MIN_EPOCH)
    parser.add_argument("--detail-focus-stall-threshold", type=int, default=v9.DEFAULT_DETAIL_FOCUS_STALL_THRESHOLD)
    parser.add_argument("--detail-focus-top-fraction", type=float, default=v9.DEFAULT_DETAIL_FOCUS_TOP_FRACTION)
    parser.add_argument("--group-block-size", type=int, default=v9.DEFAULT_GROUP_BLOCK_SIZE)
    parser.add_argument("--curation-mode", type=str, default=v9.DEFAULT_CURATION_MODE)
    parser.add_argument("--curation-diversity-block-size", type=int, default=v9.DEFAULT_CURATION_DIVERSITY_BLOCK_SIZE)
    parser.add_argument("--curation-max-per-group", type=int, default=v9.DEFAULT_CURATION_MAX_PER_GROUP)
    parser.add_argument("--target-curated-samples", type=int, default=None)
    parser.add_argument("--require-minimap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-wdl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-height-range", type=float, default=v9.DEFAULT_MIN_HEIGHT_RANGE)
    parser.add_argument("--min-minimap-variance", type=float, default=v9.DEFAULT_MIN_MINIMAP_VARIANCE)
    parser.add_argument("--min-minimap-gradient", type=float, default=v9.DEFAULT_MIN_MINIMAP_GRADIENT)
    parser.add_argument("--max-mean-wdl-delta", type=float, default=v9.DEFAULT_MAX_MEAN_WDL_DELTA)
    parser.add_argument("--max-abs-wdl-delta", type=float, default=v9.DEFAULT_MAX_ABS_WDL_DELTA)

    parser.add_argument("--prior-no-wdl-prob", type=float, default=DEFAULT_PRIOR_NO_WDL_PROB)
    parser.add_argument("--prior-real-wdl-prob", type=float, default=DEFAULT_PRIOR_REAL_WDL_PROB)
    parser.add_argument("--prior-corrupt-wdl-prob", type=float, default=DEFAULT_PRIOR_CORRUPT_WDL_PROB)
    parser.add_argument("--corrupt-shift-max", type=float, default=DEFAULT_CORRUPT_SHIFT_MAX)
    parser.add_argument("--corrupt-noise-std", type=float, default=DEFAULT_CORRUPT_NOISE_STD)

    parser.add_argument("--mid-l1-weight", type=float, default=DEFAULT_MID_L1_WEIGHT)
    parser.add_argument("--coarse-l1-weight", type=float, default=DEFAULT_COARSE_L1_WEIGHT)
    parser.add_argument("--gradient-weight", type=float, default=DEFAULT_GRADIENT_WEIGHT)
    parser.add_argument("--detail-residual-weight", type=float, default=DEFAULT_DETAIL_RESIDUAL_WEIGHT)
    parser.add_argument("--gate-suppression-weight", type=float, default=DEFAULT_GATE_SUPPRESSION_WEIGHT)
    parser.add_argument("--quality-reward", type=float, default=DEFAULT_QUALITY_REWARD)
    parser.add_argument("--low-signal-penalty", type=float, default=DEFAULT_LOW_SIGNAL_PENALTY)
    parser.add_argument("--blank-tile-penalty", type=float, default=DEFAULT_BLANK_TILE_PENALTY)

    return parser


def curate_entries(args: argparse.Namespace, cache_manifest: Path) -> tuple[list[v9.V9SampleEntry], list[v9.V9SampleEntry]]:
    print(f"\nLoading cache manifest: {cache_manifest}")
    all_entries = v9.load_cache_manifest(cache_manifest)
    print(f"  Loaded {len(all_entries)} manifest entries")

    print(f"  Auditing entries (require_minimap={args.require_minimap}, require_wdl={args.require_wdl})...")
    audited = [
        v9.audit_entry(
            entry,
            require_wdl=args.require_wdl,
            require_minimap=args.require_minimap,
            min_height_range=args.min_height_range,
            min_minimap_variance=args.min_minimap_variance,
            min_minimap_gradient=args.min_minimap_gradient,
            max_mean_wdl_delta=args.max_mean_wdl_delta,
            max_abs_wdl_delta=args.max_abs_wdl_delta,
        )
        for entry in all_entries
    ]
    accepted = [entry for entry in audited if entry.accepted]
    print(f"  Audited {len(audited)} entries: {len(accepted)} sane")
    if not accepted:
        raise SystemExit("No sane samples after audit. Check cache manifest and thresholds.")

    if args.subset is not None and args.subset > 0 and len(accepted) > args.subset:
        subset_seed = args.subset_seed if args.subset_seed is not None else args.seed
        rng = random.Random(subset_seed)
        accepted = [accepted[index] for index in sorted(rng.sample(range(len(accepted)), args.subset))]
        print(f"  Applied random subset: {len(accepted)} audited samples")

    selected_entries = v9.select_curated_entries(
        accepted,
        limit=args.target_curated_samples or args.limit,
        curation_mode=args.curation_mode,
        diversity_block_size=args.curation_diversity_block_size,
        max_per_group=args.curation_max_per_group,
    )
    print(f"  Selected {len(selected_entries)} curated sample(s) for training")

    dev_eval_entries: list[v9.V9SampleEntry] = []
    if args.dev_eval_cache_manifest:
        dev_eval_manifest = Path(args.dev_eval_cache_manifest)
        dev_eval_all = v9.load_cache_manifest(dev_eval_manifest)
        dev_eval_entries = v9.select_diverse_eval_entries(dev_eval_all, args.dev_eval_limit, v9.DEFAULT_DEV_EVAL_BLOCK_SIZE, args.seed)
        print(f"  Loaded {len(dev_eval_entries)} dev-eval entries from {dev_eval_manifest}")

    return selected_entries, dev_eval_entries


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.no_channels_last:
        args.channels_last = False
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")

    cache_manifest = Path(args.cache_manifest)
    if not cache_manifest.exists():
        raise SystemExit(f"Cache manifest not found: {cache_manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_entries, dev_eval_entries = curate_entries(args, cache_manifest)
    run_summary = train_single_run(
        selected_entries=selected_entries,
        dev_eval_entries=dev_eval_entries,
        args=args,
        output_dir=output_dir,
        device=device,
        amp_dtype=amp_dtype,
    )

    print(
        f"\nFinished V10 training run | best no-WDL val {run_summary['best_val_loss']:.6f}@{run_summary['best_epoch']} | stop_reason={run_summary['stop_reason']}"
    )


if __name__ == "__main__":
    main()