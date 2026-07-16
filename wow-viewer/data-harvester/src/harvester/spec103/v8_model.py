"""V50TerrainRefiner — current lightweight terrain refinement model.

Same pinned contract as v7 (research-v7-contract.md): 13-channel input in the same order,
WDL-trestle residual on channel 6, 2-3 output channels + 4-value bounds head, output
interpolated to `output_size`. Drop-in for `v7_losses.combined_loss`, the trainer, the
inference script, and the label-free harness — nothing downstream changes.

What changed vs the legacy v7/v8 lineage (and why): v7 spends 117.06M params, 73% of them at 8x8-16x16
(1024->2048 bottleneck + dec5), on a smooth low-frequency height field that already
receives a 17x17 prior as its trestle base. V8 is a ConvNeXt-V2-style U-Net
(arXiv 2301.00808: 7x7 depthwise conv + pointwise MLP + GRN) with capped widths
(32-64-128-256-384), a pixel-shuffle decoder (checkerboard-free), and a cheap
global-context mixer + pooled bounds head at the deepest stage. ~6M params /
~18 GFLOPs @256 — about 19x fewer params and 6x fewer FLOPs than v7, so local
training tells you whether a run is sound in minutes, not days.

Deterministic feedforward, plain PyTorch only (no custom kernels — Windows and RunPod
run the same code).  The filename remains for source-history compatibility;
all current artifacts identify this implementation as v50.1.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v7_model import (
    DEFAULT_DETAIL_RESIDUAL_SCALE,
    DEFAULT_GLOBAL_RESIDUAL_SCALE,
    DEFAULT_OUTPUT_HEAD_MODE,
    DEFAULT_OUTPUT_SIZE,
    MODEL_INPUT_CHANNELS,
    MODEL_OUTPUT_CHANNELS,
)
from harvester.v50_contract import TERRAIN_CHECKPOINT_VARIANT

MODEL_VARIANT_V50_TERRAIN = TERRAIN_CHECKPOINT_VARIANT
# Import-compatible legacy alias. New code must use MODEL_VARIANT_V50_TERRAIN.
MODEL_VARIANT_V8_LEAN = MODEL_VARIANT_V50_TERRAIN
DEFAULT_WIDTHS = (32, 64, 128, 256, 384)
DEFAULT_DEPTHS = (1, 1, 2, 2, 2)


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-V2), channels-last."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """7x7 depthwise (reflect-padded, matching v7's border behavior) + pointwise MLP + GRN."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode="reflect")
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.grn = GRN(expansion * dim)
        self.pwconv2 = nn.Linear(expansion * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(x)))))
        return identity + x.permute(0, 3, 1, 2)


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


def _icnr_init(weight: torch.Tensor, upscale_factor: int = 2) -> torch.Tensor:
    """ICNR init (Aitken et al. 2017, arXiv:1707.02937): each group of upscale_factor**2
    output channels that PixelShuffle maps into one 2x2 output block starts IDENTICAL
    (effectively a nearest-neighbor upsample), instead of independently random. Without
    this, pixel-shuffle decoders reliably show checkerboard/grid artifacts at init that
    training only partially removes — a real, well-documented banding source distinct
    from anything in v7 (v7 upsamples via bilinear + conv, not PixelShuffle)."""
    out_channels, in_channels, kh, kw = weight.shape
    sub_channels = out_channels // (upscale_factor ** 2)
    sub_kernel = torch.empty(sub_channels, in_channels, kh, kw)
    nn.init.trunc_normal_(sub_kernel, std=0.02)
    return sub_kernel.repeat_interleave(upscale_factor ** 2, dim=0)


class PixelShuffleUp(nn.Module):
    """1x1 conv to 4x channels + PixelShuffle(2): checkerboard-free 2x upsampling
    (given ICNR init — see `_icnr_init`; applied by the owning module after construction)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim * 4, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class GlobalContext(nn.Module):
    """Pooled-MLP broadcast at the deepest stage: global receptive field at negligible cost.

    Carries the absolute-height context v7 bought with its 2048-wide 8x8 bottleneck —
    needed most on prior-dropout tiles where the trestle base is a flat 0.5 fill.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.norm(x).mean(dim=(2, 3))
        g = self.fc2(self.act(self.fc1(g)))
        return x + g[:, :, None, None]


class V50TerrainRefiner(nn.Module):
    """Lean ConvNeXt-V2 U-Net honoring the exact v7 I/O contract.

    forward(inputs[N,13,S,S]) -> (outputs[N,2..3,output_size,output_size], bounds[N,4]),
    with the identical trestle/head/clamp semantics as MultiChannelUNetV7. Input spatial
    size must be divisible by 2^(len(widths)-1) (16 for the default 5 stages).
    """

    def __init__(
        self,
        in_channels: int = MODEL_INPUT_CHANNELS,
        out_channels: int = MODEL_OUTPUT_CHANNELS,
        use_wdl_global_trestle: bool = False,
        global_residual_scale: float = DEFAULT_GLOBAL_RESIDUAL_SCALE,
        use_detail_head: bool = False,
        detail_residual_scale: float = DEFAULT_DETAIL_RESIDUAL_SCALE,
        output_head_mode: str = DEFAULT_OUTPUT_HEAD_MODE,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        widths: Tuple[int, ...] = DEFAULT_WIDTHS,
        depths: Tuple[int, ...] = DEFAULT_DEPTHS,
    ):
        super().__init__()
        if len(widths) != len(depths):
            raise ValueError(f"widths ({len(widths)}) and depths ({len(depths)}) must have equal length")
        self.use_wdl_global_trestle = use_wdl_global_trestle
        self.global_residual_scale = float(global_residual_scale)
        self.use_detail_head = bool(use_detail_head)
        self.detail_residual_scale = float(detail_residual_scale)
        self.output_head_mode = str(output_head_mode).strip().lower()
        self.output_size = int(output_size)
        self.widths = tuple(int(w) for w in widths)

        self.stem = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, padding_mode="reflect")
        self.enc_stages = nn.ModuleList(
            nn.Sequential(*[ConvNeXtV2Block(w) for _ in range(d)]) for w, d in zip(widths, depths)
        )
        self.downs = nn.ModuleList(Downsample(widths[i], widths[i + 1]) for i in range(len(widths) - 1))

        deepest = widths[-1]
        self.global_context = GlobalContext(deepest)
        self.bounds_norm = nn.LayerNorm(deepest, eps=1e-6)
        self.bounds_fc = nn.Sequential(nn.Linear(deepest, 128), nn.GELU(), nn.Linear(128, 4))

        self.ups = nn.ModuleList(PixelShuffleUp(widths[i + 1], widths[i]) for i in reversed(range(len(widths) - 1)))
        self.fuses = nn.ModuleList(nn.Conv2d(2 * widths[i], widths[i], kernel_size=1) for i in reversed(range(len(widths) - 1)))
        self.dec_stages = nn.ModuleList(ConvNeXtV2Block(widths[i]) for i in reversed(range(len(widths) - 1)))

        self.out_conv = nn.Conv2d(widths[0], out_channels, kernel_size=1)

        self.apply(self._init_weights)
        for up in self.ups:
            with torch.no_grad():
                up.conv.weight.copy_(_icnr_init(up.conv.weight, upscale_factor=2))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(inputs)
        skips = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)

        x = self.global_context(x)
        bounds = self.bounds_fc(self.bounds_norm(x.mean(dim=(2, 3))))

        for up, fuse, block, skip in zip(self.ups, self.fuses, self.dec_stages, reversed(skips)):
            x = up(x)
            x = fuse(torch.cat([x, skip], dim=1))
            x = block(x)

        raw_outputs = self.out_conv(x)
        global_output = raw_outputs[:, 0:1]
        local_output = raw_outputs[:, 1:2]

        # Head semantics copied from MultiChannelUNetV7.forward — the trestle residual,
        # clamp behavior, and both output_head_modes are contract, not implementation.
        if self.output_head_mode == "linear_unclamped_train":
            if self.use_wdl_global_trestle and inputs.shape[1] > 6:
                wdl_base = inputs[:, 6:7]
                global_output = wdl_base + global_output * self.global_residual_scale
            if not self.training:
                global_output = torch.clamp(global_output, 0.0, 1.0)
                local_output = torch.clamp(local_output, 0.0, 1.0)
        else:
            if self.use_wdl_global_trestle and inputs.shape[1] > 6:
                wdl_base = inputs[:, 6:7]
                global_delta = torch.tanh(global_output) * self.global_residual_scale
                global_output = torch.clamp(wdl_base + global_delta, 0.0, 1.0)
            else:
                global_output = torch.clamp(global_output, 0.0, 1.0)
            local_output = torch.clamp(local_output, 0.0, 1.0)

        outputs = torch.cat([global_output, local_output], dim=1)
        if self.use_detail_head and raw_outputs.shape[1] > 2:
            if self.output_head_mode == "linear_unclamped_train":
                detail_output = raw_outputs[:, 2:3] * self.detail_residual_scale
            else:
                detail_output = torch.tanh(raw_outputs[:, 2:3]) * self.detail_residual_scale
            outputs = torch.cat([outputs, detail_output], dim=1)
        if outputs.shape[-2:] != (self.output_size, self.output_size):
            outputs = F.interpolate(outputs, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)

        return outputs, bounds


# Existing source/tests may import this historical name.  Its checkpoint value
# and all current command surfaces are v50-only.
V8LeanUNet = V50TerrainRefiner
