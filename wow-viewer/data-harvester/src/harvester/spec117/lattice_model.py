"""Spec 117 US2: the standalone WDL-lattice target contract and predictor.

The WDL lattice is 545 real samples (17x17 outer + 16x16 inner, Spec 108 FR-001) drawn directly
from MCVT vertex data via ``TerrainWdlLattice`` -- unlike ``height_257``, which fills every raster
position (interpolating gaps), the lattice keeps its own per-sample presence mask and a gap is
never fabricated (spec Edge Cases). ``encode_relative_height`` in ``height_relative_model.py``
cannot be reused directly here: it requires a fully finite 2D array and computes min/max over the
whole thing, which is exactly wrong for a target that may have real gaps. This module is the
lattice-specific analogue of that same v112.1 target-contract philosophy (per-tile min-max
normalization, ``RANGE_FLOOR`` floor so near-flat tiles are not amplified), adapted to respect a
per-sample presence mask throughout: normalization, loss, baseline, and row selection all treat an
absent sample as absent, never as zero.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from harvester.spec117.lattice_contract import INNER_DIM, OUTER_DIM, SAMPLE_COUNT

RANGE_FLOOR = 1.0  # same floor concept and value as height_relative_model.RANGE_FLOOR
INPUT_SIZE = 256

_OUTER_COUNT = OUTER_DIM * OUTER_DIM


class LatticeTargetError(ValueError):
    """Raised when a WDL lattice sample set cannot be encoded/decoded/selected as declared."""


def _flatten(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(outer, dtype=np.float64).reshape(-1),
                            np.asarray(inner, dtype=np.float64).reshape(-1)))


def encode_lattice_target(
    outer: np.ndarray,
    inner: np.ndarray,
    outer_present: np.ndarray,
    inner_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(target[545] in [0,1], mask[545] in {0,1}, tile_min, tile_max)``.

    ``tile_min``/``tile_max`` are computed ONLY over present samples -- an absent sample never
    contributes to the normalization range, and its encoded target value is an arbitrary 0.0
    placeholder that the paired mask marks as never-loss-bearing. Raises if a tile has zero present
    samples: callers must exclude such a row via ``select_lattice_rows`` before reaching here,
    never silently degenerate to a fabricated range.
    """
    if outer.shape != (OUTER_DIM, OUTER_DIM) or inner.shape != (INNER_DIM, INNER_DIM):
        raise LatticeTargetError(
            f"expected outer {(OUTER_DIM, OUTER_DIM)} / inner {(INNER_DIM, INNER_DIM)}, "
            f"got outer {outer.shape} / inner {inner.shape}"
        )
    values = _flatten(outer, inner)
    mask = _flatten(
        np.asarray(outer_present, dtype=bool), np.asarray(inner_present, dtype=bool)
    ).astype(np.float64)
    if mask.sum() == 0:
        raise LatticeTargetError("cannot encode a tile with zero present lattice samples")

    present_values = values[mask > 0]
    tile_min = float(present_values.min())
    tile_max = float(present_values.max())
    denominator = max(tile_max - tile_min, RANGE_FLOOR)
    normalized = np.where(mask > 0, np.clip((values - tile_min) / denominator, 0.0, 1.0), 0.0)
    return normalized.astype(np.float32), mask.astype(np.float32), tile_min, tile_max


def decode_lattice_target(normalized: np.ndarray, tile_min: float, tile_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Invert ``encode_lattice_target``; returns ``(outer[17,17], inner[16,16])`` world-unit grids."""
    denominator = max(tile_max - tile_min, RANGE_FLOOR)
    flat = np.asarray(normalized, dtype=np.float64).reshape(-1)
    if flat.size != SAMPLE_COUNT:
        raise LatticeTargetError(f"expected {SAMPLE_COUNT} values, got {flat.size}")
    world = (flat * denominator + tile_min).astype(np.float32)
    return world[:_OUTER_COUNT].reshape(OUTER_DIM, OUTER_DIM), world[_OUTER_COUNT:].reshape(INNER_DIM, INNER_DIM)


def select_lattice_rows(group, rows: list[int]) -> tuple[list[int], int]:
    """Return ``(usable_rows, excluded_count)``: drop rows with zero present lattice samples.

    Spec Edge Cases: "A held-out tile with no exportable lattice at all -- excluded from
    evaluation and counted, not scored as a zero-error or maximum-error case." Requires the store
    to carry ``wdl_outer_present``/``wdl_inner_present``; callers must check for their presence in
    the store before calling this (a store built before Spec 117's catalog amendment has neither).
    """
    usable: list[int] = []
    excluded = 0
    for row in rows:
        outer_present = np.asarray(group["wdl_outer_present"][row])
        inner_present = np.asarray(group["wdl_inner_present"][row])
        if bool(outer_present.any()) or bool(inner_present.any()):
            usable.append(row)
        else:
            excluded += 1
    return usable, excluded


def compute_lattice_tile_mean_baseline(
    targets_and_masks: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Masked MAE of predicting each tile's own mean-of-present-samples for its present samples.

    The lattice analogue of ``height_relative_train.compute_tile_mean_baseline`` (research.md
    D-02): the same "hardest honest reference" concept, adapted because a raw mean over the whole
    545-vector would be corrupted by the arbitrary 0.0 placeholder at absent positions.
    """
    if not targets_and_masks:
        raise LatticeTargetError("cannot compute a baseline over zero validation tiles")
    errors = []
    for target, mask in targets_and_masks:
        present = mask > 0
        if not present.any():
            raise LatticeTargetError("cannot score a tile with zero present lattice samples")
        tile_mean = float(target[present].mean())
        errors.append(float(np.abs(target[present] - tile_mean).mean()))
    return float(np.mean(errors))


def lattice_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked smooth-L1: absent samples never contribute a gradient."""
    per_element = nn.functional.smooth_l1_loss(predicted, target, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_element * mask).sum() / denom


def lattice_gradient_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked L1 on the 2D finite-difference gradients of each lattice grid.

    A loss-only structural term (ported from the V7 height regressor's gradient-consistency
    stack, adapted to the lattice's two regular grids). Pure point-wise smooth-L1 cannot see
    spatial coherence: a model that gets every point's *value* roughly right but scrambles
    their *arrangement* still scores poorly without a term that rewards matching the local
    slope field. A gradient between two adjacent samples is only counted when BOTH are present.
    """
    batch = predicted.shape[0]
    total = predicted.new_zeros(())
    weight = 0
    for dim, lo, hi in ((OUTER_DIM, 0, _OUTER_COUNT), (INNER_DIM, _OUTER_COUNT, SAMPLE_COUNT)):
        p = predicted[:, lo:hi].reshape(batch, dim, dim)
        t = target[:, lo:hi].reshape(batch, dim, dim)
        m = mask[:, lo:hi].reshape(batch, dim, dim)
        # x direction (last axis): valid only where both neighbours are present.
        mx = m[:, :, :-1] * m[:, :, 1:]
        dx = torch.abs((p[:, :, 1:] - p[:, :, :-1]) - (t[:, :, 1:] - t[:, :, :-1]))
        total = total + (dx * mx).sum() / mx.sum().clamp_min(1.0)
        # y direction.
        my = m[:, :-1, :] * m[:, 1:, :]
        dy = torch.abs((p[:, 1:, :] - p[:, :-1, :]) - (t[:, 1:, :] - t[:, :-1, :]))
        total = total + (dy * my).sum() / my.sum().clamp_min(1.0)
        weight += 2
    return total / max(weight, 1)


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Conv3x3 + GroupNorm + SiLU -- the same block HeightRelativeNet (direct_cnn_v112) uses."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


def _double_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Two stacked Conv3x3+GroupNorm+SiLU -- a full U-Net double-conv level (v4).

    v3 used a single conv per level (the lean ``direct_cnn_v112`` primitive). v4 stacks a second
    same-resolution conv at each level: the first conv carries the stride (down/same-sample), the
    second refines at the output resolution. This roughly doubles the per-level depth and, together
    with the wider ``base=64`` default, makes the network decisively over-capacity for a 679-tile
    corpus -- so that a "does not beat tile-mean" verdict can no longer be attributed to too few
    parameters. The structure is still fully constructable from ``base`` alone.
    """
    return nn.Sequential(
        _block(in_ch, out_ch, stride=stride),
        _block(out_ch, out_ch, stride=1),
    )


class LatticeNet(nn.Module):
    """v5: native-direct -- predict the 17x17 outer + 16x16 inner grids directly, no interpolation.

    v1-v4 all routed the prediction through a dense 256x256 field and then resampled to the lattice:
    v3/v4 bilinearly, v1/v2 by average pooling. Every one of those put a rendering-domain transform
    (interpolation or averaging) into the model's OUTPUT path, and v4 additionally trained against a
    bilinearly-upsampled target (the old ``lattice_dense_loss``). v5 removes all of it: the objective
    and the output are both at the lattice's own native resolution. Only real signals in, native
    samples out; interpolation is allowed ONLY in the downstream ``lattice_bridge.py`` (a clearly
    separate rendering step), never in this model or its loss.

    Architecture: a strided double-conv encoder reduces 256->128->64->32->16, then two native heads
    read the grids off learned convolutions -- NOT average pooling (the v1/v2 localization failure)
    and NOT bilinear resampling:
      - inner 16x16 straight off the 16x16 bottleneck (``enc5``);
      - outer 17x17 from the 32x32 map (``enc4``) via a learned k2/s2/p1 conv, floor((32+2-2)/2)+1=17
        -- a localized learned downsample, not an interpolation.
    Each output cell sees a bounded input receptive field, so the field is localized, not globally
    averaged. Kept deliberately over-capacity (double-conv, ``base=64`` default) so a failure to beat
    tile-mean is never about size. Constructable from ``base`` alone, so ``lattice_bridge.py``
    reconstructs it unchanged via ``lattice_config.base``. Independently weighted from every other
    v50 stage (constitution IV / spec FR-007).
    """

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        self.enc1 = _double_block(3, base)                       # 256, b
        self.enc2 = _double_block(base, base * 2, stride=2)      # 128, 2b
        self.enc3 = _double_block(base * 2, base * 4, stride=2)   # 64, 4b
        self.enc4 = _double_block(base * 4, base * 8, stride=2)   # 32, 8b
        self.enc5 = _double_block(base * 8, base * 8, stride=2)   # 16, 8b (inner-grid resolution)
        # Inner 16x16 head: native, straight off the 16x16 bottleneck.
        self.inner_head = nn.Sequential(
            _double_block(base * 8, base * 4),                   # 16x16
            nn.Conv2d(base * 4, 1, 1),                           # 16x16 x 1
        )
        # Outer 17x17 head: a learned k2/s2/p1 conv turns the 32x32 map into 17x17, then refine.
        self.outer_reduce = nn.Conv2d(base * 8, base * 8, kernel_size=2, stride=2, padding=1)  # 32 -> 17
        self.outer_head = nn.Sequential(
            _double_block(base * 8, base * 4),                   # 17x17
            nn.Conv2d(base * 4, 1, 1),                           # 17x17 x 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """256x256x3 minimap RGB -> (B, 545) native lattice in [0, 1] (outer 289 then inner 256)."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise LatticeTargetError(
                f"LatticeNet consumes (B, 3, H, W); got shape {tuple(x.shape)}"
            )
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)          # (B, 8b, 32, 32)
        e5 = self.enc5(e4)          # (B, 8b, 16, 16)
        inner = torch.sigmoid(self.inner_head(e5)).flatten(1)                     # (B, 256)
        outer = torch.sigmoid(self.outer_head(self.outer_reduce(e4))).flatten(1)  # (B, 289)
        return torch.cat([outer, inner], dim=1)  # (B, 545)


__all__ = [
    "LatticeTargetError",
    "RANGE_FLOOR",
    "INPUT_SIZE",
    "encode_lattice_target",
    "decode_lattice_target",
    "select_lattice_rows",
    "compute_lattice_tile_mean_baseline",
    "lattice_loss",
    "lattice_gradient_loss",
    "LatticeNet",
]
