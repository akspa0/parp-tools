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


def _conv_block(in_ch: int, out_ch: int, *, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class LatticeNet(nn.Module):
    """Lean conv encoder + two pooled heads: 256x256x3 minimap RGB -> 545 values in [0, 1].

    Independently weighted from every other v50 stage (constitution IV / spec FR-007) -- this is a
    fresh, small architecture, not a shared-weight extension of the coarse/detailer nets. Each head
    average-pools the shared feature map to its own lattice resolution (17x17 outer, 16x16 inner)
    before a small refining conv, so every output element still sees a local neighbourhood of
    features rather than one fully-global pooled vector (the failure mode the project's own prior
    WDL-prior model documented and corrected against). Kept deliberately lean (time-to-signal over
    architecture search); its real parameter count is recorded in the run's architecture identity
    rather than targeted at a specific number.
    """

    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(3, base),          # 128
            _conv_block(base, base * 2),   # 64
            _conv_block(base * 2, base * 4),  # 32
            _conv_block(base * 4, base * 4),  # 16
        )
        self.outer_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((OUTER_DIM, OUTER_DIM)),
            nn.Conv2d(base * 4, base, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, 1, 1),
        )
        self.inner_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((INNER_DIM, INNER_DIM)),
            nn.Conv2d(base * 4, base, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise LatticeTargetError(f"LatticeNet consumes (B, 3, H, W); got shape {tuple(x.shape)}")
        features = self.encoder(x)
        outer = torch.sigmoid(self.outer_head(features)).flatten(1)  # (B, 289)
        inner = torch.sigmoid(self.inner_head(features)).flatten(1)  # (B, 256)
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
    "LatticeNet",
]
