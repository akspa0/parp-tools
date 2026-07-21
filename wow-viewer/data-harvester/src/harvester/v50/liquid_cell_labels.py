"""Spec 115 follow-on: per-CELL liquid-type labels from MCNK chunk flags.

Why per-cell rather than per-pixel: MCNK chunks are the authoring unit. ``mcnk_flags_16`` is
natively ``(16, 16)`` per tile, so a 16x16 prediction grid predicts exactly what the format stores.
The Spec 115 per-pixel classifier had to upsample chunk-resolution truth to 256x256, which invented
a resolution the data never had and made boundary pixels the dominant error term.

Why liquid specifically: it passes the two filters that killed the other candidates.

1. **Visible in RGB** — water renders unmistakably in a minimap. (``impass`` is a collision marker
   with no rendered footprint; ``has_mcsh`` measured r=-0.006 against minimap luminance. Both are
   unlearnable from the image regardless of how attractive their class balance looks.)
2. **Usable class balance** — none 56.5% / river 2.9% / ocean 40.7% across ~110k sampled chunks.
   Compare Spec 115's road at 0.26%.

River-vs-ocean is a genuine *type* judgement from visual cues (extent, colour, edge shape), not just
"is there water", which is what the existing per-pixel ``water`` class already covers.

Ground truth is the MCNK flag bits. ``liquid_type_256`` is an independent per-pixel source that
agrees closely (values 0/1/2 at 59.0/1.8/39.2% vs flags at 56.5/2.9/40.7%), so it is used as a
cross-check rather than a second opinion baked into the labels.
"""

from __future__ import annotations

import numpy as np

TAXONOMY_REVISION = "liquid-cell-v1"

CHUNKS_PER_AXIS = 16

# MCNK header flag bits (wowdev.wiki). Only the liquid bits are consumed here.
FLAG_HAS_MCSH = 0x0001
FLAG_IMPASS = 0x0002
FLAG_LQ_RIVER = 0x0004
FLAG_LQ_OCEAN = 0x0008
FLAG_LQ_MAGMA = 0x0010
FLAG_LQ_SLIME = 0x0020

# Ordinal IS the output channel index; contract-stable.
NONE = 0
RIVER = 1
OCEAN = 2
MAGMA = 3
SLIME = 4

CLASS_NAMES: tuple[str, ...] = ("none", "river", "ocean", "magma", "slime")
CLASS_COUNT = len(CLASS_NAMES)

# Precedence when a chunk sets multiple liquid bits (rare: river+ocean measured at 0.01%).
# Most-specific/least-common first so a river tagged also-ocean stays a river.
_PRECEDENCE: tuple[tuple[int, int], ...] = (
    (FLAG_LQ_MAGMA, MAGMA),
    (FLAG_LQ_SLIME, SLIME),
    (FLAG_LQ_RIVER, RIVER),
    (FLAG_LQ_OCEAN, OCEAN),
)


class LiquidCellLabelError(ValueError):
    """Raised when per-cell liquid labels cannot be derived honestly."""


def labels_from_mcnk_flags(mcnk_flags_16: np.ndarray) -> np.ndarray:
    """``(16, 16) int32`` MCNK flags -> ``(16, 16) uint8`` liquid class ordinals."""
    flags = np.asarray(mcnk_flags_16)
    if flags.shape != (CHUNKS_PER_AXIS, CHUNKS_PER_AXIS):
        raise LiquidCellLabelError(
            f"mcnk_flags_16 must be ({CHUNKS_PER_AXIS}, {CHUNKS_PER_AXIS}), got {flags.shape}"
        )
    labels = np.full(flags.shape, NONE, dtype=np.uint8)
    # Applied lowest-precedence first so earlier entries overwrite later ones.
    for bit, ordinal in reversed(_PRECEDENCE):
        labels = np.where((flags & bit).astype(bool), np.uint8(ordinal), labels)
    return labels


def labels_from_liquid_type(liquid_type_256: np.ndarray) -> np.ndarray:
    """Independent cross-check: majority ``liquid_type_256`` value per 16x16 cell.

    Only used to VALIDATE flag-derived labels. Its encoding (0=none, 1=river-like, 2=ocean-like) is
    inferred from distribution agreement with the flags, not from a documented enum, so it must not
    silently become the label source.
    """
    values = np.asarray(liquid_type_256)
    if values.ndim != 2 or values.shape[0] % CHUNKS_PER_AXIS or values.shape[1] % CHUNKS_PER_AXIS:
        raise LiquidCellLabelError(
            f"liquid_type_256 must be 2D and divisible by {CHUNKS_PER_AXIS}, got {values.shape}"
        )
    step_y = values.shape[0] // CHUNKS_PER_AXIS
    step_x = values.shape[1] // CHUNKS_PER_AXIS
    blocks = values.reshape(CHUNKS_PER_AXIS, step_y, CHUNKS_PER_AXIS, step_x)
    out = np.full((CHUNKS_PER_AXIS, CHUNKS_PER_AXIS), NONE, dtype=np.uint8)
    for cy in range(CHUNKS_PER_AXIS):
        for cx in range(CHUNKS_PER_AXIS):
            cell = blocks[cy, :, cx, :].ravel()
            counts = np.bincount(cell.astype(np.int64), minlength=3)
            wet = counts[1:].sum()
            if wet == 0:
                continue
            out[cy, cx] = RIVER if counts[1] >= counts[2] else OCEAN
    return out


def labels_from_liquid_type_grid(liquid_type_256: np.ndarray, grid: int) -> np.ndarray:
    """Majority ``liquid_type_256`` value per cell on an arbitrary ``grid`` x ``grid`` lattice.

    Needed for QUAD resolution. A tile's real mesh is 128x128 quads (129 outer vertices per axis),
    which is 8x finer than the 16x16 MCNK chunk grid that ``mcnk_flags_16`` can express: a chunk
    flag marks the whole chunk wet even when only a corner holds water. ``liquid_type_256`` is
    per-pixel, so it is the only source that can label at the resolution the geometry actually has.

    At grid=16 this agrees with the flag-derived labels to ~99.7% (measured), so the two sources
    corroborate each other; finer grids simply use the source that can represent them.
    """
    values = np.asarray(liquid_type_256)
    if grid < 1 or values.ndim != 2:
        raise LiquidCellLabelError(f"need a 2D liquid_type and grid >= 1, got {values.shape}/{grid}")
    if values.shape[0] % grid or values.shape[1] % grid:
        raise LiquidCellLabelError(
            f"liquid_type_256 {values.shape} is not divisible by grid {grid}"
        )
    step_y = values.shape[0] // grid
    step_x = values.shape[1] // grid
    blocks = values.astype(np.int64).reshape(grid, step_y, grid, step_x)
    counts = np.stack([(blocks == k).sum(axis=(1, 3)) for k in range(3)])  # (3, grid, grid)
    winner = counts.argmax(axis=0).astype(np.uint8)
    # liquid_type encoding: 0=none, 1=river-like, 2=ocean-like -> our NONE/RIVER/OCEAN ordinals,
    # which happen to coincide. Mapped explicitly so a taxonomy change cannot silently break it.
    out = np.full(winner.shape, NONE, dtype=np.uint8)
    out[winner == 1] = RIVER
    out[winner == 2] = OCEAN
    return out


def agreement_rate(flag_labels: np.ndarray, type_labels: np.ndarray) -> float:
    """Fraction of cells where the two independent sources agree on wet-vs-dry."""
    a = np.asarray(flag_labels) != NONE
    b = np.asarray(type_labels) != NONE
    return float((a == b).mean())


def summarize(labels: np.ndarray) -> dict[str, int]:
    """Per-class cell counts."""
    values = np.asarray(labels)
    return {
        CLASS_NAMES[ordinal]: int(np.count_nonzero(values == ordinal))
        for ordinal in range(CLASS_COUNT)
    }
