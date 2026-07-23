"""Spec 118 US2: shared object-masked-loss helpers for the two existing geometry trainers.

The visible-object mask is a GROUND-TRUTH signal admissible loss-side only (FR-014): it
down-weights object-covered pixels during training and never becomes an inference input. The math
here deliberately mirrors the trainers' existing ``--liquid-mask-weight`` convention
(``point_weight = 1 - w * mask``) so a run's loss knobs stay composable and parity-defaulted
(``w = 0`` is bit-identical to no masking).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# The v18 placement-footprint mask, painted by AlphaTensorPackBuilder from MDDF/MODF placements —
# populated on the 0.5.3 alpha corpus, unlike the strict `object_geometry_visible_mask_257` which
# only ADtTensorPackBuilder produces (empty on alpha). Soft-edged (0..1), so `1 - w*mask` gives a
# graded down-weight around each object footprint.
OBJECT_MASK_ARRAY = "object_precise_mask"


def object_mask_available(group: Any) -> bool:
    """True when the store carries the placement-footprint object mask (post-catalog-fix stores)."""
    return OBJECT_MASK_ARRAY in group


def clamp_weight(weight: float) -> float:
    """Clamp the user flag into [0, 1]: 0 = parity (no masking), 1 = zero loss on object pixels."""
    return min(max(float(weight), 0.0), 1.0)


def object_point_weight(mask: np.ndarray, weight: float) -> np.ndarray:
    """Per-pixel loss multiplier ``1 - w * mask``.

    ``w = 0`` returns exact ones so an unmasked run stays bit-identical to the pre-flag behavior;
    ``w = 1`` zeroes exactly the visibly object-covered pixels (FR-006/FR-007 -- never more than
    the visible portion, so a mostly-underground object barely reduces trainable land area).
    """
    w = clamp_weight(weight)
    if w == 0.0:
        return np.ones_like(np.asarray(mask), dtype=np.float32)
    return 1.0 - w * np.asarray(mask, dtype=np.float32)


def mask_touched(mask: np.ndarray) -> bool:
    """True when a tile has at least one visibly object-covered pixel."""
    return bool((np.asarray(mask) > 0).any())


def object_touched_rows(group: Any, rows: Sequence[int]) -> list[bool]:
    """Per-row object-touched flags for a set of store rows (evaluation-time subset split)."""
    array = group[OBJECT_MASK_ARRAY]
    return [mask_touched(np.asarray(array[row])) for row in rows]


def subset_metrics(
    touched_abs: float,
    touched_px: int,
    untouched_abs: float,
    untouched_px: int,
    *,
    weight: float,
) -> dict[str, Any]:
    """FR-008 object-touched vs untouched MAE summary for the run record.

    Reported alongside the existing aggregate and relief-stratified metrics so a flat-tile-
    dominated aggregate cannot mask a change on the object-touched subset (spec US2 acceptance 3).
    """
    return {
        "object_touched_region_mae": (touched_abs / touched_px) if touched_px else None,
        "object_untouched_region_mae": (untouched_abs / untouched_px) if untouched_px else None,
        "object_touched_pixels": touched_px,
        "object_untouched_pixels": untouched_px,
        "object_mask_weight": clamp_weight(weight),
        "source_array": OBJECT_MASK_ARRAY,
        "ground_truth_admissible_because": "loss-side only; never an inference input (FR-014)",
        "note": "compare object_touched_region_mae against the paired --object-mask-weight 0.0 run "
        "on the same split (SC-003); a null result is a valid, reportable outcome",
    }


__all__ = [
    "OBJECT_MASK_ARRAY",
    "object_mask_available",
    "clamp_weight",
    "object_point_weight",
    "mask_touched",
    "object_touched_rows",
    "subset_metrics",
]
