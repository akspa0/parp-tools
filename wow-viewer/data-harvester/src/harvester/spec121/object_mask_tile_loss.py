"""Spec 121 US3: tile-level object-mask loss weighting for the lattice stage (D-05).

Stage A's target is a 545-point lattice — it has no pixels, so Spec 118's per-point
``1 - w * mask`` rule cannot apply literally. The mask's honest summary for a lattice tile is
"how much of this tile's minimap is object-contaminated": a per-tile trust weight
``1 - w * coverage`` where ``coverage`` is the marked fraction of
``object_geometry_visible_mask_257``. Weight 0.0 is bit-parity with the unweighted path (the
trainer branches and never calls this module), matching the parity-default convention of
``--liquid-mask-weight`` / Spec 118's ``--object-mask-weight``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

OBJECT_MASK_ARRAY = "object_geometry_visible_mask_257"
# A tile at or above this marked fraction counts as "object-touched" in the run record's
# touched/untouched MAE split (mirrors the region-level reporting idea of Spec 118 FR-008,
# summarized per tile because the lattice has no pixel grid).
OBJECT_TOUCHED_COVERAGE = 0.05


class ObjectMaskLossError(ValueError):
    """Raised when object-mask coverage data cannot be produced as declared."""


def load_tile_coverages(group, rows: list[int]) -> np.ndarray:
    """Return float32 coverage (marked fraction) per row. Caller checked array presence first."""
    if OBJECT_MASK_ARRAY not in group:
        raise ObjectMaskLossError(
            f"store lacks {OBJECT_MASK_ARRAY}; check object_mask_present() before loading coverages"
        )
    coverages = np.empty(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        mask = np.asarray(group[OBJECT_MASK_ARRAY][row])
        coverages[i] = float((mask > 0.5).mean())
    return coverages


def coverage_weights(coverages: torch.Tensor, weight: float) -> torch.Tensor:
    """Per-tile trust weights ``1 - w * coverage``, clamped so an all-object tile keeps a
    finite (tiny) gradient instead of being silently dropped (spec Edge Cases)."""
    if weight < 0.0 or weight > 1.0:
        raise ObjectMaskLossError(f"object mask weight must be in [0, 1], got {weight}")
    return (1.0 - weight * coverages).clamp_min(0.0)


def weighted_lattice_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    coverages: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Per-sample masked smooth-L1, coverage-weighted across the batch.

    Per-sample loss normalizes by that sample's own present count (same rule as
    ``lattice_loss`` but per row), then the batch mean is weighted by ``1 - w * coverage``.
    If every weight collapses to 0 the mean falls back to the unweighted one so the loss
    stays finite (an all-object batch is down-weighted, never NaN).
    """
    per_element = nn.functional.smooth_l1_loss(predicted, target, reduction="none")
    per_sample = (per_element * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    weights = coverage_weights(coverages.to(per_sample.device), weight)
    denom = weights.sum()
    if float(denom) == 0.0:
        return per_sample.mean()
    return (per_sample * weights).sum() / denom


def touched_untouched_mae(
    per_tile_mae: np.ndarray, coverages: np.ndarray
) -> dict[str, float | int | None]:
    """Split held-out tile MAE by object-touched (coverage >= threshold) vs untouched."""
    mae = np.asarray(per_tile_mae, dtype=np.float64)
    cov = np.asarray(coverages, dtype=np.float64)
    if mae.shape != cov.shape:
        raise ObjectMaskLossError(
            f"per-tile MAE and coverage shapes differ: {mae.shape} vs {cov.shape}"
        )
    touched = cov >= OBJECT_TOUCHED_COVERAGE
    result: dict[str, float | int | None] = {
        "threshold": OBJECT_TOUCHED_COVERAGE,
        "touched_tiles": int(touched.sum()),
        "untouched_tiles": int((~touched).sum()),
        "touched_mae": float(mae[touched].mean()) if touched.any() else None,
        "untouched_mae": float(mae[~touched].mean()) if (~touched).any() else None,
    }
    return result


__all__ = [
    "OBJECT_MASK_ARRAY",
    "OBJECT_TOUCHED_COVERAGE",
    "ObjectMaskLossError",
    "load_tile_coverages",
    "coverage_weights",
    "weighted_lattice_loss",
    "touched_untouched_mae",
]
