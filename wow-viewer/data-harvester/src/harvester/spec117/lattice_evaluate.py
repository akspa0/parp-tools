"""Spec 117: relief-stratified honest evaluation for the WDL-lattice predictor.

The aggregate held-out ``val_mae`` is dominated by near-flat tiles, where a per-tile constant
(tile-mean) is nearly unbeatable -- this is the project's long-documented aggregate-MAE blind spot
(the reason Spec 116 US4 stratifies by relief). A model can capture real relief structure while the
flat tiles hold the mean error above the tile-mean baseline. These helpers stratify held-out tiles
by their own raw height relief and report, per stratum, the model's native masked MAE against the
tile-mean baseline -- so the honest question ("does it beat the trivial baseline where there is
actually relief to predict?") is answerable instead of hidden inside one flat-dominated average.
"""

from __future__ import annotations

import numpy as np


def tile_relief_and_baseline(
    target: np.ndarray, mask: np.ndarray, tile_min: float, tile_max: float
) -> tuple[float, float]:
    """Return ``(relief, tile_mean_mae)`` for one tile.

    ``relief`` is the raw world-unit height range (``tile_max - tile_min``) -- the honest measure of
    how much elevation variation the tile actually contains, used to stratify. ``tile_mean_mae`` is
    the masked MAE of predicting the tile's own mean-of-present-samples (the same per-tile baseline
    ``compute_lattice_tile_mean_baseline`` averages), computed on the normalized target so it is
    directly comparable to the model's own ``val_mae``.
    """
    present = np.asarray(mask) > 0
    if not present.any():
        raise ValueError("cannot evaluate a tile with zero present lattice samples")
    values = np.asarray(target, dtype=np.float64)[present]
    tile_mean = float(values.mean())
    tile_mean_mae = float(np.abs(values - tile_mean).mean())
    return float(tile_max - tile_min), tile_mean_mae


def relief_stratified_metrics(per_tile: list[dict], *, n_strata: int = 4) -> dict:
    """Stratify held-out tiles into ``n_strata`` equal-count relief bins (lowest to highest raw
    relief) and report, per stratum and overall, model MAE vs tile-mean MAE and who wins.

    ``per_tile`` items must each carry ``model_mae`` (the model's masked native MAE on that tile),
    ``tile_mean_mae``, and ``relief``. The headline is ``relief_subset`` -- the highest-relief
    stratum, where tile-mean is a weak baseline and the model must actually add value. Bins are
    equal-count quantile bins (not equal-width), so a corpus that is mostly flat still yields a
    populated high-relief stratum instead of one giant flat bin and a near-empty tail.
    """
    if not per_tile:
        raise ValueError("relief stratification needs at least one tile")
    if n_strata < 1:
        raise ValueError("n_strata must be >= 1")
    ordered = sorted(per_tile, key=lambda t: t["relief"])
    n = len(ordered)
    strata: list[dict] = []
    for s in range(n_strata):
        lo = (s * n) // n_strata
        hi = ((s + 1) * n) // n_strata
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        model_mae = float(np.mean([t["model_mae"] for t in chunk]))
        tile_mean_mae = float(np.mean([t["tile_mean_mae"] for t in chunk]))
        strata.append({
            "index": s,
            "n_tiles": len(chunk),
            "relief_min": float(chunk[0]["relief"]),
            "relief_max": float(chunk[-1]["relief"]),
            "model_mae": model_mae,
            "tile_mean_mae": tile_mean_mae,
            "model_beats_tile_mean": bool(model_mae < tile_mean_mae),
        })
    overall_model = float(np.mean([t["model_mae"] for t in ordered]))
    overall_tile_mean = float(np.mean([t["tile_mean_mae"] for t in ordered]))
    return {
        "stratify_by": "raw_height_relief",
        "n_strata": n_strata,
        "n_tiles": n,
        "strata": strata,
        # The highest-relief stratum: the honest question is whether the model beats tile-mean HERE,
        # not on the flat tiles that dominate the aggregate.
        "relief_subset": strata[-1] if strata else None,
        "overall": {
            "model_mae": overall_model,
            "tile_mean_mae": overall_tile_mean,
            "model_beats_tile_mean": bool(overall_model < overall_tile_mean),
        },
    }


__all__ = ["tile_relief_and_baseline", "relief_stratified_metrics"]
