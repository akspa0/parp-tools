"""Spec 117: relief-stratified evaluation must expose relief-tile performance that the
flat-tile-dominated aggregate hides."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.spec117.lattice_evaluate import relief_stratified_metrics, tile_relief_and_baseline


def test_tile_relief_and_baseline_range_and_masked_mae():
    target = np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32)
    mask = np.array([1, 1, 1, 0], dtype=np.float32)  # last sample absent
    relief, tm_mae = tile_relief_and_baseline(target, mask, tile_min=10.0, tile_max=60.0)
    assert relief == pytest.approx(50.0)  # 60 - 10 raw world units, NOT the normalized range
    # present values [0, 0.5, 1.0], mean 0.5 -> MAE mean(0.5, 0, 0.5) = 1/3.
    assert tm_mae == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_tile_relief_and_baseline_refuses_all_absent():
    with pytest.raises(ValueError):
        tile_relief_and_baseline(np.zeros(4), np.zeros(4), 0.0, 1.0)


def test_relief_stratified_reveals_win_the_aggregate_hides():
    """Model loses on flat tiles (where tile-mean is unbeatable) but wins on relief tiles; the
    stratified view must surface the relief win even when the flat tiles drag the aggregate."""
    per_tile = [
        {"relief": 1.0, "model_mae": 0.10, "tile_mean_mae": 0.02},    # flat: model worse
        {"relief": 2.0, "model_mae": 0.09, "tile_mean_mae": 0.03},    # flat: model worse
        {"relief": 400.0, "model_mae": 0.15, "tile_mean_mae": 0.30},  # relief: model better
        {"relief": 500.0, "model_mae": 0.14, "tile_mean_mae": 0.32},  # relief: model better
    ]
    out = relief_stratified_metrics(per_tile, n_strata=2)
    assert out["n_tiles"] == 4
    assert len(out["strata"]) == 2
    # Low-relief stratum: tile-mean wins. High-relief stratum: model wins.
    assert out["strata"][0]["model_beats_tile_mean"] is False
    assert out["strata"][1]["model_beats_tile_mean"] is True
    # relief_subset is the highest-relief stratum -- the honest headline.
    assert out["relief_subset"] is out["strata"][-1]
    assert out["relief_subset"]["relief_min"] == pytest.approx(400.0)
    # Aggregate (flat-dominated) hides the relief win: mean model 0.12 vs mean tile-mean 0.1675.
    assert "model_beats_tile_mean" in out["overall"]


def test_relief_stratified_equal_count_bins_no_empty_tail():
    per_tile = [{"relief": float(i), "model_mae": 0.1, "tile_mean_mae": 0.2} for i in range(10)]
    out = relief_stratified_metrics(per_tile, n_strata=4)
    # 10 tiles across 4 equal-count bins -> every stratum populated, sizes summing to 10.
    assert sum(s["n_tiles"] for s in out["strata"]) == 10
    assert all(s["n_tiles"] >= 1 for s in out["strata"])
    # Strata are ordered lowest-relief to highest-relief.
    assert out["strata"][0]["relief_max"] <= out["strata"][-1]["relief_min"]


def test_relief_stratified_refuses_empty():
    with pytest.raises(ValueError):
        relief_stratified_metrics([], n_strata=4)
