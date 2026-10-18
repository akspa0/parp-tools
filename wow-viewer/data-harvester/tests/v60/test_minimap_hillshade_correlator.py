"""Unit tests for MinimapHillshadeCorrelator & Metrics (Spec 274)."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.minimap_hillshade_correlator import compute_elevation_metrics


def test_compute_elevation_metrics_exact() -> None:
    z = np.arange(100, dtype=np.float32).reshape(10, 10)
    metrics = compute_elevation_metrics(z, z)

    assert np.isclose(metrics["pearson_r"], 1.0)
    assert np.isclose(metrics["mae_yards"], 0.0)
    assert np.isclose(metrics["rmse_yards"], 0.0)
    assert np.isclose(metrics["max_error_yards"], 0.0)


def test_compute_elevation_metrics_with_offset() -> None:
    target = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    pred = target + 5.0

    metrics = compute_elevation_metrics(pred, target)
    assert np.isclose(metrics["pearson_r"], 1.0)
    assert np.isclose(metrics["mae_yards"], 5.0)
    assert np.isclose(metrics["rmse_yards"], 5.0)
    assert np.isclose(metrics["max_error_yards"], 5.0)
