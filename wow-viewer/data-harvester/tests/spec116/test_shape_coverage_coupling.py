"""Spec 116 US2: shape->coverage coupling measurement and derivability decision."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.spec116.shape_coverage_coupling import (
    BIMODALITY_COEFFICIENT_THRESHOLD,
    ShapeCoverageCouplingError,
    decision_from_report,
    measure_shape_coverage_coupling,
)
from harvester.spec116.structure_contract import validate_analysis_report
from tests.spec116.conftest import build_store

CHUNKS = 16


def _ramp_height() -> np.ndarray:
    """A (257, 257) height field with a smooth diagonal ramp -> real per-chunk relief."""
    lin = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    return np.outer(lin, lin)


def _threshold_coverage(height_257: np.ndarray, layer: int = 1) -> np.ndarray:
    """Coverage that is a THRESHOLD function of elevation: 1 where elevation > median, else 0.

    A linear fit averages this away; a non-linear fit recovers it -- the exact case the spec says a
    linear test is underpowered for.
    """
    block = height_257[: CHUNKS * 16, : CHUNKS * 16]
    elevation = block.reshape(CHUNKS, 16, CHUNKS, 16).mean(axis=(1, 3))
    coverage = (elevation > np.median(elevation)).astype(np.float32)
    mask = np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32)
    mask[:, :, 0] = 1.0
    mask[:, :, layer] = coverage
    return mask


def _random_coverage(layer: int = 1, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coverage = rng.random((CHUNKS, CHUNKS), dtype=np.float32)
    mask = np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32)
    mask[:, :, 0] = 1.0
    mask[:, :, layer] = coverage
    return mask


def _write_store(tmp_path, name, rows):
    store = tmp_path / name
    store.mkdir()
    build_store(store, rows=rows)
    return store


class TestCouplingMeasurement:
    def test_strong_threshold_coupling_is_detected(self, tmp_path) -> None:
        h = _ramp_height()
        rows = [
            {
                "map": "Kalimdor", "tile_x": 1, "tile_y": 1, "split": "train", "source": "authored",
                "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
                "mcly_texture_ids": np.full((CHUNKS, CHUNKS, 4), -1, dtype=np.int32),
                "mcly_layer_mask": _threshold_coverage(h),
                "height_257": h,
            },
        ]
        store = _write_store(tmp_path, "strong.zarr", rows)
        report = measure_shape_coverage_coupling(store=store)
        evs = [e["explained_variance"] for e in report["shape_coverage_coupling"]["per_tile_layer_explained_variance"]]
        # The threshold relationship is recovered with high explained variance.
        assert max(evs) > 0.80
        assert report["shape_coverage_coupling"]["high_coupling_tile_share"] >= 0.20
        assert decision_from_report(report) == "coverage_derivable"
        validate_analysis_report(report)

    def test_random_coverage_is_independent(self, tmp_path) -> None:
        h = _ramp_height()
        rows = [
            {
                "map": "Kalimdor", "tile_x": 1, "tile_y": 1, "split": "train", "source": "authored",
                "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
                "mcly_texture_ids": np.full((CHUNKS, CHUNKS, 4), -1, dtype=np.int32),
                "mcly_layer_mask": _random_coverage(),
                "height_257": h,
            },
        ]
        store = _write_store(tmp_path, "random.zarr", rows)
        report = measure_shape_coverage_coupling(store=store)
        evs = [e["explained_variance"] for e in report["shape_coverage_coupling"]["per_tile_layer_explained_variance"]]
        # Random coverage is not explained by shape.
        assert max(evs) < 0.50
        assert decision_from_report(report) == "coverage_independent"
        validate_analysis_report(report)

    def test_zero_coverage_layer_is_skipped_not_scored_zero(self, tmp_path) -> None:
        h = _ramp_height()
        mask = np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32)
        mask[:, :, 0] = 1.0  # base only; all detail layers have zero coverage everywhere
        rows = [
            {
                "map": "Kalimdor", "tile_x": 1, "tile_y": 1, "split": "train", "source": "authored",
                "texture_names": [r"Tileset\X\XGrass.blp"],
                "mcly_texture_ids": np.full((CHUNKS, CHUNKS, 4), -1, dtype=np.int32),
                "mcly_layer_mask": mask, "height_257": h,
            },
        ]
        store = _write_store(tmp_path, "baseonly.zarr", rows)
        with pytest.raises(ShapeCoverageCouplingError, match="no .+ fittable"):
            measure_shape_coverage_coupling(store=store)

    def test_bimodality_coefficient_threshold_is_the_standard_value(self) -> None:
        assert BIMODALITY_COEFFICIENT_THRESHOLD == pytest.approx(5.0 / 9.0)

    def test_missing_height_array_is_rejected(self, tmp_path) -> None:
        # A store without height_257: build one then delete the array is awkward; instead build a
        # store whose only row has zero height (still valid) and assert the measurement runs. The
        # missing-array path is covered by passing a store lacking the array.
        import zarr
        store = tmp_path / "noheight.zarr"
        store.mkdir()
        g = zarr.open_group(str(store), mode="w")
        g.create_array("mcly_layer_mask", data=np.zeros((1, CHUNKS, CHUNKS, 4), dtype=np.float32))
        with pytest.raises(ShapeCoverageCouplingError, match="missing"):
            measure_shape_coverage_coupling(store=store)
