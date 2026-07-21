"""Spec 116 US4: relief stratification, trivial baseline, and reused-piece overlap."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.spec116.relief_stratification import (
    RELIEF_STD_THRESHOLD,
    STRATUM_FLAT,
    STRATUM_RELIEF,
    ReliefStratificationError,
    chunk_strata,
    count_reused_piece_overlap,
    stratified_mae,
    tile_mean_baseline_mae,
)


def _flat_height() -> np.ndarray:
    return np.zeros((257, 257), dtype=np.float32)


def _bump_height() -> np.ndarray:
    h = np.zeros((257, 257), dtype=np.float32)
    # Off-chunk-aligned box (60..196) so the boundary chunks straddle 0/1 and have real std.
    h[60:196, 60:196] = 1.0
    return h


class TestStrata:
    def test_flat_height_is_all_flat(self) -> None:
        strata = chunk_strata(_flat_height())
        assert strata.shape == (16, 16)
        assert (strata == STRATUM_FLAT).all()

    def test_bump_height_has_relief_chunks(self) -> None:
        strata = chunk_strata(_bump_height())
        assert (strata == STRATUM_RELIEF).any()
        assert (strata == STRATUM_FLAT).any()

    def test_threshold_is_a_reported_constant(self) -> None:
        assert RELIEF_STD_THRESHOLD > 0.0


class TestStratifiedMae:
    def test_error_is_reported_per_stratum(self) -> None:
        target = np.zeros((256, 256), dtype=np.float32)
        pred = np.ones((256, 256), dtype=np.float32) * 0.5
        strata = np.zeros((16, 16), dtype=np.uint8)
        strata[4:8, 4:8] = STRATUM_RELIEF
        out = stratified_mae([pred], [target], [strata])
        assert out["flat"]["mae"] == pytest.approx(0.5)
        assert out["relief"]["mae"] == pytest.approx(0.5)
        assert out["flat"]["pixels"] > out["relief"]["pixels"] > 0

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ReliefStratificationError, match="mismatch"):
            stratified_mae([np.zeros((256, 256))], [np.zeros((256, 256)), np.zeros((256, 256))], [np.zeros((16, 16))])


class TestTrivialBaseline:
    def test_tile_mean_baseline_is_mean_absolute_deviation(self) -> None:
        t = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        # mean = 1.5; |dev| = 1.5,0.5,0.5,1.5 -> mean = 1.0
        assert tile_mean_baseline_mae([t]) == pytest.approx(1.0)

    def test_empty_targets_rejected(self) -> None:
        with pytest.raises(ReliefStratificationError, match="no targets"):
            tile_mean_baseline_mae([])


class TestReusedPieceOverlap:
    def test_identical_minimaps_match_fully(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).astype(np.float32)
        out = count_reused_piece_overlap([img], [img], block=32, ncc_threshold=0.99, max_held_blocks=8)
        assert out["checked_blocks"] > 0
        assert out["match_fraction"] == pytest.approx(1.0)

    def test_disjoint_minimaps_do_not_match(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).astype(np.float32)
        b = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).astype(np.float32)
        out = count_reused_piece_overlap([a], [b], block=32, ncc_threshold=0.99, max_held_blocks=8)
        assert out["match_fraction"] == pytest.approx(0.0)

    def test_empty_inputs_rejected(self) -> None:
        with pytest.raises(ReliefStratificationError, match="both train and held-out"):
            count_reused_piece_overlap([], [np.zeros((64, 64, 3))])
