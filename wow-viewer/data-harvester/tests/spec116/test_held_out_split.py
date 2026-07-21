"""Spec 116 US4: spatially-isolated held-out split."""

from __future__ import annotations

from pathlib import Path

import pytest

from harvester.spec116.held_out_split import (
    DEFAULT_BUFFER_RINGS,
    HeldOutSplitError,
    build_held_out_split,
    load_split,
    write_split,
)
from harvester.spec116.structure_contract import validate_held_out_split
from tests.spec116.conftest import build_store

CHUNKS = 16


def _grid_tiles(map_name: str, n: int) -> list[dict]:
    """An n x n grid of curriculum tiles with real chunk arrays (zeros)."""
    import numpy as np
    rows = []
    for y in range(n):
        for x in range(n):
            rows.append({
                "map": map_name, "tile_x": x, "tile_y": y, "split": "train", "source": "authored",
                "texture_names": [r"Tileset\X\XGrass.blp"],
                "mcly_texture_ids": np.zeros((CHUNKS, CHUNKS, 4), dtype=np.int32),
                "mcly_layer_mask": np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32),
            })
    return rows


def _index_rows(tiles):
    return [{"tile_row": i, "map": t["map"], "tile_x": t["tile_x"], "tile_y": t["tile_y"]}
            for i, t in enumerate(tiles)]


class TestIsolation:
    def test_grid_split_has_zero_violations(self) -> None:
        tiles = _grid_tiles("Kalimdor", 10)
        idx = _index_rows(tiles)
        split = build_held_out_split(tiles=idx, held_out_fraction=0.2, buffer_rings=1, block_size=5, seed=1)
        assert split["verified_violation_count"] == 0
        assert split["split_counts"]["held_out"] > 0
        assert split["split_counts"]["train"] > 0

        # Explicit independent re-check of the 8-neighbour invariant.
        held = {(a["map"], a["tile_x"], a["tile_y"]) for a in split["assignments"] if a["split"] == "held_out"}
        train = {(a["map"], a["tile_x"], a["tile_y"]) for a in split["assignments"] if a["split"] == "train"}
        for m, x, y in held:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    assert (m, x + dx, y + dy) not in train

    def test_buffer_tiles_are_excluded_not_train(self) -> None:
        tiles = _grid_tiles("Kalimdor", 8)
        idx = _index_rows(tiles)
        split = build_held_out_split(tiles=idx, held_out_fraction=0.2, buffer_rings=1, block_size=4, seed=2)
        splits = {a["split"] for a in split["assignments"]}
        assert "excluded" in splits
        assert split["excluded_count"] > 0

    def test_too_small_corpus_refuses(self) -> None:
        # A 2x2 grid cannot isolate a held-out tile with a 1-ring buffer (the buffer eats all).
        tiles = _grid_tiles("Kalimdor", 2)
        idx = _index_rows(tiles)
        with pytest.raises(HeldOutSplitError):
            build_held_out_split(tiles=idx, held_out_fraction=0.5, buffer_rings=1, block_size=1, seed=0)

    def test_invalid_fraction_rejected(self) -> None:
        with pytest.raises(HeldOutSplitError, match="held_out_fraction"):
            build_held_out_split(tiles=_index_rows(_grid_tiles("M", 5)), held_out_fraction=0.0)

    def test_invalid_buffer_rings_rejected(self) -> None:
        with pytest.raises(HeldOutSplitError, match="buffer_rings"):
            build_held_out_split(tiles=_index_rows(_grid_tiles("M", 5)), buffer_rings=0)


class TestPersistRoundTrip:
    def test_write_and_load_round_trips_and_validates(self, tmp_path: Path) -> None:
        tiles = _grid_tiles("Kalimdor", 10)
        store = tmp_path / "store.zarr"
        store.mkdir()
        build_store(store, rows=tiles)
        idx = _index_rows(tiles)
        split = build_held_out_split(tiles=idx, held_out_fraction=0.2, buffer_rings=1, block_size=5, seed=3)
        out = tmp_path / "split"
        manifest = write_split(store=store, output=out, split=split, build_id="0_5_3_3368")
        validate_held_out_split(manifest)
        assert manifest["verified_violation_count"] == 0
        assert manifest["absolute_comparison_to_prior_invalid"] is True

        loaded_manifest, loaded_rows = load_split(out)
        assert loaded_manifest["split_counts"] == manifest["split_counts"]
        assert len(loaded_rows) == len(split["assignments"])
        assert {r["split"] for r in loaded_rows} == {a["split"] for a in split["assignments"]}

    def test_default_buffer_rings_is_one(self) -> None:
        assert DEFAULT_BUFFER_RINGS == 1
