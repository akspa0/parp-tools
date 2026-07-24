"""Spec 121 T034: within-map split builder + apply tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from harvester.spec121.within_map_split import (
    WITHIN_MAP_SPLIT_SCHEMA,
    WithinMapSplitError,
    apply_within_map_split,
    build_within_map_split,
    detect_split_schema,
    load_within_map_split,
    validate_within_map_split,
    write_within_map_split,
)


def _index(rows: int = 20, maps: int = 2) -> list[dict]:
    rng = __import__("numpy").random.default_rng(42)
    index = []
    for i in range(rows):
        m = f"Map{rng.integers(0, maps)}"
        tx = int(rng.integers(0, 10))
        ty = int(rng.integers(0, 10))
        index.append({"tile_row": i, "map": m, "tile_x": tx, "tile_y": ty})
    return index


def test_build_produces_no_overlap():
    index = _index(50, maps=3)
    split = build_within_map_split(index, held_out_fraction=0.2, seed=121)
    assert split["verified_overlap_count"] == 0
    assert split["split_counts"]["train"] > 0
    assert split["split_counts"]["held_out"] > 0


def test_build_per_map_counts_sum_to_total():
    index = _index(100, maps=4)
    split = build_within_map_split(index, held_out_fraction=0.15, seed=42)
    total = sum(split["split_counts"].values()) + split["excluded_count"]
    per_map_total = sum(sum(mc.values()) for mc in split["per_map_counts"].values())
    assert total == per_map_total


def test_build_buffer_rings_excludes_adjacent():
    # 5x5 grid: 25 tiles, ~4 held-out, buffer_rings=1 excludes ~12-16 adjacent, leaves train.
    index = [{"tile_row": i, "map": "M", "tile_x": x, "tile_y": y}
             for i, (x, y) in enumerate([(x, y) for x in range(5) for y in range(5)])]
    split = build_within_map_split(index, held_out_fraction=0.15, buffer_rings=1, seed=0)
    assert split["excluded_count"] > 0
    assert split["split_counts"]["train"] > 0
    assert split["verified_overlap_count"] == 0


def test_build_rejects_invalid_fraction():
    with pytest.raises(WithinMapSplitError):
        build_within_map_split([], held_out_fraction=0.0)
    with pytest.raises(WithinMapSplitError):
        build_within_map_split([], held_out_fraction=1.0)


def test_build_rejects_negative_buffer():
    with pytest.raises(WithinMapSplitError):
        build_within_map_split([], buffer_rings=-1)


def test_write_and_load_roundtrip(tmp_path: Path):
    index = _index(30, maps=2)
    split = build_within_map_split(index, held_out_fraction=0.2, seed=7)
    store = tmp_path / "store.zarr"
    store.mkdir()
    (store / "index.parquet").write_text("placeholder")
    output = tmp_path / "split"
    manifest = write_within_map_split(store=store, output=output, split=split, build_id="test")
    assert manifest["schema"] == WITHIN_MAP_SPLIT_SCHEMA
    assert manifest["verified_overlap_count"] == 0
    loaded_manifest, loaded_rows = load_within_map_split(output)
    assert loaded_manifest == manifest
    assert len(loaded_rows) == len(split["assignments"])


def test_validate_rejects_overlap():
    doc = {
        "schema": WITHIN_MAP_SPLIT_SCHEMA,
        "created_utc": "2026-07-24T00:00:00Z",
        "build_id": "test",
        "store": {"path": "/x", "sha256": "a" * 64},
        "split_rule": "within-map-random",
        "leakage_rule": "none",
        "buffer_rings": 0,
        "held_out_fraction": 0.15,
        "seed": 121,
        "split_counts": {"train": 10, "held_out": 5},
        "per_map_counts": {"M": {"train": 10, "held_out": 5, "excluded": 0}},
        "verified_overlap_count": 1,
        "absolute_comparison_to_region_isolated_runs_invalid": True,
    }
    with pytest.raises(WithinMapSplitError):
        validate_within_map_split(doc)


def test_apply_mirrors_held_out_semantics():
    index = _index(20, maps=2)
    split = build_within_map_split(index, held_out_fraction=0.3, seed=5)
    store = Path("/tmp/_test_store")
    store.mkdir(parents=True, exist_ok=True)
    (store / "index.parquet").write_text("placeholder")
    output = Path("/tmp/_test_split")
    write_within_map_split(store=store, output=output, split=split, build_id="test")
    selected = list(range(20))
    train, val, manifest = apply_within_map_split(
        index_rows=index, selected_rows=selected, split_dir=output,
    )
    assert len(train) + len(val) <= 20
    assert len(train) > 0 and len(val) > 0
    assert manifest["verified_overlap_count"] == 0
    # Cleanup
    import shutil
    shutil.rmtree(store, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)


def test_detect_schema(tmp_path: Path):
    index = _index(10)
    split = build_within_map_split(index)
    store = tmp_path / "s"
    store.mkdir()
    (store / "index.parquet").write_text("placeholder")
    output = tmp_path / "d"
    write_within_map_split(store=store, output=output, split=split)
    assert detect_split_schema(output) == WITHIN_MAP_SPLIT_SCHEMA
