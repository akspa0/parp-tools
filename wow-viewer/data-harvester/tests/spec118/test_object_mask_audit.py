"""Spec 118 T013 (US1): the mask audit computes hand-checkable fractions/consistency stats on a
fixture store, detects planted violations, and refuses a store missing the strict arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.spec118.object_mask_audit import (
    INSTANCE_ARRAY,
    MASK_ARRAY,
    SOURCE_ARRAY,
    ObjectMaskAuditError,
    audit_object_masks,
)


def _build_store(
    tmp_path: Path,
    *,
    masks: list[np.ndarray],
    sources: list[np.ndarray] | None = None,
    instances: list[np.ndarray] | None = None,
    with_footprint: np.ndarray | None = None,
    rows: int | None = None,
) -> Path:
    store = tmp_path / "store.zarr"
    group = zarr.open_group(str(store), mode="w")
    n = rows if rows is not None else len(masks)
    if masks:
        group.create_array(MASK_ARRAY, data=np.stack(masks).astype(np.float32))
    if sources is not None:
        group.create_array(SOURCE_ARRAY, data=np.stack(sources).astype(np.uint8))
    if instances is not None:
        group.create_array(INSTANCE_ARRAY, data=np.stack(instances).astype(np.int32))
    if with_footprint is not None:
        group.create_array("object_mask_257", data=np.stack(with_footprint).astype(np.float32))
    index_rows = [{"map": "Kalimdor" if i % 2 == 0 else "Azeroth", "tile_x": i, "tile_y": 0} for i in range(n)]
    pq.write_table(pa.Table.from_pylist(index_rows), store / "index.parquet")
    return store


def _tile(fill: float = 0.0) -> np.ndarray:
    return np.full((257, 257), fill, dtype=np.float32)


def test_audit_reports_hand_computed_fractions_and_zero_violations(tmp_path: Path):
    # Tile 0: exactly 4 visible pixels of one doodad instance (id 1, class 1).
    mask0 = _tile()
    mask0[:2, :2] = 1.0
    source0 = np.zeros((257, 257), dtype=np.uint8)
    source0[:2, :2] = 1
    instance0 = np.zeros((257, 257), dtype=np.int32)
    instance0[:2, :2] = 1
    # Tile 1: no objects at all (valid negative).
    mask1 = _tile()
    source1 = np.zeros((257, 257), dtype=np.uint8)
    instance1 = np.zeros((257, 257), dtype=np.int32)
    # Footprint mask marks a 64x64 block on tile 0 (the old over-masking behavior).
    footprint = np.zeros((2, 257, 257), dtype=np.float32)
    footprint[0, :64, :64] = 1.0

    store = _build_store(
        tmp_path,
        masks=[mask0, mask1],
        sources=[source0, source1],
        instances=[instance0, instance1],
        with_footprint=footprint,
    )
    doc = audit_object_masks(store)

    assert doc["schema"] == "v118-object-mask-audit-v1"
    assert doc["tile_count"] == 2
    assert doc["object_touched_tile_count"] == 1
    expected_fraction = 4.0 / (257 * 257)
    assert doc["marked_fraction"]["p50"] == pytest.approx(expected_fraction / 2, rel=1e-3)
    assert doc["per_map_marked_fraction"]["Kalimdor"]["p50"] == pytest.approx(expected_fraction, rel=1e-6)
    assert doc["per_map_marked_fraction"]["Azeroth"]["p50"] == 0.0
    assert doc["instance_count_per_tile"]["p50"] == pytest.approx(0.5)
    assert doc["instance_visible_pixel_count"]["p50"] == pytest.approx(4.0)
    assert doc["class_consistency"]["violation_count"] == 0
    assert doc["mask_instance_mismatch_pixel_count"] == 0
    reduction = doc["visible_vs_footprint"]
    assert reduction is not None
    # 64x64 footprint vs 2x2 visible = 1024x reduction on the touched tile.
    assert reduction["median_footprint_to_visible_ratio"] == pytest.approx(1024.0, rel=1e-3)


def test_audit_detects_planted_class_and_instance_violations(tmp_path: Path):
    # Instance 1 with 100 pixels, 10 of them a different class (10% > 5% tolerance).
    mask0 = _tile()
    mask0[:10, :10] = 1.0
    source0 = np.zeros((257, 257), dtype=np.uint8)
    source0[:10, :10] = 1
    source0[0, :] = 0  # not in instance region; keep class counts clean below
    source0[:10, :10] = 1
    source0[1, :10] = 2  # 10 mixed-class pixels inside the instance
    instance0 = np.zeros((257, 257), dtype=np.int32)
    instance0[:10, :10] = 1
    # One pixel where the mask is 0 but the instance id is positive (FR-002 violation).
    mask0[50, 50] = 0.0
    instance0[50, 50] = 9

    store = _build_store(
        tmp_path,
        masks=[mask0],
        sources=[source0],
        instances=[instance0],
    )
    doc = audit_object_masks(store)

    assert doc["class_consistency"]["violation_count"] == 1
    assert doc["mask_instance_mismatch_pixel_count"] == 1


def test_audit_refuses_a_store_missing_the_strict_arrays(tmp_path: Path):
    store = tmp_path / "store.zarr"
    zarr.open_group(str(store), mode="w")
    pq.write_table(pa.Table.from_pylist([{"map": "Kalimdor", "tile_x": 0, "tile_y": 0}]), store / "index.parquet")

    with pytest.raises(ObjectMaskAuditError, match="object_geometry_visible_mask_257"):
        audit_object_masks(store)


def test_audit_map_filter_restricts_rows(tmp_path: Path):
    mask0 = _tile()
    mask0[:2, :2] = 1.0
    source0 = np.zeros((257, 257), dtype=np.uint8)
    source0[:2, :2] = 2
    instance0 = np.zeros((257, 257), dtype=np.int32)
    instance0[:2, :2] = 1
    mask1 = _tile()
    source1 = np.zeros((257, 257), dtype=np.uint8)
    instance1 = np.zeros((257, 257), dtype=np.int32)

    store = _build_store(
        tmp_path,
        masks=[mask0, mask1],
        sources=[source0, source1],
        instances=[instance0, instance1],
    )
    doc = audit_object_masks(store, map_filter="Azeroth")
    assert doc["tile_count"] == 1
    assert doc["object_touched_tile_count"] == 0
    assert doc["marked_fraction"]["p50"] == 0.0
