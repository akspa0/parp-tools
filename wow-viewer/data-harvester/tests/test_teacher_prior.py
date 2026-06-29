"""Tests for the spec 077 teacher deconstruction prior construction.

Covers the preference chain, suppression behavior, and metadata/index
parity used by the build_teacher_prior_dataset CLI.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
import zarr.storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _entry in (_REPO_ROOT, _SRC_DIR, _SCRIPTS_DIR):
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

from harvester.teacher_prior import (  # noqa: E402
    MaskSource,
    PRIOR_CHANNELS,
    build_prior_tensor,
    make_tile_record,
    pick_object_mask,
    suppress_object_pixels,
)
import build_teacher_prior_dataset  # noqa: E402
import review_teacher_prior_dataset  # noqa: E402
import audit_teacher_prior_visibility  # noqa: E402


# --- pick_object_mask preference chain ------------------------------------

def test_pick_prefers_object_precise_mask_by_default() -> None:
    filtered = np.zeros((256, 256), dtype=np.float32)
    filtered[0, 0] = 1.0
    precise = np.ones((256, 256), dtype=np.float32)
    obj_mask = np.ones((256, 256), dtype=np.float32)
    mask, source = pick_object_mask(
        object_filtered_mask=filtered,
        object_precise_mask=precise,
        object_mask=obj_mask,
    )
    assert source is MaskSource.ObjectPrecise
    assert int(mask.sum()) == 256 * 256


def test_pick_can_use_filtered_first_for_ablation() -> None:
    filtered = np.zeros((256, 256), dtype=np.float32)
    filtered[0, 0] = 1.0
    precise = np.ones((256, 256), dtype=np.float32)
    mask, source = pick_object_mask(
        object_filtered_mask=filtered,
        object_precise_mask=precise,
        object_mask=None,
        mask_priority=("object_filtered_mask", "object_precise_mask", "object_mask"),
    )
    assert source is MaskSource.ObjectFiltered
    assert int(mask.sum()) == 1


def test_pick_falls_back_to_filtered_when_precise_empty() -> None:
    filtered = np.zeros((256, 256), dtype=np.float32)
    filtered[10:20, 10:20] = 0.6
    precise = np.zeros((256, 256), dtype=np.float32)
    mask, source = pick_object_mask(
        object_filtered_mask=filtered,
        object_precise_mask=precise,
        object_mask=np.ones((256, 256), dtype=np.float32),
    )
    assert source is MaskSource.ObjectFiltered
    assert int(mask.sum()) == 100


def test_pick_falls_back_to_object_mask_when_precise_and_filtered_empty() -> None:
    obj_mask = np.zeros((256, 256), dtype=np.float32)
    obj_mask[5:10, 5:10] = 0.9
    mask, source = pick_object_mask(
        object_filtered_mask=np.zeros((256, 256), dtype=np.float32),
        object_precise_mask=np.zeros((256, 256), dtype=np.float32),
        object_mask=obj_mask,
    )
    assert source is MaskSource.ObjectMask
    assert int(mask.sum()) == 25


def test_pick_returns_empty_when_all_sources_empty() -> None:
    mask, source = pick_object_mask(
        object_filtered_mask=None,
        object_precise_mask=None,
        object_mask=None,
    )
    assert source is MaskSource.None_
    assert int(mask.sum()) == 0


# --- suppress_object_pixels behavior --------------------------------------

def test_suppress_passthrough_when_mask_empty() -> None:
    rng = np.random.default_rng(42)
    minimap = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    out = suppress_object_pixels(minimap, mask)
    np.testing.assert_array_equal(out, minimap)


def test_suppress_replaces_object_pixels_with_median() -> None:
    minimap = np.zeros((256, 256, 3), dtype=np.uint8)
    # Non-object region: red dominant
    minimap[:128, :, 0] = 200
    minimap[:128, :, 1] = 30
    minimap[:128, :, 2] = 30
    # Object region: blue
    minimap[128:, :, 0] = 10
    minimap[128:, :, 1] = 10
    minimap[128:, :, 2] = 250
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[128:, :] = 1
    out = suppress_object_pixels(minimap, mask)
    # Object region should now be near red (the non-object median)
    assert int(out[200, 100, 0]) > 100
    assert int(out[200, 100, 2]) < 100
    # Non-object region should be unchanged
    np.testing.assert_array_equal(out[:128, :, :], minimap[:128, :, :])


def test_suppress_handles_all_object_tile_with_neutral_fallback() -> None:
    minimap = np.full((256, 256, 3), 10, dtype=np.uint8)
    mask = np.ones((256, 256), dtype=np.uint8)
    out = suppress_object_pixels(minimap, mask)
    # Fallback is mid-gray 128
    assert int(out[0, 0, 0]) == 128
    assert int(out[0, 0, 1]) == 128
    assert int(out[0, 0, 2]) == 128


# --- build_prior_tensor shape + dtype -------------------------------------

def test_build_prior_tensor_uses_documented_channels() -> None:
    rng = np.random.default_rng(7)
    minimap = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    tensor, mask, conf, source = build_prior_tensor(
        minimap,
        object_filtered_mask=None,
        object_precise_mask=np.zeros((256, 256), dtype=np.float32),
        object_mask=np.zeros((256, 256), dtype=np.float32),
    )
    assert tensor.shape == (256, 256, 5)
    assert mask.shape == (256, 256)
    assert conf.shape == (256, 256)
    assert tensor.dtype == np.uint8
    # First three channels are suppressed RGB, last two are mask/confidence.
    np.testing.assert_array_equal(tensor[:, :, 3], mask)
    np.testing.assert_array_equal(tensor[:, :, 4], conf)
    assert source is MaskSource.None_
    assert len(PRIOR_CHANNELS) == 5


def test_build_prior_tensor_resizes_257_masks_to_minimap_grid() -> None:
    minimap = np.zeros((256, 256, 3), dtype=np.uint8)
    minimap[:, :, 0] = 200
    mask_257 = np.zeros((257, 257), dtype=np.float32)
    mask_257[128:, 128:] = 1.0
    tensor, mask, conf, source = build_prior_tensor(
        minimap,
        object_filtered_mask=mask_257,
        object_precise_mask=None,
        object_mask=None,
    )
    assert source is MaskSource.ObjectFiltered
    assert tensor.shape == (256, 256, 5)
    assert mask.shape == (256, 256)
    assert conf.shape == (256, 256)
    assert int(mask[200, 200]) == 1


# --- make_tile_record coverage / metadata ---------------------------------

def test_make_tile_record_reports_coverage() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:64, :64] = 1
    record = make_tile_record(
        build="3_3_5_12340",
        map_name="Azeroth",
        tile_id=42,
        tile_x=10,
        tile_y=20,
        mask_uint8=mask,
        source=MaskSource.ObjectFiltered,
        index=0,
    )
    assert record.has_teacher_objects is True
    assert 0.05 < record.teacher_object_cov < 0.07  # 64*64 / 256*256 ≈ 0.0625
    assert record.filtered_mask_source == "object_filtered_mask"


# --- end-to-end build via CLI ---------------------------------------------

def _make_minimap(color: tuple[int, int, int]) -> np.ndarray:
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    return arr


def test_build_teacher_prior_dataset_writes_expected_arrays() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v18_path = root / "fake_v18.zarr"
        out_root = root / "out"
        store = zarr.storage.LocalStore(str(v18_path), read_only=False)
        v18_root = zarr.group(store=store)
        minimap = np.stack([
            _make_minimap((200, 30, 30)),  # tile 0: red, no objects
            _make_minimap((30, 200, 30)),  # tile 1: green, no objects
        ], axis=0).astype(np.uint8)
        v18_root.create_array("minimap_rgb", data=minimap, chunks=(2, 256, 256, 3))
        v18_root.create_array(
            "object_filtered_mask",
            data=np.zeros((2, 257, 257), dtype=np.float32),
            chunks=(2, 257, 257),
        )
        v18_root.create_array(
            "object_precise_mask",
            data=np.zeros((2, 257, 257), dtype=np.float32),
            chunks=(2, 257, 257),
        )
        v18_root.create_array(
            "object_mask",
            data=np.zeros((2, 257, 257), dtype=np.float32),
            chunks=(2, 257, 257),
        )
        # Build a tiny index.parquet
        import pyarrow as pa
        table = pa.table({
            "tile_id": [0, 1],
            "build": ["fake_v18", "fake_v18"],
            "map": ["Test", "Test"],
            "tile_x": [10, 11],
            "tile_y": [20, 21],
            "n_mddf": [0, 0],
            "n_modf": [0, 0],
        })
        pq.write_table(table, str(v18_path / "index.parquet"))

        exit_code = build_teacher_prior_dataset.main_with_args(
            [
                "--v18-path", str(v18_path),
                "--output-root", str(out_root),
            ]
        )
        assert exit_code == 0

        out_path = out_root / "fake_v18.zarr"
        assert (out_path / "tiles.parquet").exists()
        store2 = zarr.storage.LocalStore(str(out_path), read_only=True)
        out_root_grp = zarr.open_group(store2, mode="r")
        assert "raw_minimap_rgb_256" in out_root_grp
        assert "teacher_object_mask_256" in out_root_grp
        assert "teacher_object_confidence_256" in out_root_grp
        assert "processed_minimap_prior_256" in out_root_grp
        prior = np.asarray(out_root_grp["processed_minimap_prior_256"][:])
        assert prior.shape == (2, 256, 256, 5)
        # No object pixels → mask band is zeros, prior suppressed RGB == raw RGB
        np.testing.assert_array_equal(prior[:, :, :, 0], minimap[:, :, :, 0])
        np.testing.assert_array_equal(prior[:, :, :, 3], np.zeros((2, 256, 256), dtype=np.uint8))

        tiles = pq.read_table(str(out_path / "tiles.parquet")).to_pylist()
        assert len(tiles) == 2
        for t in tiles:
            assert t["filtered_mask_source"] == "none"
            assert t["has_teacher_objects"] is False
            assert t["teacher_object_cov"] == 0.0


def test_build_teacher_prior_dataset_honors_curation_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v18_path = root / "fake_v18.zarr"
        out_root = root / "out"
        curation_dir = root / "curation"
        curation_dir.mkdir()
        store = zarr.storage.LocalStore(str(v18_path), read_only=False)
        v18_root = zarr.group(store=store)
        minimap = np.stack([
            _make_minimap((200, 30, 30)),
            _make_minimap((30, 200, 30)),
            _make_minimap((30, 30, 200)),
        ], axis=0).astype(np.uint8)
        v18_root.create_array("minimap_rgb", data=minimap, chunks=(3, 256, 256, 3))
        v18_root.create_array("object_filtered_mask", data=np.zeros((3, 257, 257), dtype=np.float32), chunks=(3, 257, 257))
        table = pa.table({
            "tile_id": [0, 1, 2],
            "build": ["fake_v18"] * 3,
            "map": ["Test"] * 3,
            "tile_x": [10, 11, 12],
            "tile_y": [20, 21, 22],
        })
        pq.write_table(table, str(v18_path / "index.parquet"))
        kept = pa.table({"build": ["fake_v18"], "tile_id": [1], "keep": [True]})
        pq.write_table(kept, str(curation_dir / "kept_tiles.parquet"))

        exit_code = build_teacher_prior_dataset.main_with_args(
            [
                "--v18-path", str(v18_path),
                "--output-root", str(out_root),
                "--curation-manifest", str(curation_dir),
            ]
        )
        assert exit_code == 0
        tiles = pq.read_table(str(out_root / "fake_v18.zarr" / "tiles.parquet")).to_pylist()
        assert [int(row["tile_id"]) for row in tiles] == [1]


def test_review_teacher_prior_can_render_specific_tile_with_source_masks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v18_path = root / "fake_v18.zarr"
        out_root = root / "out"
        review_dir = root / "review"
        store = zarr.storage.LocalStore(str(v18_path), read_only=False)
        v18_root = zarr.group(store=store)
        minimap = np.stack([
            _make_minimap((200, 30, 30)),
            _make_minimap((30, 200, 30)),
        ], axis=0).astype(np.uint8)
        precise = np.zeros((2, 257, 257), dtype=np.float32)
        precise[1, 80:120, 80:120] = 1.0
        filtered = np.zeros((2, 257, 257), dtype=np.float32)
        filtered[1, 90, 90] = 1.0
        v18_root.create_array("minimap_rgb", data=minimap, chunks=(2, 256, 256, 3))
        v18_root.create_array("object_precise_mask", data=precise, chunks=(2, 257, 257))
        v18_root.create_array("object_filtered_mask", data=filtered, chunks=(2, 257, 257))
        v18_root.create_array("object_mask", data=np.zeros((2, 257, 257), dtype=np.float32), chunks=(2, 257, 257))
        table = pa.table({
            "tile_id": [0, 1],
            "build": ["fake_v18", "fake_v18"],
            "map": ["Test", "Test"],
            "tile_x": [10, 11],
            "tile_y": [20, 21],
        })
        pq.write_table(table, str(v18_path / "index.parquet"))

        assert build_teacher_prior_dataset.main_with_args([
            "--v18-path", str(v18_path),
            "--output-root", str(out_root),
        ]) == 0
        assert review_teacher_prior_dataset.main_with_args([
            "--library", str(out_root / "fake_v18.zarr"),
            "--output-dir", str(review_dir),
            "--v18-path", str(v18_path),
            "--tile-id", "1",
        ]) == 0
        summary = json.loads((review_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["requested_tile_ids"] == [1]
        assert summary["selected_tile_ids"] == [1]
        assert summary["source_mask_arrays"] == ["object_filtered_mask", "object_mask", "object_precise_mask"]
        assert (review_dir / "contact_sheet.png").exists()


def test_review_teacher_prior_can_render_specific_compact_row() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v18_path = root / "fake_v18.zarr"
        out_root = root / "out"
        review_dir = root / "review"
        curation_dir = root / "curation"
        curation_dir.mkdir()
        store = zarr.storage.LocalStore(str(v18_path), read_only=False)
        v18_root = zarr.group(store=store)
        minimap = np.stack([
            _make_minimap((200, 30, 30)),
            _make_minimap((30, 200, 30)),
            _make_minimap((30, 30, 200)),
        ], axis=0).astype(np.uint8)
        precise = np.zeros((3, 257, 257), dtype=np.float32)
        precise[2, 80:120, 80:120] = 1.0
        v18_root.create_array("minimap_rgb", data=minimap, chunks=(3, 256, 256, 3))
        v18_root.create_array("object_precise_mask", data=precise, chunks=(3, 257, 257))
        table = pa.table({
            "tile_id": [0, 1, 2],
            "build": ["fake_v18"] * 3,
            "map": ["Test"] * 3,
            "tile_x": [10, 11, 12],
            "tile_y": [20, 21, 22],
        })
        pq.write_table(table, str(v18_path / "index.parquet"))
        kept = pa.table({"build": ["fake_v18"], "tile_id": [2], "keep": [True]})
        pq.write_table(kept, str(curation_dir / "kept_tiles.parquet"))

        assert build_teacher_prior_dataset.main_with_args([
            "--v18-path", str(v18_path),
            "--output-root", str(out_root),
            "--curation-manifest", str(curation_dir),
        ]) == 0
        assert review_teacher_prior_dataset.main_with_args([
            "--library", str(out_root / "fake_v18.zarr"),
            "--output-dir", str(review_dir),
            "--row-index", "0",
        ]) == 0
        summary = json.loads((review_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["requested_row_indices"] == [0]
        assert summary["selected_rows"] == [0]
        assert summary["selected_tile_ids"] == [2]


def test_audit_teacher_prior_visibility_buckets_mismatched_masks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v18_path = root / "fake_v18.zarr"
        out_root = root / "out"
        audit_dir = root / "audit"
        store = zarr.storage.LocalStore(str(v18_path), read_only=False)
        v18_root = zarr.group(store=store)
        minimap = np.stack([
            _make_minimap((100, 100, 100)),
            _make_minimap((100, 100, 100)),
            _make_minimap((100, 100, 100)),
        ], axis=0).astype(np.uint8)
        minimap[1, 80:120, 80:120] = np.array([20, 220, 20], dtype=np.uint8)
        precise = np.zeros((3, 257, 257), dtype=np.float32)
        precise[0, 80:120, 80:120] = 1.0  # mask exists, minimap region is terrain-like -> weak
        precise[1, 80:120, 80:120] = 1.0  # mask exists, minimap region visibly differs -> visible
        precise[2, 1, 1] = 1.0            # too tiny -> tiny
        v18_root.create_array("minimap_rgb", data=minimap, chunks=(3, 256, 256, 3))
        v18_root.create_array("object_precise_mask", data=precise, chunks=(3, 257, 257))
        table = pa.table({
            "tile_id": [0, 1, 2],
            "build": ["fake_v18"] * 3,
            "map": ["Test"] * 3,
            "tile_x": [10, 11, 12],
            "tile_y": [20, 21, 22],
        })
        pq.write_table(table, str(v18_path / "index.parquet"))

        assert build_teacher_prior_dataset.main_with_args([
            "--v18-path", str(v18_path),
            "--output-root", str(out_root),
        ]) == 0
        assert audit_teacher_prior_visibility.main_with_args([
            "--library", str(out_root / "fake_v18.zarr"),
            "--output-dir", str(audit_dir),
            "--min-mask-coverage", "0.001",
            "--visible-delta", "18",
        ]) == 0
        rows = pq.read_table(str(audit_dir / "visibility_audit.parquet")).to_pylist()
        buckets = {int(row["tile_id"]): row["bucket"] for row in rows}
        keeps = {int(row["tile_id"]): bool(row["keep"]) for row in rows}
        assert buckets == {0: "weak", 1: "visible", 2: "tiny"}
        assert keeps == {0: False, 1: True, 2: False}
        kept = pq.read_table(str(audit_dir / "kept_tiles.parquet")).to_pylist()
        assert [row["keep"] for row in kept] == [False, True, False]
