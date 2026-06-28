"""Tests for spec 077 Phase 5 (US4) ADT-free object explanation contracts."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest
import zarr
import zarr.codecs
import zarr.storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _entry in (_REPO_ROOT, _SRC_DIR, _SCRIPTS_DIR):
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

from harvester.asset_matcher import (  # noqa: E402
    _decode_png_mask,
    _decode_png_thumbnail,
    _hamming_hex,
    _masked_correlation,
    _phash,
    _resize_nearest,
    build_hypothesis_from_bbox,
    build_thumbnails_from_captures,
    score_candidates,
)
from harvester.inference_object import (  # noqa: E402
    AssetCandidate,
    InferenceObjectHypothesis,
    RecoveredObjectPlacement,
    collect_hypotheses,
    hypothesis_to_recovered,
)
import build_adt_free_prior  # noqa: E402


# --- phash + hamming -------------------------------------------------------

def test_phash_is_stable_and_different_for_different_inputs() -> None:
    rng = np.random.default_rng(123)
    a = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    hash_a = _phash(a)
    hash_a2 = _phash(a)
    hash_b = _phash(b)
    assert hash_a == hash_a2
    assert hash_a != hash_b


def test_hamming_hex_returns_max_for_length_mismatch() -> None:
    assert _hamming_hex("abc", "abcd") == 4 * 4


# --- masked correlation ---------------------------------------------------

def test_masked_correlation_zero_for_empty_masks() -> None:
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    assert _masked_correlation(a, mask, b, mask) == 0.0


def test_masked_correlation_high_for_identical_masked_regions() -> None:
    # Two solid blue squares on red background; masks mark the squares.
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    a[..., 0] = 200  # red
    a[4:12, 4:12, :] = (10, 10, 250)  # blue square
    b = a.copy()
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    c = _masked_correlation(a, mask, b, mask)
    assert c > 0.95


def test_masked_correlation_low_for_unrelated_masks() -> None:
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    a[2:6, 2:6] = (10, 10, 250)
    b = np.zeros((16, 16, 3), dtype=np.uint8)
    b[10:14, 10:14] = (250, 10, 10)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    mask_b = np.zeros((16, 16), dtype=np.uint8)
    mask_b[10:14, 10:14] = 255
    c = _masked_correlation(a, mask, b, mask_b)
    # No mask overlap -> 0
    assert c == 0.0


# --- resize nearest -------------------------------------------------------

def test_resize_nearest_passes_through_when_shape_matches() -> None:
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = _resize_nearest(arr, 4)
    np.testing.assert_array_equal(out, arr)


def test_resize_nearest_downsamples() -> None:
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    out = _resize_nearest(arr, 4)
    assert out.shape == (4, 4)
    # Nearest-neighbor downsample should land on integer index samples.
    assert out[0, 0] == arr[0, 0]


# --- InferenceObjectHypothesis contract -----------------------------------

def test_hypothesis_top_candidate_returns_highest_score() -> None:
    h = InferenceObjectHypothesis(
        tile_id=0,
        instance_id=1,
        mask_bbox=(0, 0, 10, 10),
        mask_confidence=1.0,
        asset_candidate_paths=("a.m2", "b.m2", "c.m2"),
        asset_candidate_scores=(0.2, 0.7, 0.4),
        asset_candidate_library_ids=("objlib_1", "objlib_2", "objlib_3"),
        pose_xy=(5.0, 5.0),
        pose_yaw=0.0,
    )
    top = h.top_candidate()
    assert top is not None
    assert top.asset_path == "b.m2"
    assert top.score == 0.7


def test_hypothesis_ranked_candidates_is_sorted_descending() -> None:
    h = InferenceObjectHypothesis(
        tile_id=0,
        instance_id=1,
        mask_bbox=(0, 0, 10, 10),
        mask_confidence=1.0,
        asset_candidate_paths=("a.m2", "b.m2", "c.m2", "d.m2"),
        asset_candidate_scores=(0.2, 0.7, 0.4, 0.9),
        asset_candidate_library_ids=("o1", "o2", "o3", "o4"),
    )
    ranked = h.ranked_candidates()
    assert [c.asset_path for c in ranked] == ["d.m2", "b.m2", "c.m2", "a.m2"]
    assert ranked[0].score == 0.9


def test_hypothesis_to_recovered_lifts_xy_yaw_and_z() -> None:
    h = InferenceObjectHypothesis(
        tile_id=0,
        instance_id=1,
        mask_bbox=(0, 0, 10, 10),
        mask_confidence=0.8,
        asset_candidate_paths=("foo.wmo",),
        asset_candidate_scores=(0.6,),
        asset_candidate_library_ids=("objlib_foo",),
        pose_xy=(123.0, 456.0),
        pose_yaw=1.57,
    )
    placement = hypothesis_to_recovered(h, terrain_z=42.0)
    assert isinstance(placement, RecoveredObjectPlacement)
    assert placement.asset_path == "foo.wmo"
    assert placement.x == 123.0
    assert placement.y == 456.0
    assert placement.z_from_terrain == 42.0
    assert placement.yaw == 1.57
    assert placement.confidence == 0.6
    # FR-018: pitch / roll / scale are deferred
    assert placement.pitch is None
    assert placement.roll is None
    assert placement.scale is None


def test_collect_hypotheses_sorts_by_top_score() -> None:
    hyps = [
        InferenceObjectHypothesis(
            tile_id=2,
            instance_id=1,
            mask_bbox=(0, 0, 5, 5),
            mask_confidence=1.0,
            asset_candidate_paths=("a.m2",),
            asset_candidate_scores=(0.3,),
        ),
        InferenceObjectHypothesis(
            tile_id=1,
            instance_id=1,
            mask_bbox=(0, 0, 5, 5),
            mask_confidence=1.0,
            asset_candidate_paths=("a.m2",),
            asset_candidate_scores=(0.9,),
        ),
    ]
    ordered = collect_hypotheses(hyps)
    assert ordered[0].tile_id == 1
    assert ordered[1].tile_id == 2


# --- score_candidates with synthetic thumbnails ----------------------------

def _write_thumbnail(captures_dir: Path, variant_id: str, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color).save(captures_dir / f"{variant_id}_image.png")
    Image.new("L", (32, 32), 200).save(captures_dir / f"{variant_id}_mask.png")


def test_score_candidates_returns_top_k_for_matching_crop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captures = root / "captures"
        captures.mkdir()
        # Two library entries with different colors; minimap has a blue square
        # in the top-left that should match entry "blue" better.
        for vid, color in [("v_blue", (10, 10, 250)), ("v_red", (250, 10, 10))]:
            _write_thumbnail(captures, vid, color)
        library_assets = [
            {
                "library_id": "objlib_blue",
                "original_asset_path": "blue.m2",
                "normalized_asset_path": "blue.m2",
                "asset_type": "m2",
                "capture_status": "captured",
                "preferred_variant_id": "v_blue",
            },
            {
                "library_id": "objlib_red",
                "original_asset_path": "red.m2",
                "normalized_asset_path": "red.m2",
                "asset_type": "m2",
                "capture_status": "captured",
                "preferred_variant_id": "v_red",
            },
        ]
        thumbs = build_thumbnails_from_captures(captures, library_assets)
        assert len(thumbs) == 2
        minimap = np.zeros((64, 64, 3), dtype=np.uint8)
        minimap[10:30, 10:30] = (10, 10, 250)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 200
        candidates = score_candidates(minimap, mask, (10, 10, 30, 30), thumbs, top_k=2)
        assert len(candidates) == 2
        # Blue entry should be the top match.
        assert candidates[0].library_id == "objlib_blue"
        assert candidates[0].score > candidates[1].score


def test_build_hypothesis_emits_xy_center_and_only_top_k_assets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captures = root / "captures"
        captures.mkdir()
        for vid, color in [("v_a", (10, 10, 250)), ("v_b", (250, 10, 10)), ("v_c", (10, 250, 10))]:
            _write_thumbnail(captures, vid, color)
        library_assets = [
            {"library_id": f"objlib_{c}", "original_asset_path": f"{c}.m2",
             "normalized_asset_path": f"{c}.m2", "asset_type": "m2",
             "capture_status": "captured", "preferred_variant_id": f"v_{c}"}
            for c in ("a", "b", "c")
        ]
        thumbs = build_thumbnails_from_captures(captures, library_assets)
        minimap = np.zeros((64, 64, 3), dtype=np.uint8)
        minimap[5:15, 5:15] = (10, 10, 250)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:15, 5:15] = 200
        h = build_hypothesis_from_bbox(
            tile_id=42,
            instance_id=3,
            minimap_rgb=minimap,
            object_mask=mask,
            bbox_xyxy=(5, 5, 15, 15),
            thumbnails=thumbs,
            top_k=2,
        )
        # Center of the bbox.
        assert h.pose_xy == (10.0, 10.0)
        # FR-018: yaw is 0 in the first pass.
        assert h.pose_yaw == 0.0
        # pose_z_from_terrain remains None until the height lane fills it in.
        assert h.pose_z_from_terrain is None
        # top_k honored.
        assert len(h.asset_candidate_paths) <= 2


# --- ADT-free prior builder end-to-end ------------------------------------

CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=1, shuffle="bitshuffle")


def _make_v18_for_adt_free(path: Path, n_tiles: int = 2) -> None:
    import shutil
    if path.exists():
        shutil.rmtree(path)
    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store)
    minimap = np.zeros((n_tiles, 256, 256, 3), dtype=np.uint8)
    for i in range(n_tiles):
        minimap[i, :, :, 0] = 100 + i * 20
        minimap[i, :, :, 1] = 50
        minimap[i, :, :, 2] = 200
    root.create_array("minimap_rgb", data=minimap, chunks=(n_tiles, 256, 256, 3), compressors=CODEC)


def _write_predicted_mask_npz(path: Path, masks: np.ndarray) -> None:
    np.savez(str(path), predicted_mask=masks)


def test_build_adt_free_prior_tensor_handles_empty_mask() -> None:
    minimap = np.full((256, 256, 3), 100, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    out = build_adt_free_prior.build_adt_free_prior_tensor(minimap, mask)
    assert out.shape == (256, 256, 5)
    # No object pixels: suppressed RGB equals raw, mask band is zero,
    # confidence is full.
    np.testing.assert_array_equal(out[..., 0], minimap[..., 0])
    np.testing.assert_array_equal(out[..., 3], np.zeros((256, 256), dtype=np.uint8))
    np.testing.assert_array_equal(out[..., 4], np.full((256, 256), 255, dtype=np.uint8))


def test_build_adt_free_prior_tensor_suppresses_object_pixels() -> None:
    minimap = np.zeros((256, 256, 3), dtype=np.uint8)
    minimap[:128, :, 0] = 200  # red on top half
    minimap[128:, :, 2] = 250  # blue on bottom half
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[128:, :] = 255
    out = build_adt_free_prior.build_adt_free_prior_tensor(minimap, mask)
    # Bottom half should now be near red (median of non-object).
    assert int(out[200, 100, 0]) > 100
    assert int(out[200, 100, 2]) < 100
    # Top half unchanged.
    np.testing.assert_array_equal(out[:128, :, :], minimap[:128, :, :])


def test_build_adt_free_prior_cli_writes_zarr_and_tiles_parquet() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v18_path = root / "v18.zarr"
        mask_npz = root / "masks.npz"
        out_root = root / "out"
        _make_v18_for_adt_free(v18_path, n_tiles=2)
        predicted = np.zeros((2, 256, 256), dtype=np.uint8)
        predicted[0, 60:120, 60:120] = 255
        _write_predicted_mask_npz(mask_npz, predicted)
        exit_code = build_adt_free_prior.main_with_args(
            [
                "--v18-path", str(v18_path),
                "--predicted-mask", str(mask_npz),
                "--output-root", str(out_root),
            ]
        )
        assert exit_code == 0
        out_path = out_root / "v18.zarr"
        assert (out_path / "tiles.parquet").exists()
        store = zarr.storage.LocalStore(str(out_path), read_only=True)
        root_grp = zarr.open_group(store, mode="r")
        assert "raw_minimap_rgb_256" in root_grp
        assert "predicted_object_mask_256" in root_grp
        assert "processed_minimap_prior_256" in root_grp
        prior = np.asarray(root_grp["processed_minimap_prior_256"][:])
        assert prior.shape == (2, 256, 256, 5)
        # Tile 0 has predicted object pixels; suppressed RGB != raw at
        # those pixels; mask band reflects the prediction.
        assert int(prior[0, 80, 80, 3]) > 0
        # Tile 1 is empty; pass-through.
        np.testing.assert_array_equal(prior[1, :, :, 3], np.zeros((256, 256), dtype=np.uint8))
