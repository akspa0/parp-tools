from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_near_dedupe import (  # noqa: E402
    _hamming_variants,
    _normalize_crop,
    _transforms,
    cluster_near_duplicates,
    write_near_dedupe_outputs,
)
from harvester.fractal_raw_analysis import RawComponentFingerprint  # noqa: E402


def _make_canvas(tmp_path: Path, shape: tuple[int, int, int]) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "canvas.zarr"), mode="w")
    root.create_array("alpha_256", shape=shape, dtype=np.float32, fill_value=0.0)
    return root


def _fingerprint(x: int, y: int, w: int, h: int, layer_slot: int = 0) -> RawComponentFingerprint:
    return RawComponentFingerprint(
        region_id=f"r_{x}_{y}_{w}_{h}",
        pattern_id="pat_unknown",
        build="0_5_3_3368",
        map_name="Azeroth",
        layer_idx=layer_slot,
        layer_slot=layer_slot,
        bbox_xywh=(x, y, w, h),
        area=w * h,
        crop_w=w,
        crop_h=h,
        alpha_mean=1.0,
        alpha_max=1.0,
        tile_coverage_count=1,
        tile_coverage=[{"tile_id": 0, "pixels": w * h}],
        mcly_texture_ids=[],
        mcly_active_layers=[],
    )


def test_normalize_crop_preserves_small_aspect_ratio() -> None:
    crop = np.zeros((10, 30), dtype=np.float32)
    crop[2:8, 5:25] = 1.0
    thumb = _normalize_crop(crop, size=32)
    assert thumb.shape == (32, 32)
    assert thumb.any()


def test_transforms_produce_eight_variants() -> None:
    thumb = np.zeros((32, 32), dtype=bool)
    thumb[10:20, 5:15] = True
    assert len(_transforms(thumb)) == 8


def test_hamming_variants_radius_one() -> None:
    base = "0" * 64
    variants = _hamming_variants(base, radius=1)
    assert len(variants) == 65
    assert base in variants
    assert all(len(v) == 64 for v in variants)


def test_cluster_groups_translated_duplicates(tmp_path: Path) -> None:
    canvas = _make_canvas(tmp_path, (128, 256, 1))
    # Two identical 8x8 squares at different x positions on layer 0.
    canvas["alpha_256"][0:8, 10:18, 0] = 1.0
    canvas["alpha_256"][0:8, 100:108, 0] = 1.0

    fingerprints = [
        _fingerprint(10, 0, 8, 8, 0),
        _fingerprint(100, 0, 8, 8, 0),
    ]
    clusters = cluster_near_duplicates(fingerprints, canvas, threshold=0.5, size=16, radius=0)

    assert len(clusters) == 1
    assert sum(len(members) for members in clusters.values()) == 2


def test_cluster_groups_mirrored_duplicates(tmp_path: Path) -> None:
    canvas = _make_canvas(tmp_path, (64, 64, 1))
    # An L-shape and its mirror image (dihedral transform).
    canvas["alpha_256"][0:8, 0:4, 0] = 1.0
    canvas["alpha_256"][0:4, 0:8, 0] = 1.0
    canvas["alpha_256"][20:28, 10:14, 0] = 1.0
    canvas["alpha_256"][20:24, 10:18, 0] = 1.0

    fingerprints = [_fingerprint(0, 0, 8, 8, 0), _fingerprint(10, 20, 8, 8, 0)]
    clusters = cluster_near_duplicates(fingerprints, canvas, threshold=0.5, size=16, radius=0)

    assert len(clusters) == 1
    assert sum(len(members) for members in clusters.values()) == 2


def test_cluster_respects_layer_slot(tmp_path: Path) -> None:
    canvas = _make_canvas(tmp_path, (32, 32, 2))
    canvas["alpha_256"][0:8, 0:8, 0] = 1.0
    canvas["alpha_256"][0:8, 0:8, 1] = 1.0

    fingerprints = [_fingerprint(0, 0, 8, 8, 0), _fingerprint(0, 0, 8, 8, 1)]
    clusters = cluster_near_duplicates(fingerprints, canvas, threshold=0.5, size=16, radius=0)

    # Same spatial crop on different layers should still cluster because alpha shape is identical.
    assert len(clusters) == 1


def test_write_near_dedupe_outputs(tmp_path: Path) -> None:
    canvas = _make_canvas(tmp_path, (32, 32, 1))
    canvas["alpha_256"][0:8, 0:8, 0] = 1.0
    fp = _fingerprint(0, 0, 8, 8, 0)
    clusters = {"near_abc": [fp]}
    summary = write_near_dedupe_outputs(tmp_path / "near", clusters)
    assert summary["cluster_count"] == 1
    assert summary["member_count"] == 1
    assert (tmp_path / "near" / "near_patterns.parquet").exists()
    assert (tmp_path / "near" / "near_pattern_members.parquet").exists()
