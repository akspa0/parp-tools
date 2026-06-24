from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_family_catalog import (  # noqa: E402
    BrushFamily,
    CanvasCache,
    build_families,
    discover_canvas_dirs,
    extract_alpha_crop,
    filter_families,
    group_members_by_cluster,
    load_near_clusters,
    pad_crop,
    write_family_outputs,
)


def _make_analysis_root(tmp_path: Path) -> Path:
    root = tmp_path / "analysis"
    near_dir = root / "dedupe" / "near"
    near_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        {
            "cluster_id": "near_abc",
            "member_count": 2,
            "build_count": 1,
            "map_count": 1,
            "layer_count": 1,
            "builds": ["0_5_3_3368"],
            "maps": ["Azeroth"],
            "layer_indices": [0],
            "crop_w": 8,
            "crop_h": 8,
            "area": 64,
            "example_region_id": "r_0",
            "example_bbox_xywh": [0, 0, 8, 8],
            "mcly_texture_ids": [],
            "mcly_active_layers": [0],
        }
    ]
    members = [
        {
            "cluster_id": "near_abc",
            "region_id": "r_0",
            "build": "0_5_3_3368",
            "map_name": "Azeroth",
            "layer_slot": 0,
            "layer_idx": 0,
            "bbox_xywh": [0, 0, 8, 8],
            "area": 64,
            "mcly_texture_ids": [],
            "mcly_active_layers": [0],
        },
        {
            "cluster_id": "near_abc",
            "region_id": "r_1",
            "build": "0_5_3_3368",
            "map_name": "Azeroth",
            "layer_slot": 0,
            "layer_idx": 0,
            "bbox_xywh": [20, 20, 8, 8],
            "area": 64,
            "mcly_texture_ids": [],
            "mcly_active_layers": [0],
        },
    ]

    import pyarrow.parquet as pq
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(patterns), near_dir / "near_patterns.parquet")
    pq.write_table(pa.Table.from_pylist(members), near_dir / "near_pattern_members.parquet")

    target_dir = root / "0_5_3_3368_Azeroth_tilefull" / "canvas"
    target_dir.mkdir(parents=True, exist_ok=True)
    canvas = zarr.open_group(str(target_dir / "canvas.zarr"), mode="w")
    canvas.create_array("alpha_256", shape=(64, 64, 1), dtype=np.float32, fill_value=0.0)
    canvas["alpha_256"][0:8, 0:8, 0] = 1.0
    canvas["alpha_256"][20:28, 20:28, 0] = 1.0
    canvas.create_array("alpha_layer_indices", data=np.array([0], dtype=np.int32))

    return root


def test_discover_canvas_dirs(tmp_path: Path) -> None:
    root = _make_analysis_root(tmp_path)
    index = discover_canvas_dirs(root)
    assert index == {("0_5_3_3368", "Azeroth"): root / "0_5_3_3368_Azeroth_tilefull" / "canvas"}


def test_load_near_clusters(tmp_path: Path) -> None:
    root = _make_analysis_root(tmp_path)
    patterns, members = load_near_clusters(root)
    assert len(patterns) == 1
    assert len(members) == 2


def test_filter_families(tmp_path: Path) -> None:
    root = _make_analysis_root(tmp_path)
    patterns, members = load_near_clusters(root)
    selected = filter_families(patterns, min_members=2)
    assert len(selected) == 1
    selected = filter_families(patterns, min_members=3)
    assert len(selected) == 0


def test_build_families(tmp_path: Path) -> None:
    root = _make_analysis_root(tmp_path)
    patterns, members = load_near_clusters(root)
    selected = filter_families(patterns, min_members=2)
    members_by_cluster = group_members_by_cluster(members)
    cache = CanvasCache(discover_canvas_dirs(root))
    families, tensor = build_families(selected, members_by_cluster, cache, crop_size=32)
    cache.close()
    assert len(families) == 1
    assert isinstance(families[0], BrushFamily)
    assert tensor.shape == (1, 32, 32)
    assert tensor[0].max() > 0.0


def test_pad_crop_scales_large_crop() -> None:
    crop = np.ones((20, 20), dtype=np.float32)
    padded = pad_crop(crop, target_size=16)
    assert padded.shape == (16, 16)
    assert padded.max() > 0.0


def test_extract_alpha_crop(tmp_path: Path) -> None:
    root = _make_analysis_root(tmp_path)
    canvas_dir = discover_canvas_dirs(root)[("0_5_3_3368", "Azeroth")]
    canvas = zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r")
    crop = extract_alpha_crop(canvas, (0, 0, 8, 8), 0)
    assert crop.shape == (8, 8)
    assert crop.max() == 1.0


def test_write_family_outputs(tmp_path: Path) -> None:
    root = _make_analysis_root(tmp_path)
    patterns, members = load_near_clusters(root)
    selected = filter_families(patterns, min_members=2)
    members_by_cluster = group_members_by_cluster(members)
    cache = CanvasCache(discover_canvas_dirs(root))
    families, tensor = build_families(selected, members_by_cluster, cache, crop_size=32)
    cache.close()
    out = tmp_path / "catalog"
    summary = write_family_outputs(out, families, tensor)
    assert summary["family_count"] == 1
    assert (out / "families.parquet").exists()
    assert (out / "families.zarr").exists()
