from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_segments import classify_region, segment_canvas_regions  # noqa: E402


def test_classify_region_rejects_chonkers() -> None:
    label, reason = classify_region(
        bbox_xywh=(0, 0, 900, 900),
        area=900 * 900,
        total_pixels=1024 * 1024,
        tile_coverage_count=8,
        alpha_mean=0.5,
    )

    assert label == "composite_chonker"
    assert reason == "large_full_map_region"


def test_classify_region_identifies_multi_tile_members() -> None:
    label, reason = classify_region(
        bbox_xywh=(240, 8, 96, 80),
        area=2400,
        total_pixels=512 * 256,
        tile_coverage_count=2,
        alpha_mean=0.4,
    )

    assert label == "fractal_member"
    assert reason is None


def test_classify_region_rejects_tiny_slivers_before_atomic_training() -> None:
    label, reason = classify_region(
        bbox_xywh=(240, 8, 96, 20),
        area=1200,
        total_pixels=512 * 256,
        tile_coverage_count=2,
        alpha_mean=0.4,
    )

    assert label == "too_small_unique"
    assert reason == "below_minimum_adt_footprint"


def test_segment_canvas_regions_emits_tile_coverage_and_spatial_stats(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "canvas.zarr"), mode="w")
    alpha = np.zeros((256, 512, 2), dtype=np.float32)
    alpha[20:100, 240:320, 1] = 0.8
    tile_id_256 = np.full((256, 512), -1, dtype=np.int32)
    tile_id_256[:, :256] = 10
    tile_id_256[:, 256:] = 11
    height = np.ones((257, 513), dtype=np.float32) * 42.0
    normals = np.zeros((257, 513, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    mcly_ids = np.zeros((16, 32, 4), dtype=np.int32)
    mcly_ids[:, :, 1] = 3
    mcly_mask = np.zeros((16, 32, 4), dtype=np.float32)
    mcly_mask[:, :, 1] = 1.0
    root.create_array("alpha_256", data=alpha)
    root.create_array("alpha_layer_indices", data=np.array([0, 1], dtype=np.int32))
    root.create_array("tile_id_256", data=tile_id_256)
    root.create_array("height_257", data=height)
    root.create_array("normal_xyz", data=normals)
    root.create_array("mcly_texture_ids", data=mcly_ids)
    root.create_array("mcly_layer_mask", data=mcly_mask)
    root.attrs["layout"] = {"build": "test", "map_name": "Map"}

    regions = segment_canvas_regions(root, threshold=0.05, min_area=16)

    assert len(regions) == 1
    region = regions[0]
    assert region.layer_idx == 1
    assert region.curation_label == "fractal_member"
    assert region.tile_coverage_count == 2
    assert [item["tile_id"] for item in region.tile_coverage] == [11, 10]
    assert region.height_mean == 42.0
    assert region.normal_mean_xyz == (0.0, 0.0, 1.0)
    assert 3 in region.mcly_texture_ids
    assert region.mcly_active_layers == [1]
