from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_library import (  # noqa: E402
    FractalBrushLibrary,
    build_trainable_library,
    smoke_load_library,
    stable_sample_id,
)


def test_build_trainable_library_filters_rejected_and_loads_samples(tmp_path: Path) -> None:
    canvas_dir = _write_canvas(tmp_path)
    regions_path = _write_regions(tmp_path)
    output_dir = tmp_path / "library"

    summary = build_trainable_library(
        canvas_dir=canvas_dir,
        regions_path=regions_path,
        output_dir=output_dir,
        crop_size=32,
    )

    assert summary["sample_count"] == 2
    assert summary["rejected_count"] == 2
    dataset = FractalBrushLibrary(output_dir)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["alpha"].shape == (32, 32, 2)
    assert sample["height"].shape == (33, 33)
    assert sample["normal"].shape == (33, 33, 3)
    assert sample["mcly_texture_ids"].shape == (2, 2, 4)
    assert sample["metadata"]["curation_label"] in {"accepted_candidate", "fractal_member"}
    assert sample["metadata"]["dominant_mcly_texture_id"] == 7
    assert sample["metadata"]["mcly_texture_id_counts"]["7"] > 0
    assert sample["metadata"]["mcly_active_layer_coverage"]["1"] == 1.0
    assert sample["provenance"]["source_region_id"]


def test_stable_ids_and_splits_are_deterministic(tmp_path: Path) -> None:
    canvas_dir = _write_canvas(tmp_path)
    regions_path = _write_regions(tmp_path)

    first = build_trainable_library(canvas_dir=canvas_dir, regions_path=regions_path, output_dir=tmp_path / "a", crop_size=32)
    second = build_trainable_library(canvas_dir=canvas_dir, regions_path=regions_path, output_dir=tmp_path / "b", crop_size=32)

    assert first["split_counts"] == second["split_counts"]
    rows_a = pq.read_table(tmp_path / "a" / "samples.parquet").to_pylist()
    rows_b = pq.read_table(tmp_path / "b" / "samples.parquet").to_pylist()
    assert [row["sample_id"] for row in rows_a] == [row["sample_id"] for row in rows_b]
    assert rows_a[0]["sample_id"] == stable_sample_id(rows_a[0]["region_id"], tuple(rows_a[0]["bbox_xywh"]), rows_a[0]["layer_idx"])


def test_smoke_loader_rejects_no_default_rejected_labels(tmp_path: Path) -> None:
    canvas_dir = _write_canvas(tmp_path)
    regions_path = _write_regions(tmp_path)
    output_dir = tmp_path / "library"
    build_trainable_library(canvas_dir=canvas_dir, regions_path=regions_path, output_dir=output_dir, crop_size=32)

    smoke = smoke_load_library(output_dir, count=2)

    assert smoke["loaded"] == 2
    assert set(smoke["labels"]) == {"accepted_candidate", "fractal_member"}


def _write_canvas(tmp_path: Path) -> Path:
    root = zarr.open_group(str(tmp_path / "canvas.zarr"), mode="w")
    alpha = np.zeros((64, 64, 2), dtype=np.float32)
    alpha[10:20, 10:22, 1] = 0.7
    alpha[34:52, 34:54, 1] = 0.9
    height = np.arange(65 * 65, dtype=np.float32).reshape(65, 65)
    normals = np.zeros((65, 65, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    mcly_ids = np.full((4, 4, 4), -1, dtype=np.int32)
    mcly_ids[:, :, 1] = 7
    mcly_mask = np.zeros((4, 4, 4), dtype=np.float32)
    mcly_mask[:, :, 1] = 1.0
    tile_ids = np.ones((64, 64), dtype=np.int32)
    root.create_array("alpha_256", data=alpha)
    root.create_array("height_257", data=height)
    root.create_array("normal_xyz", data=normals)
    root.create_array("mcly_texture_ids", data=mcly_ids)
    root.create_array("mcly_layer_mask", data=mcly_mask)
    root.create_array("tile_id_256", data=tile_ids)
    root.attrs["layout"] = {"build": "test", "map_name": "Map"}
    return tmp_path


def _write_regions(tmp_path: Path) -> Path:
    rows = [
        _region("r1", "accepted_candidate", None, (8, 8, 20, 20), 120),
        _region("r2", "fractal_member", None, (32, 32, 28, 28), 360),
        _region("r3", "composite_chonker", "large_full_map_region", (0, 0, 64, 64), 3000),
        _region("r4", "one_off_detail", "large_sparse_single_tile_region", (2, 2, 50, 14), 200),
    ]
    path = tmp_path / "fractal_regions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _region(region_id: str, label: str, reason: str | None, bbox: tuple[int, int, int, int], area: int) -> dict[str, object]:
    return {
        "region_id": region_id,
        "build": "test",
        "map_name": "Map",
        "layer_slot": 1,
        "layer_idx": 1,
        "bbox_xywh": list(bbox),
        "area": area,
        "tile_coverage_count": 1,
        "tile_coverage": [{"tile_id": 1, "pixel_count": area}],
        "alpha_mean": 0.5,
        "alpha_max": 0.9,
        "height_mean": 1.0,
        "height_std": 0.25,
        "height_range": 2.0,
        "normal_mean_xyz": [0.0, 0.0, 1.0],
        "mcly_texture_ids": [7],
        "mcly_active_layers": [1],
        "curation_label": label,
        "rejection_reason": reason,
        "linked_component_ids": ["c1"],
    }
