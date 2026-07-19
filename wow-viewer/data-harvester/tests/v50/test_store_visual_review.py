from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import zarr

from harvester.v50.store_visual_review import COLUMN_TITLES, render_store_review


def test_render_store_review_writes_same_tile_sheet_and_object_policy(tmp_path: Path):
    store = tmp_path / "fixture.zarr"
    group = zarr.open_group(str(store), mode="w")
    authored = np.full((1, 256, 256, 3), 80, dtype=np.uint8)
    authored[:, 32:64, 32:64] = 240  # object-like authored-only mark
    synthetic = np.full((1, 256, 256, 3), 60, dtype=np.uint8)
    detail = np.repeat(np.repeat(synthetic, 4, axis=1), 4, axis=2)
    height = np.linspace(0, 30, 257 * 257, dtype=np.float32).reshape(1, 257, 257)
    normals = np.zeros((1, 257, 257, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    group.create_array("minimap_rgb_authored", data=authored)
    group.create_array("minimap_rgb", data=synthetic)
    group.create_array("minimap_rgb_1024", data=detail)
    group.create_array("height_257", data=height)
    group.create_array("normal_xyz", data=normals)
    group.create_array("mddf_count", data=np.asarray([3], dtype=np.int32))
    group.create_array("modf_count", data=np.asarray([1], dtype=np.int32))
    pq.write_table(
        pa.Table.from_pylist([{"map": "Fixture", "tile_x": 10, "tile_y": 20}]),
        store / "index.parquet",
    )

    output = tmp_path / "review.png"
    report = render_store_review(store, output, sample_count=1)

    assert output.exists()
    with Image.open(output) as image:
        assert image.width == len(COLUMN_TITLES) * 256
    assert report["pixel_equality_required"] is False
    assert report["synthetic_object_policy"] == "terrain_only_no_objects"
    assert report["rows"][0]["mddf_count"] == 3
    assert report["rows"][0]["modf_count"] == 1
