"""Unit tests for Batch-Export Clean Pairs & DXT1 Parity (Spec 274)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v60.batch_export_clean_pairs import (
    combine_wdl_lattice,
    export_clean_pairs_dataset,
    find_zarr_map_stores,
    scan_clean_pair_candidates,
)
from harvester.v60.dxt1_parity import (
    dxt1_encode_decode_rgb,
    quantize_rgb565,
    rgb565_to_uint16,
)


def test_dxt1_parity_shape_and_dtype() -> None:
    # 256x256 RGB float input
    img_float = np.linspace(0.0, 1.0, 256 * 256 * 3, dtype=np.float32).reshape(256, 256, 3)
    out = dxt1_encode_decode_rgb(img_float)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255


def test_dxt1_block_quantization_effect() -> None:
    # Flat block with slight gradient
    block = np.full((16, 16, 3), 128, dtype=np.uint8)
    block[0, 0] = [135, 120, 130]
    out = dxt1_encode_decode_rgb(block)
    assert out.shape == (16, 16, 3)
    # Output values must be quantized to RGB565 endpoints and 4-color palette
    unique_colors = len(np.unique(out.reshape(-1, 3), axis=0))
    assert unique_colors <= 4


def test_combine_wdl_lattice() -> None:
    outer = np.ones((17, 17), dtype=np.float32) * 100.0
    inner = np.ones((16, 16), dtype=np.float32) * 50.0

    lattice = combine_wdl_lattice(outer, inner)
    assert lattice.shape == (33, 33)
    assert np.allclose(lattice[0::2, 0::2], 100.0)
    assert np.allclose(lattice[1::2, 1::2], 50.0)
    # Interpolated cells must be between outer and inner
    assert np.allclose(lattice[0::2, 1::2], 100.0)


def _create_mock_zarr_map_store(root: Path, map_name: str, count: int = 4) -> Path:
    store = root / f"3_0_1_8303-{map_name}.zarr"
    store.mkdir(parents=True)
    g = zarr.open_group(str(store), mode="w")

    rgb = np.random.randint(20, 220, (count, 256, 256, 3), dtype=np.uint8)
    shadow = np.random.uniform(0.1, 0.9, (count, 256, 256)).astype(np.float32)
    height = np.random.uniform(50.0, 350.0, (count, 257, 257)).astype(np.float32)
    outer = np.random.uniform(50.0, 350.0, (count, 17, 17)).astype(np.float32)
    inner = np.random.uniform(50.0, 350.0, (count, 16, 16)).astype(np.float32)

    g.create_array("minimap_rgb", data=rgb)
    g.create_array("terrain_shadow_256", data=shadow)
    g.create_array("height_257", data=height)
    g.create_array("wdl_outer_17", data=outer)
    g.create_array("wdl_inner_16", data=inner)

    rows = []
    for i in range(count):
        rows.append(
            {
                "map": map_name,
                "tile_x": 20 + i,
                "tile_y": 30 + i,
                "build_id": "3_0_1_8303",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), store / "index.parquet")
    return store


def test_export_clean_pairs_dry_run_and_build(tmp_path: Path) -> None:
    source_root = tmp_path / "v60_stores"
    source_root.mkdir()
    _create_mock_zarr_map_store(source_root, "Azeroth", count=4)
    _create_mock_zarr_map_store(source_root, "Kalimdor", count=3)

    stores = find_zarr_map_stores(source_root)
    assert len(stores) == 2

    out_zarr = tmp_path / "clean_pairs.zarr"

    # Dry-run
    dry_plan = export_clean_pairs_dataset(
        source_stores=stores,
        output_path=out_zarr,
        dry_run=True,
    )
    assert dry_plan["dry_run"] is True
    assert dry_plan["total_candidate_tiles"] == 7
    assert not out_zarr.exists()

    # Real build
    result = export_clean_pairs_dataset(
        source_stores=stores,
        output_path=out_zarr,
        validation_map="Kalimdor",
        dry_run=False,
    )
    assert result["exported_count"] == 7
    assert result["val_count"] == 3
    assert result["train_count"] == 4
    assert out_zarr.exists()

    # Validate output Zarr arrays
    res_g = zarr.open_group(str(out_zarr), mode="r")
    assert res_g["real_minimap_rgb"].shape == (7, 256, 256, 3)
    assert res_g["synthetic_hillshade_dxt1"].shape == (7, 256, 256, 3)
    assert res_g["wdl_lattice_33"].shape == (7, 33, 33)
    assert res_g["height_257"].shape == (7, 257, 257)

    # Validate index.parquet
    tbl = pq.read_table(out_zarr / "index.parquet")
    assert len(tbl) == 7
    assert set(tbl.column_names) >= {"map", "tile_x", "tile_y", "split", "relief_delta_z"}
    assert tbl.filter(pa.compute.equal(tbl["map"], "Kalimdor"))["split"].to_pylist() == ["val", "val", "val"]
