#!/usr/bin/env python3
"""Unit tests for ZarrElevationDataset and Felwood indentation supervision (Spec 269)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import zarr

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v60.zarr_elevation_dataset import (
    ZarrElevationDataset,
    compute_photometric_normals_from_rgb,
    extract_zarr_samples,
    is_classic_indentation_tile,
    is_felwood_tile,
)


def test_felwood_and_indentation_classification():
    """Verify Felwood canyon and classic indentation coordinate detection."""
    # Felwood in Kalimdor: X in [25, 29], Y in [19, 26]
    assert is_felwood_tile("Kalimdor", 27, 22) is True
    assert is_felwood_tile("kalimdor", 25, 19) is True
    assert is_felwood_tile("Kalimdor", 29, 26) is True
    assert is_felwood_tile("Kalimdor", 32, 40) is False  # Outside Felwood (Durotar/Barrens)
    assert is_felwood_tile("Azeroth", 27, 22) is False   # Azeroth is not Felwood

    # Classic indentation corridors
    assert is_classic_indentation_tile("Kalimdor", 27, 22) is True  # Felwood
    assert is_classic_indentation_tile("Kalimdor", 28, 30) is True  # Ashenvale
    assert is_classic_indentation_tile("Kalimdor", 30, 48) is True  # Thousand Needles
    assert is_classic_indentation_tile("Azeroth", 35, 40) is True   # Searing Gorge / Badlands
    assert is_classic_indentation_tile("Kalimdor", 10, 10) is False


def test_compute_photometric_normals():
    """Verify photometric surface normal computation from RGB."""
    rgb = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    normals = compute_photometric_normals_from_rgb(rgb)
    assert normals.shape == (256, 256, 2)
    assert normals.dtype == np.float32


def test_zarr_elevation_dataset_item():
    """Verify tensor contract and shapes produced by ZarrElevationDataset."""
    sample = {
        "stem": "3_3_5_12340_Kalimdor_27_22",
        "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "z_gt": np.random.uniform(100.0, 300.0, (257, 257)).astype(np.float32),
        "z_trestle": np.random.uniform(100.0, 300.0, (257, 257)).astype(np.float32),
        "is_felwood": True,
    }

    ds = ZarrElevationDataset([sample], augment=False)
    assert len(ds) == 1

    item = ds[0]
    assert item["features"].shape == (6, 256, 256)
    assert item["features"].dtype == torch.float32
    assert item["z_trestle"].shape == (257, 257)
    assert item["z_gt"].shape == (257, 257)
    assert item["delta_z"].shape == (257, 257)
    assert item["bounds"].shape == (2,)
    assert item["stem"] == "3_3_5_12340_Kalimdor_27_22"
    assert item["is_felwood"] is True


def test_extract_zarr_samples_from_mock_store(tmp_path: Path):
    """Verify extraction of samples from a mock Zarr store."""
    store_dir = tmp_path / "3_3_5_12340-Kalimdor.zarr"
    store_dir.mkdir(parents=True)

    g = zarr.open_group(str(store_dir), mode="w")

    # 2 tiles: one Felwood (27, 22), one flat (0, 0)
    h_felwood = np.zeros((257, 257), dtype=np.float32)
    h_felwood[100:150, 100:150] = 50.0  # canyon relief

    h_flat = np.zeros((257, 257), dtype=np.float32)  # 0 relief

    heights = np.stack([h_felwood, h_flat], axis=0)
    g.create_array("height_257", data=heights, overwrite=True)

    rgb = np.zeros((2, 256, 256, 3), dtype=np.uint8)
    g.create_array("minimap_rgb", data=rgb, overwrite=True)

    index_rows = [
        {"build_id": "3_3_5_12340", "map": "Kalimdor", "tile_x": 27, "tile_y": 22, "tile_id": 0},
        {"build_id": "3_3_5_12340", "map": "Kalimdor", "tile_x": 0, "tile_y": 0, "tile_id": 1},
    ]
    pq.write_table(pa.Table.from_pylist(index_rows), str(store_dir / "index.parquet"))

    samples = extract_zarr_samples(store_dir, target_maps=["Kalimdor"], min_relief=5.0)
    # The flat tile should be filtered out by min_relief
    assert len(samples) >= 1
    assert samples[0]["is_felwood"] is True
    assert samples[0]["tile_x"] == 27
    assert samples[0]["tile_y"] == 22
