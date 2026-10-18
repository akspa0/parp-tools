"""Unit tests for PyTorch CleanPairDataset (Spec 274)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from harvester.v60.batch_export_clean_pairs import (
    export_clean_pairs_dataset,
    find_zarr_map_stores,
)
from harvester.v60.clean_pair_dataset import CleanPairDataset
from tests.v60.test_batch_export_clean_pairs import _create_mock_zarr_map_store


@pytest.fixture
def mock_clean_pairs_store(tmp_path: Path) -> Path:
    source_root = tmp_path / "mock_v60"
    source_root.mkdir()
    _create_mock_zarr_map_store(source_root, "Azeroth", count=6)
    _create_mock_zarr_map_store(source_root, "Kalimdor", count=4)

    stores = find_zarr_map_stores(source_root)
    out_zarr = tmp_path / "clean_pairs.zarr"

    export_clean_pairs_dataset(
        source_stores=stores,
        output_path=out_zarr,
        validation_map="Kalimdor",
        dry_run=False,
    )
    return out_zarr


def test_clean_pair_dataset_splits_and_shapes(mock_clean_pairs_store: Path) -> None:
    train_ds = CleanPairDataset(mock_clean_pairs_store, split="train", augment=False)
    val_ds = CleanPairDataset(mock_clean_pairs_store, split="val", augment=False)

    assert len(train_ds) == 6
    assert len(val_ds) == 4

    item = train_ds[0]
    assert isinstance(item["real_minimap"], torch.Tensor)
    assert item["real_minimap"].shape == (3, 256, 256)
    assert item["synthetic_hillshade"].shape == (3, 256, 256)
    assert item["wdl_lattice"].shape == (1, 33, 33)
    assert item["height_257"].shape == (1, 257, 257)

    # Check value ranges for images in [0, 1]
    assert 0.0 <= float(item["real_minimap"].min())
    assert float(item["real_minimap"].max()) <= 1.0
    assert 0.0 <= float(item["synthetic_hillshade"].min())
    assert float(item["synthetic_hillshade"].max()) <= 1.0


def test_clean_pair_dataset_augmentation_and_dataloader(mock_clean_pairs_store: Path) -> None:
    aug_ds = CleanPairDataset(mock_clean_pairs_store, split="train", augment=True)
    loader = DataLoader(aug_ds, batch_size=2, shuffle=True)

    batch = next(iter(loader))
    assert batch["real_minimap"].shape == (2, 3, 256, 256)
    assert batch["synthetic_hillshade"].shape == (2, 3, 256, 256)
    assert batch["wdl_lattice"].shape == (2, 1, 33, 33)
    assert batch["height_257"].shape == (2, 1, 257, 257)
    assert len(batch["map"]) == 2
