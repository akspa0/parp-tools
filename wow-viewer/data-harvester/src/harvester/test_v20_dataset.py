import os
from pathlib import Path
import pytest
import torch

from harvester.v20_dataset import V20Dataset


def test_v20_dataset_shapes():
    # Define dataset root path relative to the test location
    # data-harvester is root, so relative to root it is output/datasets/v18
    dataset_root = Path("output/datasets/v18")
    if not dataset_root.exists():
        dataset_root = Path("i:/parp/parp-tools/wow-viewer/output/datasets/v18")

    # If the directory still does not exist (e.g. run in sandbox without staged assets), skip test
    if not (dataset_root / "0_5_3_3368.zarr").exists():
        pytest.skip("Staged Zarr build 0_5_3_3368 not found, skipping integration test.")

    # Initialize the dataset
    ds = V20Dataset(
        dataset_root=dataset_root,
        builds=["0_5_3_3368"],
        input_channels=3,
        augment=True,
        split="train",
        val_fraction=0.1,
        limit=5,
    )

    assert len(ds) > 0
    sample = ds[0]

    # Verify input shape
    assert sample["input"].shape == (3, 256, 256)
    assert sample["input"].dtype == torch.float32

    # Verify height target shapes
    assert sample["height_raw"].shape == (1, 257, 257)
    assert sample["height_norm"].shape == (1, 257, 257)
    assert sample["ground_intent_height"].shape == (1, 257, 257)
    assert sample["ground_intent_height"].dtype == torch.float32

    # Verify liquid type shapes and range
    assert sample["liquid_type_256"].shape == (1, 256, 256)
    assert sample["liquid_type_256"].dtype == torch.int64
    assert torch.all((sample["liquid_type_256"] >= 0) & (sample["liquid_type_256"] <= 4))

    # Verify precise object masks
    assert sample["object_precise_mask_256"].shape == (1, 256, 256)
    assert sample["object_precise_mask_257"].shape == (1, 257, 257)
    assert sample["object_precise_mask_256"].dtype == torch.float32

    # Verify alpha weights
    assert sample["alpha"].shape == (4, 256, 256)
    assert sample["alpha"].dtype == torch.float32
