"""Unit tests for Stage 2 Residual Elevation Model & Dataset (Spec 265)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v60.residual_elevation_dataset import ResidualElevationDataset
from harvester.v60.residual_elevation_model import (
    ResidualElevationRefiner,
    residual_composite_loss,
)


def test_residual_refiner_forward_shape() -> None:
    model = ResidualElevationRefiner(in_channels=6, base_channels=16)
    x = torch.rand(2, 6, 256, 256, dtype=torch.float32)
    out = model(x)
    assert out.shape == (2, 257, 257)
    assert out.dtype == torch.float32


def test_residual_composite_loss_gradient_flow() -> None:
    model = ResidualElevationRefiner(in_channels=6, base_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.rand(2, 6, 256, 256, dtype=torch.float32)
    z_init = torch.full((2, 257, 257), 50.0, dtype=torch.float32)
    gt_delta_z = torch.full((2, 257, 257), 20.0, dtype=torch.float32)
    gt_z = z_init + gt_delta_z

    optimizer.zero_grad()
    pred_delta = model(x)
    loss, metrics = residual_composite_loss(pred_delta, gt_delta_z, z_init, gt_z)

    assert "res_l1" in metrics
    assert "final_l1" in metrics
    assert "grad_loss" in metrics
    assert "normal_loss" in metrics
    assert loss.item() > 0.0

    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_residual_elevation_dataset_features_shape() -> None:
    mock_samples = [
        {
            "stem": "development_16_33",
            "rgb": np.random.rand(256, 256, 3).astype(np.float32),
            "z_init": np.full((257, 257), 30.0, dtype=np.float32),
            "z_gt": np.full((257, 257), 250.0, dtype=np.float32),
            "delta_z": np.full((257, 257), 220.0, dtype=np.float32),
        }
    ]
    ds = ResidualElevationDataset(mock_samples, augment=True)
    assert len(ds) == 1
    item = ds[0]
    assert item["features"].shape == (6, 256, 256)
    assert item["z_init"].shape == (257, 257)
    assert item["delta_z"].shape == (257, 257)
    assert item["z_gt"].shape == (257, 257)
