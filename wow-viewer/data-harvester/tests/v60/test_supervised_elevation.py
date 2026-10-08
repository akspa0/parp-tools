"""Unit tests for Supervised Neural Elevation Model & Dataset (Spec 264)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v60.development_elevation_dataset import DevelopmentElevationDataset
from harvester.v60.supervised_elevation_model import (
    SupervisedElevationUNet,
    compute_elevation_normals,
    elevation_composite_loss,
)


def test_elevation_unet_forward_pass_shape() -> None:
    model = SupervisedElevationUNet(in_channels=3, base_channels=16)
    x = torch.rand(2, 3, 256, 256, dtype=torch.float32)
    out = model(x)
    assert out.shape == (2, 257, 257)
    assert out.dtype == torch.float32


def test_compute_elevation_normals_unit_length() -> None:
    z = torch.zeros(2, 257, 257, dtype=torch.float32)
    # Add a slope
    yy, xx = torch.meshgrid(torch.arange(257), torch.arange(257), indexing="ij")
    z[0] = xx.float() * 2.0
    normals = compute_elevation_normals(z)
    assert normals.shape == (2, 3, 255, 255)
    lens = torch.linalg.norm(normals, dim=1)
    assert torch.allclose(lens, torch.ones_like(lens), atol=1e-5)


def test_elevation_composite_loss_gradient_flow() -> None:
    model = SupervisedElevationUNet(in_channels=3, base_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.rand(2, 3, 256, 256, dtype=torch.float32)
    target_z = torch.full((2, 257, 257), 50.0, dtype=torch.float32)

    optimizer.zero_grad()
    pred_z = model(x)
    loss, metrics = elevation_composite_loss(pred_z, target_z)

    assert "l1_yards" in metrics
    assert "grad_loss" in metrics
    assert "normal_loss" in metrics
    assert loss.item() > 0.0

    loss.backward()
    # Ensure gradients exist on model parameters
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_development_elevation_dataset_item() -> None:
    mock_samples = [
        {
            "stem": "development_16_35",
            "rgb": np.random.rand(256, 256, 3).astype(np.float32),
            "z": np.full((257, 257), 100.0, dtype=np.float32),
        }
    ]
    ds = DevelopmentElevationDataset(mock_samples, augment=True)
    assert len(ds) == 1
    item = ds[0]
    assert item["rgb"].shape == (3, 256, 256)
    assert item["z"].shape == (257, 257)
    assert item["stem"] == "development_16_35"
