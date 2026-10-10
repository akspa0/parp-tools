"""Unit tests for MinimapToWdlNet and MultiCorpusWdlDataset (Spec 268)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from harvester.v60.minimap_to_wdl_model import (
    MinimapToWdlNet,
    MultiCorpusWdlDataset,
    minimap_wdl_composite_loss,
)


def test_minimap_to_wdl_net_forward_4ch() -> None:
    model = MinimapToWdlNet(in_channels=6, base_channels=16)
    model.eval()

    x = torch.rand((2, 4, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        pred_h17, pred_shape, pred_span, pred_base = model(x)

    assert pred_h17.shape == (2, 17, 17)
    assert pred_shape.shape == (2, 17, 17)
    assert pred_span.shape == (2,)
    assert pred_base.shape == (2,)
    assert torch.isfinite(pred_h17).all()
    assert (pred_shape >= 0.0).all() and (pred_shape <= 1.0).all()
    assert (pred_span > 0.0).all()


def test_minimap_to_wdl_net_forward_3ch() -> None:
    """Verify model seamlessly accepts 3-channel RGB minimap and auto-derives shadow and normals."""
    model = MinimapToWdlNet(in_channels=6, base_channels=16)
    model.eval()

    x3 = torch.rand((2, 3, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        pred_h17, pred_shape, pred_span, pred_base = model(x3)

    assert pred_h17.shape == (2, 17, 17)
    assert torch.isfinite(pred_h17).all()


def test_minimap_to_wdl_net_predict_tile() -> None:
    model = MinimapToWdlNet(in_channels=6, base_channels=16)
    rgb = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    shadow = np.random.rand(256, 256).astype(np.float32)

    h17 = model.predict_tile(rgb, shadow=shadow, device=torch.device("cpu"))
    assert isinstance(h17, np.ndarray)
    assert h17.shape == (17, 17)
    assert h17.dtype == np.float32
    assert np.all(np.isfinite(h17))


def test_minimap_wdl_composite_loss_computation() -> None:
    pred_h = torch.full((2, 17, 17), 100.0, dtype=torch.float32)
    pred_s = torch.full((2, 17, 17), 0.5, dtype=torch.float32)
    pred_span = torch.tensor([100.0, 100.0], dtype=torch.float32)
    pred_base = torch.tensor([50.0, 50.0], dtype=torch.float32)
    gt_h = torch.full((2, 17, 17), 110.0, dtype=torch.float32)

    loss, metrics = minimap_wdl_composite_loss(pred_h, pred_s, pred_span, pred_base, gt_h)
    assert loss.item() > 0.0
    assert "shape_l1" in metrics
    assert "shape_grad" in metrics
    assert "span_loss" in metrics
    assert "base_loss" in metrics
    assert "point_loss" in metrics


def test_multi_corpus_wdl_dataset_integration() -> None:
    dev_path = Path("I:/parp/parp-tools/wow-viewer/output/datasets/development_elevation_corpus.npz")
    if not dev_path.exists():
        pytest.skip("Development corpus not available on this machine.")

    ds = MultiCorpusWdlDataset(dev_corpus_path=dev_path, augment=True)
    assert len(ds) == 270

    sample = ds[0]
    assert "features" in sample
    assert sample["features"].shape == (4, 256, 256)
    assert sample["h17"].shape == (17, 17)
