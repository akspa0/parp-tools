"""Unit tests for Spec 120 OBB Detector Model and Loss (T008)."""

from __future__ import annotations

import torch

from harvester.spec120.obb_detector_model import MinimapOBBDetector
from harvester.spec120.obb_detector_train import OBBDetectorLoss, generate_dry_run_report


def test_obb_detector_forward_shape() -> None:
    """Verify forward pass output shape (N, 16, 16, 11)."""
    model = MinimapOBBDetector(in_channels=3, num_classes=4, base=16)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    assert out.shape == (2, 16, 16, 11)


def test_obb_detector_parameter_count() -> None:
    """Verify model parameter count stays in small class (<1M params at base 16)."""
    model = MinimapOBBDetector(in_channels=3, num_classes=4, base=16)
    num_params = sum(p.numel() for p in model.parameters())

    assert 100_000 < num_params < 1_000_000


def test_obb_detector_decode_predictions() -> None:
    """Verify decoding raw output tensor to OBB detection dicts."""
    model = MinimapOBBDetector(in_channels=3, num_classes=4, base=16)
    raw_pred = torch.full((1, 16, 16, 11), -5.0)

    # Set confidence high at grid cell (8, 8)
    raw_pred[0, 8, 8, 0] = 5.0  # sigmoid(5.0) > 0.99
    raw_pred[0, 8, 8, 1] = 0.0  # dx
    raw_pred[0, 8, 8, 2] = 0.0  # dy
    raw_pred[0, 8, 8, 7] = 10.0  # class 0 (wmo)

    detections = model.decode_predictions(raw_pred, conf_thresh=0.5)

    assert len(detections) == 1
    img0_dets = detections[0]
    assert len(img0_dets) == 1

    d = img0_dets[0]
    assert d["class_id"] == 0
    assert d["conf"] > 0.9
    assert d["px"] == 136.0  # (8 + sigmoid(0)) / 16 * 256 = (8.5 / 16) * 256 = 136.0
    assert d["py"] == 136.0


def test_obb_detector_loss_computation() -> None:
    """Verify loss computation with mock predictions and targets."""
    loss_fn = OBBDetectorLoss()
    pred = torch.randn(2, 16, 16, 11)

    targets = torch.full((2, 64, 6), -1.0)
    # Ground truth target 1
    targets[0, 0] = torch.tensor([0.0, 0.5, 0.5, 0.1, 0.1, 45.0])

    loss_dict = loss_fn(pred, targets)

    assert "loss" in loss_dict
    assert "conf_loss" in loss_dict
    assert "loc_loss" in loss_dict
    assert "angle_loss" in loss_dict
    assert "cls_loss" in loss_dict
    assert loss_dict["loss"].item() > 0.0


def test_dry_run_report() -> None:
    """Verify dry-run report generation."""
    model = MinimapOBBDetector(in_channels=3, num_classes=4, base=16)
    report = generate_dry_run_report(model, epochs=10, batch_size=8, lr=1e-3)

    assert report["arch"] == "minimap_obb_detector_v1"
    assert report["epochs"] == 10
    assert report["dry_run"] is True
