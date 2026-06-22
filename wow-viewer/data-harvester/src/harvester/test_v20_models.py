import pytest
import torch

from harvester.v20_models import (
    V20SemanticSegmentor,
    V20FingerprintClassifier,
    V20TerrainInpainter,
    V20PlacementRestorer,
)


def test_v20_semantic_segmentor_forward():
    model = V20SemanticSegmentor(in_channels=3)
    x = torch.randn(2, 3, 256, 256)
    liquid_logits, object_mask, alpha_weights = model(x)

    assert liquid_logits.shape == (2, 5, 256, 256)
    assert object_mask.shape == (2, 1, 256, 256)
    assert alpha_weights.shape == (2, 4, 256, 256)

    # Check bounds / values of sigmoid heads
    assert torch.all(object_mask >= 0.0) and torch.all(object_mask <= 1.0)
    assert torch.all(alpha_weights >= 0.0) and torch.all(alpha_weights <= 1.0)


def test_v20_fingerprint_classifier_forward():
    model = V20FingerprintClassifier(in_channels=7, num_classes=150)
    x = torch.randn(2, 7, 64, 64)
    cls_logits, reg_params = model(x)

    assert cls_logits.shape == (2, 150)
    assert reg_params.shape == (2, 4)


def test_v20_terrain_inpainter_forward():
    model = V20TerrainInpainter(in_channels=10)
    x = torch.randn(2, 10, 256, 256)
    pred_height = model(x)

    assert pred_height.shape == (2, 1, 257, 257)
    # Check that heights are clamped safely
    assert torch.all(pred_height >= -10.0) and torch.all(pred_height <= 10.0)


def test_v20_placement_restorer_forward():
    model = V20PlacementRestorer(in_channels=4, num_models=300)
    x = torch.randn(2, 4, 64, 64)
    model_logits, reg_params = model(x)

    assert model_logits.shape == (2, 300)
    assert reg_params.shape == (2, 5)
