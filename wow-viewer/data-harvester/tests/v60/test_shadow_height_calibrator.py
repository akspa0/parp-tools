"""Tests for ShadowHeightCalibrator (Spec 264 Phase 2)."""

from pathlib import Path
import numpy as np
import pytest

from harvester.v60.development_ground_truth import DevelopmentGroundTruthExtractor
from harvester.v60.shadow_height_calibrator import ShadowHeightCalibrator

MAPS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_data" / "original_development" / "World" / "Maps" / "development"
TEXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "test_data" / "original_development" / "World" / "Textures" / "Minimap"


def test_feature_extraction_and_scale_prediction():
    """Verify feature extractor and scale predictor produce valid physical outputs."""
    calibrator = ShadowHeightCalibrator()
    rgb = np.full((256, 256, 3), 0.5, dtype=np.float32)
    shadow = np.full((256, 256), 0.5, dtype=np.float32)
    raw_h = np.linspace(0.0, 5.0, 256 * 256, dtype=np.float32).reshape((256, 256))

    feats = calibrator.extract_features(rgb, shadow, raw_h)
    assert feats.shape == (5,)
    assert np.all(np.isfinite(feats))

    scale = calibrator.predict_scale(rgb, shadow, raw_h)
    assert scale >= 2.0
    assert scale <= 800.0


def test_supervised_calibration_against_authentic_adt():
    """Verify supervised calibration on authentic development tile achieves high correlation."""
    adt_16_33 = MAPS_DIR / "development_16_33.adt"
    if not adt_16_33.is_file():
        pytest.skip("development_16_33.adt not found")

    extractor = DevelopmentGroundTruthExtractor(MAPS_DIR, TEXTURES_DIR)
    gt_257, is_sculpted = extractor.extract_height_257(adt_16_33)
    assert is_sculpted is True

    # Generate synthetic raw height surface correlated with ground truth
    noise = np.random.normal(0.0, 5.0, gt_257.shape).astype(np.float32)
    synthetic_raw = gt_257 * 0.15 + noise

    calibrator = ShadowHeightCalibrator()
    calibrated, metrics = calibrator.calibrate_height(
        raw_height=synthetic_raw,
        ground_truth_257=gt_257,
    )

    assert calibrated.shape == (257, 257)
    assert metrics.pearson_r >= 0.70, f"Expected Pearson r >= 0.70, got {metrics.pearson_r:.4f}"
    assert metrics.r2 >= 0.50, f"Expected R^2 >= 0.50, got {metrics.r2:.4f}"
    assert metrics.mae <= 25.0, f"Expected low MAE in yards, got {metrics.mae:.2f}"


def test_unsupervised_calibration_scaling():
    """Verify unsupervised calibration produces finite, bounded surfaces in yards."""
    calibrator = ShadowHeightCalibrator(default_scale=50.0, default_base_elevation=100.0)
    raw_h = np.random.uniform(0.0, 10.0, (256, 256)).astype(np.float32)

    calibrated, metrics = calibrator.calibrate_height(raw_height=raw_h)

    assert calibrated.shape == (257, 257)
    assert np.all(np.isfinite(calibrated))
    assert float(np.min(calibrated)) >= 99.9  # Approximately base elevation 100.0
    assert float(np.max(calibrated)) <= 150.1  # Base + scale 50.0
    assert metrics.scale_yards == 50.0
    assert metrics.base_elevation == 100.0


def test_fit_training_corpus():
    """Verify fitting on multiple development tiles updates model weights."""
    calibrator = ShadowHeightCalibrator()
    samples = []
    for i in range(5):
        rgb = np.random.uniform(0.2, 0.8, (256, 256, 3)).astype(np.float32)
        shadow = np.random.uniform(0.3, 0.7, (256, 256)).astype(np.float32)
        raw_h = np.random.uniform(0.0, 5.0 + i * 2.0, (256, 256)).astype(np.float32)
        gt = np.random.uniform(10.0, 50.0 + i * 30.0, (257, 257)).astype(np.float32)
        samples.append((rgb, shadow, raw_h, gt))

    result = calibrator.fit_training_corpus(samples)
    assert result["samples"] == 5
    assert np.all(calibrator._weights >= 0.0), "Regression weights must be non-negative"
