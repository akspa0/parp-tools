"""Unit tests for Spec 120 VLM Crop Patch Annotator (T013)."""

from __future__ import annotations

import numpy as np

from harvester.spec120.vlm_crop_annotator import annotate_crop_patch


def test_annotate_crop_patch() -> None:
    """Verify VLM crop patch annotation output format."""
    mock_patch = np.zeros((64, 64, 3), dtype=np.uint8)
    annotation = annotate_crop_patch(mock_patch, mock=True)

    assert "vlm_model" in annotation
    assert "visual_description" in annotation
    assert "condition" in annotation
    assert "detected_features" in annotation
    assert annotation["vlm_confidence"] > 0.9
