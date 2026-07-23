"""Spec 118 T004: the widened ``STAGES`` enum accepts ``object_segmentation`` in a well-formed
``v50-model-stage-run-v1`` document and still rejects an unlisted stage."""

from __future__ import annotations

import pytest

from harvester.v50.model_stage_contract import (
    STAGES,
    ContractViolationError,
    validate_model_stage_run,
)


def _document(stage: str) -> dict:
    return {
        "schema": "v50-model-stage-run-v1",
        "run_id": "objects-authored-v1",
        "created_utc": "2026-07-22T00:00:00Z",
        "stage": stage,
        "output_signal": "object_class_3",
        "architecture": {"id": "object_segment_net", "config_sha256": "c" * 64, "parameter_count": 1000},
        "curriculum": {"path": "store/index.parquet", "sha256": "a" * 64},
        "upstream_models": [],
        "checkpoint": {"path": "run/checkpoint_best.pt", "sha256": "b" * 64, "best_epoch": 5},
        "baselines": {"majority_class": {"val_iou": 0.0}},
        "metrics": {"best_epoch": 5, "best_val_iou": 0.5},
        "visual_evidence": {},
        "promotion_verdict": "pending",
    }


def test_object_segmentation_stage_is_listed():
    assert "object_segmentation" in STAGES


def test_object_segmentation_document_validates():
    validate_model_stage_run(_document("object_segmentation"))


def test_unlisted_stage_still_rejects():
    with pytest.raises(ContractViolationError):
        validate_model_stage_run(_document("not_a_real_stage"))
