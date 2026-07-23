"""Spec 118 T006: ``object_contract.build_object_stage_run`` assembles a self-validating
``v50-model-stage-run-v1`` document and refuses a malformed one before it can be written to disk."""

from __future__ import annotations

import pytest
from torch import nn

from harvester.spec118.object_contract import (
    BRIDGE_CLASS_COUNT,
    CLASS_COUNT,
    CLASS_NAMES,
    OUTPUT_SIGNAL,
    STAGE,
    ObjectContractError,
    architecture_identity,
    build_object_stage_run,
)


def test_class_table_matches_data_model():
    assert STAGE == "object_segmentation"
    assert OUTPUT_SIGNAL == "object_class_2"
    assert CLASS_NAMES == ("none", "object")
    assert CLASS_COUNT == 2
    assert BRIDGE_CLASS_COUNT == 1


def test_architecture_identity_is_deterministic_and_content_hashed():
    model = nn.Linear(4, 4)
    config = {"base": 24, "input": "3x256x256"}
    first = architecture_identity(model, architecture_id="object_segment_net", config=config)
    second = architecture_identity(model, architecture_id="object_segment_net", config=config)
    assert first == second
    assert first["parameter_count"] == sum(p.numel() for p in model.parameters())

    different = architecture_identity(model, architecture_id="object_segment_net", config={"base": 32})
    assert different["config_sha256"] != first["config_sha256"]


def _stage_run_kwargs(**overrides):
    model = nn.Linear(4, 4)
    kwargs = {
        "run_id": "objects-authored-v1",
        "architecture": architecture_identity(model, architecture_id="object_segment_net", config={"base": 24}),
        "curriculum": {"path": "store/index.parquet", "sha256": "a" * 64},
        "checkpoint": {"path": "run/checkpoint_best.pt", "sha256": "b" * 64, "best_epoch": 5},
        "baselines": {"majority_class": {"val_iou": 0.0}},
        "metrics": {"best_epoch": 5, "best_val_iou": 0.5},
    }
    kwargs.update(overrides)
    return kwargs


def test_build_object_stage_run_self_validates():
    summary = build_object_stage_run(**_stage_run_kwargs())
    assert summary["schema"] == "v50-model-stage-run-v1"
    assert summary["stage"] == STAGE
    assert summary["output_signal"] == OUTPUT_SIGNAL
    assert summary["upstream_models"] == []
    assert summary["promotion_verdict"] == "pending"


def test_build_object_stage_run_rejects_a_malformed_checkpoint():
    kwargs = _stage_run_kwargs(checkpoint={"path": "run/checkpoint_best.pt"})  # missing sha256/best_epoch
    with pytest.raises(ObjectContractError):
        build_object_stage_run(**kwargs)
