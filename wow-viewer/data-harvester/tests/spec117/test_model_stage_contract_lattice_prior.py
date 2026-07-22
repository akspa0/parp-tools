"""Spec 117 T004: ``v50-model-stage-run-v1`` must accept ``stage="lattice_prior"`` now, and must
still reject stages that were never declared -- the widened enum is additive, not permissive."""

from __future__ import annotations

import pytest

from harvester.v50.model_stage_contract import ContractViolationError, validate_model_stage_run

_VALID_DOC = {
    "schema": "v50-model-stage-run-v1",
    "run_id": "lattice-authored-v1",
    "created_utc": "2026-07-21T00:00:00Z",
    "stage": "lattice_prior",
    "output_signal": "wdl_lattice_545",
    "architecture": {
        "id": "lattice_net",
        "config_sha256": "a" * 64,
        "parameter_count": 123456,
    },
    "curriculum": {"path": "store/index.parquet", "sha256": "b" * 64},
    "upstream_models": [],
    "checkpoint": {"path": "run/checkpoint_best.pt", "sha256": "c" * 64, "best_epoch": 3},
    "baselines": {"tile_mean": {"val_mae": 0.2}},
    "metrics": {"best_epoch": 3, "best_val_mae": 0.15},
    "visual_evidence": {},
    "promotion_verdict": "pending",
}


def test_lattice_prior_stage_validates():
    validate_model_stage_run(_VALID_DOC)  # must not raise


def test_unknown_stage_still_rejected():
    doc = {**_VALID_DOC, "stage": "not_a_real_stage"}
    with pytest.raises(ContractViolationError, match="stage"):
        validate_model_stage_run(doc)
