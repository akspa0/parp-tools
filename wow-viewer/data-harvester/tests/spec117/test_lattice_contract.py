"""Spec 117 T006: ``lattice_contract.build_lattice_stage_run`` assembles a self-validating
``v50-model-stage-run-v1`` document and refuses a malformed one before it can be written to disk."""

from __future__ import annotations

import pytest
from torch import nn

from harvester.spec117.lattice_contract import (
    OUTPUT_SIGNAL,
    SAMPLE_COUNT,
    STAGE,
    LatticeContractError,
    architecture_identity,
    build_lattice_stage_run,
)


def test_sample_count_matches_spec_108_contract():
    assert SAMPLE_COUNT == 545
    assert STAGE == "lattice_prior"
    assert OUTPUT_SIGNAL == "wdl_lattice_545"


def test_architecture_identity_is_deterministic_and_content_hashed():
    model = nn.Linear(4, 4)
    config = {"base": 24, "input": "3x256x256"}
    first = architecture_identity(model, architecture_id="lattice_net", config=config)
    second = architecture_identity(model, architecture_id="lattice_net", config=config)
    assert first == second
    assert first["parameter_count"] == sum(p.numel() for p in model.parameters())

    different = architecture_identity(model, architecture_id="lattice_net", config={"base": 32})
    assert different["config_sha256"] != first["config_sha256"]


def _stage_run_kwargs(**overrides):
    model = nn.Linear(4, 4)
    kwargs = {
        "run_id": "lattice-authored-v1",
        "architecture": architecture_identity(model, architecture_id="lattice_net", config={"base": 24}),
        "curriculum": {"path": "store/index.parquet", "sha256": "a" * 64},
        "checkpoint": {"path": "run/checkpoint_best.pt", "sha256": "b" * 64, "best_epoch": 5},
        "baselines": {"tile_mean": {"val_mae": 0.21}},
        "metrics": {"best_epoch": 5, "best_val_mae": 0.17},
    }
    kwargs.update(overrides)
    return kwargs


def test_build_lattice_stage_run_self_validates():
    summary = build_lattice_stage_run(**_stage_run_kwargs())
    assert summary["schema"] == "v50-model-stage-run-v1"
    assert summary["stage"] == STAGE
    assert summary["output_signal"] == OUTPUT_SIGNAL
    assert summary["upstream_models"] == []
    assert summary["promotion_verdict"] == "pending"


def test_build_lattice_stage_run_rejects_a_malformed_checkpoint():
    kwargs = _stage_run_kwargs(checkpoint={"path": "run/checkpoint_best.pt"})  # missing sha256/best_epoch
    with pytest.raises(LatticeContractError):
        build_lattice_stage_run(**kwargs)
