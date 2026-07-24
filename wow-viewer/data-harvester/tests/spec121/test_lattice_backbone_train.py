"""Spec 121 T011: Stage A trainer plan + CLI surface tests (no CUDA, no training)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from harvester.spec121.lattice_backbone_train import (
    LR_SCHEDULES,
    STAGE,
    build_stage_a_plan,
)
from harvester.v50.height_relative_train import TrainerContractError

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "spec121_train_lattice_prior.py"


def _plan_kwargs(**overrides):
    kwargs = {
        "architecture": {"id": "mit_b0_lattice", "config_sha256": "c" * 64,
                         "parameter_count": 3_469_922},
        "architecture_id": "mit_b0_lattice",
        "source": "authored",
        "train_rows": 500,
        "val_rows": 100,
        "excluded_train": 3,
        "excluded_val": 1,
        "batch_size": 16,
        "epochs": 100,
        "seed": 121,
        "lr": 2e-4,
        "lr_schedule": "onecycle",
        "object_mask_weight": 0.0,
        "object_mask_signal_present": True,
        "pretrained": None,
        "parameter_band": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_plan_records_the_recipe_and_boundaries():
    plan = build_stage_a_plan(**_plan_kwargs())
    assert plan["schema"] == "v121-stage-a-plan-v1"
    assert plan["stage"] == STAGE == "lattice_prior"
    assert plan["architecture_id"] == "mit_b0_lattice"
    assert plan["deployment_inputs"] == ["minimap_rgb"]
    assert plan["training_target"] == "wdl_outer_17+wdl_inner_16 -> wdl_lattice_545"
    assert plan["no_gan_no_adversarial_no_generative_image"] is True
    assert plan["object_mask_weight"] == 0.0
    assert plan["object_mask_signal_present"] is True
    assert plan["parameter_band_ok"] is True
    assert plan["train_steps_per_epoch"] == 32


def test_plan_rejects_invalid_lr_schedule():
    with pytest.raises(TrainerContractError):
        build_stage_a_plan(**_plan_kwargs(lr_schedule="cosine"))
    assert "onecycle" in LR_SCHEDULES and "constant" in LR_SCHEDULES


def test_plan_rejects_nonpositive_batch_or_epochs():
    with pytest.raises(TrainerContractError):
        build_stage_a_plan(**_plan_kwargs(batch_size=0))
    with pytest.raises(TrainerContractError):
        build_stage_a_plan(**_plan_kwargs(epochs=0))


def test_plan_rejects_out_of_range_mask_weight():
    with pytest.raises(TrainerContractError):
        build_stage_a_plan(**_plan_kwargs(object_mask_weight=1.5))
    with pytest.raises(TrainerContractError):
        build_stage_a_plan(**_plan_kwargs(object_mask_weight=-0.1))


def test_plan_records_pretrained_provenance_and_band_flag():
    pretrained = {"hub_id": "nvidia/mit-b0", "revision": "main",
                  "license": "Apache-2.0", "optional": True}
    plan = build_stage_a_plan(**_plan_kwargs(pretrained=pretrained, parameter_band=False))
    assert plan["pretrained"]["hub_id"] == "nvidia/mit-b0"
    assert plan["pretrained"]["revision"] == "main"
    assert plan["parameter_band_ok"] is False  # dry-run flags; --confirm-run refuses


def test_cli_help_lists_the_spec121_flags():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    for flag in (
        "--architecture", "--pretrained", "--pretrained-hub-id", "--pretrained-revision",
        "--object-mask-weight", "--held-out-split", "--confirm-run", "--gradient-weight",
        "--pct-start",
    ):
        assert flag in result.stdout, f"missing {flag} in --help"
    assert "lattice_net" in result.stdout and "mit_b0_lattice" in result.stdout
