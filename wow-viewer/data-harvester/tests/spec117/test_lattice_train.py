"""Spec 117 T013: the lattice trainer's pure gates -- missing-array refusal and plan construction.

Mirrors ``tests/v50/test_direct_geometry_train.py``'s convention of testing the trainer's pure
functions directly (CLI wiring is exercised for real via the quickstart dry-run, not re-simulated
here with a subprocess). ``apply_held_out_split``'s leaky/unspecified-split refusal is already
covered by that same test module against the identical, reused function -- not duplicated here.
"""

from __future__ import annotations

import pytest

from harvester.spec117.lattice_train import (
    REQUIRED_WDL_ARRAYS,
    build_lattice_plan,
    require_wdl_arrays,
)
from harvester.v50.height_relative_train import TrainerContractError


def test_required_wdl_arrays_matches_the_us1_catalog_names():
    assert set(REQUIRED_WDL_ARRAYS) == {
        "wdl_outer_17", "wdl_inner_16", "wdl_outer_present", "wdl_inner_present",
    }


def test_require_wdl_arrays_passes_when_all_present():
    group = dict.fromkeys(REQUIRED_WDL_ARRAYS, object())
    require_wdl_arrays(group)  # must not raise


def test_require_wdl_arrays_refuses_a_store_missing_any_one():
    group = {name: object() for name in REQUIRED_WDL_ARRAYS if name != "wdl_inner_present"}
    with pytest.raises(TrainerContractError, match="wdl_inner_present"):
        require_wdl_arrays(group)


def _architecture() -> dict:
    return {"id": "lattice_net", "config_sha256": "a" * 64, "parameter_count": 200_000}


def test_build_lattice_plan_records_the_recipe_and_no_gan_boundary():
    plan = build_lattice_plan(
        architecture=_architecture(), source="authored",
        train_rows=100, val_rows=20, excluded_train=3, excluded_val=1,
        batch_size=16, epochs=100, seed=117, lr=2e-4, lr_schedule="onecycle",
    )
    assert plan["schema"] == "v117-lattice-plan-v1"
    assert plan["stage"] == "lattice_prior"
    assert plan["deployment_inputs"] == ["minimap_rgb"]
    assert plan["training_target"] == "wdl_outer_17+wdl_inner_16 -> wdl_lattice_545"
    assert plan["no_gan_no_adversarial_no_generative_image"] is True
    assert plan["split_counts"] == {"train": 100, "val": 20}
    assert plan["excluded_no_present_lattice"] == {"train": 3, "val": 1}
    assert plan["train_steps_per_epoch"] == 7  # ceil(100/16)


def test_build_lattice_plan_rejects_invalid_lr_schedule():
    with pytest.raises(TrainerContractError, match="lr schedule"):
        build_lattice_plan(
            architecture=_architecture(), source="authored",
            train_rows=100, val_rows=20, excluded_train=0, excluded_val=0,
            batch_size=16, epochs=100, seed=117, lr=2e-4, lr_schedule="cosmic",
        )


def test_build_lattice_plan_rejects_nonpositive_batch_or_epochs():
    with pytest.raises(TrainerContractError):
        build_lattice_plan(
            architecture=_architecture(), source="authored",
            train_rows=100, val_rows=20, excluded_train=0, excluded_val=0,
            batch_size=0, epochs=100, seed=117, lr=2e-4, lr_schedule="constant",
        )
