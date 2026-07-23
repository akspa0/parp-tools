"""Spec 118 T022 (US3): the segmenter trainer's pure gates -- missing-array refusal and plan
construction.

Mirrors ``tests/spec117/test_lattice_train.py``'s convention of testing the trainer's pure
functions directly (CLI wiring is exercised for real via the quickstart dry-run, not re-simulated
here with a subprocess). ``apply_held_out_split``'s leaky/unspecified-split refusal is already
covered by ``tests/v50/test_direct_geometry_train.py`` against the identical, reused function --
not duplicated here.
"""

from __future__ import annotations

import pytest

from harvester.spec118.object_segment_train import (
    GATE_MEDIAN_VISIBLE_IOU,
    GATE_PER_CLASS_RECALL,
    REQUIRED_ARRAYS,
    build_object_plan,
    require_object_arrays,
)
from harvester.v50.height_relative_train import TrainerContractError


def test_required_arrays_match_the_us1_catalog_names():
    assert set(REQUIRED_ARRAYS) == {"minimap_rgb", "object_geometry_visible_source_257"}


def test_require_object_arrays_passes_when_all_present():
    require_object_arrays(dict.fromkeys(REQUIRED_ARRAYS, object()))


def test_require_object_arrays_refuses_a_store_missing_the_source_array():
    with pytest.raises(TrainerContractError, match="object_geometry_visible_source_257"):
        require_object_arrays({"minimap_rgb": object()})


def _architecture() -> dict:
    return {"id": "object_segment_net", "config_sha256": "a" * 64, "parameter_count": 300_000}


def test_build_object_plan_records_the_recipe_and_gate_thresholds():
    plan = build_object_plan(
        architecture=_architecture(), source="authored",
        train_rows=100, val_rows=20, object_touched_train=54, object_touched_val=11,
        batch_size=16, epochs=100, seed=118, lr=2e-4, lr_schedule="onecycle",
    )
    assert plan["schema"] == "v118-object-plan-v1"
    assert plan["stage"] == "object_segmentation"
    assert plan["deployment_inputs"] == ["minimap_rgb"]
    assert plan["training_target"] == "object_geometry_visible_source_257 -> object_class_3"
    assert plan["no_gan_no_adversarial_no_generative_image"] is True
    assert plan["split_counts"] == {"train": 100, "val": 20}
    assert plan["object_touched_tiles"] == {"train": 54, "val": 11}
    assert plan["train_steps_per_epoch"] == 7  # ceil(100/16)
    assert plan["gate_thresholds"]["median_visible_object_iou"] == GATE_MEDIAN_VISIBLE_IOU
    assert plan["gate_thresholds"]["per_class_recall"] == GATE_PER_CLASS_RECALL


def test_build_object_plan_rejects_invalid_lr_schedule():
    with pytest.raises(TrainerContractError, match="lr schedule"):
        build_object_plan(
            architecture=_architecture(), source="authored",
            train_rows=100, val_rows=20, object_touched_train=0, object_touched_val=0,
            batch_size=16, epochs=100, seed=118, lr=2e-4, lr_schedule="cosmic",
        )


def test_build_object_plan_rejects_nonpositive_batch_or_epochs():
    with pytest.raises(TrainerContractError):
        build_object_plan(
            architecture=_architecture(), source="authored",
            train_rows=100, val_rows=20, object_touched_train=0, object_touched_val=0,
            batch_size=0, epochs=100, seed=118, lr=2e-4, lr_schedule="constant",
        )
