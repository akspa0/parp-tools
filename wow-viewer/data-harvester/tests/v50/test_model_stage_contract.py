"""Spec 114 T005: JSON-schema contract tests for the three published document variants."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from harvester.v50.model_stage_contract import (
    ContractViolation,
    append_generated_input,
    identity_for_path,
    sha256_json,
    validate_curriculum_summary,
    validate_document,
    validate_model_stage_run,
    validate_object_visibility_summary,
    verify_identity,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "v50" / "spec114"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture()
def curriculum() -> dict:
    return _load("curriculum_summary.json")


@pytest.fixture()
def visibility() -> dict:
    return _load("object_visibility_summary.json")


@pytest.fixture()
def stage_run() -> dict:
    return _load("model_stage_run.json")


def test_all_three_fixtures_validate(curriculum: dict, visibility: dict, stage_run: dict) -> None:
    validate_curriculum_summary(curriculum)
    validate_object_visibility_summary(visibility)
    validate_model_stage_run(stage_run)


def test_dispatch_routes_each_fixture(curriculum: dict, visibility: dict, stage_run: dict) -> None:
    validate_document(curriculum)
    validate_document(visibility)
    validate_document(stage_run)


def test_dispatch_refuses_unknown_schema(curriculum: dict) -> None:
    mutated = copy.deepcopy(curriculum)
    mutated["schema"] = "v50-mixed-curriculum-v1"
    with pytest.raises(ContractViolation, match="unknown Spec 114 schema"):
        validate_document(mutated)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("curriculum_id"),
        lambda d: d.update(extra_key=True),
        lambda d: d.update(target_signal="height_257"),
        lambda d: d.update(synthetic_lighting_contract="AuthoredEra"),
        lambda d: d.update(group_leak_count=1),
        lambda d: d.update(row_count=0),
        lambda d: d["split_counts"].update(validation=-1),
        lambda d: d["input_origins"].pop("synthetic_noon_white"),
        lambda d: d.update(created_utc="not-a-date"),
        lambda d: d["source_stores"][0].update(sha256="xyz"),
        lambda d: d.update(source_stores=[]),
        lambda d: d["excluded_counts"].update(synthetic_stale_lighting=-3),
    ],
)
def test_curriculum_mutations_fail(curriculum: dict, mutate) -> None:
    mutated = copy.deepcopy(curriculum)
    mutate(mutated)
    with pytest.raises(ContractViolation):
        validate_curriculum_summary(mutated)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("renderer_revision"),
        lambda d: d.update(alignment_fixture_count=0),
        lambda d: d.update(mask_shape=[256]),
        lambda d: d.update(mask_shape=[0, 256]),
        lambda d: d.update(mask_shape="256x256"),
        lambda d: d.update(unavailable_rows=-1),
        lambda d: d.update(placement_sources=[]),
        lambda d: d.update(schema="v50-model-stage-run-v1"),
    ],
)
def test_visibility_mutations_fail(visibility: dict, mutate) -> None:
    mutated = copy.deepcopy(visibility)
    mutate(mutated)
    with pytest.raises(ContractViolation):
        validate_object_visibility_summary(mutated)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(stage="super_resolution"),
        lambda d: d.update(promotion_verdict="maybe"),
        lambda d: d["architecture"].update(parameter_count=0),
        lambda d: d["architecture"].update(config_sha256="z" * 64),
        lambda d: d["checkpoint"].update(best_epoch=0),
        lambda d: d.update(pretrained_source={"hub_id": "x", "revision": "y"}),
        lambda d: d.update(
            pretrained_source={
                "hub_id": "nvidia/mit-b0",
                "revision": "r" * 40,
                "sha256": "e" * 64,
                "license": "Apache-2.0",
                "optional": False,
            }
        ),
        lambda d: d.update(metrics=[]),
        lambda d: d["upstream_models"].append({"path": "x"}),
    ],
)
def test_stage_run_mutations_fail(stage_run: dict, mutate) -> None:
    mutated = copy.deepcopy(stage_run)
    mutate(mutated)
    with pytest.raises(ContractViolation):
        validate_model_stage_run(mutated)


def test_valid_pretrained_source_is_accepted(stage_run: dict) -> None:
    mutated = copy.deepcopy(stage_run)
    mutated["pretrained_source"] = {
        "hub_id": "nvidia/mit-b0",
        "revision": "r" * 40,
        "sha256": "e" * 64,
        "license": "Apache-2.0",
        "optional": True,
    }
    validate_model_stage_run(mutated)


def test_identity_roundtrip_and_drift_detection(tmp_path: Path) -> None:
    target = tmp_path / "evidence.bin"
    target.write_bytes(b"spec114")
    identity = identity_for_path(target)
    verify_identity(identity, target)
    target.write_bytes(b"tampered")
    with pytest.raises(ContractViolation, match="identity drift"):
        verify_identity(identity, target)


def test_identity_for_path_refuses_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="does not exist"):
        identity_for_path(tmp_path / "absent.bin")


def test_sha256_json_is_key_order_invariant() -> None:
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


def test_append_generated_input_requires_valid_identity(curriculum: dict) -> None:
    with pytest.raises(ContractViolation):
        append_generated_input(curriculum, {"path": "mask.ckpt", "sha256": "short"})
    updated = append_generated_input(
        curriculum, {"path": "mask.ckpt", "sha256": "f" * 64}
    )
    validate_curriculum_summary(updated)
    assert len(updated["generated_input_models"]) == 1
    assert "generated_input_models" not in curriculum
