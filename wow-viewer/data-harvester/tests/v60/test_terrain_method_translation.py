from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvester.v60.terrain_method_translation import (
    FORBIDDEN_INPUTS,
    ExternalMethodRecord,
    TerrainMethodTranslationError,
    TranslationDecision,
    audit_input_reads,
    build_combined_contract,
    build_height_prior_contract,
    build_method_translation_report,
    build_point_cloud_contract,
    build_rgb_only_contract,
    canonical_signal_name,
    initial_method_records,
    validate_method_records,
)

FIXTURE = Path(__file__).parent / "fixtures" / "terrain_method_translation_methods.json"


def test_initial_method_ledger_is_complete_and_deterministic() -> None:
    records = initial_method_records()
    report = validate_method_records(records)

    assert report["valid"] is True
    assert report["method_count"] == 6
    assert report["method_ids"] == [
        "aerial_object_mask_models",
        "cloth_simulation_filter",
        "dsm2dtm",
        "pdal_smrf",
        "prithvi_eo_2",
        "resdepth",
    ]
    assert report["by_translation_status"] == {"diagnostic": 2, "reference": 4}


def test_fixture_records_round_trip_from_json() -> None:
    values = json.loads(FIXTURE.read_text(encoding="utf-8"))
    records = tuple(ExternalMethodRecord.from_mapping(value) for value in values)

    assert validate_method_records(records)["valid"] is True
    assert records[0].to_dict()["input_modalities"] == ("rgb",)


def test_input_contracts_cover_rgb_height_point_cloud_and_combined() -> None:
    contracts = (
        build_rgb_only_contract(),
        build_height_prior_contract(),
        build_point_cloud_contract(),
        build_combined_contract(),
    )

    assert [contract.branch for contract in contracts] == [
        "rgb_only",
        "height_prior",
        "point_cloud",
        "combined",
    ]
    assert contracts[0].runtime_claim == "deployment_candidate"
    assert all(contract.runtime_claim == "offline_diagnostic" for contract in contracts[1:])


def test_rgb_audit_accepts_observable_and_predicted_inputs() -> None:
    report = audit_input_reads(build_rgb_only_contract(), ("rgb", "predicted_mask"))

    assert report["valid"] is True
    assert report["canonical_input_reads"] == ["minimap_rgb", "predicted_object_mask"]
    assert report["decision"] == "candidate"


@pytest.mark.parametrize(
    "signal",
    ["height_257", "terrain_shadow_256", "mcsh", "shadow_mask", "wdl", "object_mask"],
)
def test_rgb_audit_rejects_forbidden_inputs(signal: str) -> None:
    report = audit_input_reads(build_rgb_only_contract(), ("minimap_rgb", signal))

    assert report["valid"] is False
    assert report["decision"] == "rejected"
    assert report["forbidden_reads"]


def test_dsm_and_point_cloud_audits_remain_offline_only() -> None:
    dsm_report = audit_input_reads(build_height_prior_contract(), ("dsm",))
    point_report = audit_input_reads(build_point_cloud_contract(), ("lidar",))

    assert dsm_report["valid"] is True
    assert dsm_report["runtime_claim"] == "offline_diagnostic"
    assert point_report["valid"] is True
    assert point_report["canonical_input_reads"] == ["point_cloud"]


def test_unknown_inputs_are_rejected_even_when_not_forbidden() -> None:
    report = audit_input_reads(build_rgb_only_contract(), ("minimap_rgb", "secret_prior"))

    assert report["valid"] is False
    assert report["undeclared_reads"] == ["secret_prior"]


def test_invalid_combined_contract_is_rejected() -> None:
    with pytest.raises(TerrainMethodTranslationError, match="combined"):
        build_rgb_only_contract().__class__(
            contract_id="bad",
            branch="combined",
            observable_inputs=("minimap_rgb",),
            predicted_inputs=(),
            supervision_only_inputs=(),
            forbidden_inputs=tuple(sorted(FORBIDDEN_INPUTS)),
            runtime_claim="offline_diagnostic",
        )


def test_aliases_are_canonicalized() -> None:
    assert canonical_signal_name("RGB") == "minimap_rgb"
    assert canonical_signal_name("MCSH") == "shadow_mask"
    assert canonical_signal_name("LiDAR") == "point_cloud"


def test_promoted_translation_decision_requires_evidence_artifact() -> None:
    with pytest.raises(TerrainMethodTranslationError, match="reviewer_artifacts"):
        TranslationDecision(
            subject_id="dsm2dtm",
            status="promoted",
            reason="not enough evidence",
            required_next_gate="none",
            reviewed_at="2026-08-10",
            reviewer_artifacts=(),
        )

    decision = TranslationDecision(
        subject_id="pdal_smrf",
        status="diagnostic",
        reason="offline point-cloud baseline",
        required_next_gate="configured point-cloud source",
        reviewed_at="2026-08-10",
        reviewer_artifacts=("report.json",),
    )
    assert decision.to_dict()["status"] == "diagnostic"


def test_report_contains_valid_and_invalid_sample_audits() -> None:
    report = build_method_translation_report()

    assert report["valid"] is True
    assert len(report["ledger"]["methods"]) == 6
    assert any(not audit["valid"] for audit in report["sample_audits"])
