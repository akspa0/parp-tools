from __future__ import annotations

from harvester.v60.terrain_method_translation import (
    INPUT_CONTRACT_SCHEMA,
    build_method_translation_report,
)


def test_report_contract_is_versioned_and_lists_all_branches() -> None:
    report = build_method_translation_report()

    assert report["schema"] == "v60-terrain-method-translation-v1"
    assert all(item["schema"] == INPUT_CONTRACT_SCHEMA for item in report["contracts"])
    assert {item["branch"] for item in report["contracts"]} == {
        "rgb_only",
        "height_prior",
        "point_cloud",
        "combined",
    }
    assert report["ledger"]["by_input_modality"]["dsm"] == 2
