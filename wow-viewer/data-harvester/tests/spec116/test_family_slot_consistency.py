"""Spec 116 US1: family->slot consistency measurement and vocabulary decision."""

from __future__ import annotations

import pytest

from harvester.spec116.family_slot_consistency import (
    DEFAULT_THRESHOLD,
    FamilySlotConsistencyError,
    measure_family_slot_consistency,
    recommendation_from_report,
)
from harvester.spec116.structure_contract import validate_analysis_report


class TestConsistencyMeasurement:
    def test_consistent_store_recommends_slot_keyed(self, consistent_store) -> None:
        report = measure_family_slot_consistency(
            store=consistent_store["store"], dumps=consistent_store["dumps"],
        )
        fsc = report["family_slot_consistency"]
        # terrain always slot 0, road always slot 1 -> perfect consistency
        assert fsc["summary_consistency_score"] == pytest.approx(1.0)
        assert fsc["recommendation"] == "slot_keyed"
        assert report["decision"]["value"] == "slot_keyed"
        validate_analysis_report(report)

    def test_spread_store_recommends_family_keyed(self, spread_store) -> None:
        report = measure_family_slot_consistency(
            store=spread_store["store"], dumps=spread_store["dumps"],
        )
        fsc = report["family_slot_consistency"]
        # road spreads across slots 0,1,2 -> max_p 1/3; terrain max_p 0.75; mean < 0.70
        assert fsc["summary_consistency_score"] < DEFAULT_THRESHOLD
        assert fsc["recommendation"] == "family_keyed"
        assert recommendation_from_report(report) == "family_keyed"
        validate_analysis_report(report)

    def test_threshold_is_configurable_and_reported(self, consistent_store) -> None:
        # A very low threshold still classifies the consistent store as slot_keyed.
        report = measure_family_slot_consistency(
            store=consistent_store["store"], dumps=consistent_store["dumps"], threshold=0.10,
        )
        assert report["family_slot_consistency"]["threshold"] == pytest.approx(0.10)
        assert report["family_slot_consistency"]["recommendation"] == "slot_keyed"

    def test_invalid_threshold_rejected(self, consistent_store) -> None:
        with pytest.raises(FamilySlotConsistencyError, match="threshold"):
            measure_family_slot_consistency(
                store=consistent_store["store"], dumps=consistent_store["dumps"], threshold=1.5,
            )

    def test_per_family_distribution_sums_to_one(self, spread_store) -> None:
        report = measure_family_slot_consistency(
            store=spread_store["store"], dumps=spread_store["dumps"],
        )
        for entry in report["family_slot_consistency"]["per_family"]:
            if entry["max_slot_probability"] > 0.0:
                assert sum(entry["slot_distribution"]) == pytest.approx(1.0)
