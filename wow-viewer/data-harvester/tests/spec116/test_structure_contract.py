"""Spec 116: contract validators accept valid documents and reject tampered ones."""

from __future__ import annotations

import copy

import pytest

from harvester.spec116.structure_contract import (
    ANALYSIS_REPORT_SCHEMA,
    HELD_OUT_SPLIT_SCHEMA,
    STRUCTURE_GEOMETRY_COMPARISON_SCHEMA,
    STRUCTURE_INFER_SCHEMA,
    STRUCTURE_RUN_SCHEMA,
    Spec116ContractError,
    validate_analysis_report,
    validate_document,
    validate_held_out_split,
    validate_structure_geometry_comparison,
    validate_structure_infer,
    validate_structure_run,
)

_SHA = "a" * 64
_DT = "2026-07-21T00:00:00Z"


def _held_out_split() -> dict:
    return {
        "schema": HELD_OUT_SPLIT_SCHEMA,
        "created_utc": _DT,
        "build_id": "0_5_3_3368",
        "store": {"path": "/s", "sha256": _SHA},
        "taxonomy_revision": "v115.1",
        "adjacency_rule": "8-neighbour",
        "buffer_rings": 1,
        "seed": 116,
        "split_counts": {"train": 100, "held_out": 20},
        "verified_violation_count": 0,
        "absolute_comparison_to_prior_invalid": True,
        "baseline_requiring_rerun": "tile-mean",
    }


def _analysis_report() -> dict:
    return {
        "schema": ANALYSIS_REPORT_SCHEMA,
        "created_utc": _DT,
        "report_kind": "family_slot_consistency",
        "identity": {
            "store": {"path": "/s", "sha256": _SHA},
            "taxonomy_revision": "v115.1",
            "rule_set_sha256": _SHA,
            "texture_name_dumps": [{"path": "/d", "sha256": _SHA}],
        },
        "family_slot_consistency": {
            "per_family": [
                {"family": "terrain", "slot_distribution": [1.0, 0.0, 0.0, 0.0], "max_slot_probability": 1.0},
                {"family": "road", "slot_distribution": [0.0, 1.0, 0.0, 0.0], "max_slot_probability": 1.0},
            ],
            "summary_consistency_score": 1.0,
            "threshold": 0.7,
            "recommendation": "slot_keyed",
        },
        "decision": {"kind": "vocabulary", "value": "slot_keyed"},
    }


def _structure_run() -> dict:
    return {
        "schema": STRUCTURE_RUN_SCHEMA,
        "created_utc": _DT,
        "feature": "116-relational-terrain-layers",
        "slot": 1,
        "vocabulary_decision": "family_keyed",
        "identity": {"path": "/r", "sha256": _SHA},
        "inputs": {
            "store": {"path": "/s", "sha256": _SHA},
            "held_out_split": {"path": "/sp", "sha256": _SHA, "verified_violation_count": 0},
            "texture_name_dumps": [{"path": "/d", "sha256": _SHA}],
            "taxonomy_revision": "v115.1",
            "rule_set_sha256": _SHA,
        },
        "architecture": {"class": "StructureSlotNet", "base": 32, "slot": 1, "num_classes": 5, "param_count": 1500000},
        "config": {"batch_size": 16, "epochs": 100, "lr": 1e-3, "max_class_weight": 15.0, "device": "cuda"},
        "split_counts": {"train": 100, "held_out": 20},
        "baselines": {"majority_class": {"family": "terrain", "per_class_iou": {}, "per_class_recall": {}}},
        "best_epoch": 50,
        "metrics": {
            "per_class": {"terrain": {"iou": 0.7, "recall": 0.8}},
            "macro_iou": 0.5,
            "rarest_class_iou": 0.4,
            "rarest_class_recall": 0.5,
            "aggregate_accuracy_reported_only": 0.9,
        },
        "promotion_verdict": "pending",
        "gate": {"rule": "per_class_iou_recall", "rarest_class": "structure", "sc003": False},
    }


def _structure_infer() -> dict:
    return {
        "schema": STRUCTURE_INFER_SCHEMA,
        "created_utc": _DT,
        "checkpoint": {"path": "/c", "sha256": _SHA, "taxonomy_revision": "v115.1"},
        "inputs": [{"path": "/i", "sha256": _SHA}],
        "legal_table_available": True,
        "sc004_all_references_legal": True,
        "per_tile": [{"input_sha256": _SHA, "class_fractions": {}, "low_confidence_chunks": 0}],
    }


def _geometry_comparison() -> dict:
    return {
        "schema": STRUCTURE_GEOMETRY_COMPARISON_SCHEMA,
        "held_out_split": {"path": "/sp", "sha256": _SHA},
        "without_structure": {"checkpoint_sha256": _SHA, "relief_mae": 0.2, "flat_mae": 0.1, "trivial_baseline_relief_mae": 0.18},
        "with_structure": {"checkpoint_sha256": _SHA, "relief_mae": 0.15, "flat_mae": 0.1, "trivial_baseline_relief_mae": 0.18},
        "sc007_beats_trivial_on_relief": True,
        "absolute_comparison_to_prior_runs_invalid": True,
    }


class TestHeldOutSplit:
    def test_valid(self) -> None:
        validate_held_out_split(_held_out_split())

    def test_nonzero_violation_count_rejected(self) -> None:
        doc = _held_out_split()
        doc["verified_violation_count"] = 1
        with pytest.raises(Spec116ContractError, match="verified_violation_count"):
            validate_held_out_split(doc)

    def test_wrong_adjacency_rejected(self) -> None:
        doc = _held_out_split()
        doc["adjacency_rule"] = "4-neighbour"
        with pytest.raises(Spec116ContractError, match="adjacency_rule"):
            validate_held_out_split(doc)

    def test_comparison_must_be_invalid(self) -> None:
        doc = _held_out_split()
        doc["absolute_comparison_to_prior_invalid"] = False
        with pytest.raises(Spec116ContractError, match="absolute_comparison_to_prior_invalid"):
            validate_held_out_split(doc)


class TestAnalysisReport:
    def test_valid_family_slot(self) -> None:
        validate_analysis_report(_analysis_report())

    def test_bad_recommendation_rejected(self) -> None:
        doc = _analysis_report()
        doc["family_slot_consistency"]["recommendation"] = "bogus"
        doc["decision"]["value"] = "bogus"
        with pytest.raises(Spec116ContractError):
            validate_analysis_report(doc)

    def test_bad_sha256_rejected(self) -> None:
        doc = _analysis_report()
        doc["identity"]["rule_set_sha256"] = "xyz"
        with pytest.raises(Spec116ContractError, match="sha256"):
            validate_analysis_report(doc)

    def test_shape_coverage_requires_branch(self) -> None:
        doc = _analysis_report()
        doc["report_kind"] = "shape_coverage_coupling"
        doc["decision"] = {"kind": "derivability", "value": "coverage_derivable"}
        # missing shape_coverage_coupling branch -> rejected
        with pytest.raises(Spec116ContractError, match="shape_coverage_coupling"):
            validate_analysis_report(doc)


class TestStructureRun:
    def test_valid(self) -> None:
        validate_structure_run(_structure_run())

    def test_leaky_split_rejected(self) -> None:
        doc = _structure_run()
        doc["inputs"]["held_out_split"]["verified_violation_count"] = 2
        with pytest.raises(Spec116ContractError, match="verified_violation_count"):
            validate_structure_run(doc)

    def test_slot_out_of_range_rejected(self) -> None:
        doc = _structure_run()
        doc["slot"] = 4
        doc["architecture"]["slot"] = 4
        with pytest.raises(Spec116ContractError, match="slot"):
            validate_structure_run(doc)

    def test_gate_must_not_reference_accuracy(self) -> None:
        doc = _structure_run()
        doc["gate"]["rule"] = "aggregate_accuracy"
        with pytest.raises(Spec116ContractError, match="gate.rule"):
            validate_structure_run(doc)

    def test_bad_promotion_verdict_rejected(self) -> None:
        doc = _structure_run()
        doc["promotion_verdict"] = "maybe"
        with pytest.raises(Spec116ContractError, match="promotion_verdict"):
            validate_structure_run(doc)


class TestStructureInfer:
    def test_valid(self) -> None:
        validate_structure_infer(_structure_infer())

    def test_bad_checkpoint_sha_rejected(self) -> None:
        doc = _structure_infer()
        doc["checkpoint"]["sha256"] = "nothex"
        with pytest.raises(Spec116ContractError, match="sha256"):
            validate_structure_infer(doc)


class TestGeometryComparison:
    def test_valid(self) -> None:
        validate_structure_geometry_comparison(_geometry_comparison())

    def test_comparison_must_be_invalid(self) -> None:
        doc = _geometry_comparison()
        doc["absolute_comparison_to_prior_runs_invalid"] = False
        with pytest.raises(Spec116ContractError, match="absolute_comparison_to_prior_runs_invalid"):
            validate_structure_geometry_comparison(doc)


class TestDispatch:
    @pytest.mark.parametrize(
        "builder",
        [
            _held_out_split, _analysis_report, _structure_run, _structure_infer, _geometry_comparison,
        ],
    )
    def test_dispatch_validates_each(self, builder) -> None:
        validate_document(copy.deepcopy(builder()))

    def test_unknown_schema_rejected(self) -> None:
        with pytest.raises(Spec116ContractError, match="unknown schema"):
            validate_document({"schema": "bogus-v1"})
