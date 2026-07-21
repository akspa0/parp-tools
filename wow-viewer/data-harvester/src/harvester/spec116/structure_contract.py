"""Spec 116 contract validation.

Hand-rolled, dependency-free structural validation for the documents published in
``specs/116-relational-terrain-layers/contracts/`` and ``data-model.md``:

- ``v50-held-out-split-v1``        (US4 spatially-isolated held-out split manifest)
- ``v116-analysis-report-v1``      (US1 family->slot consistency / US2 shape->coverage coupling)
- ``v50-structure-run-v1``         (US3 per-slot structure training run record)
- ``v50-structure-infer-v1``       (US3 deployment inference audit)
- ``v50-structure-geometry-comparison-v1`` (US5 paired geometry comparison)

Mirrors the Spec 114 ``model_stage_contract`` discipline: required keys, ``additionalProperties:
false`` semantics, const values, enums, sha256 patterns, integer bounds. Identity helpers bind a
document to real file content so a split, report, or checkpoint cannot be silently swapped after a
run record names it. sha256 helpers are reused from ``harvester.v50.model_stage_contract`` (one
canonical owner; no duplication).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from harvester.v50.model_stage_contract import sha256_file, sha256_json  # canonical owner

HELD_OUT_SPLIT_SCHEMA = "v50-held-out-split-v1"
ANALYSIS_REPORT_SCHEMA = "v116-analysis-report-v1"
STRUCTURE_RUN_SCHEMA = "v50-structure-run-v1"
STRUCTURE_INFER_SCHEMA = "v50-structure-infer-v1"
STRUCTURE_GEOMETRY_COMPARISON_SCHEMA = "v50-structure-geometry-comparison-v1"

REPORT_KINDS = frozenset({"family_slot_consistency", "shape_coverage_coupling"})
VOCABULARY_DECISIONS = frozenset({"slot_keyed", "family_keyed"})
DERIVABILITY_DECISIONS = frozenset({"coverage_derivable", "coverage_independent"})
PROMOTION_VERDICTS = frozenset({"pending", "promoted", "refused"})

_HEX = frozenset("0123456789abcdefABCDEF")


class Spec116ContractError(ValueError):
    """Raised when a document does not satisfy a published Spec 116 contract."""


def _fail(path: str, message: str) -> None:
    raise Spec116ContractError(f"{path}: {message}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _require_keys(obj: Any, required: set[str], optional: set[str], path: str) -> dict:
    if not isinstance(obj, dict):
        _fail(path, f"must be an object, got {type(obj).__name__}")
    missing = sorted(required - obj.keys())
    if missing:
        _fail(path, f"missing required keys {missing}")
    extra = sorted(obj.keys() - required - optional)
    if extra:
        _fail(path, f"unexpected keys {extra} (additionalProperties: false)")
    return obj


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, f"must be a non-empty string, got {value!r}")
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}, got {value!r}")
    return value


def _require_number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"must be a number, got {value!r}")
    f = float(value)
    if minimum is not None and f < minimum:
        _fail(path, f"must be >= {minimum}, got {value!r}")
    if maximum is not None and f > maximum:
        _fail(path, f"must be <= {maximum}, got {value!r}")
    return f


def _require_const(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        _fail(path, f"must be {expected!r}, got {value!r}")


def _require_enum(value: Any, allowed: frozenset[str], path: str) -> str:
    text = _require_str(value, path)
    if text not in allowed:
        _fail(path, f"must be one of {sorted(allowed)}, got {text!r}")
    return text


def _require_datetime(value: Any, path: str) -> None:
    text = _require_str(value, path)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _fail(path, f"must be an ISO-8601 date-time, got {text!r}")


def _require_sha256(value: Any, path: str) -> None:
    if not _is_sha256(value):
        _fail(path, f"must be 64 hex characters, got {value!r}")


def _require_identity(obj: Any, path: str) -> None:
    _require_keys(obj, {"path", "sha256"}, set(), path)
    _require_str(obj["path"], f"{path}.path")
    _require_sha256(obj["sha256"], f"{path}.sha256")


def _require_store_ref(obj: Any, path: str) -> None:
    _require_keys(obj, {"path", "sha256"}, set(), path)
    _require_str(obj["path"], f"{path}.path")
    _require_sha256(obj["sha256"], f"{path}.sha256")


def _require_dump_list(value: Any, path: str) -> None:
    if not isinstance(value, list) or not value:
        _fail(path, "must be a non-empty array of {path,sha256}")
    for index, item in enumerate(value):
        _require_store_ref(item, f"{path}[{index}]")


# --------------------------------------------------------------------------- #
# v50-held-out-split-v1
# --------------------------------------------------------------------------- #
def validate_held_out_split(doc: Any) -> None:
    path = "$"
    _require_keys(
        doc,
        {
            "schema", "created_utc", "build_id", "store", "taxonomy_revision",
            "adjacency_rule", "buffer_rings", "seed", "split_counts",
            "verified_violation_count", "absolute_comparison_to_prior_invalid",
            "baseline_requiring_rerun",
        },
        {"excluded_count"},
        path,
    )
    _require_const(doc["schema"], HELD_OUT_SPLIT_SCHEMA, f"{path}.schema")
    _require_datetime(doc["created_utc"], f"{path}.created_utc")
    _require_str(doc["build_id"], f"{path}.build_id")
    _require_store_ref(doc["store"], f"{path}.store")
    _require_str(doc["taxonomy_revision"], f"{path}.taxonomy_revision")
    _require_const(doc["adjacency_rule"], "8-neighbour", f"{path}.adjacency_rule")
    _require_int(doc["buffer_rings"], f"{path}.buffer_rings", minimum=1)
    _require_int(doc["seed"], f"{path}.seed", minimum=0)
    counts = _require_keys(doc["split_counts"], {"train", "held_out"}, set(), f"{path}.split_counts")
    _require_int(counts["train"], f"{path}.split_counts.train", minimum=0)
    _require_int(counts["held_out"], f"{path}.split_counts.held_out", minimum=0)
    # SC-005: the violation count MUST be exactly zero.
    _require_const(doc["verified_violation_count"], 0, f"{path}.verified_violation_count")
    _require_const(doc["absolute_comparison_to_prior_invalid"], True, f"{path}.absolute_comparison_to_prior_invalid")
    _require_str(doc["baseline_requiring_rerun"], f"{path}.baseline_requiring_rerun")


# --------------------------------------------------------------------------- #
# v116-analysis-report-v1
# --------------------------------------------------------------------------- #
def validate_analysis_report(doc: Any) -> None:
    path = "$"
    _require_keys(
        doc,
        {"schema", "created_utc", "report_kind", "identity", "decision"},
        {"family_slot_consistency", "shape_coverage_coupling",
         "row_count", "layer_entry_count", "excluded", "build_id",
         "fittable_tile_layer_count", "skipped_zero_coverage",
         "high_coupling_variance_threshold", "bimodality_coefficient_threshold"},
        path,
    )
    _require_const(doc["schema"], ANALYSIS_REPORT_SCHEMA, f"{path}.schema")
    _require_datetime(doc["created_utc"], f"{path}.created_utc")
    kind = _require_enum(doc["report_kind"], REPORT_KINDS, f"{path}.report_kind")

    ident = _require_keys(
        doc["identity"], {"store", "taxonomy_revision"},
        {"rule_set_sha256", "texture_name_dumps"}, f"{path}.identity",
    )
    _require_store_ref(ident["store"], f"{path}.identity.store")
    _require_str(ident["taxonomy_revision"], f"{path}.identity.taxonomy_revision")
    if "rule_set_sha256" in ident:
        _require_sha256(ident["rule_set_sha256"], f"{path}.identity.rule_set_sha256")
    if "texture_name_dumps" in ident:
        _require_dump_list(ident["texture_name_dumps"], f"{path}.identity.texture_name_dumps")

    decision = _require_keys(doc["decision"], {"kind", "value"}, set(), f"{path}.decision")
    if kind == "family_slot_consistency":
        _require_const(decision["kind"], "vocabulary", f"{path}.decision.kind")
        _require_enum(decision["value"], VOCABULARY_DECISIONS, f"{path}.decision.value")
        fsc = _require_keys(
            doc["family_slot_consistency"],
            {"per_family", "summary_consistency_score", "threshold", "recommendation"},
            set(), f"{path}.family_slot_consistency",
        )
        _require_number(fsc["summary_consistency_score"], f"{path}.family_slot_consistency.summary_consistency_score",
                        minimum=0.0, maximum=1.0)
        _require_number(fsc["threshold"], f"{path}.family_slot_consistency.threshold", minimum=0.0, maximum=1.0)
        _require_enum(fsc["recommendation"], VOCABULARY_DECISIONS, f"{path}.family_slot_consistency.recommendation")
        if not isinstance(fsc["per_family"], list):
            _fail(f"{path}.family_slot_consistency.per_family", "must be an array")
    else:  # shape_coverage_coupling
        _require_const(decision["kind"], "derivability", f"{path}.decision.kind")
        _require_enum(decision["value"], DERIVABILITY_DECISIONS, f"{path}.decision.value")
        if "shape_coverage_coupling" not in doc:
            _fail(f"{path}.shape_coverage_coupling", "required when report_kind is shape_coverage_coupling")
        scc = _require_keys(
            doc["shape_coverage_coupling"],
            {"per_tile_layer_explained_variance", "bimodality_coefficient", "mixture_bic",
             "high_coupling_tile_share", "linear_underpowered_note"},
            set(), f"{path}.shape_coverage_coupling",
        )
        _require_number(scc["bimodality_coefficient"], f"{path}.shape_coverage_coupling.bimodality_coefficient",
                        minimum=0.0)
        _require_number(scc["high_coupling_tile_share"], f"{path}.shape_coverage_coupling.high_coupling_tile_share",
                        minimum=0.0, maximum=1.0)
        _require_str(scc["linear_underpowered_note"], f"{path}.shape_coverage_coupling.linear_underpowered_note")
        bic = _require_keys(
            scc["mixture_bic"], {"one_component", "two_component", "two_preferred"}, set(),
            f"{path}.shape_coverage_coupling.mixture_bic",
        )
        _require_number(bic["one_component"], f"{path}.shape_coverage_coupling.mixture_bic.one_component")
        _require_number(bic["two_component"], f"{path}.shape_coverage_coupling.mixture_bic.two_component")
        if not isinstance(bic["two_preferred"], bool):
            _fail(f"{path}.shape_coverage_coupling.mixture_bic.two_preferred", "must be a boolean")
        if not isinstance(scc["per_tile_layer_explained_variance"], list):
            _fail(f"{path}.shape_coverage_coupling.per_tile_layer_explained_variance", "must be an array")


# --------------------------------------------------------------------------- #
# v50-structure-run-v1
# --------------------------------------------------------------------------- #
def validate_structure_run(doc: Any) -> None:
    path = "$"
    _require_keys(
        doc,
        {
            "schema", "created_utc", "feature", "slot", "vocabulary_decision",
            "identity", "inputs", "architecture", "config", "split_counts",
            "baselines", "best_epoch", "metrics", "promotion_verdict", "gate",
        },
        set(),
        path,
    )
    _require_const(doc["schema"], STRUCTURE_RUN_SCHEMA, f"{path}.schema")
    _require_datetime(doc["created_utc"], f"{path}.created_utc")
    _require_const(doc["feature"], "116-relational-terrain-layers", f"{path}.feature")
    _require_int(doc["slot"], f"{path}.slot", minimum=1, maximum=3)
    _require_enum(doc["vocabulary_decision"], VOCABULARY_DECISIONS, f"{path}.vocabulary_decision")
    _require_identity(doc["identity"], f"{path}.identity")

    inputs = _require_keys(
        doc["inputs"],
        {"store", "held_out_split", "texture_name_dumps", "taxonomy_revision", "rule_set_sha256"},
        set(), f"{path}.inputs",
    )
    _require_store_ref(inputs["store"], f"{path}.inputs.store")
    held = _require_keys(
        inputs["held_out_split"], {"path", "sha256", "verified_violation_count"}, set(),
        f"{path}.inputs.held_out_split",
    )
    _require_str(held["path"], f"{path}.inputs.held_out_split.path")
    _require_sha256(held["sha256"], f"{path}.inputs.held_out_split.sha256")
    # A structure run may never proceed on a leaky split.
    _require_const(held["verified_violation_count"], 0, f"{path}.inputs.held_out_split.verified_violation_count")
    _require_dump_list(inputs["texture_name_dumps"], f"{path}.inputs.texture_name_dumps")
    _require_str(inputs["taxonomy_revision"], f"{path}.inputs.taxonomy_revision")
    _require_sha256(inputs["rule_set_sha256"], f"{path}.inputs.rule_set_sha256")

    arch = _require_keys(
        doc["architecture"], {"class", "base", "slot", "num_classes", "param_count"}, set(),
        f"{path}.architecture",
    )
    _require_str(arch["class"], f"{path}.architecture.class")
    _require_int(arch["base"], f"{path}.architecture.base", minimum=1)
    _require_int(arch["slot"], f"{path}.architecture.slot", minimum=1, maximum=3)
    _require_int(arch["num_classes"], f"{path}.architecture.num_classes", minimum=2)
    _require_int(arch["param_count"], f"{path}.architecture.param_count", minimum=0)

    cfg = _require_keys(
        doc["config"], {"batch_size", "epochs", "lr", "max_class_weight", "device"}, set(),
        f"{path}.config",
    )
    _require_int(cfg["batch_size"], f"{path}.config.batch_size", minimum=1)
    _require_int(cfg["epochs"], f"{path}.config.epochs", minimum=1)
    _require_number(cfg["lr"], f"{path}.config.lr", minimum=0.0)
    _require_number(cfg["max_class_weight"], f"{path}.config.max_class_weight", minimum=0.0)
    _require_str(cfg["device"], f"{path}.config.device")

    counts = _require_keys(doc["split_counts"], {"train", "held_out"}, set(), f"{path}.split_counts")
    _require_int(counts["train"], f"{path}.split_counts.train", minimum=0)
    _require_int(counts["held_out"], f"{path}.split_counts.held_out", minimum=0)

    _require_keys(doc["baselines"], {"majority_class"}, set(), f"{path}.baselines")
    _require_int(doc["best_epoch"], f"{path}.best_epoch", minimum=0)

    metrics = _require_keys(
        doc["metrics"],
        {"per_class", "macro_iou", "rarest_class_iou", "rarest_class_recall", "aggregate_accuracy_reported_only"},
        set(), f"{path}.metrics",
    )
    if not isinstance(metrics["per_class"], dict):
        _fail(f"{path}.metrics.per_class", "must be an object of family -> {iou, recall}")
    _require_number(metrics["macro_iou"], f"{path}.metrics.macro_iou", minimum=0.0, maximum=1.0)
    _require_number(metrics["rarest_class_iou"], f"{path}.metrics.rarest_class_iou", minimum=0.0, maximum=1.0)
    _require_number(metrics["rarest_class_recall"], f"{path}.metrics.rarest_class_recall", minimum=0.0, maximum=1.0)
    _require_number(metrics["aggregate_accuracy_reported_only"], f"{path}.metrics.aggregate_accuracy_reported_only",
                    minimum=0.0, maximum=1.0)

    _require_enum(doc["promotion_verdict"], PROMOTION_VERDICTS, f"{path}.promotion_verdict")
    gate = _require_keys(doc["gate"], {"rule", "rarest_class", "sc003"}, set(), f"{path}.gate")
    _require_const(gate["rule"], "per_class_iou_recall", f"{path}.gate.rule")
    _require_str(gate["rarest_class"], f"{path}.gate.rarest_class")
    if not isinstance(gate["sc003"], bool):
        _fail(f"{path}.gate.sc003", "must be a boolean")


# --------------------------------------------------------------------------- #
# v50-structure-infer-v1
# --------------------------------------------------------------------------- #
def validate_structure_infer(doc: Any) -> None:
    path = "$"
    _require_keys(
        doc,
        {"schema", "created_utc", "checkpoint", "inputs", "legal_table_available",
         "sc004_all_references_legal", "per_tile"},
        set(), path,
    )
    _require_const(doc["schema"], STRUCTURE_INFER_SCHEMA, f"{path}.schema")
    _require_datetime(doc["created_utc"], f"{path}.created_utc")
    ckpt = _require_keys(doc["checkpoint"], {"path", "sha256", "taxonomy_revision"}, set(), f"{path}.checkpoint")
    _require_str(ckpt["path"], f"{path}.checkpoint.path")
    _require_sha256(ckpt["sha256"], f"{path}.checkpoint.sha256")
    _require_str(ckpt["taxonomy_revision"], f"{path}.checkpoint.taxonomy_revision")
    if not isinstance(doc["inputs"], list) or not doc["inputs"]:
        _fail(f"{path}.inputs", "must be a non-empty array of {path,sha256}")
    for index, item in enumerate(doc["inputs"]):
        _require_store_ref(item, f"{path}.inputs[{index}]")
    if not isinstance(doc["legal_table_available"], bool):
        _fail(f"{path}.legal_table_available", "must be a boolean")
    if not isinstance(doc["sc004_all_references_legal"], bool):
        _fail(f"{path}.sc004_all_references_legal", "must be a boolean")
    if not isinstance(doc["per_tile"], list):
        _fail(f"{path}.per_tile", "must be an array")


# --------------------------------------------------------------------------- #
# v50-structure-geometry-comparison-v1
# --------------------------------------------------------------------------- #
def validate_structure_geometry_comparison(doc: Any) -> None:
    path = "$"
    _require_keys(
        doc,
        {"schema", "held_out_split", "without_structure", "with_structure",
         "sc007_beats_trivial_on_relief", "absolute_comparison_to_prior_runs_invalid"},
        set(), path,
    )
    _require_const(doc["schema"], STRUCTURE_GEOMETRY_COMPARISON_SCHEMA, f"{path}.schema")
    held = _require_keys(doc["held_out_split"], {"path", "sha256"}, set(), f"{path}.held_out_split")
    _require_str(held["path"], f"{path}.held_out_split.path")
    _require_sha256(held["sha256"], f"{path}.held_out_split.sha256")
    for side in ("without_structure", "with_structure"):
        s = _require_keys(
            doc[side], {"checkpoint_sha256", "relief_mae", "flat_mae", "trivial_baseline_relief_mae"},
            set(), f"{path}.{side}",
        )
        _require_sha256(s["checkpoint_sha256"], f"{path}.{side}.checkpoint_sha256")
        _require_number(s["relief_mae"], f"{path}.{side}.relief_mae", minimum=0.0)
        _require_number(s["flat_mae"], f"{path}.{side}.flat_mae", minimum=0.0)
        _require_number(s["trivial_baseline_relief_mae"], f"{path}.{side}.trivial_baseline_relief_mae", minimum=0.0)
    if not isinstance(doc["sc007_beats_trivial_on_relief"], bool):
        _fail(f"{path}.sc007_beats_trivial_on_relief", "must be a boolean")
    _require_const(doc["absolute_comparison_to_prior_runs_invalid"], True,
                    f"{path}.absolute_comparison_to_prior_runs_invalid")


def validate_document(doc: Any) -> None:
    """Dispatch on the document's ``schema`` const across all Spec 116 variants."""
    if not isinstance(doc, dict) or "schema" not in doc:
        raise Spec116ContractError("$: document must be an object with a 'schema' key")
    schema = doc["schema"]
    if schema == HELD_OUT_SPLIT_SCHEMA:
        validate_held_out_split(doc)
    elif schema == ANALYSIS_REPORT_SCHEMA:
        validate_analysis_report(doc)
    elif schema == STRUCTURE_RUN_SCHEMA:
        validate_structure_run(doc)
    elif schema == STRUCTURE_INFER_SCHEMA:
        validate_structure_infer(doc)
    elif schema == STRUCTURE_GEOMETRY_COMPARISON_SCHEMA:
        validate_structure_geometry_comparison(doc)
    else:
        raise Spec116ContractError(f"$: unknown schema {schema!r}")


__all__ = [
    "Spec116ContractError",
    "validate_held_out_split",
    "validate_analysis_report",
    "validate_structure_run",
    "validate_structure_infer",
    "validate_structure_geometry_comparison",
    "validate_document",
    "sha256_file",
    "sha256_json",
    "HELD_OUT_SPLIT_SCHEMA",
    "ANALYSIS_REPORT_SCHEMA",
    "STRUCTURE_RUN_SCHEMA",
    "STRUCTURE_INFER_SCHEMA",
    "STRUCTURE_GEOMETRY_COMPARISON_SCHEMA",
]
