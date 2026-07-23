"""Spec 114 reconstruction contract validation (T006).

Hand-rolled structural validation for the three document variants published in
``specs/114-direct-terrain-reconstruction/contracts/model-stage-and-curriculum.schema.json``:

- ``v50-reconstruction-curriculum-v1`` (dual-view geometry curriculum summary)
- ``v50-object-visibility-v1`` (trusted object-label summary)
- ``v50-model-stage-run-v1`` (one model stage's run/evaluation record)

The validator is deliberately dependency-free (no ``jsonschema`` package) so the gates stay pure
CPU functions. Every check mirrors the published schema: required keys, ``additionalProperties:
false`` semantics, const values, enums, sha256 patterns, and integer bounds. Identity helpers bind
documents to real file content so a curriculum or checkpoint cannot be silently swapped after a
run record names it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

CURRICULUM_SCHEMA = "v50-reconstruction-curriculum-v1"
OBJECT_VISIBILITY_SCHEMA = "v50-object-visibility-v1"
MODEL_STAGE_RUN_SCHEMA = "v50-model-stage-run-v1"

TARGET_SIGNAL = "relative_height_257"
SYNTHETIC_LIGHTING_CONTRACT = "NoonWhiteGlobal"
STAGES = frozenset(
    {
        "direct_geometry", "object_visibility", "terrain_features", "texture_families", "alpha_stack",
        # Spec 117: the standalone WDL-lattice predictor is one more independently trained/
        # checkpointed stage in this same residual chain (research.md D-01) -- reusing this schema
        # verbatim rather than inventing a parallel one.
        "lattice_prior",
        # Spec 118: the from-scratch visible-object segmenter is likewise one more independent
        # stage (research.md D-06), reusing this schema verbatim.
        "object_segmentation",
        # Spec 119: the object-library classifier and segmenter are two more independent,
        # separately checkpointed stages (research.md D-02/D-05), reusing this schema verbatim.
        "object_library_classifier",
        "object_library_segmenter",
    }
)
PROMOTION_VERDICTS = frozenset({"pending", "promoted", "rejected"})

_HEX = frozenset("0123456789abcdefABCDEF")


class ContractViolationError(ValueError):
    """Raised when a document does not satisfy the published Spec 114 contract."""


def _fail(path: str, message: str) -> None:
    raise ContractViolationError(f"{path}: {message}")


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


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}, got {value!r}")
    return value


def _require_const(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        _fail(path, f"must be {expected!r}, got {value!r}")


def _require_datetime(value: Any, path: str) -> None:
    text = _require_str(value, path)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _fail(path, f"must be an ISO-8601 date-time, got {text!r}")


def validate_identity(obj: Any, path: str = "$") -> None:
    _require_keys(obj, {"path", "sha256"}, set(), path)
    _require_str(obj["path"], f"{path}.path")
    if not _is_sha256(obj["sha256"]):
        _fail(f"{path}.sha256", f"must be 64 hex characters, got {obj['sha256']!r}")


def _validate_identities(value: Any, path: str, *, min_items: int) -> None:
    if not isinstance(value, list) or len(value) < min_items:
        _fail(path, f"must be an array with at least {min_items} item(s)")
    for index, item in enumerate(value):
        validate_identity(item, f"{path}[{index}]")


def _validate_split_counts(obj: Any, path: str) -> None:
    _require_keys(obj, {"train", "validation", "test"}, set(), path)
    for key in ("train", "validation", "test"):
        _require_int(obj[key], f"{path}.{key}")


def _validate_excluded_counts(obj: Any, path: str) -> None:
    if not isinstance(obj, dict):
        _fail(path, "must be an object of reason -> count")
    for reason, count in obj.items():
        _require_int(count, f"{path}.{reason}")


def validate_curriculum_summary(doc: Any) -> None:
    """Validate a ``v50-reconstruction-curriculum-v1`` document."""
    path = "$"
    _require_keys(
        doc,
        {
            "schema", "curriculum_id", "created_utc", "source_stores", "row_count",
            "source_group_count", "split_counts", "input_origins", "target_signal",
            "synthetic_lighting_contract", "group_leak_count", "excluded_counts",
        },
        {"generated_input_models"},
        path,
    )
    _require_const(doc["schema"], CURRICULUM_SCHEMA, f"{path}.schema")
    _require_str(doc["curriculum_id"], f"{path}.curriculum_id")
    _require_datetime(doc["created_utc"], f"{path}.created_utc")
    _validate_identities(doc["source_stores"], f"{path}.source_stores", min_items=1)
    _require_int(doc["row_count"], f"{path}.row_count", minimum=1)
    _require_int(doc["source_group_count"], f"{path}.source_group_count", minimum=1)
    _validate_split_counts(doc["split_counts"], f"{path}.split_counts")
    origins = _require_keys(
        doc["input_origins"], {"authored", "synthetic_noon_white"}, set(), f"{path}.input_origins"
    )
    _require_int(origins["authored"], f"{path}.input_origins.authored")
    _require_int(origins["synthetic_noon_white"], f"{path}.input_origins.synthetic_noon_white")
    _require_const(doc["target_signal"], TARGET_SIGNAL, f"{path}.target_signal")
    _require_const(
        doc["synthetic_lighting_contract"], SYNTHETIC_LIGHTING_CONTRACT,
        f"{path}.synthetic_lighting_contract",
    )
    _require_const(doc["group_leak_count"], 0, f"{path}.group_leak_count")
    _validate_excluded_counts(doc["excluded_counts"], f"{path}.excluded_counts")
    if "generated_input_models" in doc:
        _validate_identities(
            doc["generated_input_models"], f"{path}.generated_input_models", min_items=0
        )


def validate_object_visibility_summary(doc: Any) -> None:
    """Validate a ``v50-object-visibility-v1`` document."""
    path = "$"
    _require_keys(
        doc,
        {
            "schema", "evidence_id", "created_utc", "renderer_revision", "placement_sources",
            "populated_rows", "empty_rows", "unavailable_rows", "alignment_fixture_count",
        },
        {"mask_shape"},
        path,
    )
    _require_const(doc["schema"], OBJECT_VISIBILITY_SCHEMA, f"{path}.schema")
    _require_str(doc["evidence_id"], f"{path}.evidence_id")
    _require_datetime(doc["created_utc"], f"{path}.created_utc")
    _require_str(doc["renderer_revision"], f"{path}.renderer_revision")
    _validate_identities(doc["placement_sources"], f"{path}.placement_sources", min_items=1)
    for key in ("populated_rows", "empty_rows", "unavailable_rows"):
        _require_int(doc[key], f"{path}.{key}")
    _require_int(doc["alignment_fixture_count"], f"{path}.alignment_fixture_count", minimum=1)
    if "mask_shape" in doc:
        shape = doc["mask_shape"]
        if not isinstance(shape, list) or len(shape) != 2:
            _fail(f"{path}.mask_shape", "must be [height, width]")
        _require_int(shape[0], f"{path}.mask_shape[0]", minimum=1)
        _require_int(shape[1], f"{path}.mask_shape[1]", minimum=1)


def _validate_pretrained_source(value: Any, path: str) -> None:
    if value is None:
        return
    _require_keys(value, {"hub_id", "revision", "sha256", "license", "optional"}, set(), path)
    _require_str(value["hub_id"], f"{path}.hub_id")
    _require_str(value["revision"], f"{path}.revision")
    if not _is_sha256(value["sha256"]):
        _fail(f"{path}.sha256", "must be 64 hex characters")
    _require_str(value["license"], f"{path}.license")
    _require_const(value["optional"], True, f"{path}.optional")


def validate_model_stage_run(doc: Any) -> None:
    """Validate a ``v50-model-stage-run-v1`` document."""
    path = "$"
    _require_keys(
        doc,
        {
            "schema", "run_id", "created_utc", "stage", "output_signal", "architecture",
            "curriculum", "upstream_models", "checkpoint", "baselines", "metrics",
            "visual_evidence", "promotion_verdict",
        },
        {"pretrained_source"},
        path,
    )
    _require_const(doc["schema"], MODEL_STAGE_RUN_SCHEMA, f"{path}.schema")
    _require_str(doc["run_id"], f"{path}.run_id")
    _require_datetime(doc["created_utc"], f"{path}.created_utc")
    if doc["stage"] not in STAGES:
        _fail(f"{path}.stage", f"must be one of {sorted(STAGES)}, got {doc['stage']!r}")
    _require_str(doc["output_signal"], f"{path}.output_signal")

    architecture = _require_keys(
        doc["architecture"], {"id", "config_sha256", "parameter_count"}, set(), f"{path}.architecture"
    )
    _require_str(architecture["id"], f"{path}.architecture.id")
    if not _is_sha256(architecture["config_sha256"]):
        _fail(f"{path}.architecture.config_sha256", "must be 64 hex characters")
    _require_int(
        architecture["parameter_count"], f"{path}.architecture.parameter_count", minimum=1
    )
    _validate_pretrained_source(doc.get("pretrained_source"), f"{path}.pretrained_source")

    validate_identity(doc["curriculum"], f"{path}.curriculum")
    _validate_identities(doc["upstream_models"], f"{path}.upstream_models", min_items=0)

    checkpoint = _require_keys(
        doc["checkpoint"], {"path", "sha256", "best_epoch"}, set(), f"{path}.checkpoint"
    )
    _require_str(checkpoint["path"], f"{path}.checkpoint.path")
    if not _is_sha256(checkpoint["sha256"]):
        _fail(f"{path}.checkpoint.sha256", "must be 64 hex characters")
    _require_int(checkpoint["best_epoch"], f"{path}.checkpoint.best_epoch", minimum=1)

    for key in ("baselines", "metrics", "visual_evidence"):
        if not isinstance(doc[key], dict):
            _fail(f"{path}.{key}", "must be an object")
    if doc["promotion_verdict"] not in PROMOTION_VERDICTS:
        _fail(
            f"{path}.promotion_verdict",
            f"must be one of {sorted(PROMOTION_VERDICTS)}, got {doc['promotion_verdict']!r}",
        )


def validate_document(doc: Any) -> None:
    """Dispatch on the document's ``schema`` const across all three Spec 114 variants."""
    if not isinstance(doc, dict):
        _fail("$", "document must be an object")
    schema = doc.get("schema")
    validators = {
        CURRICULUM_SCHEMA: validate_curriculum_summary,
        OBJECT_VISIBILITY_SCHEMA: validate_object_visibility_summary,
        MODEL_STAGE_RUN_SCHEMA: validate_model_stage_run,
    }
    validator = validators.get(schema)
    if validator is None:
        _fail("$.schema", f"unknown Spec 114 schema {schema!r}")
    validator(doc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Content hash of a JSON-serializable object under canonical key ordering."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def identity_for_path(path: Path, *, display_path: str | None = None) -> dict[str, str]:
    """Build a schema ``identity`` binding a document to the file's current content."""
    if not path.is_file():
        raise ContractViolationError(f"identity target does not exist or is not a file: {path}")
    return {"path": display_path or str(path), "sha256": sha256_file(path)}


def verify_identity(identity: dict, path: Path) -> None:
    """Fail closed when the named file no longer matches the recorded content hash."""
    validate_identity(identity)
    actual = sha256_file(path)
    if actual.lower() != identity["sha256"].lower():
        raise ContractViolationError(
            f"identity drift for {identity['path']!r}: recorded {identity['sha256']}, "
            f"file now hashes {actual}"
        )


def append_generated_input(summary: dict, identity: dict) -> dict:
    """Return a copy of a curriculum summary with one generated upstream signal named.

    FR-006/FR-015: any downstream consumer of a generated signal must record the checkpoint and
    inference-run identity that produced it. The identity is validated before it is attached, so a
    malformed provenance entry can never enter a persisted curriculum.
    """
    validate_identity(identity, "$.generated_input")
    updated = dict(summary)
    models = list(updated.get("generated_input_models", []))
    models.append(identity)
    updated["generated_input_models"] = models
    validate_curriculum_summary(updated)
    return updated
