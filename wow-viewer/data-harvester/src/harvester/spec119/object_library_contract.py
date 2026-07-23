"""Spec 119 object-library classifier/segmenter contract (T003).

Pure-function label/family/target derivations shared by the split builder, both trainers, the
inference CLI, and the quality lens. No I/O here: callers pass asset rows (dicts from
``assets.parquet``) and mask arrays; this module owns the CoarseClassLabel map, the AssetFamily
derivation (the leakage-safe split key, research D-01), the heuristic FineFamilyLabel (D-03),
the SegmentationTarget derivation, and blank-capture handling (D-04, FR-006).

Run-record helpers (``architecture_identity``/``build_stage_run``) mirror Spec 118's
``object_contract.py`` so every stage in the residual chain records identity the same way; the
``v50-model-stage-run-v1`` schema is reused verbatim with two new stages (T004).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import numpy as np

from harvester.v50.model_stage_contract import (
    ContractViolationError,
    identity_for_path,
    sha256_file,
    sha256_json,
    validate_model_stage_run,
)

# ---- Stages (T004 widens ``model_stage_contract.STAGES`` with these exact strings) ----------

STAGE_CLASSIFIER = "object_library_classifier"
STAGE_SEGMENTER = "object_library_segmenter"
OUTPUT_SIGNAL_CLASSIFIER = "object_library_class_coarse"
OUTPUT_SIGNAL_SEGMENTER = "object_library_foreground_mask"

# ---- Coarse class labels (data-model.md: CoarseClassLabel) ----------------------------------

# empty=0 so it is the default/background class; the rest alphabetical for readability.
COARSE_CLASS_INDEX: dict[str, int] = {"empty": 0, "m2": 1, "mdx": 2, "wmo": 3}
COARSE_INDEX_CLASS: dict[int, str] = {index: name for name, index in COARSE_CLASS_INDEX.items()}
COARSE_ASSET_TYPES: frozenset[str] = frozenset({"m2", "mdx", "wmo"})

# D-04 / FR-006: captures whose mask coverage falls below this fraction are "empty".
BLANK_THRESHOLD_DEFAULT = 0.01

EMPTY_CLASS = "empty"

# Numeric-suffix variant patterns for the leakage check (research D-01): ``castle01``/``castle02``
# and ``name_000``/``name_001`` are near-duplicate numbered variants of one asset family.
_VARIANT_SUFFIX_RE = re.compile(r"^(?P<stem>.*?)(?:_\d{3}|\d+)$")


class ObjectLibraryContractError(ValueError):
    """Raised when a Spec 119 row or document violates its contract."""


def mask_coverage(mask: np.ndarray) -> float:
    """Fraction of pixels marked object (``mask > 0``)."""
    array = np.asarray(mask)
    if array.size == 0:
        return 0.0
    return float((array > 0).mean())


def is_blank_capture(coverage: float, threshold: float = BLANK_THRESHOLD_DEFAULT) -> bool:
    """D-04: a capture below the coverage threshold is empty/blank (failed or near-blank render)."""
    return float(coverage) < float(threshold)


def coarse_label_for_row(
    asset_type: str,
    coverage: float,
    threshold: float = BLANK_THRESHOLD_DEFAULT,
) -> str:
    """The classifier's primary label for one asset row (D-03 + D-04).

    Blank captures are relabeled ``empty`` so the classifier learns to flag them rather than
    confidently misclassify a blank image (FR-006). Non-blank rows use the authoritative
    ``asset_type`` from ``assets.parquet``; anything outside the coarse set is a data error,
    not a fourth class.
    """
    if is_blank_capture(coverage, threshold):
        return EMPTY_CLASS
    if asset_type not in COARSE_ASSET_TYPES:
        raise ObjectLibraryContractError(
            f"unknown asset_type {asset_type!r}; expected one of {sorted(COARSE_ASSET_TYPES)}"
        )
    return asset_type


def coarse_index_for_label(label: str) -> int:
    if label not in COARSE_CLASS_INDEX:
        raise ObjectLibraryContractError(
            f"unknown coarse label {label!r}; expected one of {sorted(COARSE_CLASS_INDEX)}"
        )
    return COARSE_CLASS_INDEX[label]


def _path_parts(normalized_asset_path: str) -> list[str]:
    return [part for part in normalized_asset_path.replace("\\", "/").strip("/").split("/") if part]


def derive_asset_family(normalized_asset_path: str) -> str:
    """The leakage-safe split key (research D-01): the asset's parent directory.

    ``world/wmo/azeroth/buildings/castle/castle01.wmo`` → ``world/wmo/azeroth/buildings/castle``.
    Numbered variants in one directory share one family and therefore never straddle the
    train/held-out split (FR-004). A top-level file (no directory) is its own family.
    """
    parts = _path_parts(normalized_asset_path)
    if len(parts) < 2:
        return parts[0] if parts else ""
    return "/".join(parts[:-1])


def derive_fine_family_label(normalized_asset_path: str) -> str:
    """The heuristic finer family token (research D-3): the containing directory's name.

    ``world/wmo/azeroth/buildings/castle/castle01.wmo`` → ``castle``. Explicitly heuristic,
    never the primary success metric (SC-001 is coarse-only).
    """
    parts = _path_parts(normalized_asset_path)
    if len(parts) < 2:
        return "unknown"
    return parts[-2]


def variant_stem(normalized_asset_path: str) -> str:
    """Filename stem with a trailing numeric variant suffix stripped (``castle01`` → ``castle``).

    Used by the split's leakage check: two assets sharing (family-ish path, variant stem) are
    near-duplicate numbered variants and must not straddle train/held-out.
    """
    parts = _path_parts(normalized_asset_path)
    if not parts:
        return ""
    filename = parts[-1]
    # Strip compound suffixes like ``.wmo.mpq`` then simple ``.m2``/``.mdx``/``.wmo``.
    for suffix in (".wmo.mpq", ".wmo", ".m2", ".mdx"):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
            break
    match = _VARIANT_SUFFIX_RE.match(filename)
    return match.group("stem") if match else filename


def segmentation_target(mask: np.ndarray) -> np.ndarray:
    """The segmenter's per-pixel target (D-04): binary foreground int64 ``(H, W)``."""
    return (np.asarray(mask) > 0).astype(np.int64)


# ---- Run-record helpers (mirror spec118.object_contract; schema reused verbatim, D-05) ------


def architecture_identity(model: Any, *, architecture_id: str, config: dict[str, Any]) -> dict:
    """Schema-conformant ``architecture`` block: local id, content-hashed config, parameter count."""
    return {
        "id": architecture_id,
        "config_sha256": sha256_json({key: config[key] for key in sorted(config)}),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_stage_run(
    *,
    stage: str,
    output_signal: str,
    run_id: str,
    architecture: dict,
    curriculum: dict,
    checkpoint: dict,
    baselines: dict,
    metrics: dict,
    visual_evidence: dict | None = None,
    created_utc: str | None = None,
    promotion_verdict: str = "pending",
) -> dict:
    """Assemble and self-validate the ``v50-model-stage-run-v1`` record for a Spec 119 stage.

    ``upstream_models`` is always ``[]``: both specialists consume the object library directly
    and have no upstream generated input (Rule 7 independence).
    """
    if stage not in (STAGE_CLASSIFIER, STAGE_SEGMENTER):
        raise ObjectLibraryContractError(f"unknown Spec 119 stage {stage!r}")
    summary = {
        "schema": "v50-model-stage-run-v1",
        "run_id": run_id,
        "created_utc": created_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "output_signal": output_signal,
        "architecture": architecture,
        "curriculum": curriculum,
        "upstream_models": [],
        "checkpoint": checkpoint,
        "baselines": baselines,
        "metrics": metrics,
        "visual_evidence": visual_evidence or {},
        "promotion_verdict": promotion_verdict,
    }
    try:
        validate_model_stage_run(summary)
    except ContractViolationError as exc:
        raise ObjectLibraryContractError(
            f"Spec 119 stage-run record violates its own contract: {exc}"
        ) from exc
    return summary


__all__ = [
    "BLANK_THRESHOLD_DEFAULT",
    "COARSE_ASSET_TYPES",
    "COARSE_CLASS_INDEX",
    "COARSE_INDEX_CLASS",
    "EMPTY_CLASS",
    "OUTPUT_SIGNAL_CLASSIFIER",
    "OUTPUT_SIGNAL_SEGMENTER",
    "ObjectLibraryContractError",
    "STAGE_CLASSIFIER",
    "STAGE_SEGMENTER",
    "architecture_identity",
    "build_stage_run",
    "coarse_index_for_label",
    "coarse_label_for_row",
    "derive_asset_family",
    "derive_fine_family_label",
    "identity_for_path",
    "is_blank_capture",
    "mask_coverage",
    "segmentation_target",
    "sha256_file",
    "sha256_json",
    "variant_stem",
]
