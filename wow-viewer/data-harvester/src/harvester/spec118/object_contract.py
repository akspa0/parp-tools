"""Spec 118 contract helpers: the from-scratch visible-object segmenter's schema identity.

Per research.md D-06, this feature deliberately reuses two already-validated schemas verbatim
rather than inventing new ones:

- ``v50-model-stage-run-v1`` (``harvester.v50.model_stage_contract``) for the segmenter's own
  training run record, with a new ``stage="object_segmentation"`` value (the one addition that
  schema needed -- see its widened ``STAGES`` frozenset).
- ``v115-feature-map-v1`` for the bridged generated-object feature store
  (``object_feature_bridge.py``), which requires no code here at all since
  ``direct_geometry_train.py``/``geometry_detailer_train.py`` already validate that schema
  structurally.

This module owns only what's new: the stage/output-signal constants, the class table
(``none``/``doodad``/``building`` from ``ObjectGeometryPixelSource``), an architecture-identity
builder matching ``direct_geometry_model.architecture_identity``'s shape, and a thin
``build_object_stage_run`` assembler that self-validates before returning. sha256/identity
helpers are re-exported from ``harvester.v50.model_stage_contract`` (single canonical owner; no
duplication, mirroring ``harvester.spec117.lattice_contract``'s own re-export pattern).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from torch import nn

from harvester.v50.model_stage_contract import (
    ContractViolationError,
    identity_for_path,
    sha256_file,
    sha256_json,
    validate_model_stage_run,
)

STAGE = "object_segmentation"
OUTPUT_SIGNAL = "object_class_2"

# Binary per-pixel target: the v18 placement-footprint mask (`object_mask`) carries no
# doodad-vs-building split (both are painted into one mask, and the count boundary needed to split
# them via `object_instance_mask` is a 1-D per-tile field the curriculum does not copy). So the
# segmenter predicts none-vs-object, which is exactly what deployment needs: identify object pixels
# in a real minimap so they can be masked out of the terrain/shadow signal.
CLASS_NAMES = ("none", "object")
CLASS_COUNT = len(CLASS_NAMES)  # 2
# The bridge drops the redundant ``none`` channel (1 - object), leaving one object-probability map.
BRIDGE_CLASS_COUNT = 1


class ObjectContractError(ValueError):
    """Raised when a Spec 118 document violates its (reused) contract."""


def architecture_identity(model: nn.Module, *, architecture_id: str, config: dict[str, Any]) -> dict:
    """Schema-conformant ``architecture`` block: local id, content-hashed config, parameter count.

    Mirrors ``harvester.v50.direct_geometry_model.architecture_identity`` exactly so every stage in
    the residual chain records identity the same way.
    """
    return {
        "id": architecture_id,
        "config_sha256": sha256_json({key: config[key] for key in sorted(config)}),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_object_stage_run(
    *,
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
    """Assemble and self-validate the published ``v50-model-stage-run-v1`` record for this stage.

    ``upstream_models`` is always ``[]``: the object segmenter has no upstream generated input of
    its own (data-model.md Run-Record Schema).
    """
    summary = {
        "schema": "v50-model-stage-run-v1",
        "run_id": run_id,
        "created_utc": created_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": STAGE,
        "output_signal": OUTPUT_SIGNAL,
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
        raise ObjectContractError(f"object stage-run record violates its own contract: {exc}") from exc
    return summary


__all__ = [
    "ObjectContractError",
    "STAGE",
    "OUTPUT_SIGNAL",
    "CLASS_NAMES",
    "CLASS_COUNT",
    "BRIDGE_CLASS_COUNT",
    "architecture_identity",
    "build_object_stage_run",
    "sha256_file",
    "sha256_json",
    "identity_for_path",
]
