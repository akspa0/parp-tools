"""Spec 117 contract helpers: the standalone WDL-lattice predictor's schema identity.

Per research.md D-01, this feature deliberately reuses two already-validated schemas verbatim
rather than inventing new ones:

- ``v50-model-stage-run-v1`` (``harvester.v50.model_stage_contract``) for the predictor's own
  training run record, with a new ``stage="lattice_prior"`` value (the one addition that schema
  needed -- see its widened ``STAGES`` frozenset).
- ``v115-feature-map-v1`` for the bridged generated-lattice feature store (``lattice_bridge.py``),
  which requires no code here at all since ``direct_geometry_train.py``/``geometry_detailer_train.py``
  already validate that schema structurally.

This module owns only what's new: the stage/output-signal constants, the lattice's own dimension
constants (545 = 17x17 outer + 16x16 inner, Spec 108 FR-001 / ``TerrainWdlLattice``), an
architecture-identity builder matching ``direct_geometry_model.architecture_identity``'s shape, and
a thin ``build_lattice_stage_run`` assembler that self-validates before returning. sha256/identity
helpers are re-exported from ``harvester.v50.model_stage_contract`` (single canonical owner; no
duplication, mirroring ``harvester.spec116.structure_contract``'s own re-export pattern).
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

STAGE = "lattice_prior"
OUTPUT_SIGNAL = "wdl_lattice_545"
OUTER_DIM = 17
INNER_DIM = 16
SAMPLE_COUNT = OUTER_DIM * OUTER_DIM + INNER_DIM * INNER_DIM  # 545


class LatticeContractError(ValueError):
    """Raised when a Spec 117 document violates its (reused) contract."""


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


def build_lattice_stage_run(
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

    ``upstream_models`` is always ``[]``: the lattice predictor is the coarsest stage in the chain
    (data-model.md Run-Record Schema) -- it has no upstream generated input of its own.
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
        raise LatticeContractError(f"lattice stage-run record violates its own contract: {exc}") from exc
    return summary


__all__ = [
    "LatticeContractError",
    "STAGE",
    "OUTPUT_SIGNAL",
    "OUTER_DIM",
    "INNER_DIM",
    "SAMPLE_COUNT",
    "architecture_identity",
    "build_lattice_stage_run",
    "sha256_file",
    "sha256_json",
    "identity_for_path",
]
