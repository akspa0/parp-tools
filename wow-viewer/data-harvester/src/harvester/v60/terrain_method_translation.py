"""Evidence and input-contract gates for translating external terrain methods.

This module is deliberately metadata-only. It does not download external weights, read client
arrays, or run a model. Its job is to make modality assumptions and forbidden inference reads
explicit before a later benchmark or training path is allowed to exist.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

METHOD_TRANSLATION_SCHEMA = "v60-terrain-method-translation-v1"
METHOD_LEDGER_SCHEMA = "v60-terrain-method-ledger-v1"
INPUT_CONTRACT_SCHEMA = "v60-terrain-input-contract-v1"
INPUT_AUDIT_SCHEMA = "v60-terrain-input-audit-v1"

INPUT_MODALITIES = frozenset({"rgb", "dsm", "point_cloud", "mask", "multispectral", "metadata"})
BRANCHES = frozenset({"rgb_only", "height_prior", "point_cloud", "combined"})
RUNTIME_CLAIMS = frozenset({"none", "offline_diagnostic", "deployment_candidate"})
SOURCE_KINDS = frozenset({"paper", "github", "huggingface", "documentation", "dataset"})
LICENSE_STATUSES = frozenset({"verified", "needs_review", "unknown", "not_applicable"})
WEIGHTS_STATUSES = frozenset({"not_required", "available", "unavailable", "not_reviewed"})
TRANSLATION_STATUSES = frozenset(
    {"reference", "diagnostic", "candidate", "hold", "rejected", "promoted"}
)

FORBIDDEN_INPUTS = frozenset(
    {
        "height_257",
        "terrain_shadow_256",
        "shadow_mask",
        "wdl",
        "object_mask",
        "object_precise_mask",
        "source_object_mask",
        "target",
        "alpha",
        "normal_xyz",
        "liquid_mask",
    }
)

_SIGNAL_ALIASES = {
    "rgb": "minimap_rgb",
    "minimap_rgb": "minimap_rgb",
    "dsm": "dsm",
    "digital_surface_model": "dsm",
    "height_prior": "dsm",
    "point_cloud": "point_cloud",
    "lidar": "point_cloud",
    "xyz": "point_cloud",
    "minimap_object_mask": "predicted_object_mask",
    "predicted_mask": "predicted_object_mask",
    "predicted_object_mask": "predicted_object_mask",
    "height": "height_257",
    "height_257": "height_257",
    "terrain_shadow": "terrain_shadow_256",
    "terrain_shadow_256": "terrain_shadow_256",
    "mcsh": "shadow_mask",
    "shadow": "shadow_mask",
    "shadow_mask": "shadow_mask",
    "object_mask": "object_mask",
    "object_precise_mask": "object_precise_mask",
    "source_object_mask": "source_object_mask",
    "wdl": "wdl",
    "alpha": "alpha",
    "normal": "normal_xyz",
    "normal_xyz": "normal_xyz",
    "liquid": "liquid_mask",
    "liquid_mask": "liquid_mask",
    "target": "target",
}


class TerrainMethodTranslationError(ValueError):
    """Raised when method or input-contract metadata violates the v60 contract."""


def _clean_strings(values: Iterable[Any], *, field: str) -> tuple[str, ...]:
    cleaned = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
    if not cleaned:
        raise TerrainMethodTranslationError(f"{field} must contain at least one non-empty value")
    return cleaned


def canonical_signal_name(value: Any) -> str:
    """Return the contract name for a signal alias while preserving unknown names."""

    cleaned = str(value).strip().lower()
    return _SIGNAL_ALIASES.get(cleaned, cleaned)


def _canonical_strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({canonical_signal_name(value) for value in values if str(value).strip()}))


@dataclass(frozen=True)
class ExternalMethodRecord:
    """One external method and its current WoW translation status."""

    method_id: str
    name: str
    source_urls: tuple[str, ...]
    source_kind: str
    accessed_at: str
    input_modalities: tuple[str, ...]
    output_signals: tuple[str, ...]
    domain: str
    license_status: str
    weights_status: str
    translation_status: str
    translation_reason: str

    def __post_init__(self) -> None:
        if not self.method_id.strip():
            raise TerrainMethodTranslationError("method_id must not be empty")
        if not self.name.strip():
            raise TerrainMethodTranslationError("name must not be empty")
        if not self.source_urls or any(not value.strip() for value in self.source_urls):
            raise TerrainMethodTranslationError("source_urls must contain non-empty URLs")
        if self.source_kind not in SOURCE_KINDS:
            raise TerrainMethodTranslationError(f"invalid source_kind {self.source_kind!r}")
        if not self.accessed_at.strip():
            raise TerrainMethodTranslationError("accessed_at must not be empty")
        if not self.input_modalities or any(value not in INPUT_MODALITIES for value in self.input_modalities):
            raise TerrainMethodTranslationError(
                f"input_modalities must use {sorted(INPUT_MODALITIES)!r}"
            )
        if not self.output_signals or any(not value.strip() for value in self.output_signals):
            raise TerrainMethodTranslationError("output_signals must contain non-empty values")
        if not self.domain.strip():
            raise TerrainMethodTranslationError("domain must not be empty")
        if self.license_status not in LICENSE_STATUSES:
            raise TerrainMethodTranslationError(f"invalid license_status {self.license_status!r}")
        if self.weights_status not in WEIGHTS_STATUSES:
            raise TerrainMethodTranslationError(f"invalid weights_status {self.weights_status!r}")
        if self.translation_status not in TRANSLATION_STATUSES:
            raise TerrainMethodTranslationError(
                f"invalid translation_status {self.translation_status!r}"
            )
        if not self.translation_reason.strip():
            raise TerrainMethodTranslationError("translation_reason must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExternalMethodRecord:
        return cls(
            method_id=str(value["method_id"]),
            name=str(value["name"]),
            source_urls=_clean_strings(value["source_urls"], field="source_urls"),
            source_kind=str(value["source_kind"]),
            accessed_at=str(value["accessed_at"]),
            input_modalities=_clean_strings(value["input_modalities"], field="input_modalities"),
            output_signals=_clean_strings(value["output_signals"], field="output_signals"),
            domain=str(value["domain"]),
            license_status=str(value["license_status"]),
            weights_status=str(value["weights_status"]),
            translation_status=str(value["translation_status"]),
            translation_reason=str(value["translation_reason"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InputContract:
    """Declared observable, predicted, supervision-only, and forbidden input boundary."""

    contract_id: str
    branch: str
    observable_inputs: tuple[str, ...]
    predicted_inputs: tuple[str, ...]
    supervision_only_inputs: tuple[str, ...]
    forbidden_inputs: tuple[str, ...]
    runtime_claim: str

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise TerrainMethodTranslationError("contract_id must not be empty")
        if self.branch not in BRANCHES:
            raise TerrainMethodTranslationError(f"invalid branch {self.branch!r}")
        if self.runtime_claim not in RUNTIME_CLAIMS:
            raise TerrainMethodTranslationError(f"invalid runtime_claim {self.runtime_claim!r}")
        observable = _canonical_strings(self.observable_inputs)
        predicted = _canonical_strings(self.predicted_inputs)
        supervision = _canonical_strings(self.supervision_only_inputs)
        forbidden = _canonical_strings(self.forbidden_inputs)
        if not observable:
            raise TerrainMethodTranslationError("observable_inputs must not be empty")
        overlap = (set(observable) & set(predicted)) | (set(observable) & set(supervision))
        if overlap:
            raise TerrainMethodTranslationError(f"observable inputs overlap another input class: {sorted(overlap)}")
        if set(predicted) & set(supervision):
            raise TerrainMethodTranslationError("predicted_inputs overlap supervision_only_inputs")
        missing_forbidden = FORBIDDEN_INPUTS - set(forbidden)
        if missing_forbidden:
            raise TerrainMethodTranslationError(
                f"forbidden_inputs must declare all standard forbidden signals; missing {sorted(missing_forbidden)}"
            )
        if set(observable) & set(forbidden):
            raise TerrainMethodTranslationError("observable_inputs contain forbidden signals")
        if set(predicted) & set(forbidden):
            raise TerrainMethodTranslationError("predicted_inputs contain forbidden signals")
        if self.branch == "rgb_only" and "minimap_rgb" not in set(observable):
            raise TerrainMethodTranslationError("rgb_only contracts require observable minimap_rgb")
        if self.branch == "height_prior" and not {"dsm"} & set(observable):
            raise TerrainMethodTranslationError("height_prior contracts require observable dsm")
        if self.branch == "point_cloud" and "point_cloud" not in set(observable):
            raise TerrainMethodTranslationError("point_cloud contracts require observable point_cloud")
        if self.branch == "combined" and len(set(observable) & {"minimap_rgb", "dsm", "point_cloud"}) < 2:
            raise TerrainMethodTranslationError("combined contracts require at least two primary observables")
        if self.runtime_claim == "deployment_candidate" and self.branch != "rgb_only":
            raise TerrainMethodTranslationError(
                "only rgb_only contracts may claim current deployment candidacy"
            )
        object.__setattr__(self, "observable_inputs", observable)
        object.__setattr__(self, "predicted_inputs", predicted)
        object.__setattr__(self, "supervision_only_inputs", supervision)
        object.__setattr__(self, "forbidden_inputs", forbidden)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INPUT_CONTRACT_SCHEMA,
            **asdict(self),
        }


@dataclass(frozen=True)
class TranslationDecision:
    """Evidence-backed status for a method or evidence run."""

    subject_id: str
    status: str
    reason: str
    required_next_gate: str
    reviewed_at: str
    reviewer_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise TerrainMethodTranslationError("subject_id must not be empty")
        if self.status not in TRANSLATION_STATUSES:
            raise TerrainMethodTranslationError(f"invalid decision status {self.status!r}")
        if not self.reason.strip():
            raise TerrainMethodTranslationError("decision reason must not be empty")
        if not self.required_next_gate.strip():
            raise TerrainMethodTranslationError("required_next_gate must not be empty")
        if not self.reviewed_at.strip():
            raise TerrainMethodTranslationError("reviewed_at must not be empty")
        artifacts = tuple(str(value).strip() for value in self.reviewer_artifacts if str(value).strip())
        if self.status == "promoted" and not artifacts:
            raise TerrainMethodTranslationError("promoted decisions require reviewer_artifacts")
        object.__setattr__(self, "reviewer_artifacts", tuple(sorted(set(artifacts))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_method_records(records: Sequence[ExternalMethodRecord]) -> dict[str, Any]:
    """Validate a ledger and return a deterministic JSON-safe report."""

    failures: list[str] = []
    method_ids = [record.method_id for record in records]
    duplicates = sorted(method_id for method_id, count in Counter(method_ids).items() if count > 1)
    if duplicates:
        failures.append(f"duplicate method_id values: {duplicates}")
    if not records:
        failures.append("method ledger must contain at least one record")
    for record in records:
        if record.translation_status != "promoted" and not record.translation_reason:
            failures.append(f"method {record.method_id!r} has no translation reason")
    return {
        "schema": METHOD_LEDGER_SCHEMA,
        "method_count": len(records),
        "method_ids": sorted(method_ids),
        "by_source_kind": dict(sorted(Counter(record.source_kind for record in records).items())),
        "by_translation_status": dict(
            sorted(Counter(record.translation_status for record in records).items())
        ),
        "by_input_modality": dict(
            sorted(
                Counter(modality for record in records for modality in record.input_modalities).items()
            )
        ),
        "failures": failures,
        "valid": not failures,
    }


def initial_method_records(accessed_at: str = "2026-08-10") -> tuple[ExternalMethodRecord, ...]:
    """Return the six methods frozen by Spec 141 Phase 0 research."""

    return (
        ExternalMethodRecord(
            method_id="dsm2dtm",
            name="DSM2DTM",
            source_urls=(
                "https://elib.dlr.de/198246/1/ISPRS2023_Bittner_final.pdf",
                "https://github.com/KseniaBittner/DSM2DTM",
            ),
            source_kind="github",
            accessed_at=accessed_at,
            input_modalities=("dsm",),
            output_signals=("dtm", "ground_mask"),
            domain="Swiss aerial DSM patches; bare-earth terrain beneath above-ground objects",
            license_status="needs_review",
            weights_status="available",
            translation_status="reference",
            translation_reason="Closest neural DSM-to-DTM formulation, but not an RGB-minimap input contract.",
        ),
        ExternalMethodRecord(
            method_id="resdepth",
            name="ResDepth",
            source_urls=("https://github.com/prs-eth/ResDepth",),
            source_kind="github",
            accessed_at=accessed_at,
            input_modalities=("dsm", "rgb", "mask"),
            output_signals=("height_residual", "refined_dsm"),
            domain="Registered DSM plus orthophoto or panchromatic imagery",
            license_status="needs_review",
            weights_status="available",
            translation_status="reference",
            translation_reason="Useful residual-correction architecture only when an observable height prior exists.",
        ),
        ExternalMethodRecord(
            method_id="pdal_smrf",
            name="PDAL SMRF",
            source_urls=("https://pdal.org/en/stable/stages/filters.smrf.html",),
            source_kind="documentation",
            accessed_at=accessed_at,
            input_modalities=("point_cloud",),
            output_signals=("ground_mask", "non_ground_mask"),
            domain="Airborne LiDAR point returns",
            license_status="verified",
            weights_status="not_required",
            translation_status="diagnostic",
            translation_reason="Classical point-cloud ground baseline; no minimap RGB input.",
        ),
        ExternalMethodRecord(
            method_id="cloth_simulation_filter",
            name="Cloth Simulation Filter",
            source_urls=("https://github.com/jianboqi/CSF",),
            source_kind="github",
            accessed_at=accessed_at,
            input_modalities=("point_cloud",),
            output_signals=("ground_mask", "non_ground_mask"),
            domain="Airborne LiDAR XYZ points",
            license_status="needs_review",
            weights_status="not_required",
            translation_status="diagnostic",
            translation_reason="Classical airborne-LiDAR baseline; no minimap RGB input.",
        ),
        ExternalMethodRecord(
            method_id="aerial_object_mask_models",
            name="Aerial tree/building object-mask models",
            source_urls=(
                "https://huggingface.co/restor/tcd-segformer-mit-b3",
                "https://huggingface.co/dnovak232/kastela-dof5-building-segmentation",
            ),
            source_kind="huggingface",
            accessed_at=accessed_at,
            input_modalities=("rgb",),
            output_signals=("predicted_object_mask",),
            domain="High-resolution aerial imagery; tree or building classes",
            license_status="needs_review",
            weights_status="available",
            translation_status="reference",
            translation_reason="Potential predicted-mask auxiliary, but domain and resolution transfer are unproven.",
        ),
        ExternalMethodRecord(
            method_id="prithvi_eo_2",
            name="Prithvi EO 2.0",
            source_urls=(
                "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                "https://github.com/torchgeo/terratorch",
            ),
            source_kind="huggingface",
            accessed_at=accessed_at,
            input_modalities=("multispectral", "metadata"),
            output_signals=("geospatial_features",),
            domain="Earth-observation imagery and geospatial foundation tasks",
            license_status="needs_review",
            weights_status="available",
            translation_status="reference",
            translation_reason="General geospatial encoder, not a direct bare-earth predictor for WoW minimap RGB.",
        ),
    )


def build_rgb_only_contract() -> InputContract:
    return InputContract(
        contract_id="rgb-only-minimap-v1",
        branch="rgb_only",
        observable_inputs=("minimap_rgb",),
        predicted_inputs=("predicted_object_mask",),
        supervision_only_inputs=(
            "height_257",
            "terrain_shadow_256",
            "shadow_mask",
            "object_mask",
        ),
        forbidden_inputs=tuple(sorted(FORBIDDEN_INPUTS)),
        runtime_claim="deployment_candidate",
    )


def build_height_prior_contract() -> InputContract:
    return InputContract(
        contract_id="offline-height-prior-v1",
        branch="height_prior",
        observable_inputs=("dsm",),
        predicted_inputs=(),
        supervision_only_inputs=("height_257", "terrain_shadow_256", "object_mask"),
        forbidden_inputs=tuple(sorted(FORBIDDEN_INPUTS)),
        runtime_claim="offline_diagnostic",
    )


def build_point_cloud_contract() -> InputContract:
    return InputContract(
        contract_id="offline-point-cloud-v1",
        branch="point_cloud",
        observable_inputs=("point_cloud",),
        predicted_inputs=(),
        supervision_only_inputs=("height_257", "object_mask"),
        forbidden_inputs=tuple(sorted(FORBIDDEN_INPUTS)),
        runtime_claim="offline_diagnostic",
    )


def build_combined_contract() -> InputContract:
    return InputContract(
        contract_id="offline-rgb-dsm-combined-v1",
        branch="combined",
        observable_inputs=("minimap_rgb", "dsm"),
        predicted_inputs=(),
        supervision_only_inputs=("height_257", "object_mask"),
        forbidden_inputs=tuple(sorted(FORBIDDEN_INPUTS)),
        runtime_claim="offline_diagnostic",
    )


def initial_input_contracts() -> tuple[InputContract, ...]:
    return (
        build_rgb_only_contract(),
        build_height_prior_contract(),
        build_point_cloud_contract(),
        build_combined_contract(),
    )


def validate_input_contract(contract: InputContract) -> dict[str, Any]:
    """Return the normalized contract and a validation report."""

    return {
        "schema": INPUT_CONTRACT_SCHEMA,
        "contract_id": contract.contract_id,
        "branch": contract.branch,
        "runtime_claim": contract.runtime_claim,
        "observable_inputs": list(contract.observable_inputs),
        "predicted_inputs": list(contract.predicted_inputs),
        "supervision_only_inputs": list(contract.supervision_only_inputs),
        "forbidden_inputs": list(contract.forbidden_inputs),
        "failures": [],
        "valid": True,
    }


def audit_input_reads(contract: InputContract, input_reads: Iterable[Any]) -> dict[str, Any]:
    """Audit model input reads against a declared contract.

    ``input_reads`` describes arrays actually supplied to the model. Evaluation-only target reads
    belong in a separate provenance record and must never be mixed into this list.
    """

    raw_reads = tuple(sorted({str(value).strip() for value in input_reads if str(value).strip()}))
    reads = tuple(sorted({canonical_signal_name(value) for value in raw_reads}))
    allowed = set(contract.observable_inputs) | set(contract.predicted_inputs)
    forbidden = sorted(set(reads) & (set(contract.forbidden_inputs) | FORBIDDEN_INPUTS))
    undeclared = sorted(set(reads) - allowed - set(contract.forbidden_inputs) - FORBIDDEN_INPUTS)
    failures: list[str] = []
    if forbidden:
        failures.append(f"forbidden model input reads: {forbidden}")
    if undeclared:
        failures.append(f"undeclared model input reads: {undeclared}")
    return {
        "schema": INPUT_AUDIT_SCHEMA,
        "contract_id": contract.contract_id,
        "branch": contract.branch,
        "runtime_claim": contract.runtime_claim,
        "raw_input_reads": list(raw_reads),
        "canonical_input_reads": list(reads),
        "allowed_inputs": sorted(allowed),
        "forbidden_reads": forbidden,
        "undeclared_reads": undeclared,
        "failures": failures,
        "valid": not failures,
        "decision": "candidate" if not failures else "rejected",
    }


def build_method_translation_report() -> dict[str, Any]:
    """Build the deterministic Phase 0/1 report used by the audit CLI and tests."""

    methods = initial_method_records()
    ledger = validate_method_records(methods)
    contracts = initial_input_contracts()
    contract_reports = [validate_input_contract(contract) for contract in contracts]
    sample_audits = [
        audit_input_reads(contracts[0], ("minimap_rgb", "predicted_object_mask")),
        audit_input_reads(contracts[0], ("minimap_rgb", "height_257")),
        audit_input_reads(contracts[1], ("dsm",)),
        audit_input_reads(contracts[2], ("point_cloud",)),
        audit_input_reads(contracts[3], ("minimap_rgb", "dsm")),
    ]
    return {
        "schema": METHOD_TRANSLATION_SCHEMA,
        "ledger": {
            **ledger,
            "methods": [method.to_dict() for method in methods],
        },
        "contracts": [contract.to_dict() for contract in contracts],
        "contract_reports": contract_reports,
        "sample_audits": sample_audits,
        "valid": ledger["valid"]
        and all(report["valid"] for report in contract_reports)
        and sample_audits[0]["valid"]
        and not sample_audits[1]["valid"]
        and all(audit["valid"] for audit in sample_audits[2:]),
    }


__all__ = [
    "BRANCHES",
    "FORBIDDEN_INPUTS",
    "INPUT_AUDIT_SCHEMA",
    "INPUT_CONTRACT_SCHEMA",
    "INPUT_MODALITIES",
    "METHOD_LEDGER_SCHEMA",
    "METHOD_TRANSLATION_SCHEMA",
    "ExternalMethodRecord",
    "InputContract",
    "TerrainMethodTranslationError",
    "TranslationDecision",
    "audit_input_reads",
    "build_combined_contract",
    "build_height_prior_contract",
    "build_method_translation_report",
    "build_point_cloud_contract",
    "build_rgb_only_contract",
    "canonical_signal_name",
    "initial_input_contracts",
    "initial_method_records",
    "validate_input_contract",
    "validate_method_records",
]
