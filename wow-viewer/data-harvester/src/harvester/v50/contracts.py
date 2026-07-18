"""Canonical v50 contracts: ArtifactRecord, DatasetStoreManifest, DatasetSignal, RowLineage, and
the verification/trust/disposition enums, plus the release-identity gates migrated from
``harvester.v50_contract`` (Spec 109 T008/T012).

These types are the single Python source of truth for
``specs/109-v50-clean-room-audit/contracts/v50-provenance.schema.json``: ``DatasetStoreManifest``
and ``DatasetSignal`` serialize (``to_dict``) to exactly that schema's shape, and ``validate()``
enforces its constraints without a runtime ``jsonschema`` dependency (the checks here are simple
enough -- hash format, release format, enum membership -- that hand validation is clearer than
adding a schema-engine dependency for it).

Spec 109 FR-002: a v50 name, directory, attribute, or release string must not by itself establish
trust. Nothing in this module marks anything ``verified`` or ``promoted`` on construction --
callers (inventory/verify_store, still to come) are responsible for only ever setting those states
after the checks FR-005 requires.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

MODEL_FAMILY = "v50"
DEFAULT_RELEASE = "v50.1"
STORE_SCHEMA = "v50-complete-store-v1"
MIXED_STORE_SCHEMA = "v50-mixed-curriculum-v1"
WDL_ARCHIVE_SCHEMA = "v50-generated-wdl-v1"
WDL_CHECKPOINT_VARIANT = "v50.1-spatial-wdl-prior"
TERRAIN_CHECKPOINT_VARIANT = "v50.1-terrain-refiner-convnextv2"

WL_LIQUID_SURFACE_QUADS_SIGNAL = "wl_liquid_surface_quads_v1"
WL_LIQUID_ABOVE_TERRAIN_SIGNAL = "wl_liquid_above_terrain_v1"
WL_LIQUID_BASIC_TYPE_SIGNAL = "wl_liquid_basic_type_header_v1"
WL_LIQUID_REQUIRED_PROVENANCE = frozenset(
    {
        WL_LIQUID_SURFACE_QUADS_SIGNAL,
        WL_LIQUID_ABOVE_TERRAIN_SIGNAL,
        WL_LIQUID_BASIC_TYPE_SIGNAL,
    }
)
WL_LIQUID_SOURCE_SIGNALS = frozenset({"wl_liquid_mask", "wl_liquid_height"})
V50_FRESH_ONLY_SIGNALS = frozenset({"liquid_mask", "liquid_height"})

_RELEASE = re.compile(r"^v50\.[1-9][0-9]*$")
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Enums (data-model.md's ArtifactRecord.proof_level/trust_state/disposition,
# DatasetSignal.migration_policy, DatasetStoreManifest.finalization_state)
# ---------------------------------------------------------------------------


class ProofLevel(str, Enum):
    INVENTORY = "inventory"
    CONTRACT = "contract"
    SAMPLED = "sampled"
    FULL = "full"
    QUALITY = "quality"


class TrustState(str, Enum):
    UNVERIFIED = "unverified"
    REJECTED = "rejected"
    VERIFIED = "verified"
    PROMOTED = "promoted"


class Disposition(str, Enum):
    KEEP = "keep"
    QUARANTINE = "quarantine"
    VERIFY = "verify"
    MIGRATE = "migrate"
    REMOVE_CANDIDATE = "remove-candidate"


class MigrationPolicy(str, Enum):
    COPY_IF_VERIFIED = "copy-if-verified"
    FRESH_ONLY = "fresh-only"
    DERIVED_AFTER_BUILD = "derived-after-build"
    UNAVAILABLE = "unavailable"


class FinalizationState(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


def _require_hash(value: str, *, field_name: str) -> str:
    if not HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match 'sha256:<64 hex chars>', got {value!r}")
    return value


# ---------------------------------------------------------------------------
# ArtifactRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRecord:
    """One discovered dataset, model, checkpoint, prior archive, manifest, report, local
    environment, command surface, or documentation owner and its audit state.

    Constructing a record does NOT verify anything; ``trust_state`` defaults to ``UNVERIFIED`` and
    callers must justify any other value (FR-001, FR-002)."""

    artifact_id: str
    kind: str
    resolved_path: str
    observed_bytes: int
    content_identity: str
    owner: str
    proof_level: ProofLevel = ProofLevel.INVENTORY
    trust_state: TrustState = TrustState.UNVERIFIED
    disposition: Disposition = Disposition.QUARANTINE
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    evidence_paths: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_hash(self.artifact_id, field_name="artifact_id")
        _require_hash(self.content_identity, field_name="content_identity")
        if self.observed_bytes < 0:
            raise ValueError("observed_bytes must be non-negative")
        if not self.resolved_path:
            raise ValueError("resolved_path is required")


# ---------------------------------------------------------------------------
# DatasetSignal / DatasetStoreManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSignal:
    name: str
    dtype: str
    row_shape: tuple[int, ...]
    required: bool
    authoritative_source: str
    content_identity: str
    coverage_count: int
    migration_policy: MigrationPolicy

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("signal name is required")
        _require_hash(self.content_identity, field_name="content_identity")
        if self.coverage_count < 0:
            raise ValueError("coverage_count must be non-negative")
        if any(dimension < 1 for dimension in self.row_shape):
            raise ValueError(f"row_shape entries must be >= 1, got {self.row_shape!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "row_shape": list(self.row_shape),
            "required": self.required,
            "authoritative_source": self.authoritative_source,
            "content_identity": self.content_identity,
            "coverage_count": self.coverage_count,
            "migration_policy": MigrationPolicy(self.migration_policy).value,
        }


@dataclass(frozen=True)
class UnavailableSignal:
    name: str
    reason: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("unavailable signal name is required")
        if not self.reason:
            raise ValueError("unavailable signal reason is required (never silent)")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason}


# Spec 112 (T003): closed vocabulary of machine-readable reason PREFIXES for UnavailableSignal.
# Additive and opt-in — existing free-text reasons (Spec 109 era) remain valid; new corrective code
# paths use these so tooling can distinguish "cannot exist for this build" from "this tile lacks it"
# without parsing prose (research.md Decision 4).
REASON_ERA_UNAVAILABLE = "era_unavailable:"
REASON_NO_SOURCE_DATA = "no_source_data:"
REASON_NOT_YET_EXTRACTED = "not_yet_extracted:"

_REASON_PREFIXES = (REASON_ERA_UNAVAILABLE, REASON_NO_SOURCE_DATA, REASON_NOT_YET_EXTRACTED)


def classify_reason(reason: str) -> str | None:
    """Return the vocabulary prefix a reason uses, or None for legacy free-text reasons."""
    for prefix in _REASON_PREFIXES:
        if reason.startswith(prefix):
            return prefix
    return None


@dataclass(frozen=True)
class DatasetStoreManifest:
    """Matches ``specs/109-v50-clean-room-audit/contracts/v50-provenance.schema.json`` exactly;
    ``to_dict()`` output is that schema's shape and ``validate()`` enforces its constraints."""

    release: str
    store_id: str
    build_id: str
    producer_identity: str
    client_build_evidence_id: str
    index_identity: str
    row_count: int
    signals: tuple[DatasetSignal, ...]
    row_lineage_identity: str
    finalization_state: FinalizationState
    unavailable_signals: tuple[UnavailableSignal, ...] = field(default_factory=tuple)
    model_family: str = MODEL_FAMILY
    schema: str = STORE_SCHEMA

    def validate(self) -> None:
        if self.model_family != MODEL_FAMILY:
            raise ValueError(f"model_family must be {MODEL_FAMILY!r}, got {self.model_family!r}")
        if self.schema != STORE_SCHEMA:
            raise ValueError(f"schema must be {STORE_SCHEMA!r}, got {self.schema!r}")
        validate_release(self.release)
        for label, value in (
            ("store_id", self.store_id),
            ("producer_identity", self.producer_identity),
            ("client_build_evidence_id", self.client_build_evidence_id),
            ("index_identity", self.index_identity),
            ("row_lineage_identity", self.row_lineage_identity),
        ):
            _require_hash(value, field_name=label)
        if not self.build_id:
            raise ValueError("build_id is required")
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if not self.signals:
            raise ValueError("a v50 store manifest must declare at least one signal")
        names = [signal.name for signal in self.signals]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate signal names in manifest: {names!r}")
        if self.finalization_state not in FinalizationState:
            raise ValueError(f"invalid finalization_state {self.finalization_state!r}")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "model_family": self.model_family,
            "release": self.release,
            "schema": self.schema,
            "store_id": self.store_id,
            "build_id": self.build_id,
            "producer_identity": self.producer_identity,
            "client_build_evidence_id": self.client_build_evidence_id,
            "index_identity": self.index_identity,
            "row_count": self.row_count,
            "signals": [signal.to_dict() for signal in self.signals],
            "row_lineage_identity": self.row_lineage_identity,
            "finalization_state": FinalizationState(self.finalization_state).value,
        }
        if self.unavailable_signals:
            payload["unavailable_signals"] = [signal.to_dict() for signal in self.unavailable_signals]
        return payload


# ---------------------------------------------------------------------------
# RowLineage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowLineage:
    """Per-row record binding a v50 store row to its source evidence and per-signal action.

    ``signal_actions`` values must be one of ``copied``, ``freshly_extracted``, ``derived``, or
    ``unavailable`` -- mirroring ``DatasetSignal.migration_policy`` but recorded per row, since a
    signal's policy is a store-wide default while any individual row can still fall back (a
    ``copy-if-verified`` signal that failed verification for one specific row, for example)."""

    _VALID_ACTIONS = frozenset({"copied", "freshly_extracted", "derived", "unavailable"})

    store_row: int
    build_id: str
    map_name: str
    tile_x: int
    tile_y: int
    source_group: str
    signal_actions: Mapping[str, str]
    signal_source_hashes: Mapping[str, str] = field(default_factory=dict)
    signal_destination_hashes: Mapping[str, str] = field(default_factory=dict)
    source_artifact_id: str | None = None
    source_row: int | None = None
    split_group: str | None = None

    def __post_init__(self) -> None:
        if self.store_row < 0:
            raise ValueError("store_row must be non-negative")
        if not self.build_id:
            raise ValueError("build_id is required")
        if self.source_artifact_id is not None:
            _require_hash(self.source_artifact_id, field_name="source_artifact_id")
        invalid = {action for action in self.signal_actions.values() if action not in self._VALID_ACTIONS}
        if invalid:
            raise ValueError(f"invalid signal actions {sorted(invalid)!r}; must be one of {sorted(self._VALID_ACTIONS)}")
        for label, hashes in (
            ("signal_source_hashes", self.signal_source_hashes),
            ("signal_destination_hashes", self.signal_destination_hashes),
        ):
            for signal_name, value in hashes.items():
                _require_hash(value, field_name=f"{label}[{signal_name}]")


# ---------------------------------------------------------------------------
# Release-identity gates (T012: migrated verbatim from harvester.v50_contract;
# v50_contract.py now re-exports these so existing callers/tests are unaffected)
# ---------------------------------------------------------------------------


def validate_release(value: str) -> str:
    release = str(value).strip().lower()
    if not _RELEASE.fullmatch(release):
        raise ValueError("--release must be v50.N (for example v50.1)")
    return release


def require_store_release(group, expected_release: str, *, store: Path) -> None:
    """Fail before a run can consume an unversioned or mismatched store."""
    release = validate_release(expected_release)
    actual_family = str(group.attrs.get("model_family", ""))
    actual_release = str(group.attrs.get("release", ""))
    actual_schema = str(group.attrs.get("schema", ""))
    if (actual_family, actual_release, actual_schema) != (MODEL_FAMILY, release, MIXED_STORE_SCHEMA):
        raise ValueError(
            "store is not the requested v50 release: "
            f"expected family={MODEL_FAMILY!r}, release={release!r}, schema={MIXED_STORE_SCHEMA!r}; "
            f"got family={actual_family!r}, release={actual_release!r}, schema={actual_schema!r} at {store}"
        )


def require_metadata_release(metadata: Mapping[str, object], expected_release: str, *, artifact: str) -> None:
    """Reject a checkpoint/archive whose declared release cannot feed this run."""
    release = validate_release(expected_release)
    family = str(metadata.get("model_family", ""))
    actual = str(metadata.get("release", ""))
    if family != MODEL_FAMILY or actual != release:
        raise ValueError(
            f"{artifact} is not compatible with {release}: "
            f"family={family!r}, release={actual!r}"
        )


def release_identity(release: str) -> dict[str, str]:
    release = validate_release(release)
    return {"model_family": MODEL_FAMILY, "release": release}


def migration_policy_for_signal(signal_name: str) -> str:
    """Return the frozen V50 migration policy for a canonical signal.

    Historic liquid payloads cannot prove whether their WL fallback was the former
    sparse-stamp projection. They are therefore never eligible for bit-preserving
    V18 migration: V50 obtains them only from a corrected client-backed extraction.
    """
    return "fresh-only" if str(signal_name) in V50_FRESH_ONLY_SIGNALS else "copy-if-verified"


def require_liquid_source_provenance(available_signals: Iterable[object] | object, *, artifact: str) -> None:
    """Reject WL liquid arrays that lack complete visibility/type provenance.

    Non-WL liquid readers (for example MH2O/MCLQ) do not carry the WL marker. The
    caller must still record their authoritative reader in the V50 row lineage.
    """
    if isinstance(available_signals, str):
        signals = {available_signals}
    else:
        try:
            signals = {str(signal) for signal in available_signals}
        except TypeError:
            signals = set()
    if signals.intersection(WL_LIQUID_SOURCE_SIGNALS) and not WL_LIQUID_REQUIRED_PROVENANCE.issubset(signals):
        missing = ", ".join(sorted(WL_LIQUID_REQUIRED_PROVENANCE.difference(signals)))
        raise ValueError(
            f"{artifact} contains WL liquid data without required provenance ({missing}); "
            "reject it and re-extract from an approved client build"
        )
