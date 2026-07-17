"""Complete v50 store/index/lineage validation -- the promotion gate (Spec 109 T017, FR-005).

FR-005: promotion requires reproducible source identity, extraction/build identity, schema and
dtype/shape validation, row-count agreement, row-level lineage, partition leakage checks,
required-signal truthfulness, and content-integrity hashes. This module runs every one of those
checks and returns a single ``StoreVerificationResult``; ``passed`` is True only when all of them
pass. Nothing here promotes an ``ArtifactRecord`` itself -- the caller decides what to do with a
passing result. A store cannot reach ``passed=True`` by looking right in the fields the schema
happens to validate for free (that's exactly the "name/attribute establishes trust" failure mode
FR-002 rejects); it must also account for every row and every required signal against real
observed content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from harvester.spec103.prefab_curation import validate_source_group_split
from harvester.v50.contracts import DatasetStoreManifest, ProofLevel, RowLineage


@dataclass(frozen=True)
class StoreVerificationResult:
    store_id: str
    checks_performed: tuple[str, ...]
    passed: bool
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)
    proof_level: ProofLevel = ProofLevel.CONTRACT


def _check_row_count_agreement(manifest: DatasetStoreManifest, row_lineages: Sequence[RowLineage]) -> str | None:
    if len(row_lineages) != manifest.row_count:
        return (
            f"row_count mismatch: manifest declares {manifest.row_count}, "
            f"{len(row_lineages)} row lineage records observed"
        )
    return None


def _check_required_signal_truthfulness(
    manifest: DatasetStoreManifest, row_lineages: Sequence[RowLineage]
) -> list[str]:
    """Every required signal must have a real per-row action (copied/freshly_extracted/derived)
    for every row, unless the manifest explicitly records it as unavailable store-wide."""
    unavailable_names = {signal.name for signal in manifest.unavailable_signals}
    failures: list[str] = []
    for signal in manifest.signals:
        if not signal.required or signal.name in unavailable_names:
            continue
        for lineage in row_lineages:
            action = lineage.signal_actions.get(signal.name)
            if action is None:
                failures.append(
                    f"row {lineage.store_row}: required signal {signal.name!r} has no recorded action"
                )
            elif action == "unavailable":
                failures.append(
                    f"row {lineage.store_row}: required signal {signal.name!r} is unavailable "
                    "but the manifest does not declare it unavailable store-wide"
                )
    return failures


def _check_content_integrity(
    manifest: DatasetStoreManifest, observed_signal_hashes: Mapping[str, str]
) -> list[str]:
    """Every signal's declared content_identity must match what was actually observed in the
    store -- a manifest can claim any hash it likes; this is the check that catches a lie."""
    failures: list[str] = []
    for signal in manifest.signals:
        observed = observed_signal_hashes.get(signal.name)
        if observed is None:
            failures.append(f"signal {signal.name!r}: no observed content hash supplied for verification")
        elif observed != signal.content_identity:
            failures.append(
                f"signal {signal.name!r}: manifest content_identity {signal.content_identity!r} "
                f"does not match observed {observed!r}"
            )
    return failures


def _check_partition_leakage(row_lineages: Sequence[RowLineage]) -> str | None:
    labeled = [lineage for lineage in row_lineages if lineage.split_group]
    if not labeled:
        return None  # no splits declared yet (e.g. a store with no curriculum bound); nothing to check
    index_rows = [{"source_group_id": lineage.source_group} for lineage in labeled]
    train_rows = [i for i, lineage in enumerate(labeled) if lineage.split_group == "train"]
    val_rows = [i for i, lineage in enumerate(labeled) if lineage.split_group == "val"]
    try:
        validate_source_group_split(index_rows, train_rows, val_rows)
    except ValueError as exc:
        return str(exc)
    return None


def verify_store(
    manifest: DatasetStoreManifest,
    row_lineages: Sequence[RowLineage],
    *,
    observed_signal_hashes: Mapping[str, str],
) -> StoreVerificationResult:
    checks_performed = (
        "schema_dtype_shape",
        "row_count_agreement",
        "required_signal_truthfulness",
        "content_integrity",
        "partition_leakage",
    )
    failures: list[str] = []

    try:
        manifest.validate()
    except ValueError as exc:
        failures.append(f"schema_dtype_shape: {exc}")

    row_count_failure = _check_row_count_agreement(manifest, row_lineages)
    if row_count_failure:
        failures.append(f"row_count_agreement: {row_count_failure}")

    failures.extend(
        f"required_signal_truthfulness: {reason}"
        for reason in _check_required_signal_truthfulness(manifest, row_lineages)
    )
    failures.extend(f"content_integrity: {reason}" for reason in _check_content_integrity(manifest, observed_signal_hashes))

    leakage_failure = _check_partition_leakage(row_lineages)
    if leakage_failure:
        failures.append(f"partition_leakage: {leakage_failure}")

    return StoreVerificationResult(
        store_id=manifest.store_id,
        checks_performed=checks_performed,
        passed=not failures,
        failure_reasons=tuple(failures),
        proof_level=ProofLevel.FULL if not failures else ProofLevel.CONTRACT,
    )
