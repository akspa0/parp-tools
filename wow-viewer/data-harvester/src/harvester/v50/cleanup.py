"""Size inventory and deterministic cleanup-plan generation (Spec 109 T024, FR-020/FR-021).

Matches ``specs/109-v50-clean-room-audit/contracts/v50-cleanup-plan.schema.json`` exactly: a
``CleanupTarget`` only ever appears in a plan's ``targets`` if it already has ``dependency_check =
"pass"`` and ``approved = true`` -- both are schema *consts*, meaning a candidate that fails either
check is not degraded into the plan with a "rejected" marker, it is simply absent from ``targets``
entirely. Read-only: this module only ever measures and proposes; nothing here deletes anything
(FR-010, research.md Decision 5's read-only-inventory / reviewed-manifest / user-run-apply split).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from harvester.v50.contracts import ArtifactRecord, Disposition
from harvester.v50.identity import hash_manifest
from harvester.v50.path_policy import PathPolicy, PathPolicyError


@dataclass(frozen=True)
class CleanupTarget:
    artifact_id: str
    resolved_path: str
    kind: str
    observed_identity: str
    observed_bytes: int
    replacement_proof: str
    dependency_check: str = "pass"
    approved: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "resolved_path": self.resolved_path,
            "kind": self.kind,
            "observed_identity": self.observed_identity,
            "observed_bytes": self.observed_bytes,
            "replacement_proof": self.replacement_proof,
            "dependency_check": self.dependency_check,
            "approved": self.approved,
        }


@dataclass(frozen=True)
class CleanupPlan:
    plan_id: str
    inventory_identity: str
    release_manifest_identity: str
    approved_roots: tuple[str, ...]
    protected_artifact_ids: tuple[str, ...]
    targets: tuple[CleanupTarget, ...]
    expected_recovered_bytes: int
    dry_run_complete: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "v50-cleanup-plan-v1",
            "plan_id": self.plan_id,
            "inventory_identity": self.inventory_identity,
            "release_manifest_identity": self.release_manifest_identity,
            "approved_roots": list(self.approved_roots),
            "protected_artifact_ids": list(self.protected_artifact_ids),
            "targets": [target.to_dict() for target in self.targets],
            "expected_recovered_bytes": self.expected_recovered_bytes,
            "dry_run_complete": self.dry_run_complete,
        }


def build_cleanup_plan(
    records: list[ArtifactRecord],
    *,
    path_policy: PathPolicy,
    inventory_identity: str,
    release_manifest_identity: str,
    replacement_proof_for: Callable[[ArtifactRecord], str | None],
) -> CleanupPlan:
    """Build a dry-run cleanup plan. ``records`` must already have ``dependencies`` populated (see
    ``dependencies.discover_dependencies``) and only artifacts already dispositioned
    ``REMOVE_CANDIDATE`` are considered at all -- this function does not decide disposition, it
    only turns already-dispositioned candidates into a bound, hash-identified plan.

    ``replacement_proof_for`` returns a non-empty proof string when the caller has verified a
    replacement exists for this artifact (e.g. "superseded by v50.1 store <id>"); returning
    ``None`` excludes the candidate from the plan rather than including it unproven.
    """
    targets: list[CleanupTarget] = []

    for record in records:
        if record.disposition is not Disposition.REMOVE_CANDIDATE:
            continue
        if record.dependencies:
            # Something still references this artifact; it is not a safe candidate no matter how
            # obsolete its label suggests it is.
            continue
        try:
            path_policy.resolve_within_approved_root(Path(record.resolved_path))
        except PathPolicyError:
            continue

        proof = replacement_proof_for(record)
        if not proof:
            continue

        targets.append(
            CleanupTarget(
                artifact_id=record.artifact_id,
                resolved_path=record.resolved_path,
                kind=record.kind,
                observed_identity=record.content_identity,
                observed_bytes=record.observed_bytes,
                replacement_proof=proof,
            )
        )

    expected_recovered_bytes = sum(target.observed_bytes for target in targets)
    # Artifacts the plan explicitly recognizes as protected (KEEP), not merely "not yet approved
    # for removal" -- QUARANTINE/VERIFY/MIGRATE records are neither targets nor protected, they are
    # simply not part of this plan at all.
    protected_ids = tuple(
        sorted(record.artifact_id for record in records if record.disposition is Disposition.KEEP)
    )
    plan_id = hash_manifest(
        {
            "inventory_identity": inventory_identity,
            "release_manifest_identity": release_manifest_identity,
            "target_ids": sorted(target.artifact_id for target in targets),
        }
    )

    return CleanupPlan(
        plan_id=plan_id,
        inventory_identity=inventory_identity,
        release_manifest_identity=release_manifest_identity,
        approved_roots=tuple(str(root) for root in path_policy.approved_roots),
        protected_artifact_ids=protected_ids,
        targets=tuple(targets),
        expected_recovered_bytes=expected_recovered_bytes,
    )
