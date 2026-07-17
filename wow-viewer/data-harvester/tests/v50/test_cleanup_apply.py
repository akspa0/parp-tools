"""Spec 109 T045: cleanup-apply identity, interrupted-run, and post-check tests (FR-020/FR-021,
Phase 7). ``apply_cleanup_plan`` is the only thing in this codebase allowed to delete a v50
approved-root artifact, and only when every one of these holds: the caller passes the exact
plan hash back (so a stale or hand-edited plan cannot be applied), the plan itself claims
``dry_run_complete``, and each target's on-disk content still hashes to what the plan recorded
(so a file that changed between planning and apply is skipped, not silently deleted).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harvester.v50.cleanup import (
    CleanupApplyError,
    apply_cleanup_plan,
    build_cleanup_plan,
)
from harvester.v50.contracts import ArtifactRecord, Disposition
from harvester.v50.identity import hash_file
from harvester.v50.path_policy import PathPolicy

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


def _record(path: Path, *, kind: str, disposition: Disposition, content_hash: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=content_hash,
        kind=kind,
        resolved_path=str(path.resolve()),
        observed_bytes=path.stat().st_size if path.is_file() else 0,
        content_identity=content_hash,
        owner="test",
        disposition=disposition,
    )


def _policy(approved_root: Path, protected_root: Path | None = None) -> PathPolicy:
    return PathPolicy(
        approved_roots=[approved_root],
        protected_roots=[protected_root] if protected_root else [],
    )


def _one_target_plan(tmp_path: Path, *, protected_root: Path | None = None):
    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    candidate_file = approved_root / "obsolete.pt"
    candidate_file.write_bytes(b"safe-to-remove-bytes")
    # apply_cleanup_plan rehashes the real file before deleting it, so the fixture's declared
    # content_identity must be the file's real hash, not an arbitrary placeholder.
    record = _record(candidate_file, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE, content_hash=hash_file(candidate_file))
    policy = _policy(approved_root, protected_root)
    plan = build_cleanup_plan(
        [record], path_policy=policy, inventory_identity=_HASH_A, release_manifest_identity=_HASH_B,
        replacement_proof_for=lambda r: "superseded by v50.1 store",
    )
    return plan, policy, candidate_file


class TestApplyIdentityGate:
    def test_apply_refuses_a_plan_hash_that_does_not_match(self, tmp_path: Path):
        plan, policy, candidate_file = _one_target_plan(tmp_path)

        with pytest.raises(CleanupApplyError, match="plan_id"):
            apply_cleanup_plan(plan, path_policy=policy, expected_plan_id="sha256:" + "0" * 64, confirm=True)

        assert candidate_file.exists(), "nothing may be deleted when the plan-hash check fails"

    def test_apply_refuses_without_explicit_confirm(self, tmp_path: Path):
        plan, policy, candidate_file = _one_target_plan(tmp_path)

        with pytest.raises(CleanupApplyError, match="confirm"):
            apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=plan.plan_id, confirm=False)

        assert candidate_file.exists()

    def test_apply_removes_the_target_when_plan_hash_and_confirm_match(self, tmp_path: Path):
        plan, policy, candidate_file = _one_target_plan(tmp_path)

        result = apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=plan.plan_id, confirm=True)

        assert not candidate_file.exists()
        assert result.removed == (plan.targets[0].artifact_id,)
        assert result.recovered_bytes == plan.expected_recovered_bytes
        assert result.skipped == ()

    def test_apply_skips_a_target_whose_content_changed_since_the_plan_was_built(self, tmp_path: Path):
        plan, policy, candidate_file = _one_target_plan(tmp_path)
        candidate_file.write_bytes(b"someone modified this file after the plan was built")

        result = apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=plan.plan_id, confirm=True)

        assert candidate_file.exists(), "a target whose content drifted from the plan must never be deleted"
        assert result.removed == ()
        assert result.skipped[0]["artifact_id"] == plan.targets[0].artifact_id
        assert "changed" in result.skipped[0]["reason"]
        assert result.recovered_bytes == 0

    def test_apply_never_touches_a_path_inside_a_protected_root_even_if_the_plan_says_so(self, tmp_path: Path):
        protected_root = tmp_path / "protected"
        protected_root.mkdir()
        plan, policy, candidate_file = _one_target_plan(tmp_path, protected_root=None)

        # Simulate a tampered/stale plan whose target now resolves into a newly-declared
        # protected root -- apply must re-check policy at execution time, not trust the plan.
        retampered_policy = PathPolicy(approved_roots=policy.approved_roots, protected_roots=[candidate_file.parent])

        result = apply_cleanup_plan(plan, path_policy=retampered_policy, expected_plan_id=plan.plan_id, confirm=True)

        assert candidate_file.exists()
        assert result.removed == ()
        assert "protected" in result.skipped[0]["reason"]


class TestApplyResumability:
    def test_reapplying_the_same_plan_after_partial_completion_is_idempotent(self, tmp_path: Path):
        approved_root = tmp_path / "approved"
        approved_root.mkdir()
        first_file = approved_root / "first.pt"
        second_file = approved_root / "second.pt"
        first_file.write_bytes(b"first-bytes")
        second_file.write_bytes(b"second-bytes-longer")
        first_record = _record(first_file, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE, content_hash=hash_file(first_file))
        second_record = _record(second_file, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE, content_hash=hash_file(second_file))
        policy = _policy(approved_root)
        plan = build_cleanup_plan(
            [first_record, second_record], path_policy=policy, inventory_identity=_HASH_A,
            release_manifest_identity=_HASH_B, replacement_proof_for=lambda r: "proof",
        )

        # Simulate an interrupted first run: only one of the two targets was actually removed
        # before the process died (e.g. killed between the two deletions).
        first_file.unlink()

        result = apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=plan.plan_id, confirm=True)

        assert not second_file.exists()
        assert set(result.removed) == {first_record.artifact_id, second_record.artifact_id}
        assert result.skipped == ()
        assert result.recovered_bytes == len(b"second-bytes-longer")

    def test_a_target_missing_before_apply_ever_runs_is_reported_not_raised(self, tmp_path: Path):
        plan, policy, candidate_file = _one_target_plan(tmp_path)
        candidate_file.unlink()

        result = apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=plan.plan_id, confirm=True)

        assert result.removed == (plan.targets[0].artifact_id,)
        assert result.recovered_bytes == 0, "bytes already reclaimed by a prior run are not recounted"


class TestApplyPostCheck:
    def test_result_records_zero_bytes_recovered_for_a_run_that_removes_nothing(self, tmp_path: Path):
        plan, policy, candidate_file = _one_target_plan(tmp_path)
        candidate_file.write_bytes(b"drifted content, not the plan's content")

        result = apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=plan.plan_id, confirm=True)

        assert result.plan_id == plan.plan_id
        assert result.removed == ()
        assert result.recovered_bytes == 0
        assert candidate_file.exists()

    def test_result_to_dict_round_trips_every_field(self, tmp_path: Path):
        plan, policy, _candidate_file = _one_target_plan(tmp_path)

        result = apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=plan.plan_id, confirm=True)
        payload = result.to_dict()

        assert payload["schema"] == "v50-cleanup-apply-result-v1"
        assert payload["plan_id"] == plan.plan_id
        assert payload["removed"] == list(result.removed)
        assert payload["skipped"] == list(result.skipped)
        assert payload["recovered_bytes"] == result.recovered_bytes
