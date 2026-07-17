"""Spec 109 T022: dependency and cleanup-plan tests (FR-020/FR-021). Independent test from
tasks.md: "A fixture inventory containing protected, depended-on, safe obsolete, linked, and
out-of-root targets includes only the safe obsolete target in the approved dry-run plan."""

from __future__ import annotations

import json
from pathlib import Path

from harvester.v50.cleanup import build_cleanup_plan
from harvester.v50.contracts import ArtifactRecord, Disposition
from harvester.v50.dependencies import discover_dependencies
from harvester.v50.path_policy import PathPolicy

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


def _record(path: Path, *, kind: str, disposition: Disposition, content_hash: str = _HASH_A) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=content_hash,
        kind=kind,
        resolved_path=str(path.resolve()),
        observed_bytes=path.stat().st_size if path.is_file() else 0,
        content_identity=content_hash,
        owner="test",
        disposition=disposition,
    )


class TestDiscoverDependencies:
    def test_a_manifest_referencing_another_artifacts_path_creates_a_dependency(self, tmp_path: Path):
        dataset_dir = tmp_path / "dataset.zarr"
        dataset_dir.mkdir()
        manifest_path = tmp_path / "release-manifest.json"
        manifest_path.write_text(json.dumps({"store_path": str(dataset_dir.resolve())}), encoding="utf-8")

        dataset_record = _record(dataset_dir, kind="dataset", disposition=Disposition.REMOVE_CANDIDATE, content_hash=_HASH_A)
        manifest_record = _record(manifest_path, kind="manifest", disposition=Disposition.KEEP, content_hash=_HASH_B)

        updated = discover_dependencies([dataset_record, manifest_record])
        by_id = {r.artifact_id: r for r in updated}

        assert by_id[_HASH_A].dependencies == (_HASH_B,)
        assert by_id[_HASH_B].dependencies == ()

    def test_an_artifact_referenced_by_its_content_hash_is_also_found(self, tmp_path: Path):
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint_path.write_bytes(b"fake-checkpoint-bytes")
        report_path = tmp_path / "eval-report.json"
        report_path.write_text(json.dumps({"checkpoint_identity": _HASH_A}), encoding="utf-8")

        checkpoint_record = _record(checkpoint_path, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE, content_hash=_HASH_A)
        report_record = _record(report_path, kind="report", disposition=Disposition.KEEP, content_hash=_HASH_B)

        updated = discover_dependencies([checkpoint_record, report_record])
        by_id = {r.artifact_id: r for r in updated}

        assert by_id[_HASH_A].dependencies == (_HASH_B,)

    def test_an_unreferenced_artifact_has_no_dependencies(self, tmp_path: Path):
        lone_path = tmp_path / "lone.pt"
        lone_path.write_bytes(b"data")
        record = _record(lone_path, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE)

        updated = discover_dependencies([record])

        assert updated[0].dependencies == ()


class TestBuildCleanupPlan:
    def _policy(self, approved_root: Path, protected_root: Path | None = None) -> PathPolicy:
        return PathPolicy(
            approved_roots=[approved_root],
            protected_roots=[protected_root] if protected_root else [],
        )

    def test_only_the_safe_obsolete_target_survives_a_mixed_fixture(self, tmp_path: Path):
        approved_root = tmp_path / "approved_output"
        protected_root = tmp_path / "protected_specs"
        outside_root = tmp_path / "outside"
        approved_root.mkdir()
        protected_root.mkdir()
        outside_root.mkdir()

        # 1. protected: KEEP disposition, never a target regardless of anything else
        protected_file = approved_root / "protected.pt"
        protected_file.write_bytes(b"keep-me")
        protected_record = _record(protected_file, kind="checkpoint", disposition=Disposition.KEEP, content_hash=_HASH_A)

        # 2. depended-on: REMOVE_CANDIDATE but something still references it
        depended_file = approved_root / "depended.zarr"
        depended_file.mkdir()
        depended_record = ArtifactRecord(
            artifact_id=_HASH_B,
            kind="dataset",
            resolved_path=str(depended_file.resolve()),
            observed_bytes=0,
            content_identity=_HASH_B,
            owner="test",
            disposition=Disposition.REMOVE_CANDIDATE,
            dependencies=("sha256:" + "9" * 64,),  # something still depends on this
        )

        # 3. out-of-root: REMOVE_CANDIDATE, resolves outside every approved root
        outside_file = outside_root / "orphan.pt"
        outside_file.write_bytes(b"orphan")
        outside_record = _record(outside_file, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE, content_hash="sha256:" + "c" * 64)

        # 4. safe obsolete: REMOVE_CANDIDATE, no dependents, inside approved root, has proof
        safe_file = approved_root / "obsolete.pt"
        safe_file.write_bytes(b"safe-to-remove-bytes")
        safe_record = _record(safe_file, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE, content_hash="sha256:" + "d" * 64)

        records = [protected_record, depended_record, outside_record, safe_record]
        policy = self._policy(approved_root, protected_root)

        plan = build_cleanup_plan(
            records,
            path_policy=policy,
            inventory_identity=_HASH_A,
            release_manifest_identity=_HASH_B,
            replacement_proof_for=lambda r: "superseded by v50.1 store" if r.artifact_id == safe_record.artifact_id else None,
        )

        assert len(plan.targets) == 1
        assert plan.targets[0].artifact_id == safe_record.artifact_id
        assert plan.targets[0].observed_bytes == len(b"safe-to-remove-bytes")
        assert plan.expected_recovered_bytes == len(b"safe-to-remove-bytes")
        assert protected_record.artifact_id in plan.protected_artifact_ids
        assert plan.dry_run_complete is True

    def test_a_candidate_without_replacement_proof_is_excluded_not_marked_unapproved(self, tmp_path: Path):
        approved_root = tmp_path / "approved"
        approved_root.mkdir()
        candidate_file = approved_root / "unproven.pt"
        candidate_file.write_bytes(b"x")
        record = _record(candidate_file, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE)

        plan = build_cleanup_plan(
            [record],
            path_policy=self._policy(approved_root),
            inventory_identity=_HASH_A,
            release_manifest_identity=_HASH_B,
            replacement_proof_for=lambda r: None,
        )

        assert plan.targets == ()

    def test_plan_id_is_deterministic_for_the_same_inputs(self, tmp_path: Path):
        approved_root = tmp_path / "approved"
        approved_root.mkdir()
        candidate_file = approved_root / "obsolete.pt"
        candidate_file.write_bytes(b"bytes")
        record = _record(candidate_file, kind="checkpoint", disposition=Disposition.REMOVE_CANDIDATE)
        policy = self._policy(approved_root)

        plan_a = build_cleanup_plan(
            [record], path_policy=policy, inventory_identity=_HASH_A, release_manifest_identity=_HASH_B,
            replacement_proof_for=lambda r: "proof",
        )
        plan_b = build_cleanup_plan(
            [record], path_policy=policy, inventory_identity=_HASH_A, release_manifest_identity=_HASH_B,
            replacement_proof_for=lambda r: "proof",
        )

        assert plan_a.plan_id == plan_b.plan_id

    def test_non_remove_candidate_dispositions_are_never_targets(self, tmp_path: Path):
        approved_root = tmp_path / "approved"
        approved_root.mkdir()
        for disposition in (Disposition.QUARANTINE, Disposition.VERIFY, Disposition.MIGRATE):
            candidate_file = approved_root / f"{disposition.value}.pt"
            candidate_file.write_bytes(b"x")
            record = _record(candidate_file, kind="checkpoint", disposition=disposition)

            plan = build_cleanup_plan(
                [record], path_policy=self._policy(approved_root), inventory_identity=_HASH_A,
                release_manifest_identity=_HASH_B, replacement_proof_for=lambda r: "proof",
            )
            assert plan.targets == (), f"{disposition} must never become a cleanup target"
