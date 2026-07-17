#!/usr/bin/env python3
"""Spec 109 T025/T047: cleanup planning (read-only, dry-run) and reviewed apply (destructive,
user-run-only). ``plan`` never deletes, moves, or modifies any file (FR-010). ``apply`` deletes
exactly the targets in a plan the caller has already reviewed, and only when the caller passes
back the plan's own ``plan_id`` verbatim plus an explicit ``--confirm`` -- there is no default-on
deletion path, and approved/protected roots are re-supplied and re-checked at apply time rather
than trusted from the plan file.

Usage:
    uv run python scripts/v50_cleanup_artifacts.py plan \
        --inventory <inventory.json from v50_audit_artifacts.py inventory> \
        --approved-root <root> [--approved-root <root> ...] \
        --protected-root <root> [--protected-root <root> ...] \
        --dispositions <dispositions.json: {"<artifact_id>": "remove-candidate", ...}> \
        --replacement-proofs <proofs.json: {"<artifact_id>": "proof text", ...}> \
        --output <cleanup-plan.json>

    uv run python scripts/v50_cleanup_artifacts.py apply \
        --plan <cleanup-plan.json written by the plan subcommand above> \
        --plan-id <the exact plan_id field copied from that same file> \
        --approved-root <root> [--approved-root <root> ...] \
        --protected-root <root> [--protected-root <root> ...] \
        --confirm \
        --output <cleanup-apply-result.json>

``--dispositions`` and ``--replacement-proofs`` are separate, explicit, human-reviewed inputs
rather than something this script infers on its own -- disposition and replacement proof are
judgment calls this audit deliberately keeps out of an automatic classifier (FR-004: every
disposition needs a stated reason; that reason is supplied by whoever reviews the inventory, not
guessed from a file's name or extension).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harvester.v50.cleanup import CleanupApplyError, CleanupPlan, CleanupTarget, apply_cleanup_plan, build_cleanup_plan
from harvester.v50.contracts import ArtifactRecord, Disposition
from harvester.v50.dependencies import discover_dependencies
from harvester.v50.path_policy import PathPolicy


def _load_records(inventory_path: Path, dispositions: dict[str, str]) -> list[ArtifactRecord]:
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = []
    for entry in payload["artifacts"]:
        disposition_value = dispositions.get(entry["artifact_id"], entry["disposition"])
        records.append(
            ArtifactRecord(
                artifact_id=entry["artifact_id"],
                kind=entry["kind"],
                resolved_path=entry["resolved_path"],
                observed_bytes=entry["observed_bytes"],
                content_identity=entry["content_identity"],
                owner=entry["owner"],
                disposition=Disposition(disposition_value),
            )
        )
    return records


def _cmd_plan(args: argparse.Namespace) -> int:
    inventory_path = Path(args.inventory)
    inventory_identity_source = inventory_path.read_bytes()
    import hashlib

    inventory_identity = f"sha256:{hashlib.sha256(inventory_identity_source).hexdigest()}"

    dispositions = {}
    if args.dispositions:
        dispositions = json.loads(Path(args.dispositions).read_text(encoding="utf-8"))
    proofs: dict[str, str] = {}
    if args.replacement_proofs:
        proofs = json.loads(Path(args.replacement_proofs).read_text(encoding="utf-8"))

    records = _load_records(inventory_path, dispositions)
    records = discover_dependencies(records)

    policy = PathPolicy(
        approved_roots=[Path(root) for root in args.approved_root],
        protected_roots=[Path(root) for root in (args.protected_root or [])],
    )

    release_manifest_identity = args.release_manifest_identity or ("sha256:" + "0" * 64)

    plan = build_cleanup_plan(
        records,
        path_policy=policy,
        inventory_identity=inventory_identity,
        release_manifest_identity=release_manifest_identity,
        replacement_proof_for=lambda r: proofs.get(r.artifact_id),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    print(
        f"cleanup plan: {len(plan.targets)} approved target(s), "
        f"{plan.expected_recovered_bytes:,} bytes expected recovered -> {output_path}"
    )
    print("This is a dry run. Nothing has been deleted. Review the plan, then run the separate "
          "'apply' subcommand with this exact plan_id and --confirm only after explicit review.")
    return 0


def _load_plan(plan_path: Path) -> CleanupPlan:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    targets = tuple(
        CleanupTarget(
            artifact_id=t["artifact_id"],
            resolved_path=t["resolved_path"],
            kind=t["kind"],
            observed_identity=t["observed_identity"],
            observed_bytes=t["observed_bytes"],
            replacement_proof=t["replacement_proof"],
            dependency_check=t["dependency_check"],
            approved=t["approved"],
        )
        for t in payload["targets"]
    )
    return CleanupPlan(
        plan_id=payload["plan_id"],
        inventory_identity=payload["inventory_identity"],
        release_manifest_identity=payload["release_manifest_identity"],
        approved_roots=tuple(payload["approved_roots"]),
        protected_artifact_ids=tuple(payload["protected_artifact_ids"]),
        targets=targets,
        expected_recovered_bytes=payload["expected_recovered_bytes"],
        dry_run_complete=payload["dry_run_complete"],
    )


def _cmd_apply(args: argparse.Namespace) -> int:
    plan = _load_plan(Path(args.plan))
    policy = PathPolicy(
        approved_roots=[Path(root) for root in args.approved_root],
        protected_roots=[Path(root) for root in (args.protected_root or [])],
    )

    try:
        result = apply_cleanup_plan(plan, path_policy=policy, expected_plan_id=args.plan_id, confirm=args.confirm)
    except CleanupApplyError as exc:
        raise SystemExit(str(exc)) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print(
        f"cleanup apply: {len(result.removed)} removed, {len(result.skipped)} skipped, "
        f"{result.recovered_bytes:,} bytes recovered -> {output_path}"
    )
    if result.skipped:
        print("Skipped targets were left untouched; see the reason field in the output for each.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="build a dry-run cleanup plan")
    plan_parser.add_argument("--inventory", required=True)
    plan_parser.add_argument("--approved-root", action="append", required=True)
    plan_parser.add_argument("--protected-root", action="append", default=None)
    plan_parser.add_argument("--dispositions", default=None,
                              help="JSON {artifact_id: disposition} overriding the inventory's default")
    plan_parser.add_argument("--replacement-proofs", default=None,
                              help="JSON {artifact_id: proof string}; candidates without an entry are excluded")
    plan_parser.add_argument("--release-manifest-identity", default=None)
    plan_parser.add_argument("--output", required=True)
    plan_parser.set_defaults(handler=_cmd_plan)

    apply_parser = subparsers.add_parser("apply", help="delete exactly the targets in a reviewed plan (destructive)")
    apply_parser.add_argument("--plan", required=True, help="cleanup-plan.json written by the plan subcommand")
    apply_parser.add_argument("--plan-id", required=True,
                               help="the exact plan_id field copied from --plan; a mismatch refuses to run")
    apply_parser.add_argument("--approved-root", action="append", required=True)
    apply_parser.add_argument("--protected-root", action="append", default=None)
    apply_parser.add_argument("--confirm", action="store_true",
                               help="required to actually delete anything; omit to see the refusal")
    apply_parser.add_argument("--output", required=True)
    apply_parser.set_defaults(handler=_cmd_apply)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
