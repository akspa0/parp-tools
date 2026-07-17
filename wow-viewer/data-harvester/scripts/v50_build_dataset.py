#!/usr/bin/env python3
"""Spec 109 T034: v50 dataset build commands -- migrate-v18, build, verify, finalize, curriculum.

Replaces the former fail-closed placeholder now that the fixture-proven migration, store-write,
and finalization owners exist (``harvester.v50.migrate``/``store``/``verify_store``/``curriculum``).
Nothing here fabricates trust: ``migrate-v18`` only copies signals that pass the V18 audit AND
whose policy is ``copy-if-verified`` (FR-006/FR-017); everything else is recorded
``unavailable`` with a reason, leaving the store ``finalization_state=incomplete`` until ``build``
fills the fresh-extraction gaps and ``finalize`` reconciles the whole thing.

``build`` is the one subcommand that can launch a real subprocess against a configured client root
-- it never does so without ``--confirm-run`` (see ``harvester.v50.build.run_fresh_extraction``),
matching the same execution-gate discipline already used in Spec 111's
``train_spec111_reconstruction.py``. Preparing/testing these tools does not authorize spending real
harvest time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from harvester.v50.build import build_harvest_stream_command, run_fresh_extraction
from harvester.v50.contracts import (
    DatasetSignal,
    DatasetStoreManifest,
    FinalizationState,
    MigrationPolicy,
    RowLineage,
    UnavailableSignal,
)
from harvester.v50.curriculum import CurriculumRowRef, build_curriculum
from harvester.v50.identity import hash_array
from harvester.v50.migrate import ACTION_COPY, MigrationLedger, copy_signal_row, plan_signal_migration
from harvester.v50.store import finalize_store, write_v50_store
from harvester.v50.verify_store import verify_store
from harvester.v50.verify_v18 import V18RowSignalObservation, V18SignalSpec, audit_v18_signal


def _load_signal_specs(config_path: Path) -> list[V18SignalSpec]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return [
        V18SignalSpec(
            name=entry["name"],
            blacklisted=bool(entry.get("blacklisted", False)),
            blacklist_reason=str(entry.get("blacklist_reason", "")),
            has_flag_name=entry.get("has_flag_name"),
        )
        for entry in raw["signals"]
    ]


def _load_manifest(path: Path) -> DatasetStoreManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signals = tuple(
        DatasetSignal(
            name=s["name"],
            dtype=s["dtype"],
            row_shape=tuple(s["row_shape"]),
            required=s["required"],
            authoritative_source=s["authoritative_source"],
            content_identity=s["content_identity"],
            coverage_count=s["coverage_count"],
            migration_policy=MigrationPolicy(s["migration_policy"]),
        )
        for s in payload["signals"]
    )
    unavailable = tuple(UnavailableSignal(name=u["name"], reason=u["reason"]) for u in payload.get("unavailable_signals", []))
    return DatasetStoreManifest(
        release=payload["release"],
        store_id=payload["store_id"],
        build_id=payload["build_id"],
        producer_identity=payload["producer_identity"],
        client_build_evidence_id=payload["client_build_evidence_id"],
        index_identity=payload["index_identity"],
        row_count=payload["row_count"],
        signals=signals,
        row_lineage_identity=payload["row_lineage_identity"],
        finalization_state=FinalizationState(payload["finalization_state"]),
        unavailable_signals=unavailable,
    )


def _load_row_lineages(path: Path) -> list[RowLineage]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RowLineage(
            store_row=row["store_row"],
            build_id=row["build_id"],
            map_name=row["map_name"],
            tile_x=row["tile_x"],
            tile_y=row["tile_y"],
            source_group=row["source_group"],
            signal_actions=row["signal_actions"],
            signal_source_hashes=row.get("signal_source_hashes", {}),
            signal_destination_hashes=row.get("signal_destination_hashes", {}),
            source_artifact_id=row.get("source_artifact_id"),
            source_row=row.get("source_row"),
            split_group=row.get("split_group"),
        )
        for row in payload["rows"]
    ]


def _cmd_migrate_v18(args: argparse.Namespace) -> int:
    v18_store = Path(args.v18_store)
    root = zarr.open_group(str(v18_store), mode="r")
    index_rows = pq.read_table(str(v18_store / "index.parquet")).to_pylist()
    if args.sample is not None:
        index_rows = index_rows[: args.sample]

    specs = _load_signal_specs(Path(args.signals_config))
    manifest_template = _load_manifest(Path(args.manifest_template))

    ledger = MigrationLedger()
    copied_arrays: dict[str, list[np.ndarray]] = {}
    row_lineage_actions: list[dict[str, str]] = [dict() for _ in index_rows]
    unavailable: list[UnavailableSignal] = []
    per_signal_report = []

    for spec in specs:
        array = root[spec.name] if spec.name in root else None
        observations = [
            V18RowSignalObservation(
                row_id=i,
                value=(np.asarray(array[int(row["tile_id"])]) if array is not None else None),
                has_flag_value=(bool(row.get(spec.has_flag_name, False)) if spec.has_flag_name else None),
            )
            for i, row in enumerate(index_rows)
        ]
        audit = audit_v18_signal(spec, observations)
        plan = plan_signal_migration(audit)

        copy_eligible_rows = [row_id for row_id, action in plan.items() if action == ACTION_COPY]
        if copy_eligible_rows and array is not None:
            copied = []
            for row_id in copy_eligible_rows:
                copied_row, ledger = copy_signal_row(row_id, spec.name, np.asarray(array[row_id]), ledger)
                copied.append(copied_row)
                row_lineage_actions[row_id][spec.name] = "copied"
            copied_arrays[spec.name] = copied
        else:
            unavailable.append(
                UnavailableSignal(name=spec.name, reason="no rows passed audit as copy-if-verified; requires fresh extraction")
            )

        per_signal_report.append(
            {
                "signal_name": spec.name,
                "migration_policy": audit.migration_policy,
                "blacklisted": audit.blacklisted,
                "copy_eligible_row_count": len(copy_eligible_rows),
                "total_row_count": len(observations),
            }
        )

    report = {
        "schema": "v50-migration-report-v1",
        "v18_store": str(v18_store),
        "rows_processed": len(index_rows),
        "signals": per_signal_report,
        "ledger_entry_count": len(ledger.entries),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"migrate-v18: processed {len(index_rows)} rows, {len(ledger.entries)} bit-preserving copies -> {report_path}")

    if args.write_store:
        # Only fully-copied signals get a real array; everything else is honestly unavailable, and
        # finalization_state stays incomplete until `build` fills the fresh-extraction gaps.
        copyable_signals = tuple(
            signal for signal in manifest_template.signals
            if signal.name in copied_arrays and len(copied_arrays[signal.name]) == len(index_rows)
        )
        stacked_arrays = {name: np.stack(rows) for name, rows in copied_arrays.items()}
        recomputed_signals = tuple(
            DatasetSignal(
                name=signal.name,
                dtype=signal.dtype,
                row_shape=signal.row_shape,
                required=signal.required,
                authoritative_source=signal.authoritative_source,
                content_identity=hash_array(stacked_arrays[signal.name]),
                coverage_count=len(index_rows),
                migration_policy=signal.migration_policy,
            )
            for signal in copyable_signals
        )
        remaining_unavailable = tuple(unavailable) + tuple(
            UnavailableSignal(name=signal.name, reason="not fully copy-eligible across all rows in this pass")
            for signal in manifest_template.signals
            if signal.name not in {s.name for s in recomputed_signals}
        )
        output_manifest = DatasetStoreManifest(
            release=manifest_template.release,
            store_id=manifest_template.store_id,
            build_id=manifest_template.build_id,
            producer_identity=manifest_template.producer_identity,
            client_build_evidence_id=manifest_template.client_build_evidence_id,
            index_identity=manifest_template.index_identity,
            row_count=len(index_rows),
            signals=recomputed_signals or manifest_template.signals[:1],
            row_lineage_identity=manifest_template.row_lineage_identity,
            finalization_state=FinalizationState.INCOMPLETE,
            unavailable_signals=remaining_unavailable,
        )
        write_v50_store(Path(args.write_store), output_manifest, stacked_arrays)
        print(f"migrate-v18: wrote partial v50 store (finalization_state=incomplete) -> {args.write_store}")
        print("Remaining unavailable signals require `build` (fresh extraction) before `finalize`.")

    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    command = build_harvest_stream_command(
        Path(args.harvest_project),
        client_root=Path(args.clients_root),
        map_name=args.map,
        stream_profile=args.stream_profile,
    )
    process = run_fresh_extraction(command, confirm_run=args.confirm_run)
    if process is None:
        return 0
    print("Fresh extraction launched; consuming its stream is not yet wired into a store writer in this pass.")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    row_lineages = _load_row_lineages(Path(args.row_lineages))
    observed_hashes = json.loads(Path(args.observed_hashes).read_text(encoding="utf-8"))

    result = verify_store(manifest, row_lineages, observed_signal_hashes=observed_hashes)
    payload = {
        "store_id": result.store_id,
        "checks_performed": list(result.checks_performed),
        "passed": result.passed,
        "failure_reasons": list(result.failure_reasons),
        "proof_level": result.proof_level.value,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


def _cmd_finalize(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    row_lineages = _load_row_lineages(Path(args.row_lineages))

    finalized = finalize_store(Path(args.store), manifest, row_lineages)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(finalized.to_dict(), indent=2), encoding="utf-8")
    print(f"finalize: finalization_state={finalized.finalization_state.value} -> {output_path}")
    return 0 if finalized.finalization_state is FinalizationState.COMPLETE else 1


def _cmd_curriculum(args: argparse.Namespace) -> int:
    rows_payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = [
        CurriculumRowRef(store_id=r["store_id"], row_id=r["row_id"], source_group=r["source_group"], split=r["split"])
        for r in rows_payload["rows"]
    ]
    manifest = build_curriculum(
        release=args.release,
        rows=rows,
        selection_reason=args.selection_reason,
        policy_identity=args.policy_identity,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(f"curriculum: {len(rows)} row references -> {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate_parser = subparsers.add_parser("migrate-v18", help="bit-preserving copy of verified V18 signals")
    migrate_parser.add_argument("--v18-store", required=True)
    migrate_parser.add_argument("--signals-config", required=True)
    migrate_parser.add_argument("--manifest-template", required=True)
    migrate_parser.add_argument("--sample", type=int, default=None)
    migrate_parser.add_argument("--report", required=True)
    migrate_parser.add_argument("--write-store", default=None, help="also write a (likely partial) v50 store here")
    migrate_parser.set_defaults(handler=_cmd_migrate_v18)

    build_parser = subparsers.add_parser("build", help="fresh client-backed extraction (execution-gated)")
    build_parser.add_argument("--harvest-project", required=True)
    build_parser.add_argument("--clients-root", required=True)
    build_parser.add_argument("--map", required=True)
    build_parser.add_argument("--stream-profile", default="v22")
    build_parser.add_argument("--confirm-run", action="store_true",
                               help="actually launch the C# harvester; omit to only print the command")
    build_parser.set_defaults(handler=_cmd_build)

    verify_parser = subparsers.add_parser("verify", help="run the FR-005 promotion gate against a manifest")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--row-lineages", required=True)
    verify_parser.add_argument("--observed-hashes", required=True)
    verify_parser.set_defaults(handler=_cmd_verify)

    finalize_parser = subparsers.add_parser("finalize", help="recompute observed hashes and finalize a written store")
    finalize_parser.add_argument("--store", required=True)
    finalize_parser.add_argument("--manifest", required=True)
    finalize_parser.add_argument("--row-lineages", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.set_defaults(handler=_cmd_finalize)

    curriculum_parser = subparsers.add_parser("curriculum", help="build an immutable row-selection curriculum manifest")
    curriculum_parser.add_argument("--release", required=True)
    curriculum_parser.add_argument("--rows", required=True, help="JSON {rows: [{store_id, row_id, source_group, split}, ...]}")
    curriculum_parser.add_argument("--selection-reason", required=True)
    curriculum_parser.add_argument("--policy-identity", required=True)
    curriculum_parser.add_argument("--output", required=True)
    curriculum_parser.set_defaults(handler=_cmd_curriculum)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
