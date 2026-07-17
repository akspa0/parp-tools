#!/usr/bin/env python3
"""Spec 109 T019: read-only v50 audit commands. Neither subcommand promotes, mutates, or deletes
anything (FR-010) -- both only write a JSON report.

Subcommands:

    inventory     Metadata-only artifact discovery across one or more roots.
    verify-v18    Per-signal audit of an existing V18 store's already-decoded content (known
                  defects, non-finite values, has_* truthfulness) against a caller-supplied signal
                  catalog. Signal specs are supplied via --signals-config rather than hardcoded,
                  since Spec 109's frozen v50 signal table (T002) is not yet finalized.

                  Gap, stated plainly: this does not yet cross-validate against a fresh extraction
                  from the configured client build (plan.md Phase 2 step 2) -- that requires
                  invoking the existing C# harvester as a subprocess and is deferred to when a
                  concrete V18 signal catalog and client build are chosen for a real audit run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from harvester.v50.inventory import InventoryRoot, discover_artifacts
from harvester.v50.verify_v18 import V18RowSignalObservation, V18SignalSpec, audit_v18_signal


def _cmd_inventory(args: argparse.Namespace) -> int:
    roots = [
        InventoryRoot(path=Path(root), default_owner=owner)
        for root, owner in (
            (args.dataset_root, "datasets"),
            (args.output_root, "output"),
            (args.model_root, "models"),
        )
        if root is not None
    ]
    for extra in args.extra_root or []:
        if "=" not in extra:
            print(f"Error: --extra-root must be OWNER=PATH, got {extra!r}", file=sys.stderr)
            return 2
        owner, _, path = extra.partition("=")
        roots.append(InventoryRoot(path=Path(path), default_owner=owner))
    if not roots:
        print("Error: at least one of --dataset-root/--output-root/--model-root/--extra-root is required", file=sys.stderr)
        return 2

    records = discover_artifacts(roots)
    payload = {
        "schema": "v50-inventory-report-v1",
        "roots": [str(root.path) for root in roots],
        "artifact_count": len(records),
        "artifacts": [
            {
                "artifact_id": record.artifact_id,
                "kind": record.kind,
                "resolved_path": record.resolved_path,
                "observed_bytes": record.observed_bytes,
                "content_identity": record.content_identity,
                "owner": record.owner,
                "proof_level": record.proof_level.value,
                "trust_state": record.trust_state.value,
                "disposition": record.disposition.value,
            }
            for record in records
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"inventory: {len(records)} artifacts discovered (all unverified) -> {report_path}")
    return 0


def _load_signal_specs(config_path: Path) -> list[V18SignalSpec]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    specs: list[V18SignalSpec] = []
    for entry in raw["signals"]:
        specs.append(
            V18SignalSpec(
                name=entry["name"],
                blacklisted=bool(entry.get("blacklisted", False)),
                blacklist_reason=str(entry.get("blacklist_reason", "")),
                has_flag_name=entry.get("has_flag_name"),
            )
        )
    return specs


def _observations_for_signal(root, index_rows: list[dict[str, Any]], spec: V18SignalSpec) -> list[V18RowSignalObservation]:
    import numpy as np

    observations: list[V18RowSignalObservation] = []
    array = root[spec.name] if spec.name in root else None
    for row in index_rows:
        row_id = int(row["tile_id"])
        value = None
        if array is not None:
            try:
                value = np.asarray(array[row_id])
            except (IndexError, KeyError):
                value = None
        has_flag_value = None
        if spec.has_flag_name is not None:
            has_flag_value = bool(row.get(spec.has_flag_name, False))
        observations.append(V18RowSignalObservation(row_id=row_id, value=value, has_flag_value=has_flag_value))
    return observations


def _cmd_verify_v18(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq
    import zarr

    store_path = Path(args.store)
    root = zarr.open_group(str(store_path), mode="r")
    index_path = store_path / "index.parquet"
    if not index_path.exists():
        print(f"Error: no index.parquet found under {store_path}", file=sys.stderr)
        return 2
    index_rows = pq.read_table(str(index_path)).to_pylist()
    if args.sample is not None:
        index_rows = index_rows[: args.sample]

    specs = _load_signal_specs(Path(args.signals_config))

    signal_reports = []
    for spec in specs:
        observations = _observations_for_signal(root, index_rows, spec)
        result = audit_v18_signal(spec, observations)
        signal_reports.append(
            {
                "signal_name": result.signal_name,
                "migration_policy": result.migration_policy,
                "blacklisted": result.blacklisted,
                "rows_checked": len(result.row_results),
                "passed_row_ids": list(result.passed_row_ids),
                "rejected_row_ids": list(result.rejected_row_ids),
                "rejection_reasons": sorted({r.reason for r in result.row_results if not r.passed}),
            }
        )

    payload = {
        "schema": "v50-v18-signal-audit-report-v1",
        "store": str(store_path),
        "build": args.build,
        "rows_sampled": len(index_rows),
        "cross_validated_against_fresh_client_extraction": False,
        "signals": signal_reports,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"verify-v18: audited {len(specs)} signals over {len(index_rows)} rows -> {report_path}")
    print("NOTE: this does not yet cross-validate against a fresh client extraction (see module docstring).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="metadata-only artifact discovery")
    inventory_parser.add_argument("--dataset-root", default=None)
    inventory_parser.add_argument("--output-root", default=None)
    inventory_parser.add_argument("--model-root", default=None)
    inventory_parser.add_argument("--extra-root", action="append", default=None,
                                   help="repeatable OWNER=PATH for roots beyond the three named ones")
    inventory_parser.add_argument("--report", required=True)
    inventory_parser.set_defaults(handler=_cmd_inventory)

    verify_parser = subparsers.add_parser("verify-v18", help="per-signal V18 audit")
    verify_parser.add_argument("--store", required=True)
    verify_parser.add_argument("--build", required=True)
    verify_parser.add_argument("--signals-config", required=True,
                                help="JSON file: {\"signals\": [{\"name\":..., \"blacklisted\":..., "
                                     "\"blacklist_reason\":..., \"has_flag_name\":...}, ...]}")
    verify_parser.add_argument("--sample", type=int, default=None)
    verify_parser.add_argument("--report", required=True)
    verify_parser.add_argument("--clients-root", default=None,
                                help="reserved for the future fresh-extraction cross-check; unused today")
    verify_parser.set_defaults(handler=_cmd_verify_v18)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
