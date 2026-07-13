"""Bind the full staged 3.3.5 map inventory to raw V18 and the M0 corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from harvester.spec102.m0_coverage import (
    build_m0_coverage_report,
    parse_discovery_inventory,
)
from harvester.spec102.m0_scope import validate_m0_build_local_scope


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit complete 3.3.5 M0 map/row provenance")
    parser.add_argument("--client-root", required=True, type=Path)
    parser.add_argument("--harvest-tool", required=True, type=Path)
    parser.add_argument("--raw-v18-store", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--curation-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite coverage report: {args.output}")
    if not args.client_root.is_dir():
        raise FileNotFoundError(f"staged client root does not exist: {args.client_root}")
    if not args.harvest_tool.is_file():
        raise FileNotFoundError(f"harvest tool does not exist: {args.harvest_tool}")

    result = subprocess.run(
        [
            "dotnet", str(args.harvest_tool), "discover-maps",
            "--client-root", str(args.client_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "staged-client map discovery failed: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )
    inventory = parse_discovery_inventory(result.stdout)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    store_index = pq.read_table(args.store / "index.parquet").to_pylist()
    scope = validate_m0_build_local_scope(split, source_index=store_index)
    report = build_m0_coverage_report(
        inventory=inventory,
        client_root=args.client_root,
        raw_v18_store=args.raw_v18_store,
        numeric_store=args.store,
        curation_manifest=args.curation_manifest,
        split_manifest=args.split_manifest,
        expected_scope=scope.audit_binding,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "safe_for_requested_3_3_5_m0_training": report["safe_for_requested_3_3_5_m0_training"],
        "coverage": {
            key: report["coverage"][key]
            for key in ("raw_v18_rows", "raw_v18_maps", "numeric_rows", "numeric_maps",
                        "full_map_identity_coverage", "full_row_identity_coverage",
                        "eligible_m0_by_split")
        },
        "hard_failures": report["hard_failures"],
    }, indent=2))
    return 0 if report["safe_for_requested_3_3_5_m0_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
