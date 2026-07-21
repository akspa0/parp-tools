#!/usr/bin/env python3
"""Spec 116 US1 CLI: family->slot consistency measurement (dry-run by default).

Prints the per-family slot distribution, summary consistency score, and the slot_keyed/
family_keyed recommendation. Writes the durable ``v116-analysis-report-v1`` artifact only with
``--write``. No model is trained.

Run from ``wow-viewer/data-harvester/``::

    uv run python scripts/spec116_family_slot_consistency.py \
        --store <v50 curriculum store> --dumps <texture-name dump json> \
        --threshold 0.70 [--output <report.json>] [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harvester.spec116.family_slot_consistency import (
    DEFAULT_THRESHOLD,
    measure_family_slot_consistency,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 116 US1: family->slot consistency measurement (dry-run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="v50 curriculum Zarr store")
    ap.add_argument(
        "--dumps", required=True, nargs="+", type=Path,
        help="one or more terrain-texture-name-dump-v1 JSON files",
    )
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"consistency decision threshold (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--build-id", default="", help="build id recorded for provenance")
    ap.add_argument("--output", type=Path, default=None, help="output report JSON path (required with --write)")
    ap.add_argument("--write", action="store_true", help="write the report artifact (default: print only)")
    args = ap.parse_args(argv)

    if args.write and args.output is None:
        ap.error("--write requires --output")

    report = measure_family_slot_consistency(
        store=args.store, dumps=args.dumps, threshold=args.threshold, build_id=args.build_id,
    )

    fsc = report["family_slot_consistency"]
    print("=== Spec 116 US1: family->slot consistency ===")
    print(f"store: {report['identity']['store']['path']}")
    print(f"taxonomy_revision: {report['identity']['taxonomy_revision']}")
    print(f"layer_entry_count: {report['layer_entry_count']}  excluded: {report['excluded']}")
    print()
    print("per-family slot distribution (P(slot | family)):")
    for entry in fsc["per_family"]:
        dist = "  ".join(f"{p:.3f}" for p in entry["slot_distribution"])
        print(f"  {entry['family']:<10} [{dist}]  max={entry['max_slot_probability']:.3f}")
    print()
    print(f"summary_consistency_score: {fsc['summary_consistency_score']:.3f}")
    print(f"threshold: {fsc['threshold']:.3f}")
    print(f"recommendation: {fsc['recommendation']}")

    if not args.write:
        print("\nDRY RUN ONLY -- pass --write to emit the report artifact.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
