#!/usr/bin/env python3
"""Spec 111 US2 CLI: turn a lighting-bucket distribution report into a rebalanced
synthetic-lighting-variant sampling plan.

Usage:
    uv run --directory wow-viewer/data-harvester python scripts/rebalance_lighting_variants.py \
        --distribution-report <report.json from scripts/report_lighting_buckets.py> \
        [--variant-count N] [--output <plan.json>] [--dry-run]

``--dry-run`` prints the per-bucket sampling weights and flags any ``no_real_baseline`` buckets
without writing anything. The plan/allocation logic lives in
``harvester.spec111.rebalance_lighting_variants`` -- this script is a thin CLI wrapper
(constitution: CLI tools are thin wrappers). The resulting ``lighting_times`` list feeds the
existing ``spec103_build_synthetic_store.py`` generator unchanged, which retains sole ownership of
all source-group/variant leak-safety tagging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harvester.spec111.rebalance_lighting_variants import (
    compute_sampling_plan,
    rebalanced_lighting_times,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-report", required=True, type=Path)
    parser.add_argument("--variant-count", type=int, default=24,
                        help="How many synthetic lighting variants to allocate across buckets.")
    parser.add_argument("--output", default=None, type=Path, help="Write the plan JSON here.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print weights and sparse buckets only; write nothing.")
    args = parser.parse_args()

    report = json.loads(args.distribution_report.read_text(encoding="utf-8"))
    plan = compute_sampling_plan(report)

    payload = plan.to_metadata()
    if plan.bucket_weights:
        payload["lighting_times"] = rebalanced_lighting_times(plan, args.variant_count)
    else:
        payload["lighting_times"] = None
        print(
            "WARNING: every bucket is sparse (no real-example coverage). No rebalanced "
            "lighting_times were produced; the existing documented sweep policy remains in effect.",
            file=sys.stderr,
        )

    text = json.dumps(payload, indent=2)
    if args.dry_run or args.output is None:
        print(text)
        if args.dry_run:
            print("(dry run: nothing written)", file=sys.stderr)
        return 0

    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote rebalanced sampling plan to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
