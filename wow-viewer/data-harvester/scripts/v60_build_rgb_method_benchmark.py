#!/usr/bin/env python3
"""Build a dry-run RGB-only method benchmark plan from existing v60 manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.rgb_method_benchmark import (  # noqa: E402
    RGBMethodBenchmarkError,
    build_rgb_method_benchmark_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=("authored", "object_library", "both"))
    parser.add_argument("--authored-corpus", type=Path)
    parser.add_argument("--object-library-sieve", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true", help="write the plan JSON; refuses existing output")
    args = parser.parse_args()
    if args.write and args.output is None:
        parser.error("--write requires --output")
    if args.write and args.output.exists():
        parser.error(f"refusing to overwrite existing benchmark plan: {args.output}")

    try:
        plan = build_rgb_method_benchmark_plan(
            source=args.source,
            authored_corpus=args.authored_corpus,
            object_library_sieve=args.object_library_sieve,
        )
    except (OSError, ValueError, json.JSONDecodeError, RGBMethodBenchmarkError) as exc:
        print(json.dumps({"schema": "v60-rgb-method-benchmark-v1", "valid": False, "failures": [str(exc)]}, indent=2))
        return 2

    plan["dry_run"] = not args.write
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        plan["plan_path"] = str(args.output.resolve())
    else:
        print("DRY RUN ONLY: add --write --output <path> to publish the benchmark plan.", flush=True)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    return 0 if plan["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
