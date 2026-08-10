#!/usr/bin/env python3
"""Audit Spec 141 external-method records and input contracts without running models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.terrain_method_translation import (  # noqa: E402
    build_method_translation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the report without writing an artifact (the default)",
    )
    parser.add_argument("--output", type=Path, help="JSON report path used with --write")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the report to --output; refuses an existing file",
    )
    args = parser.parse_args()
    if args.write and args.output is None:
        parser.error("--write requires --output")
    if args.write and args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")

    report = build_method_translation_report()
    report["dry_run"] = not args.write
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["report_path"] = str(args.output.resolve())
    else:
        print("DRY RUN ONLY: add --write --output <path> to publish the method audit report.", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
