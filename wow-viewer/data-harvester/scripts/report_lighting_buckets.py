#!/usr/bin/env python3
"""Spec 111 T008 CLI: print or write the lighting-bucket distribution report for a build store.

Usage:
    uv run --directory wow-viewer/data-harvester python scripts/report_lighting_buckets.py \
        --store-path <path to a build store directory containing decoded_metadata.parquet> \
        [--map <single map name>] [--output <report.json>]

The report-building logic lives in ``harvester.spec111.lighting_buckets`` -- this script is a thin
CLI wrapper over that library module (constitution: CLI tools are thin wrappers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harvester.spec111.lighting_buckets import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-path", required=True, type=Path)
    parser.add_argument("--map", default=None, help="Restrict the report to a single map name.")
    parser.add_argument("--output", default=None, type=Path, help="Write JSON report here instead of stdout.")
    args = parser.parse_args()

    report = build_report(args.store_path, args.map)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote lighting bucket report to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
