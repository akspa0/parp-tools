#!/usr/bin/env python3
"""Dry-run-first builder for a minimap-observable raw RGB baseline corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.v60.real_minimap_rgb import (
    build_real_minimap_rgb_corpus,
    real_minimap_rgb_build_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--source-filter", required=True, choices=("synthetic", "authored"))
    parser.add_argument("--validation-map", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirm-build", action="store_true")
    args = parser.parse_args()
    if args.confirm_build:
        result = build_real_minimap_rgb_corpus(
            args.store,
            args.output,
            source_filter=args.source_filter,
            validation_map=args.validation_map,
        )
    else:
        result = real_minimap_rgb_build_plan(
            args.store,
            source_filter=args.source_filter,
            validation_map=args.validation_map,
        )
        result["output_root"] = str(args.output.resolve())
        print("DRY RUN ONLY: add --confirm-build to publish the raw-RGB baseline corpus.", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
