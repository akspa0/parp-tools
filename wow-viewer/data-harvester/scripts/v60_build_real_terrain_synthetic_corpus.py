#!/usr/bin/env python3
"""Dry-run-first builder for the real-terrain synthetic clean-signal bridge corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.v60.real_terrain_synthetic import (
    build_real_terrain_synthetic_corpus,
    real_terrain_synthetic_build_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, help="Directory of harvested real-terrain NPZ rows")
    parser.add_argument("--output", required=True, type=Path, help="Fresh clean-signal output directory")
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7137)
    parser.add_argument("--confirm-build", action="store_true", help="Materialize the corpus")
    args = parser.parse_args()
    if args.confirm_build:
        result = build_real_terrain_synthetic_corpus(
            args.inputs,
            args.output,
            confidence_value=args.confidence,
            seed=args.seed,
        )
    else:
        result = real_terrain_synthetic_build_plan(
            args.inputs,
            confidence_value=args.confidence,
            seed=args.seed,
        )
        result["output_root"] = str(args.output.resolve())
        print("DRY RUN ONLY: add --confirm-build to publish the bridge corpus.", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
