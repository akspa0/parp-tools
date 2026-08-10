#!/usr/bin/env python3
"""Dry-run-first builder for the complete v50.1 real-terrain synthetic bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.v60.real_terrain_synthetic_zarr import (
    DEFAULT_INPUT_SIGNAL,
    SUPPORTED_INPUT_SIGNALS,
    build_zarr_real_terrain_synthetic_corpus,
    zarr_real_terrain_synthetic_build_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--validation-map", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-signal", choices=sorted(SUPPORTED_INPUT_SIGNALS), default=DEFAULT_INPUT_SIGNAL)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--confirm-build", action="store_true")
    args = parser.parse_args()
    if args.confirm_build:
        result = build_zarr_real_terrain_synthetic_corpus(
            args.store,
            args.output,
            validation_map=args.validation_map,
            input_signal=args.input_signal,
            confidence_value=args.confidence,
        )
    else:
        result = zarr_real_terrain_synthetic_build_plan(
            args.store,
            validation_map=args.validation_map,
            input_signal=args.input_signal,
            confidence_value=args.confidence,
        )
        result["output_root"] = str(args.output.resolve())
        print("DRY RUN ONLY: add --confirm-build to publish the full bridge corpus.", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
