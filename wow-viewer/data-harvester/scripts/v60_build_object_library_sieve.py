#!/usr/bin/env python3
"""Build a v60 object-sieve corpus from the real v50 object library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_library_sieve import (  # noqa: E402
    ObjectLibrarySieveError,
    build_object_library_sieve_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build v60-object-library-sieve-v1 from a read-only v50 library and control corpus"
    )
    parser.add_argument("--control-corpus", required=True, type=Path)
    parser.add_argument("--object-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-terrain", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--blank-threshold", type=float, default=0.01)
    args = parser.parse_args()
    try:
        result = build_object_library_sieve_corpus(
            control_corpus=args.control_corpus,
            object_library=args.object_library,
            output=args.output,
            samples_per_terrain=args.samples_per_terrain,
            seed=args.seed,
            blank_threshold=args.blank_threshold,
        )
    except (FileNotFoundError, ObjectLibrarySieveError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
