#!/usr/bin/env python3
"""Build a v60 footprint-guided object-marker corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_marker import ObjectMarkerError, build_object_marker_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build v60-object-marker-v1 from the corrected v50-library sieve corpus"
    )
    parser.add_argument("--sieve-corpus", required=True, type=Path)
    parser.add_argument("--object-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=6001)
    args = parser.parse_args()
    try:
        result = build_object_marker_corpus(
            sieve_corpus=args.sieve_corpus,
            object_library=args.object_library,
            output=args.output,
            seed=args.seed,
        )
    except (FileNotFoundError, ObjectMarkerError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
