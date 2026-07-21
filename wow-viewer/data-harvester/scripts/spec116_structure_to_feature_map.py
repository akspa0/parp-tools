#!/usr/bin/env python3
"""Spec 116 US5 CLI: adapt a derived structure store into the geometry trainer's feature-store
shape (dry-run by default).

The existing Spec 114/115 geometry trainer's ``--feature-store`` requires a
``v115-feature-map-v1`` store with a ``feature_map`` array; ``spec116_materialize_structure.py``
writes ``v116-structure-store-v1`` instead. This bridge upsamples the structure store's per-chunk
predicted family + confidence into that exact per-pixel shape, so quickstart 5b's paired geometry
comparison runs against the existing trainer unmodified.

Run from ``wow-viewer/data-harvester/``::

    uv run python scripts/spec116_structure_to_feature_map.py \\
        --structure-store <derived structure store> \\
        --output <feature-map store> [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harvester.spec116.structure_feature_bridge import structure_to_feature_map


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 116 US5: structure store -> v115-feature-map-v1 bridge (dry-run by default)"
    )
    ap.add_argument("--structure-store", required=True, type=Path,
                    help="v116-structure-store-v1 store from spec116_materialize_structure.py")
    ap.add_argument("--output", required=True, type=Path, help="output feature-map store directory")
    ap.add_argument("--write", action="store_true", help="write the derived store (default: print plan only)")
    args = ap.parse_args(argv)

    result = structure_to_feature_map(
        structure_store=args.structure_store, output=args.output, write=args.write,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)

    if not args.write:
        print("DRY RUN ONLY -- pass --write to emit the feature-map store.", flush=True)
        return 0

    print(f"wrote feature-map store: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
