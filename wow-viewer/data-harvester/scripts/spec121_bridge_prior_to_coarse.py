#!/usr/bin/env python3
"""Spec 121 T015: bridge a frozen Stage A checkpoint's lattice predictions into a coarse store
that the existing detailer trainer consumes as ``--coarse-store``.

Dry-run-first: without ``--write`` it prints the plan and exits. ``--write`` persists the coarse
store matching ``v114-coarse-relief-v1`` schema.

Usage (from wow-viewer/data-harvester):
    uv run python scripts/spec121_bridge_prior_to_coarse.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec121.prior_coarse_bridge import bridge_prior_to_coarse


def main() -> int:
    import argparse
    import json

    from harvester.v50.contracts import validate_release
    from harvester.v50.height_relative_train import SOURCE_CHOICES

    ap = argparse.ArgumentParser(
        description="Spec 121: bridge Stage A lattice predictions into a coarse store for the detailer"
    )
    ap.add_argument("--store", required=True, type=Path, help="source curriculum store")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen Stage A checkpoint")
    ap.add_argument("--output", required=True, type=Path, help="output coarse store directory")
    ap.add_argument("--source", default="authored", choices=sorted(SOURCE_CHOICES))
    ap.add_argument("--release", default="v50.2", type=validate_release)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--write", action="store_true",
                    help="persist the coarse store (default: dry-run prints plan only)")
    args = ap.parse_args()

    result = bridge_prior_to_coarse(
        store=args.store, checkpoint_path=args.checkpoint, output=args.output,
        source=args.source, release=args.release, device=args.device, write=args.write,
    )
    print(json.dumps(result, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to persist the coarse store.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
