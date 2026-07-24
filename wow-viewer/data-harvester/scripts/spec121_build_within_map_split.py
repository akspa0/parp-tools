#!/usr/bin/env python3
"""Spec 121 B-reframe: build a within-map random held-out split for WDL completion training.

Dry-run-first: without ``--write`` it prints the split plan and exits. ``--write`` persists
the split as ``v121-within-map-split-v1`` schema, consumed by the existing Stage A trainer via
auto-detection (``detect_split_schema`` dispatches to ``apply_within_map_split``).

Usage (from wow-viewer/data-harvester):
    uv run python scripts/spec121_build_within_map_split.py --help
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from harvester.spec121.within_map_split import (
    build_within_map_split,
    write_within_map_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Spec 121 within-map WDL completion split (dry-run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="v50 curriculum Zarr store")
    ap.add_argument("--held-out-fraction", type=float, default=0.15)
    ap.add_argument("--buffer-rings", type=int, default=0,
                    help="exclude tiles adjacent to held-out tiles (≥0; default 0 = no buffer)")
    ap.add_argument("--seed", type=int, default=121)
    ap.add_argument("--build-id", default="")
    ap.add_argument("--output", required=True, type=Path, help="output split directory")
    ap.add_argument("--write", action="store_true",
                    help="persist the split (default: dry-run prints plan only)")
    args = ap.parse_args()

    index = pq.read_table(args.store / "index.parquet").to_pylist()
    split = build_within_map_split(
        index,
        held_out_fraction=args.held_out_fraction,
        buffer_rings=args.buffer_rings,
        seed=args.seed,
    )
    plan = {
        "schema": "v121-within-map-split-plan-v1",
        "store": str(args.store.resolve()),
        "output": str(args.output.resolve()),
        "held_out_fraction": args.held_out_fraction,
        "buffer_rings": args.buffer_rings,
        "seed": args.seed,
        "split_counts": split["split_counts"],
        "per_map_counts": split["per_map_counts"],
        "excluded_count": split["excluded_count"],
        "verified_overlap_count": split["verified_overlap_count"],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if split["verified_overlap_count"] != 0:
        raise SystemExit(
            f"split has {split['verified_overlap_count']} tile-level overlaps (bug)"
        )
    if not args.write:
        print("DRY RUN ONLY: add --write to persist the split.", flush=True)
        return 0
    manifest = write_within_map_split(
        store=args.store, output=args.output, split=split, build_id=args.build_id,
    )
    print(f"Split written: {manifest['schema']} @ {args.output.resolve()}", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
