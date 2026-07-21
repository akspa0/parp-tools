#!/usr/bin/env python3
"""Spec 116 US4 CLI: build a spatially-isolated held-out split (dry-run by default).

Prints the train/held_out/excluded counts and the verified_violation_count (must be 0). Writes
``split.parquet`` + ``split.json`` only with ``--write``.

Run from ``wow-viewer/data-harvester/``::

    uv run python scripts/spec116_build_held_out_split.py \
        --store <v50 curriculum store> --buffer-rings 1 --seed 116 \
        --output <split dir> [--write]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harvester.spec116.held_out_split import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BUFFER_RINGS,
    DEFAULT_HELD_OUT_FRACTION,
    build_held_out_split,
    write_split,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 116 US4: spatially-isolated held-out split (dry-run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="v50 curriculum Zarr store")
    ap.add_argument("--held-out-fraction", type=float, default=DEFAULT_HELD_OUT_FRACTION)
    ap.add_argument("--buffer-rings", type=int, default=DEFAULT_BUFFER_RINGS)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-id", default="")
    ap.add_argument("--output", type=Path, default=None, help="output split dir (required with --write)")
    ap.add_argument("--write", action="store_true", help="write the split (default: print only)")
    args = ap.parse_args(argv)

    if args.write and args.output is None:
        ap.error("--write requires --output")

    import pyarrow.parquet as pq

    index_rows = pq.read_table(args.store / "index.parquet").to_pylist()
    tiles = [
        {"tile_row": i, "map": str(r.get("map")), "tile_x": int(r.get("tile_x", -1)),
         "tile_y": int(r.get("tile_y", -1))}
        for i, r in enumerate(index_rows)
    ]

    # v50-held-out-split-v1 requires a non-empty build_id (unlike the analysis-report schema,
    # where it's optional). Auto-derive it from the curriculum's own index.parquet "build" column
    # rather than requiring the operator to type it by hand.
    build_id = args.build_id
    if not build_id and index_rows:
        build_id = str(index_rows[0].get("build") or "")
    if args.write and not build_id:
        ap.error(
            "--build-id could not be auto-derived from the store's index.parquet 'build' "
            "column; pass --build-id explicitly"
        )

    split = build_held_out_split(
        tiles=tiles, held_out_fraction=args.held_out_fraction, buffer_rings=args.buffer_rings,
        block_size=args.block_size, seed=args.seed,
    )

    print("=== Spec 116 US4: spatially-isolated held-out split ===")
    print(f"store: {args.store}")
    print(f"tiles: {len(tiles)}  held_out_fraction: {args.held_out_fraction}  "
          f"buffer_rings: {args.buffer_rings}  block_size: {args.block_size}  seed: {args.seed}")
    print(f"split_counts: {split['split_counts']}")
    print(f"excluded_count (isolation buffer): {split['excluded_count']}")
    print(f"verified_violation_count: {split['verified_violation_count']}  (MUST be 0)")

    if split["verified_violation_count"] != 0:
        print("\nERROR: held-out set is NOT isolated from training; refusing to write.")
        return 1

    if not args.write:
        print("\nDRY RUN ONLY -- pass --write to emit split.parquet + split.json.")
        return 0

    manifest = write_split(store=args.store, output=args.output, split=split, build_id=build_id)
    print(f"\nwrote split: {args.output}")
    print(f"  absolute_comparison_to_prior_invalid: {manifest['absolute_comparison_to_prior_invalid']}")
    print("  Rebuilding this split INVALIDATES absolute comparison with all prior results (FR-017).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
