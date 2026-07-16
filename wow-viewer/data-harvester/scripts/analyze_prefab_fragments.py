"""Discover repeated bounded terrain-art fragments from V18 alpha/height pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_canvas import load_tile_records  # noqa: E402, I001
from harvester.prefab_fragments import extract_prefab_fragments, write_fragment_outputs  # noqa: E402, I001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find repeated bounded local terrain-art fragments; never full-zone components.")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supports", default="32,64,128", help="Chunk-aligned alpha-pixel supports.")
    parser.add_argument("--stride", type=int, default=16, help="Chunk-aligned anchor stride in alpha pixels.")
    parser.add_argument("--min-alpha-coverage", type=float, default=0.08)
    parser.add_argument("--min-height-range", type=float, default=4.0)
    parser.add_argument("--max-candidates-per-tile", type=int, default=48)
    parser.add_argument("--tile-limit", type=int, default=0, help="0 means all matching tile pages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supports = tuple(int(value) for value in args.supports.split(",") if value.strip())
    root = zarr.open_group(store=zarr.storage.LocalStore(str(args.store), read_only=True), mode="r")
    all_fragments = []
    for map_name in args.maps:
        records = load_tile_records(args.store, build=args.build, map_name=map_name, require_alpha=True, tile_limit=None if args.tile_limit <= 0 else args.tile_limit)
        fragments = extract_prefab_fragments(root, records, supports=supports, stride=args.stride, min_alpha_coverage=args.min_alpha_coverage, min_height_range=args.min_height_range, max_candidates_per_tile=args.max_candidates_per_tile)
        all_fragments.extend(fragments)
        print(f"[prefab-fragments] {map_name}: tiles={len(records)} fragments={len(fragments)}", flush=True)
    summary = write_fragment_outputs(args.output, root, all_fragments, alpha_threshold=0.05)
    print(f"[prefab-fragments] repeated_families={summary['repeated_family_count']} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
