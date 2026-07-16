"""Mine repeated irregular terrain motifs from chunk-cell graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.chunk_motifs import build_chunk_cells, extract_chunk_motifs, write_motif_outputs  # noqa: E402, I001
from harvester.fractal_canvas import load_tile_records  # noqa: E402, I001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine repeated irregular chunk-cell motifs; never zones or rectangular crops.")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-alpha-variation", type=float, default=0.025)
    parser.add_argument("--min-height-relief", type=float, default=2.0)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--max-cells", type=int, default=32)
    parser.add_argument("--min-occurrences", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = zarr.open_group(store=zarr.storage.LocalStore(str(args.store), read_only=True), mode="r")
    # Tile coordinates are map-local.  Build each map separately so an Azeroth
    # cell can never become an accidental neighbor of a Kalimdor cell with the
    # same coordinate; family canonicalization happens only after that boundary.
    records = []
    motifs = []
    active_cell_count = 0
    for map_name in args.maps:
        map_records = load_tile_records(args.store, build=args.build, map_name=map_name, require_alpha=True, tile_limit=None)
        map_cells = build_chunk_cells(root, map_records, min_alpha_variation=args.min_alpha_variation, min_height_relief=args.min_height_relief)
        records.extend(map_records)
        active_cell_count += len(map_cells)
        motifs.extend(extract_chunk_motifs(map_cells, max_hops=args.max_hops, max_cells=args.max_cells))
    summary = write_motif_outputs(args.output, motifs, min_occurrences=args.min_occurrences)
    print(f"[chunk-motifs] tiles={len(records)} active_cells={active_cell_count} candidates={len(motifs)} repeated_families={summary['repeated_family_count']}", flush=True)
    print(f"[chunk-motifs] output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
