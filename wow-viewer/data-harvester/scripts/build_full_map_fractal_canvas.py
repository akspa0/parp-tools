"""Build bounded full-map signal canvases for spec 076 Phase 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_canvas import (  # noqa: E402
    assemble_full_map_canvas,
    load_tile_records,
    write_canvas_outputs,
    write_debug_overlay,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output" / "analysis" / "full-map-fractal-brush-library"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded full-map fractal brush canvas.")
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--build", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument("--tile-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zarr_path = Path(args.dataset_dir) / f"{args.build}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Build Zarr not found: {zarr_path}")
    out_dir = Path(args.output_dir) if args.output_dir is not None else _DEFAULT_OUTPUT_DIR / args.build / args.map
    layers = tuple(int(part.strip()) for part in str(args.layers).split(",") if part.strip())

    records = load_tile_records(
        zarr_path,
        build=str(args.build),
        map_name=str(args.map),
        require_alpha=True,
        tile_limit=args.tile_limit,
    )
    if not records:
        raise RuntimeError(f"No alpha-bearing records found for build={args.build} map={args.map}")

    root = zarr.open_group(store=zarr.storage.LocalStore(str(zarr_path), read_only=True), mode="r")
    layout, arrays, index_rows = assemble_full_map_canvas(root, records, layers=layers)
    write_canvas_outputs(out_dir, layout, arrays, index_rows)
    overlay_path = None
    if not bool(args.no_overlay):
        overlay_path = write_debug_overlay(out_dir, layout, arrays["alpha_256"], layer_slot=0)

    print("Full-map fractal canvas built", flush=True)
    print(f"  build: {args.build}", flush=True)
    print(f"  map: {args.map}", flush=True)
    print(f"  tiles: {len(records)}", flush=True)
    print(f"  alpha_shape: {arrays['alpha_256'].shape}", flush=True)
    print(f"  height_shape: {arrays['height_257'].shape}", flush=True)
    print(f"  mcly_shape: {arrays['mcly_texture_ids'].shape}", flush=True)
    print(f"  output_dir: {out_dir}", flush=True)
    print(f"  canvas_zarr: {out_dir / 'canvas.zarr'}", flush=True)
    print(f"  canvas_index: {out_dir / 'canvas_index.parquet'}", flush=True)
    if overlay_path is not None:
        print(f"  overlay: {overlay_path}", flush=True)


if __name__ == "__main__":
    main()
