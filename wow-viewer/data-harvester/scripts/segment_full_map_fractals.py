"""Segment full-map alpha/fractal regions from a Phase 1 canvas."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_segments import (  # noqa: E402
    load_canvas_group,
    load_catalog_rows,
    render_region_overlay,
    save_regions,
    save_regions_jsonl,
    segment_canvas_regions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment full-map alpha/fractal regions.")
    parser.add_argument("--canvas-dir", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path, default=None, help="Optional 074 catalog dir or catalog.jsonl path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--min-atomic-footprint-px", type=int, default=8, help="Minimum bbox width and height for default atomic samples (8x8 alpha pixels).")
    parser.add_argument("--curation-mode", choices=("default", "raw"), default="default", help="Use raw to emit all regions as analysis components without curation labels.")
    parser.add_argument("--max-regions-per-layer", type=int, default=200)
    parser.add_argument("--chonker-area-fraction", type=float, default=0.18)
    parser.add_argument("--one-off-min-area", type=int, default=4096)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = load_canvas_group(args.canvas_dir)
    catalog_rows = load_catalog_rows(args.catalog_dir)
    regions = segment_canvas_regions(
        canvas,
        threshold=float(args.threshold),
        min_area=int(args.min_area),
        min_atomic_footprint_px=int(args.min_atomic_footprint_px),
        curation_mode=str(args.curation_mode),
        chonker_area_fraction=float(args.chonker_area_fraction),
        one_off_min_area=int(args.one_off_min_area),
        max_regions_per_layer=int(args.max_regions_per_layer),
        catalog_rows=catalog_rows,
    )
    save_regions(out_dir / "fractal_regions.parquet", regions)
    save_regions_jsonl(out_dir / "fractal_regions.jsonl", regions)
    if not bool(args.no_overlay):
        render_region_overlay(canvas, regions, out_dir / "overlays" / "fractal_regions_overlay.png")

    counts = Counter(region.curation_label for region in regions)
    summary = {
        "region_count": int(len(regions)),
        "curation_counts": dict(sorted(counts.items())),
        "threshold": float(args.threshold),
        "min_area": int(args.min_area),
        "min_atomic_footprint_px": int(args.min_atomic_footprint_px),
        "curation_mode": str(args.curation_mode),
        "catalog_rows_loaded": int(len(catalog_rows)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("Full-map fractal segmentation complete", flush=True)
    print(f"  canvas_dir: {args.canvas_dir}", flush=True)
    print(f"  output_dir: {out_dir}", flush=True)
    print(f"  regions: {len(regions)}", flush=True)
    print(f"  curation_counts: {dict(sorted(counts.items()))}", flush=True)
    print(f"  regions_parquet: {out_dir / 'fractal_regions.parquet'}", flush=True)
    print(f"  overlay: {out_dir / 'overlays' / 'fractal_regions_overlay.png'}", flush=True)


if __name__ == "__main__":
    main()
