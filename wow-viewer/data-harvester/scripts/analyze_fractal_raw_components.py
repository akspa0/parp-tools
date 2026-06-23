"""Run raw full-map component analysis and cross-target exact dedupe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
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
from harvester.fractal_raw_analysis import (  # noqa: E402
    fingerprint_raw_regions,
    write_raw_dedupe_outputs,
)
from harvester.fractal_segments import (  # noqa: E402
    render_region_overlay,
    save_regions,
    save_regions_jsonl,
    segment_canvas_regions,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "analysis" / "full-map-fractal-brush-library" / "raw_two_build_dedupe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw alpha/fractal components across builds/maps and dedupe exact shapes.")
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--builds", nargs="+", default=["0_5_3_3368", "3_3_5_12340"])
    parser.add_argument("--maps", nargs="+", required=True, help="Map names to process for each build, or 'all' for every map in each build index. Missing maps are skipped by default.")
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument("--tile-limit", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-area", type=int, default=64, help="Minimum component area in alpha pixels (8x8).")
    parser.add_argument("--min-footprint-px", type=int, default=8, help="Minimum bbox width and height for raw components (8x8 alpha pixels).")
    parser.add_argument("--max-regions-per-layer", type=int, default=5000)
    parser.add_argument("--skip-missing-maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Render contact sheets for the dedupe catalog after analysis.")
    parser.add_argument("--max-patterns", type=int, default=200)
    parser.add_argument("--max-per-pattern", type=int, default=6)
    parser.add_argument("--repeated-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    layers = tuple(int(part.strip()) for part in str(args.layers).split(",") if part.strip())
    all_fingerprints = []
    target_summaries = []

    for build in args.builds:
        zarr_path = Path(args.dataset_dir) / f"{build}.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"Build Zarr not found: {zarr_path}")
        available_maps = _available_maps(zarr_path)
        target_maps = resolve_target_maps(args.maps, available_maps)
        for map_name in target_maps:
            if map_name not in available_maps:
                if bool(args.skip_missing_maps):
                    print(f"Skipping missing map build={build} map={map_name}", flush=True)
                    continue
                raise RuntimeError(f"Map {map_name!r} not found in build {build}; available={available_maps[:20]}")

            target_dir = out_root / f"{build}_{map_name}_tile{int(args.tile_limit)}"
            canvas_dir = target_dir / "canvas"
            segments_dir = target_dir / "segments_raw"
            canvas_dir.mkdir(parents=True, exist_ok=True)
            segments_dir.mkdir(parents=True, exist_ok=True)

            print(f"Analyzing build={build} map={map_name}", flush=True)
            root = zarr.open_group(store=zarr.storage.LocalStore(str(zarr_path), read_only=True), mode="r")
            records = load_tile_records(zarr_path, build=str(build), map_name=str(map_name), require_alpha=True, tile_limit=args.tile_limit)
            if not records:
                print("  no alpha-bearing records, skipped", flush=True)
                continue
            layout, arrays, index_rows = assemble_full_map_canvas(root, records, layers=layers)
            write_canvas_outputs(canvas_dir, layout, arrays, index_rows)
            canvas = zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r")
            if not bool(args.no_overlay):
                write_debug_overlay(canvas_dir, layout, arrays["alpha_256"], layer_slot=0)

            regions = segment_canvas_regions(
                canvas,
                threshold=float(args.threshold),
                min_area=int(args.min_area),
                min_atomic_footprint_px=int(args.min_footprint_px),
                curation_mode="raw",
                max_regions_per_layer=int(args.max_regions_per_layer),
            )
            save_regions(segments_dir / "fractal_regions.parquet", regions)
            save_regions_jsonl(segments_dir / "fractal_regions.jsonl", regions)
            if not bool(args.no_overlay):
                render_region_overlay(canvas, regions, segments_dir / "overlays" / "fractal_regions_overlay.png")
            fingerprints = fingerprint_raw_regions(canvas, regions, threshold=float(args.threshold))
            all_fingerprints.extend(fingerprints)

            counts = Counter(region.curation_label for region in regions)
            target_summary = {
                "build": str(build),
                "map": str(map_name),
                "tile_count": int(len(records)),
                "region_count": int(len(regions)),
                "curation_counts": dict(sorted(counts.items())),
                "canvas_dir": str(canvas_dir),
                "segments_dir": str(segments_dir),
            }
            target_summaries.append(target_summary)
            (segments_dir / "summary.json").write_text(json.dumps(target_summary, indent=2, sort_keys=True), encoding="utf-8")
            print(f"  regions={len(regions)}", flush=True)

    dedupe_summary = write_raw_dedupe_outputs(out_root / "dedupe", all_fingerprints)
    summary = {
        "target_count": int(len(target_summaries)),
        "targets": target_summaries,
        "dedupe": dedupe_summary,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("Raw two-build analysis complete", flush=True)
    print(f"  output_root: {out_root}", flush=True)
    print(f"  targets: {len(target_summaries)}", flush=True)
    print(f"  raw_components: {dedupe_summary['raw_component_count']}", flush=True)
    print(f"  exact_patterns: {dedupe_summary['exact_pattern_count']}", flush=True)
    print(f"  duplicate_patterns: {dedupe_summary['duplicate_pattern_count']}", flush=True)

    if bool(args.visualize):
        _run_visualizer(out_root, args)


def _run_visualizer(out_root: Path, args: argparse.Namespace) -> None:
    import subprocess

    script = Path(__file__).resolve().parent / "visualize_fractal_raw_patterns.py"
    contact_dir = out_root / "contact_sheets"
    cmd = [
        sys.executable,
        str(script),
        "--analysis-root",
        str(out_root),
        "--output-dir",
        str(contact_dir),
        "--max-patterns",
        str(int(args.max_patterns)),
        "--max-per-pattern",
        str(int(args.max_per_pattern)),
    ]
    if bool(args.repeated_only):
        cmd.append("--repeated-only")
    print("Rendering contact sheets...", flush=True)
    subprocess.run(cmd, check=True)


def _available_maps(zarr_path: Path) -> list[str]:
    table = pq.read_table(zarr_path / "index.parquet", columns=["map"])
    return sorted({str(value.as_py()) for value in table.column("map")})


def resolve_target_maps(requested_maps: list[str], available_maps: list[str]) -> list[str]:
    if any(str(item).lower() == "all" for item in requested_maps):
        return list(available_maps)
    return list(requested_maps)


if __name__ == "__main__":
    main()
