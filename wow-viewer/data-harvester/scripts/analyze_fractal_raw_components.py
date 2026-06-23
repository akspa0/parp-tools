"""Run raw full-map component analysis and cross-target exact dedupe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr

zarr.config.set({"async.concurrency": 1})

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_canvas import (  # noqa: E402
    ALPHA_TILE_SIZE,
    CanvasLayout,
    CanvasTileRecord,
    alpha_origin,
    build_canvas_layout,
    create_chunked_canvas_group,
    load_tile_records,
    mcly_origin,
    write_canvas_outputs,
    write_debug_overlay,
    write_tile_to_canvas,
)
from harvester.fractal_near_dedupe import (  # noqa: E402
    cluster_near_duplicates,
    write_near_dedupe_outputs,
)
from harvester.fractal_raw_analysis import (  # noqa: E402
    fingerprint_raw_regions,
    write_raw_dedupe_outputs,
)
from harvester.fractal_segments import (  # noqa: E402
    FractalRegion,
    detect_rectangle_pages,
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
    parser.add_argument("--tile-limit", type=int, default=64, help="Number of tiles per map axis to load (0 or all = full map).")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-area", type=int, default=64, help="Minimum component area in alpha pixels (8x8).")
    parser.add_argument("--min-footprint-px", type=int, default=8, help="Minimum bbox width and height for raw components (8x8 alpha pixels).")
    parser.add_argument("--max-regions-per-layer", type=int, default=5000)
    parser.add_argument("--strip-tiles", type=int, default=8, help="Horizontal strip width in tiles for full-map processing (0 = disable strips).")
    parser.add_argument("--strip-overlap-alpha-tiles", type=int, default=1, help="Overlap between horizontal strips in alpha tiles.")
    parser.add_argument("--skip-missing-maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Render contact sheets for the dedupe catalog after analysis.")
    parser.add_argument("--max-patterns", type=int, default=200)
    parser.add_argument("--max-per-pattern", type=int, default=6)
    parser.add_argument("--repeated-only", action="store_true")
    parser.add_argument("--near-dedupe", action="store_true", help="Run translation/mirror/rotation-invariant near-duplicate clustering after exact dedupe.")
    parser.add_argument("--near-dedupe-size", type=int, default=32, help="Normalized thumbnail edge length for near-dedupe.")
    parser.add_argument("--near-dedupe-radius", type=int, default=0, help="Hamming-radius for thumbnail matching (0 = exact invariant match).")
    parser.add_argument("--detect-rectangle-pages", action="store_true", help="Detect solid rectangular alpha pages separately from fractal components.")
    parser.add_argument("--rectangle-page-min-area", type=int, default=256, help="Minimum area for a rectangle page candidate.")
    parser.add_argument("--rectangle-page-min-extent", type=float, default=0.85, help="Minimum area / bbox_area for a rectangle page candidate.")
    parser.add_argument("--rectangle-page-max-aspect", type=float, default=8.0, help="Maximum aspect ratio for a rectangle page candidate.")
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

            use_full_map = int(args.tile_limit) <= 0
            tile_limit_tag = "full" if use_full_map else int(args.tile_limit)
            target_dir = out_root / f"{build}_{map_name}_tile{tile_limit_tag}"
            canvas_dir = target_dir / "canvas"
            segments_dir = target_dir / "segments_raw"
            canvas_dir.mkdir(parents=True, exist_ok=True)
            segments_dir.mkdir(parents=True, exist_ok=True)

            print(f"Analyzing build={build} map={map_name}", flush=True)
            root = zarr.open_group(store=zarr.storage.LocalStore(str(zarr_path), read_only=True), mode="r")
            if use_full_map:
                records = load_tile_records(zarr_path, build=str(build), map_name=str(map_name), require_alpha=True, tile_limit=None)
            else:
                records = load_tile_records(zarr_path, build=str(build), map_name=str(map_name), require_alpha=True, tile_limit=args.tile_limit)
            if not records:
                print("  no alpha-bearing records, skipped", flush=True)
                continue

            if use_full_map and int(args.strip_tiles) > 0:
                regions = process_map_in_strips(
                    root,
                    records,
                    layers,
                    canvas_dir,
                    segments_dir,
                    threshold=float(args.threshold),
                    min_area=int(args.min_area),
                    min_footprint_px=int(args.min_footprint_px),
                    max_regions_per_layer=int(args.max_regions_per_layer),
                    strip_tiles=int(args.strip_tiles),
                    overlap_alpha_tiles=int(args.strip_overlap_alpha_tiles),
                    no_overlay=bool(args.no_overlay),
                )
            else:
                from harvester.fractal_canvas import assemble_full_map_canvas

                layout, arrays, index_rows = assemble_full_map_canvas(root, records, layers=layers)
                write_canvas_outputs(canvas_dir, layout, arrays, index_rows)
                if not bool(args.no_overlay):
                    write_debug_overlay(canvas_dir, layout, arrays["alpha_256"], layer_slot=0)
                regions = segment_canvas_regions(
                    zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r"),
                    threshold=float(args.threshold),
                    min_area=int(args.min_area),
                    min_atomic_footprint_px=int(args.min_footprint_px),
                    curation_mode="raw",
                    max_regions_per_layer=int(args.max_regions_per_layer),
                )
                save_regions(segments_dir / "fractal_regions.parquet", regions)
                save_regions_jsonl(segments_dir / "fractal_regions.jsonl", regions)
                if not bool(args.no_overlay):
                    render_region_overlay(zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r"), regions, segments_dir / "overlays" / "fractal_regions_overlay.png")

            canvas = zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r")
            if bool(args.detect_rectangle_pages):
                rectangle_regions = detect_rectangle_pages(
                    canvas,
                    threshold=float(args.threshold),
                    min_area=int(args.rectangle_page_min_area),
                    min_extent=float(args.rectangle_page_min_extent),
                    max_aspect_ratio=float(args.rectangle_page_max_aspect),
                    max_regions_per_layer=int(args.max_regions_per_layer),
                )
                regions.extend(rectangle_regions)
                if not bool(args.no_overlay):
                    render_region_overlay(
                        canvas,
                        rectangle_regions,
                        segments_dir / "overlays" / "rectangle_pages_overlay.png",
                    )

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
    near_summary: dict[str, Any] | None = None
    if bool(args.near_dedupe):
        print("Running near-duplicate clustering...", flush=True)
        near_clusters = cluster_near_duplicates(
            all_fingerprints,
            canvas,
            threshold=float(args.threshold),
            size=int(args.near_dedupe_size),
            radius=int(args.near_dedupe_radius),
        )
        near_summary = write_near_dedupe_outputs(out_root / "dedupe" / "near", near_clusters)
        print(
            f"  near_clusters: {near_summary['cluster_count']} "
            f"dupe_clusters: {near_summary['duplicate_cluster_count']} "
            f"max_size: {near_summary['max_cluster_size']}",
            flush=True,
        )

    summary = {
        "target_count": int(len(target_summaries)),
        "targets": target_summaries,
        "dedupe": dedupe_summary,
        "near_dedupe": near_summary,
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


def process_map_in_strips(
    root: zarr.Group,
    records: list[CanvasTileRecord],
    layers: tuple[int, ...],
    canvas_dir: Path,
    segments_dir: Path,
    *,
    threshold: float,
    min_area: int,
    min_footprint_px: int,
    max_regions_per_layer: int,
    strip_tiles: int,
    overlap_alpha_tiles: int,
    no_overlay: bool,
) -> list[FractalRegion]:
    """Process a full map in horizontal strips to keep memory bounded."""
    full_layout = build_canvas_layout(records)
    full_group = create_chunked_canvas_group(canvas_dir, full_layout, layers=layers)
    _write_all_tiles(full_group, root, records, full_layout, layers, canvas_dir)

    min_tile_x = int(full_layout.min_tile_x)
    min_tile_y = int(full_layout.min_tile_y)
    width_tiles = int(full_layout.tile_count_x)

    all_regions: list[FractalRegion] = []
    strip_id = 0
    step = max(1, strip_tiles - overlap_alpha_tiles)
    for start_tile_x in range(0, width_tiles, step):
        end_tile_x = min(start_tile_x + strip_tiles, width_tiles)
        tile_x_lo = min_tile_x + start_tile_x
        tile_x_hi = min_tile_x + end_tile_x
        if start_tile_x > 0:
            tile_x_lo -= overlap_alpha_tiles
        strip_records = [r for r in records if tile_x_lo <= int(r.tile_x) < tile_x_hi]
        if not strip_records:
            continue
        strip_layout = build_canvas_layout(strip_records)
        print(f"  strip {strip_id}: x_tiles={tile_x_lo}..{tile_x_hi - 1} records={len(strip_records)}", flush=True)
        strip_group = _open_strip_view(full_group, strip_layout, full_layout, layers)
        regions = segment_canvas_regions(
            strip_group,
            threshold=threshold,
            min_area=min_area,
            min_atomic_footprint_px=min_footprint_px,
            curation_mode="raw",
            max_regions_per_layer=max_regions_per_layer,
        )
        offset_x = int((strip_layout.min_tile_x - min_tile_x) * ALPHA_TILE_SIZE)
        offset_y = int((strip_layout.min_tile_y - min_tile_y) * ALPHA_TILE_SIZE)
        for region in regions:
            region = _offset_region(region, offset_x, offset_y)
            provenance = dict(region.provenance)
            provenance["strip_id"] = strip_id
            provenance.setdefault("tile_x_range", [int(tile_x_lo), int(tile_x_hi)])
            region = replace(region, provenance=provenance)
            all_regions.append(region)
        if not no_overlay:
            overlay_path = segments_dir / "overlays" / f"fractal_regions_overlay_strip{strip_id:03d}.png"
            render_region_overlay(strip_group, regions, overlay_path)
        strip_id += 1

    all_regions = _dedupe_regions_across_strips(all_regions)
    save_regions(segments_dir / "fractal_regions.parquet", all_regions)
    save_regions_jsonl(segments_dir / "fractal_regions.jsonl", all_regions)
    if not no_overlay:
        full_canvas = zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r")
        render_region_overlay(full_canvas, all_regions, segments_dir / "overlays" / "fractal_regions_overlay.png")
    return all_regions


def _write_all_tiles(
    group: zarr.Group,
    root: zarr.Group,
    records: list[CanvasTileRecord],
    layout: CanvasLayout,
    layers: tuple[int, ...],
    canvas_dir: Path,
) -> None:
    index_rows: list[dict[str, Any]] = []
    for record in records:
        write_tile_to_canvas(group, record, layout, root, layer_indices=layers)
        ax, ay = alpha_origin(record, layout)
        mx, my = mcly_origin(record, layout)
        row = {
            "tile_id": int(record.tile_id),
            "map": str(record.map_name),
            "tile_x": int(record.tile_x),
            "tile_y": int(record.tile_y),
            "alpha_px_x": int(ax),
            "alpha_px_y": int(ay),
            "mcly_px_x": int(mx),
            "mcly_px_y": int(my),
        }
        index_rows.append(row)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(index_rows), canvas_dir / "canvas_index.parquet")


class _StripViewGroup:
    """A lightweight view of a cropped region of the full chunked canvas."""

    def __init__(self, full_group: zarr.Group, strip_layout: CanvasLayout, full_layout: CanvasLayout, layers: tuple[int, ...]) -> None:
        self._full = full_group
        self._strip_layout = strip_layout
        self._full_layout = full_layout
        self._layers = layers
        self.attrs = {"layout": _json_ready(asdict(strip_layout))}
        self._offset_alpha_x = int((strip_layout.min_tile_x - full_layout.min_tile_x) * ALPHA_TILE_SIZE)
        self._offset_alpha_y = int((strip_layout.min_tile_y - full_layout.min_tile_y) * ALPHA_TILE_SIZE)
        self._offset_height_x = int((strip_layout.min_tile_x - full_layout.min_tile_x) * (ALPHA_TILE_SIZE + 1))
        self._offset_height_y = int((strip_layout.min_tile_y - full_layout.min_tile_y) * (ALPHA_TILE_SIZE + 1))
        self._offset_mcly_x = int((strip_layout.min_tile_x - full_layout.min_tile_x) * 16)
        self._offset_mcly_y = int((strip_layout.min_tile_y - full_layout.min_tile_y) * 16)

    _KNOWN_KEYS = frozenset(
        {
            "alpha_256",
            "tile_id_256",
            "height_257",
            "tile_id_257",
            "normal_xyz",
            "mcly_texture_ids",
            "mcly_layer_mask",
            "tile_id_16",
            "alpha_layer_indices",
        }
    )

    def __contains__(self, key: object) -> bool:
        return str(key) in self._KNOWN_KEYS and key in self._full

    def __iter__(self):
        return iter(self._KNOWN_KEYS)

    def __getitem__(self, key: str):
        strip_layout = self._strip_layout
        if key == "alpha_256":
            ax = self._offset_alpha_x
            ay = self._offset_alpha_y
            return self._full["alpha_256"][ay : ay + int(strip_layout.alpha_shape[0]), ax : ax + int(strip_layout.alpha_shape[1])]
        if key == "tile_id_256":
            ax = self._offset_alpha_x
            ay = self._offset_alpha_y
            return self._full["tile_id_256"][ay : ay + int(strip_layout.alpha_shape[0]), ax : ax + int(strip_layout.alpha_shape[1])]
        if key == "height_257":
            hx = self._offset_height_x
            hy = self._offset_height_y
            return self._full["height_257"][hy : hy + int(strip_layout.height_shape[0]), hx : hx + int(strip_layout.height_shape[1])]
        if key == "normal_xyz":
            hx = self._offset_height_x
            hy = self._offset_height_y
            return self._full["normal_xyz"][hy : hy + int(strip_layout.height_shape[0]), hx : hx + int(strip_layout.height_shape[1]), :]
        if key == "tile_id_257":
            hx = self._offset_height_x
            hy = self._offset_height_y
            return self._full["tile_id_257"][hy : hy + int(strip_layout.height_shape[0]), hx : hx + int(strip_layout.height_shape[1])]
        if key == "mcly_texture_ids":
            mx = self._offset_mcly_x
            my = self._offset_mcly_y
            return self._full["mcly_texture_ids"][my : my + int(strip_layout.mcly_shape[0]), mx : mx + int(strip_layout.mcly_shape[1])]
        if key == "mcly_layer_mask":
            mx = self._offset_mcly_x
            my = self._offset_mcly_y
            return self._full["mcly_layer_mask"][my : my + int(strip_layout.mcly_shape[0]), mx : mx + int(strip_layout.mcly_shape[1])]
        if key == "tile_id_16":
            mx = self._offset_mcly_x
            my = self._offset_mcly_y
            return self._full["tile_id_16"][my : my + int(strip_layout.mcly_shape[0]), mx : mx + int(strip_layout.mcly_shape[1])]
        if key == "alpha_layer_indices":
            return self._full["alpha_layer_indices"][list(self._layers)]
        raise KeyError(key)


def _open_strip_view(full_group: zarr.Group, strip_layout: CanvasLayout, full_layout: CanvasLayout, layers: tuple[int, ...]) -> zarr.Group:
    """Return a zarr group whose arrays are a crop of the full canvas."""
    return _StripViewGroup(full_group, strip_layout, full_layout, layers)  # type: ignore[return-value]


def _offset_region(region: FractalRegion, dx: int, dy: int) -> FractalRegion:
    x, y, w, h = region.bbox_xywh
    provenance = dict(region.provenance)
    provenance["bbox_offset"] = [int(dx), int(dy)]
    return replace(region, bbox_xywh=(int(x + dx), int(y + dy), int(w), int(h)), provenance=provenance)


def _dedupe_regions_across_strips(regions: list[FractalRegion], *, iou_threshold: float = 0.5) -> list[FractalRegion]:
    """Remove near-duplicates that cross strip boundaries."""
    if not regions:
        return regions
    kept: list[FractalRegion] = []
    for region in regions:
        duplicate = False
        for other in kept:
            if _region_iou(region.bbox_xywh, other.bbox_xywh) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(region)
    return kept


def _region_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    x1 = max(ax0, bx0)
    y1 = max(ay0, by0)
    x2 = min(ax1, bx1)
    y2 = min(ay1, by1)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _json_ready(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _json_ready(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _available_maps(zarr_path: Path) -> list[str]:
    table = pq.read_table(zarr_path / "index.parquet", columns=["map"])
    return sorted({str(value.as_py()) for value in table.column("map")})


def resolve_target_maps(requested_maps: list[str], available_maps: list[str]) -> list[str]:
    if any(str(item).lower() == "all" for item in requested_maps):
        return list(available_maps)
    return list(requested_maps)


if __name__ == "__main__":
    main()
