"""Run raw full-map component analysis and cross-target exact dedupe."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw, ImageFont

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
    segment_blocky_pastes,
    segment_canvas_regions,
    segment_macro_pastes,
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
    parser.add_argument("--visualize", action="store_true", help="Render contact sheets for the exact dedupe catalog after analysis.")
    parser.add_argument("--visualize-near", action="store_true", help="Render contact sheets for the near-duplicate cluster catalog after analysis.")
    parser.add_argument("--max-clusters", type=int, default=200)
    parser.add_argument("--max-per-cluster", type=int, default=6)
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
    parser.add_argument("--macro-pastes", action="store_true", help="Segment macro paste/scar objects via morphological closing instead of individual brush strokes.")
    parser.add_argument("--macro-close-radius", type=int, default=32, help="Morphological closing radius in alpha pixels (merges strokes within this distance).")
    parser.add_argument("--macro-min-area", type=int, default=4096, help="Minimum area for a macro paste candidate (64x64 alpha pixels = 4096).")
    parser.add_argument("--macro-min-footprint", type=int, default=64, help="Minimum bbox width/height for a macro paste candidate.")
    parser.add_argument("--macro-max-aspect", type=float, default=12.0, help="Maximum aspect ratio for a macro paste candidate.")
    parser.add_argument("--macro-downsample-factor", type=int, default=8, help="Max-pool alpha by this factor before macro closing for full-map speed.")
    parser.add_argument("--blocky-pastes", action="store_true", help="Segment middle-scale dense blocky paste/scar children instead of giant parent zones.")
    parser.add_argument("--block-size", type=int, default=16, help="Alpha-pixel block size for blocky paste coverage grid.")
    parser.add_argument("--block-min-coverage", type=float, default=0.08, help="Minimum painted coverage per block for blocky paste segmentation.")
    parser.add_argument("--block-close-radius", type=int, default=1, help="Closing radius in block units for blocky paste segmentation.")
    parser.add_argument("--block-min-area", type=int, default=512, help="Minimum original alpha pixels for a blocky paste region.")
    parser.add_argument("--block-min-footprint", type=int, default=16, help="Minimum bbox width/height in alpha pixels for blocky paste regions.")
    parser.add_argument("--block-max-footprint", type=int, default=0, help="Optional maximum bbox width/height in alpha pixels for blocky paste regions (0 = unlimited).")
    parser.add_argument("--block-max-aspect", type=float, default=12.0, help="Maximum aspect ratio for blocky paste regions.")
    parser.add_argument("--visualize-macro", action="store_true", help="Render macro paste overview/contact sheets even when --no-overlay is set.")
    parser.add_argument("--visualize-composite-signal", action="store_true", help="Render V18-style composite hard-region overview for macro paste review.")
    parser.add_argument("--macro-max-preview-side", type=int, default=4096, help="Maximum side length for macro full-map overview image.")
    parser.add_argument("--macro-max-contact-regions", type=int, default=120, help="Maximum macro paste regions to include in contact sheets per target.")
    parser.add_argument("--macro-contact-per-page", type=int, default=24, help="Macro paste contact-sheet rows per page.")
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
                    skip_raw_segments=bool(args.macro_pastes) or bool(args.blocky_pastes),
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
            if bool(args.macro_pastes) or bool(args.blocky_pastes):
                if bool(args.blocky_pastes):
                    regions = segment_blocky_pastes(
                        canvas,
                        threshold=float(args.threshold),
                        block_size=int(args.block_size),
                        min_block_coverage=float(args.block_min_coverage),
                        block_close_radius=int(args.block_close_radius),
                        min_area=int(args.block_min_area),
                        min_footprint_px=int(args.block_min_footprint),
                        max_footprint_px=int(args.block_max_footprint) if int(args.block_max_footprint) > 0 else None,
                        max_aspect_ratio=float(args.block_max_aspect),
                        max_regions_per_layer=int(args.max_regions_per_layer),
                    )
                else:
                    regions = segment_macro_pastes(
                        canvas,
                        threshold=float(args.threshold),
                        close_radius=int(args.macro_close_radius),
                        min_area=int(args.macro_min_area),
                        min_footprint_px=int(args.macro_min_footprint),
                        max_aspect_ratio=float(args.macro_max_aspect),
                        max_regions_per_layer=int(args.max_regions_per_layer),
                        downsample_factor=int(args.macro_downsample_factor),
                    )
                save_regions(segments_dir / "fractal_regions.parquet", regions)
                save_regions_jsonl(segments_dir / "fractal_regions.jsonl", regions)
                if bool(args.visualize_macro) or not bool(args.no_overlay):
                    review_dir = segments_dir / "macro_review"
                    if bool(args.visualize_composite_signal):
                        _write_composite_signal_review(
                            root,
                            records,
                            canvas,
                            regions,
                            review_dir,
                            max_preview_side=int(args.macro_max_preview_side),
                        )
                    _write_macro_review(
                        canvas,
                        regions,
                        review_dir,
                        max_preview_side=int(args.macro_max_preview_side),
                        max_contact_regions=int(args.macro_max_contact_regions),
                        contact_per_page=int(args.macro_contact_per_page),
                        threshold=float(args.threshold),
                    )
                print(f"  {'blocky_pastes' if bool(args.blocky_pastes) else 'macro_pastes'}={len(regions)}", flush=True)
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
    if bool(args.visualize_near):
        _run_near_visualizer(out_root, args)

    if target_summaries:
        _write_analysis_index(out_root, summary)


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
    print("Rendering exact-pattern contact sheets...", flush=True)
    subprocess.run(cmd, check=True)


def _run_near_visualizer(out_root: Path, args: argparse.Namespace) -> None:
    import subprocess

    script = Path(__file__).resolve().parent / "visualize_fractal_near_patterns.py"
    contact_dir = out_root / "contact_sheets_near"
    cmd = [
        sys.executable,
        str(script),
        "--analysis-root",
        str(out_root),
        "--output-dir",
        str(contact_dir),
        "--max-clusters",
        str(int(args.max_clusters)),
        "--max-per-cluster",
        str(int(args.max_per_cluster)),
    ]
    if bool(args.repeated_only):
        cmd.append("--repeated-only")
    print("Rendering near-duplicate cluster contact sheets...", flush=True)
    subprocess.run(cmd, check=True)


_MACRO_LAYER_COLORS = {
    0: (180, 180, 180),
    1: (80, 170, 255),
    2: (100, 230, 120),
    3: (255, 185, 70),
}


def _write_macro_review(
    canvas: zarr.Group,
    regions: list[FractalRegion],
    output_dir: Path,
    *,
    max_preview_side: int,
    max_contact_regions: int,
    contact_per_page: int,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(regions, key=lambda item: item.area, reverse=True)[: max(0, int(max_contact_regions))]
    overview_path = output_dir / "macro_paste_overview.png"
    scale, overview = _render_macro_overview(canvas, selected, overview_path, max_preview_side=max_preview_side)
    pages = _render_macro_contact_sheets(canvas, selected, output_dir, contact_per_page=contact_per_page, threshold=threshold)
    summary = {
        "region_count": int(len(regions)),
        "regions_rendered": int(len(selected)),
        "overview_path": str(overview_path),
        "overview_scale": float(scale),
        "overview_size": list(overview.size),
        "contact_pages": [str(path) for path in pages],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_macro_review_index(output_dir, overview_path, pages, selected, summary)


def _render_macro_overview(
    canvas: zarr.Group,
    regions: list[FractalRegion],
    output_path: Path,
    *,
    max_preview_side: int,
) -> tuple[float, Image.Image]:
    scale, image = _stream_alpha_overview(canvas, max_preview_side=max_preview_side)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, region in enumerate(regions, start=1):
        x, y, w, h = region.bbox_xywh
        sx0 = int(round(x * scale))
        sy0 = int(round(y * scale))
        sx1 = max(sx0 + 1, int(round((x + w) * scale)))
        sy1 = max(sy0 + 1, int(round((y + h) * scale)))
        color = _MACRO_LAYER_COLORS.get(int(region.layer_idx), (255, 255, 255))
        width = 4 if idx <= 20 else 2
        draw.rectangle((sx0, sy0, sx1, sy1), outline=color, width=width)
        if idx <= 60:
            label = f"{idx}:L{region.layer_idx} {w}x{h}"
            draw.text((sx0 + 3, sy0 + 3), label, fill=color, font=font)
    _draw_macro_legend(draw, font, image.size, regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return scale, image


def _stream_alpha_overview(canvas: zarr.Group, *, max_preview_side: int) -> tuple[float, Image.Image]:
    alpha = canvas["alpha_256"]
    h, w = int(alpha.shape[0]), int(alpha.shape[1])
    factor = max(1, int(np.ceil(max(h, w) / max(1, int(max_preview_side)))))
    out_h = (h + factor - 1) // factor
    out_w = (w + factor - 1) // factor
    out = np.zeros((out_h, out_w), dtype=np.float32)
    block_rows = max(factor, factor * max(1, 512 // factor))
    for y0 in range(0, h, block_rows):
        y1 = min(h, y0 + block_rows)
        block = alpha[y0:y1, :, :].astype(np.float32)
        composite = np.clip(block.max(axis=2), 0.0, 1.0)
        pad_h = (-composite.shape[0]) % factor
        pad_w = (-composite.shape[1]) % factor
        if pad_h or pad_w:
            composite = np.pad(composite, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        pooled = composite.reshape(composite.shape[0] // factor, factor, composite.shape[1] // factor, factor).max(axis=(1, 3))
        ds_y0 = y0 // factor
        out[ds_y0 : ds_y0 + pooled.shape[0], : pooled.shape[1]] = pooled[:, :out_w]
    image = Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").convert("RGB")
    return 1.0 / float(factor), image


def _write_composite_signal_review(
    root: zarr.Group,
    records: list[CanvasTileRecord],
    canvas: zarr.Group,
    regions: list[FractalRegion],
    output_dir: Path,
    *,
    max_preview_side: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = build_canvas_layout(records)
    signal, scale = _build_composite_signal_overview(root, records, layout, max_preview_side=max_preview_side)
    image = Image.fromarray((np.clip(signal, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, region in enumerate(sorted(regions, key=lambda item: item.area, reverse=True), start=1):
        x, y, w, h = region.bbox_xywh
        sx0 = int(round(x * scale))
        sy0 = int(round(y * scale))
        sx1 = max(sx0 + 1, int(round((x + w) * scale)))
        sy1 = max(sy0 + 1, int(round((y + h) * scale)))
        color = _MACRO_LAYER_COLORS.get(int(region.layer_idx), (255, 255, 255))
        draw.rectangle((sx0, sy0, sx1, sy1), outline=color, width=3 if idx <= 20 else 2)
        if idx <= 60:
            draw.text((sx0 + 3, sy0 + 3), f"{idx}:L{region.layer_idx}", fill=color, font=font)
    _draw_composite_legend(draw, font, image.size, len(records), len(regions))
    output_path = output_dir / "composite_signal_overview.png"
    image.save(output_path)
    (output_dir / "composite_signal_summary.json").write_text(
        json.dumps(
            {
                "output_path": str(output_path),
                "tile_count": int(len(records)),
                "region_count": int(len(regions)),
                "overview_scale": float(scale),
                "overview_size": list(image.size),
                "signals": ["height_gradient", "normal_gradient", "alpha_gradient", "mcly_gradient", "normal_mask", "object_masks", "liquid_mask"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _build_composite_signal_overview(
    root: zarr.Group,
    records: list[CanvasTileRecord],
    layout: CanvasLayout,
    *,
    max_preview_side: int,
) -> tuple[np.ndarray, float]:
    alpha_h, alpha_w = layout.alpha_shape
    factor = max(1, int(np.ceil(max(alpha_h, alpha_w) / max(1, int(max_preview_side)))))
    out_h = (alpha_h + factor - 1) // factor
    out_w = (alpha_w + factor - 1) // factor
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for record in records:
        tile_id = int(record.tile_id)
        score = _tile_composite_signal(root, tile_id)
        if score is None:
            continue
        tile_img = Image.fromarray((np.clip(score, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        tile_w = max(1, int(np.ceil(ALPHA_TILE_SIZE / factor)))
        tile_img = tile_img.resize((tile_w, tile_w), Image.Resampling.BILINEAR)
        tile_arr = np.asarray(tile_img, dtype=np.float32) / 255.0
        ax, ay = alpha_origin(record, layout)
        dx, dy = int(ax // factor), int(ay // factor)
        y1 = min(out_h, dy + tile_arr.shape[0])
        x1 = min(out_w, dx + tile_arr.shape[1])
        out[dy:y1, dx:x1] = np.maximum(out[dy:y1, dx:x1], tile_arr[: y1 - dy, : x1 - dx])
    return out, 1.0 / float(factor)


def _tile_composite_signal(root: zarr.Group, tile_id: int) -> np.ndarray | None:
    if "height_257" not in root or "alpha_256" not in root:
        return None
    height = root["height_257"][tile_id].astype(np.float32)
    alpha = root["alpha_256"][tile_id].astype(np.float32)
    height_grad = _crop_257_to_256(_gradient_magnitude(height))
    alpha_grad = _gradient_magnitude(alpha.max(axis=2))
    normal_grad = np.zeros((256, 256), dtype=np.float32)
    if "normal_xyz" in root:
        normals = root["normal_xyz"][tile_id].astype(np.float32)
        normal_grad = _crop_257_to_256(np.mean([_gradient_magnitude(normals[:, :, axis]) for axis in range(3)], axis=0))
    mcly_grad = np.zeros((256, 256), dtype=np.float32)
    if "mcly_layer_mask" in root:
        mcly_any = root["mcly_layer_mask"][tile_id].astype(np.float32).max(axis=2)
        mcly_grad = _gradient_magnitude(np.kron(mcly_any, np.ones((16, 16), dtype=np.float32)))[:256, :256]
    transition = np.maximum(alpha_grad, mcly_grad)
    base_mask = np.ones((256, 256), dtype=np.float32)
    if "normal_mask" in root:
        base_mask *= _crop_257_to_256(root["normal_mask"][tile_id].astype(np.float32))
    if "liquid_mask" in root:
        base_mask *= 1.0 - 0.85 * np.clip(_as_256(root["liquid_mask"][tile_id].astype(np.float32)), 0.0, 1.0)
    object_mask = np.zeros((256, 256), dtype=np.float32)
    for key in ("mddf_mask", "modf_mask", "object_mask", "object_precise_mask"):
        if key in root:
            object_mask = np.maximum(object_mask, np.clip(_as_256(root[key][tile_id].astype(np.float32)), 0.0, 1.0))
    base_mask *= 1.0 - 0.75 * object_mask
    score = (0.50 * _masked_norm(height_grad, base_mask)) + (0.25 * _masked_norm(normal_grad, base_mask)) + (0.25 * _masked_norm(transition, base_mask))
    score = np.clip(score * np.clip(base_mask, 0.0, 1.0), 0.0, 4.0)
    max_value = float(score.max())
    return (score / max_value).astype(np.float32) if max_value > 1e-6 else score.astype(np.float32)


def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x, axis=1, append=x[:, -1:])
    dy = np.diff(x, axis=0, append=x[-1:, :])
    return np.sqrt((dx * dx) + (dy * dy) + 1e-8).astype(np.float32)


def _crop_257_to_256(x: np.ndarray) -> np.ndarray:
    return x[:256, :256].astype(np.float32, copy=False)


def _as_256(x: np.ndarray) -> np.ndarray:
    if x.shape[:2] == (256, 256):
        return x.astype(np.float32, copy=False)
    return x[:256, :256].astype(np.float32, copy=False)


def _masked_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denom = float((x * mask).sum() / max(1e-6, float(mask.sum())))
    if denom <= 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / denom, 0.0, 4.0).astype(np.float32)


def _render_macro_contact_sheets(
    canvas: zarr.Group,
    regions: list[FractalRegion],
    output_dir: Path,
    *,
    contact_per_page: int,
    threshold: float,
) -> list[Path]:
    pages: list[Path] = []
    per_page = max(1, int(contact_per_page))
    columns = 4
    cell_w, cell_h = 260, 238
    legend_h = 82
    font = ImageFont.load_default()
    for page_idx, start in enumerate(range(0, len(regions), per_page), start=1):
        page_regions = regions[start : start + per_page]
        rows = max(1, int(np.ceil(len(page_regions) / columns)))
        image = Image.new("RGB", (columns * cell_w, legend_h + rows * cell_h), color=(14, 14, 18))
        draw = ImageDraw.Draw(image)
        _draw_contact_legend(draw, font, image.size)
        for idx, region in enumerate(page_regions, start=start + 1):
            local = idx - start - 1
            col = local % columns
            row = local // columns
            cell = _render_macro_region_cell(canvas, region, rank=idx, threshold=threshold, cell_size=180)
            x = col * cell_w
            y = legend_h + row * cell_h
            image.paste(cell, (x + 8, y + 8))
        page_path = output_dir / f"macro_paste_contact_sheet_{page_idx:03d}.png"
        image.save(page_path)
        pages.append(page_path)
    return pages


def _render_macro_region_cell(canvas: zarr.Group, region: FractalRegion, *, rank: int, threshold: float, cell_size: int) -> Image.Image:
    x, y, w, h = region.bbox_xywh
    step = max(1, int(np.ceil(max(w, h) / 512)))
    alpha = canvas["alpha_256"][y : y + h : step, x : x + w : step, int(region.layer_slot)].astype(np.float32)
    if alpha.size == 0:
        alpha = np.zeros((1, 1), dtype=np.float32)
    alpha = np.where(alpha > float(threshold), alpha, 0.0)
    alpha_img = Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    alpha_img.thumbnail((cell_size, cell_size), Image.Resampling.BILINEAR)
    cell = Image.new("RGB", (244, 222), color=(20, 20, 24))
    px = (cell_size - alpha_img.width) // 2 + 8
    py = (cell_size - alpha_img.height) // 2 + 8
    cell.paste(Image.merge("RGB", (alpha_img, alpha_img, alpha_img)), (px, py))
    draw = ImageDraw.Draw(cell)
    font = ImageFont.load_default()
    color = _MACRO_LAYER_COLORS.get(int(region.layer_idx), (255, 255, 255))
    draw.rectangle((0, 0, cell.width - 1, cell.height - 1), outline=color, width=3)
    extent = region.provenance.get("extent") if isinstance(region.provenance, dict) else None
    label_lines = [
        f"#{rank} {region.region_id[:12]}",
        f"{region.build} / {region.map_name}",
        f"L{region.layer_idx} box {w}x{h} area {region.area}",
        f"xy {x},{y} tiles {region.tile_coverage_count} extent {extent}",
    ]
    draw.rectangle((4, 184, cell.width - 5, cell.height - 5), fill=(0, 0, 0))
    for line_idx, text in enumerate(label_lines):
        draw.text((8, 187 + line_idx * 8), text, fill=color if line_idx == 2 else (235, 235, 235), font=font)
    return cell


def _draw_macro_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, image_size: tuple[int, int], regions: list[FractalRegion]) -> None:
    width, _height = image_size
    draw.rectangle((0, 0, min(width - 1, 860), 72), fill=(0, 0, 0), outline=(80, 80, 88))
    draw.text((10, 8), "Spec 076 Macro/Blocky Paste-Scar Overview", fill=(245, 245, 245), font=font)
    draw.text((10, 25), "Boxes are paste/scar regions over max alpha composite. Numbers match contact sheets.", fill=(210, 210, 215), font=font)
    draw.text((10, 42), f"Rendered regions: {len(regions)}. Border colors show alpha layer.", fill=(255, 205, 120), font=font)
    x = 620
    for layer_idx, color in _MACRO_LAYER_COLORS.items():
        draw.rectangle((x, 18, x + 16, 34), fill=color, outline=(255, 255, 255))
        draw.text((x + 20, 20), f"L{layer_idx}", fill=(235, 235, 235), font=font)
        x += 56


def _draw_composite_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, image_size: tuple[int, int], tile_count: int, region_count: int) -> None:
    width, _height = image_size
    draw.rectangle((0, 0, min(width - 1, 1040), 72), fill=(0, 0, 0), outline=(80, 80, 88))
    draw.text((10, 8), "Spec 076 Composite Hard-Region Overview", fill=(245, 245, 245), font=font)
    draw.text((10, 25), "Base image = height/normal gradients + alpha/MCLY transitions, masked by normal/object/liquid signals.", fill=(210, 210, 215), font=font)
    draw.text((10, 42), f"Tiles: {tile_count}. Macro boxes: {region_count}. Use this to judge whether alpha-only macro boxes match terrain-art signal.", fill=(255, 205, 120), font=font)


def _draw_contact_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, image_size: tuple[int, int]) -> None:
    width, _height = image_size
    draw.rectangle((0, 0, width - 1, 76), fill=(20, 20, 24), outline=(70, 70, 78))
    draw.text((10, 8), "Spec 076 Macro/Blocky Paste-Scar Contact Sheet", fill=(245, 245, 245), font=font)
    draw.text((10, 26), "Each cell is one paste/scar alpha crop. This is block/macro scale, not raw brush-dot scale.", fill=(205, 205, 210), font=font)
    draw.text((10, 44), "Review box size, location, layer, tile span, and whether grouping is too coarse/fine.", fill=(255, 205, 120), font=font)


def _write_macro_review_index(output_dir: Path, overview_path: Path, pages: list[Path], regions: list[FractalRegion], summary: dict[str, Any]) -> None:
    rows = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Paste/Scar Review</title></head><body>",
        "<h1>Spec 076 Macro/Blocky Paste-Scar Review</h1>",
        f"<p>Regions rendered: {int(summary.get('regions_rendered', 0))} / {int(summary.get('region_count', 0))}</p>",
        f"<p><a href='{html.escape(overview_path.name)}'>Macro overview PNG</a></p>",
        f"<img src='{html.escape(overview_path.name)}' style='max-width:100%; image-rendering: pixelated;'>",
    ]
    composite_path = output_dir / "composite_signal_overview.png"
    if composite_path.exists():
        rows.extend(
            [
                "<h2>Composite Signal Overview</h2>",
                f"<p><a href='{html.escape(composite_path.name)}'>Composite hard-region overview PNG</a></p>",
                f"<img src='{html.escape(composite_path.name)}' style='max-width:100%; image-rendering: pixelated;'>",
            ]
        )
    rows.extend(
        [
        "<h2>Contact Sheets</h2>",
        ]
    )
    for page in pages:
        rows.append(f"<h3>{html.escape(page.name)}</h3><img src='{html.escape(page.name)}' style='max-width:100%; image-rendering: pixelated;'>")
    rows.append("<h2>Largest Regions</h2><table border='1' cellpadding='4'>")
    rows.append("<tr><th>Rank</th><th>ID</th><th>Layer</th><th>BBox</th><th>Area</th><th>Tiles</th><th>Extent</th></tr>")
    for rank, region in enumerate(regions[:100], start=1):
        extent = region.provenance.get("extent") if isinstance(region.provenance, dict) else ""
        rows.append(
            f"<tr><td>{rank}</td><td>{html.escape(region.region_id)}</td><td>{region.layer_idx}</td>"
            f"<td>{html.escape(str(region.bbox_xywh))}</td><td>{region.area}</td><td>{region.tile_coverage_count}</td><td>{extent}</td></tr>"
        )
    rows.append("</table></body></html>")
    (output_dir / "index.html").write_text("\n".join(rows), encoding="utf-8")


def _write_analysis_index(out_root: Path, summary: dict[str, Any]) -> None:
    """Write a human-readable HTML index for an --maps all analysis run."""
    targets = summary.get("targets", [])
    dedupe = summary.get("dedupe", {})
    near = summary.get("near_dedupe") or {}
    rows: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'><title>076 Raw Analysis Index</title></head><body>",
        "<h1>Spec 076 Full-Map Raw Component Analysis</h1>",
        f"<p>Output root: <code>{html.escape(str(out_root))}</code></p>",
        "<h2>Cross-Map Summary</h2><ul>",
        f"<li>Targets processed: {int(summary.get('target_count', 0))}</li>",
        f"<li>Raw components: {int(dedupe.get('raw_component_count', 0))}</li>",
        f"<li>Exact patterns: {int(dedupe.get('exact_pattern_count', 0))}</li>",
        f"<li>Exact duplicates: {int(dedupe.get('duplicate_pattern_count', 0))}</li>",
    ]
    if near:
        rows.extend(
            [
                f"<li>Near clusters: {int(near.get('cluster_count', 0))}</li>",
                f"<li>Near duplicate clusters: {int(near.get('duplicate_cluster_count', 0))}</li>",
                f"<li>Near max cluster size: {int(near.get('max_cluster_size', 0))}</li>",
            ]
        )
    rows.append("</ul>")

    rows.append("<h2>Catalogs</h2><ul>")
    rows.append(f"<li><a href='{html.escape(str(out_root / 'dedupe' / 'exact_patterns.parquet'))}'>exact_patterns.parquet</a></li>")
    if near:
        rows.append(f"<li><a href='{html.escape(str(out_root / 'dedupe' / 'near' / 'near_patterns.parquet'))}'>near_patterns.parquet</a></li>")
    if (out_root / "contact_sheets").exists():
        rows.append("<li><a href='contact_sheets/index.html'>Exact-pattern contact sheets</a></li>")
    if (out_root / "contact_sheets_near").exists():
        rows.append("<li><a href='contact_sheets_near/index.html'>Near-duplicate cluster contact sheets</a></li>")
    rows.append("</ul>")

    rows.append("<h2>Per-Map Artifacts</h2><table border='1' cellpadding='6'>")
    rows.append("<tr><th>Build</th><th>Map</th><th>Tiles</th><th>Regions</th><th>Curation counts</th><th>Canvas</th><th>Overlays</th></tr>")
    for target in targets:
        build = html.escape(str(target.get("build", "")))
        map_name = html.escape(str(target.get("map", "")))
        tile_count = int(target.get("tile_count", 0))
        region_count = int(target.get("region_count", 0))
        curation_counts = html.escape(json.dumps(target.get("curation_counts", {}), sort_keys=True))
        canvas_dir = Path(str(target.get("canvas_dir", "")))
        segments_dir = Path(str(target.get("segments_dir", "")))
        overlay_dir = segments_dir / "overlays"
        macro_review_dir = segments_dir / "macro_review"
        links: list[str] = []
        if overlay_dir.exists():
            for png in sorted(overlay_dir.glob("*.png")):
                rel = html.escape(str(png.relative_to(out_root).as_posix()))
                links.append(f"<a href='{rel}'>{html.escape(png.name)}</a>")
        if macro_review_dir.exists():
            rel = html.escape(str((macro_review_dir / "index.html").relative_to(out_root).as_posix()))
            links.append(f"<a href='{rel}'>macro_review</a>")
        rows.append(
            f"<tr><td>{build}</td><td>{map_name}</td><td>{tile_count}</td><td>{region_count}</td>"
            f"<td>{curation_counts}</td><td><code>{html.escape(str(canvas_dir))}</code></td>"
            f"<td>{'<br>'.join(links)}</td></tr>"
        )
    rows.append("</table></body></html>")

    (out_root / "index.html").write_text("\n".join(rows), encoding="utf-8")


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
    skip_raw_segments: bool = False,
) -> list[FractalRegion]:
    """Process a full map in horizontal strips to keep memory bounded."""
    full_layout = build_canvas_layout(records)
    full_group = create_chunked_canvas_group(canvas_dir, full_layout, layers=layers, aux_arrays=False)
    _write_all_tiles(full_group, root, records, full_layout, layers, canvas_dir)

    min_tile_x = int(full_layout.min_tile_x)
    min_tile_y = int(full_layout.min_tile_y)
    width_tiles = int(full_layout.tile_count_x)

    all_regions: list[FractalRegion] = []
    if skip_raw_segments:
        save_regions(segments_dir / "fractal_regions.parquet", all_regions)
        save_regions_jsonl(segments_dir / "fractal_regions.jsonl", all_regions)
        return all_regions

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
