"""Render a full-map overview with highly-repeated near-duplicate regions highlighted.

This produces a downscaled grayscale alpha composite of an entire map and overlays
bounding boxes for regions that belong to near-duplicate clusters with many members,
so you can see where repeated brush families are placed spatially.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw

_DEFAULT_ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "output" / "analysis" / "full-map-fractal-brush-library" / "full_map_Azeroth_0_5_3_3368_rectangles"

_CLUSTER_COLORS = [
    (255, 60, 60),
    (60, 255, 60),
    (60, 120, 255),
    (255, 200, 60),
    (200, 60, 255),
    (60, 255, 255),
    (255, 120, 180),
    (180, 255, 120),
    (255, 140, 60),
    (140, 60, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render full-map repeat-region overview.")
    parser.add_argument("--analysis-root", type=Path, default=_DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--build", type=str, default="0_5_3_3368")
    parser.add_argument("--map", type=str, default="Azeroth")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--max-clusters", type=int, default=20)
    parser.add_argument("--max-preview-side", type=int, default=4096)
    parser.add_argument("--bbox-width", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    output_path = Path(args.output_path) if args.output_path else analysis_root / f"repeat_overview_{args.build}_{args.map}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_dir = _find_canvas_dir(analysis_root, str(args.build), str(args.map))
    canvas = zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r")
    alpha = canvas["alpha_256"][:].astype(np.float32)
    base = np.clip(alpha.max(axis=2), 0.0, 1.0)

    scale, img = _downscale(base, int(args.max_preview_side))
    draw = ImageDraw.Draw(img)

    patterns, members = _load_near_clusters(analysis_root)
    selected = _select_clusters(patterns, int(args.min_cluster_size), int(args.max_clusters))
    if not selected:
        print("No clusters met the filter criteria.", flush=True)
        img.save(output_path)
        return

    color_by_cluster = {row["cluster_id"]: _CLUSTER_COLORS[idx % len(_CLUSTER_COLORS)] for idx, row in enumerate(selected)}

    for pattern in selected:
        cluster_id = pattern["cluster_id"]
        color = color_by_cluster[cluster_id]
        cluster_members = [m for m in members if str(m.get("cluster_id", "")) == cluster_id and str(m.get("build", "")) == str(args.build) and str(m.get("map_name", "")) == str(args.map)]
        for member in cluster_members:
            x, y, w, h = [int(v) for v in member.get("bbox_xywh", [0, 0, 1, 1])]
            sx0 = int(x * scale)
            sy0 = int(y * scale)
            sx1 = int((x + w) * scale)
            sy1 = int((y + h) * scale)
            draw.rectangle((sx0, sy0, sx1, sy1), outline=color, width=int(args.bbox_width))

    _draw_legend(draw, selected, color_by_cluster)
    img.save(output_path)

    summary = {
        "analysis_root": str(analysis_root),
        "build": str(args.build),
        "map": str(args.map),
        "output_path": str(output_path),
        "clusters_drawn": int(len(selected)),
        "regions_highlighted": int(sum(len([m for m in members if str(m.get("cluster_id", "")) == p["cluster_id"] and str(m.get("build", "")) == str(args.build) and str(m.get("map_name", "")) == str(args.map)]) for p in selected)),
    }
    (output_path.parent / f"{output_path.stem}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("Repeat-region overview complete", flush=True)
    print(f"  output_path: {output_path}", flush=True)
    print(f"  clusters_drawn: {summary['clusters_drawn']}", flush=True)
    print(f"  regions_highlighted: {summary['regions_highlighted']}", flush=True)


def _find_canvas_dir(analysis_root: Path, build: str, map_name: str) -> Path:
    for target_dir in sorted(analysis_root.glob("*_tile*")):
        if not target_dir.is_dir():
            continue
        parts = target_dir.name.split("_")
        tile_marker = -1
        for idx, part in enumerate(parts):
            if part.startswith("tile"):
                tile_marker = idx
                break
        if tile_marker <= 1:
            continue
        target_build = "_".join(parts[: tile_marker - 1])
        target_map = parts[tile_marker - 1]
        if target_build == build and target_map == map_name:
            canvas_dir = target_dir / "canvas"
            if canvas_dir.exists():
                return canvas_dir
    raise FileNotFoundError(f"No canvas target found for build={build} map={map_name} under {analysis_root}")


def _downscale(base: np.ndarray, max_preview_side: int) -> tuple[float, Image.Image]:
    h, w = base.shape
    scale = min(max_preview_side / max(1, w), max_preview_side / max(1, h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    gray = Image.fromarray((base * 255.0).astype(np.uint8), mode="L").resize((new_w, new_h), Image.Resampling.BILINEAR)
    return scale, gray.convert("RGB")


def _load_near_clusters(analysis_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    near_dir = analysis_root / "dedupe" / "near"
    patterns = pq.read_table(near_dir / "near_patterns.parquet").to_pylist()
    members = pq.read_table(near_dir / "near_pattern_members.parquet").to_pylist()
    return patterns, members


def _select_clusters(patterns: list[dict[str, Any]], min_size: int, max_clusters: int) -> list[dict[str, Any]]:
    selected = [row for row in patterns if int(row.get("member_count", 0)) >= min_size]
    selected.sort(key=lambda row: (-int(row.get("member_count", 0)), -int(row.get("area", 0)), str(row.get("cluster_id", ""))))
    return selected[:max_clusters]


def _draw_legend(draw: ImageDraw.ImageDraw, clusters: list[dict[str, Any]], color_by_cluster: dict[str, tuple[int, int, int]]) -> None:
    x, y = 12, 12
    for pattern in clusters:
        cluster_id = str(pattern.get("cluster_id", ""))
        color = color_by_cluster[cluster_id]
        members = int(pattern.get("member_count", 0))
        draw.rectangle((x, y, x + 18, y + 18), fill=color, outline=(255, 255, 255))
        draw.text((x + 24, y + 2), f"{cluster_id[:16]} x{members}", fill=(255, 255, 255))
        y += 22


if __name__ == "__main__":
    main()
