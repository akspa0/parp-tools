"""Render contact sheets for spec 076 near-duplicate cluster output."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw, ImageFont

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEFAULT_ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "output" / "analysis" / "full-map-fractal-brush-library" / "full_map_Azeroth_0_5_3_3368_rectangles"

_LAYER_COLORS = {
    0: (145, 145, 145),
    1: (92, 177, 255),
    2: (124, 224, 123),
    3: (255, 184, 76),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render near-duplicate cluster contact sheets from analyze_fractal_raw_components output.")
    parser.add_argument("--analysis-root", type=Path, default=_DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-clusters", type=int, default=200)
    parser.add_argument("--max-per-cluster", type=int, default=8)
    parser.add_argument("--clusters-per-page", type=int, default=20)
    parser.add_argument("--cell-size", type=int, default=128)
    parser.add_argument("--label-width", type=int, default=260)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-members", type=int, default=1)
    parser.add_argument("--repeated-only", action="store_true", help="Only render clusters with member_count > 1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_root / "contact_sheets_near"
    output_dir.mkdir(parents=True, exist_ok=True)

    near_dir = analysis_root / "dedupe" / "near"
    clusters = _read_rows(near_dir / "near_patterns.parquet")
    members = _read_rows(near_dir / "near_pattern_members.parquet")
    target_index = _target_index(analysis_root)
    by_cluster = _members_by_cluster(members, max_per_cluster=int(args.max_per_cluster))

    min_members = max(1, int(args.min_members))
    selected = [row for row in clusters if int(row.get("member_count", 0)) >= min_members]
    if bool(args.repeated_only):
        selected = [row for row in selected if int(row.get("member_count", 0)) > 1]
    selected.sort(key=lambda row: (-int(row.get("member_count", 0)), -int(row.get("area", 0)), str(row.get("cluster_id", ""))))
    selected = selected[: max(1, int(args.max_clusters))]

    cache = _CanvasCache(target_index)
    pages: list[Path] = []
    for page_idx, start in enumerate(range(0, len(selected), max(1, int(args.clusters_per_page))), start=1):
        page_rows = selected[start : start + int(args.clusters_per_page)]
        page_path = output_dir / f"near_patterns_page_{page_idx:03d}.png"
        _draw_page(page_rows, by_cluster, cache, page_path, args)
        pages.append(page_path)
    cache.close()

    _write_index(output_dir, pages, selected, analysis_root)
    summary = {
        "analysis_root": str(analysis_root),
        "output_dir": str(output_dir),
        "clusters_rendered": int(len(selected)),
        "pages": [str(path) for path in pages],
        "repeated_only": bool(args.repeated_only),
        "min_members": int(min_members),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("Near-duplicate cluster contact sheets complete", flush=True)
    print(f"  analysis_root: {analysis_root}", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)
    print(f"  clusters_rendered: {len(selected)}", flush=True)
    print(f"  pages: {len(pages)}", flush=True)


class _CanvasCache:
    def __init__(self, target_index: dict[tuple[str, str], Path]) -> None:
        self.target_index = target_index
        self._roots: dict[tuple[str, str], zarr.Group] = {}

    def canvas(self, build: str, map_name: str) -> zarr.Group:
        key = (str(build), str(map_name))
        root = self._roots.get(key)
        if root is None:
            canvas_dir = self.target_index.get(key)
            if canvas_dir is None:
                raise FileNotFoundError(f"No canvas target for build={build} map={map_name}")
            path = canvas_dir / "canvas.zarr"
            root = zarr.open_group(str(path), mode="r")
            self._roots[key] = root
        return root

    def close(self) -> None:
        self._roots.clear()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _target_index(analysis_root: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
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
        build = "_".join(parts[: tile_marker - 1])
        map_name = parts[tile_marker - 1]
        canvas_dir = target_dir / "canvas"
        if canvas_dir.exists():
            out[(build, map_name)] = canvas_dir
    if out:
        return out
    summary_path = analysis_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for target in summary.get("targets", []):
            build = str(target.get("build", ""))
            map_name = str(target.get("map", ""))
            canvas_dir = Path(str(target.get("canvas_dir", "")))
            if build and map_name and canvas_dir.exists():
                out[(build, map_name)] = canvas_dir
    return out


def _members_by_cluster(rows: list[dict[str, Any]], *, max_per_cluster: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: (str(item.get("build", "")), str(item.get("map_name", "")), int(item.get("area", 0)), str(item.get("region_id", "")))):
        cluster_id = str(row.get("cluster_id", ""))
        if not cluster_id:
            continue
        bucket = out.setdefault(cluster_id, [])
        if len(bucket) < int(max_per_cluster):
            bucket.append(row)
    return out


def _draw_page(
    clusters: list[dict[str, Any]],
    by_cluster: dict[str, list[dict[str, Any]]],
    cache: _CanvasCache,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    row_h = int(args.cell_size) + int(args.padding)
    legend_h = 76
    width = int(args.label_width) + (int(args.cell_size) + int(args.padding)) * int(args.max_per_cluster)
    height = legend_h + max(1, len(clusters)) * row_h + int(args.padding)
    image = Image.new("RGB", (width, height), color=(10, 10, 12))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    _draw_legend(draw, width, font)

    for row_idx, cluster in enumerate(clusters):
        y = legend_h + int(args.padding) + row_idx * row_h
        cluster_id = str(cluster.get("cluster_id", ""))
        label = "\n".join(
            [
                cluster_id,
                f"members {int(cluster.get('member_count', 0))} builds {int(cluster.get('build_count', 0))}",
                f"maps {int(cluster.get('map_count', 0))} layers {cluster.get('layer_indices', [])}",
                f"box {int(cluster.get('crop_w', 0))}x{int(cluster.get('crop_h', 0))} area {int(cluster.get('area', 0))}",
            ]
        )
        draw.text((8, y + 8), label, fill=(230, 230, 230), font=font)
        for col_idx, member in enumerate(by_cluster.get(cluster_id, [])):
            x = int(args.label_width) + col_idx * (int(args.cell_size) + int(args.padding))
            try:
                cell = _render_member(member, cache, int(args.cell_size), threshold=float(args.threshold))
            except Exception as exc:
                cell = Image.new("RGB", (int(args.cell_size), int(args.cell_size)), color=(80, 20, 20))
                ImageDraw.Draw(cell).text((6, 6), f"ERR\n{type(exc).__name__}", fill=(255, 220, 220))
            image.paste(cell, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _render_member(row: dict[str, Any], cache: _CanvasCache, cell_size: int, *, threshold: float) -> Image.Image:
    build = str(row.get("build", ""))
    map_name = str(row.get("map_name", ""))
    layer_slot = int(row.get("layer_slot", 0))
    layer_idx = int(row.get("layer_idx", layer_slot))
    x, y, w, h = [int(value) for value in row.get("bbox_xywh", [0, 0, 1, 1])]
    canvas = cache.canvas(build, map_name)
    alpha = canvas["alpha_256"][y : y + h, x : x + w, layer_slot].astype(np.float32)
    if alpha.size == 0:
        alpha = np.zeros((1, 1), dtype=np.float32)
    alpha = np.where(alpha > float(threshold), alpha, 0.0)

    pad = max(2, int(cell_size * 0.08))
    boxed = np.pad(alpha, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    src_h, src_w = boxed.shape
    max_draw = max(8, cell_size - 18)
    scale = min(max_draw / max(1, src_w), max_draw / max(1, src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    gray = Image.fromarray((np.clip(boxed, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    gray = gray.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
    cell = Image.new("RGB", (cell_size, cell_size), color=(18, 18, 20))
    cell.paste(Image.merge("RGB", (gray, gray, gray)), ((cell_size - dst_w) // 2, (cell_size - dst_h) // 2))
    draw = ImageDraw.Draw(cell)
    color = _LAYER_COLORS.get(layer_idx, (255, 255, 255))
    draw.rectangle((0, 0, cell_size - 1, cell_size - 1), outline=color, width=3)
    label_lines = [build, map_name[:13], f"L{layer_idx} {w}x{h}", f"{x},{y} area {int(row.get('area', 0))}"]
    text_h = 13 * len(label_lines) + 4
    draw.rectangle((2, cell_size - text_h - 2, cell_size - 3, cell_size - 3), fill=(0, 0, 0))
    for idx, label in enumerate(label_lines):
        draw.text((5, cell_size - text_h + idx * 13), label, fill=color if idx == 2 else (235, 235, 235))
    return cell


def _draw_legend(draw: ImageDraw.ImageDraw, width: int, font: ImageFont.ImageFont) -> None:
    draw.rectangle((0, 0, width - 1, 70), fill=(20, 20, 24), outline=(58, 58, 64), width=1)
    draw.text((10, 8), "Spec 076 Near-Duplicate Cluster Contact Sheets", fill=(245, 245, 245), font=font)
    draw.text((10, 25), "Each row is one near-duplicate cluster. Cells are member examples.", fill=(190, 190, 195), font=font)
    draw.text((10, 42), "Clustering is translation/mirror/rotation-invariant on normalized binary thumbnails.", fill=(255, 190, 110), font=font)
    x = 760
    for layer_idx, color in _LAYER_COLORS.items():
        draw.rectangle((x, 12, x + 18, 30), fill=color, outline=(255, 255, 255))
        draw.text((x + 23, 15), f"L{layer_idx}", fill=(230, 230, 230), font=font)
        x += 62


def _write_index(output_dir: Path, pages: list[Path], clusters: list[dict[str, Any]], analysis_root: Path) -> None:
    rows = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Near-Duplicate Cluster Contact Sheets</title></head><body>",
        f"<h1>Near-Duplicate Cluster Contact Sheets</h1><p>Analysis root: <code>{html.escape(str(analysis_root))}</code></p>",
        f"<p>Clusters rendered: {len(clusters)}</p>",
    ]
    for page in pages:
        rows.append(f"<h2>{html.escape(page.name)}</h2><img src='{html.escape(page.name)}' style='max-width:100%; image-rendering: pixelated;'>")
    rows.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
