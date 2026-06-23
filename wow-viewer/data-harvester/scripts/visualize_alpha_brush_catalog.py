"""Render alpha-brush catalog contact sheets from spec 074 outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_CATALOG_DIR = _PROJECT_ROOT / "output" / "analysis" / "alpha-brush-library" / "two-build-full"

_LAYER_COLORS = {
    0: (120, 120, 120),
    1: (92, 177, 255),
    2: (124, 224, 123),
    3: (255, 184, 76),
}

_LAYER_LABELS = {
    0: "L0 base/fill",
    1: "L1 primary brush",
    2: "L2 transition/detail",
    3: "L3 highlight/detail",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render alpha-brush catalog contact sheets.")
    parser.add_argument("--catalog-dir", type=Path, default=_DEFAULT_CATALOG_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-clusters", type=int, default=200)
    parser.add_argument("--max-per-cluster", type=int, default=8)
    parser.add_argument("--clusters-per-page", type=int, default=50)
    parser.add_argument("--cell-size", type=int, default=112)
    parser.add_argument("--label-width", type=int, default=220)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None, help="Override per-row threshold when rendering crops.")
    parser.add_argument(
        "--example-source",
        default="representatives",
        choices=["representatives", "catalog-first"],
        help="Use centroid-nearest representative_component_ids from clusters.jsonl, or first catalog rows.",
    )
    parser.add_argument("--split-by-layer", action="store_true", help="Write separate contact-sheet sets by dominant cluster layer.")
    parser.add_argument("--write-cluster-pages", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_top_clusters(catalog_dir: Path, max_clusters: int) -> list[dict]:
    rows = list(_read_jsonl(catalog_dir / "clusters.jsonl"))
    rows.sort(key=lambda row: (-int(row.get("member_count", 0)), int(row.get("cluster_id", 0))))
    return rows[: max(1, int(max_clusters))]


def _collect_examples(catalog_dir: Path, cluster_ids: set[int], max_per_cluster: int) -> dict[int, list[dict]]:
    examples = {cluster_id: [] for cluster_id in cluster_ids}
    for row in _read_jsonl(catalog_dir / "catalog.jsonl"):
        cluster_id = int(row.get("cluster_id", -1))
        if cluster_id not in examples:
            continue
        bucket = examples[cluster_id]
        if len(bucket) < int(max_per_cluster):
            bucket.append(row)
        if all(len(items) >= int(max_per_cluster) for items in examples.values()):
            break
    return examples


def _collect_representative_examples(
    catalog_dir: Path,
    clusters: list[dict],
    max_per_cluster: int,
) -> dict[int, list[dict]]:
    wanted_by_component: dict[str, int] = {}
    wanted_order: dict[str, int] = {}
    for cluster in clusters:
        cluster_id = int(cluster.get("cluster_id", -1))
        for order, component_id in enumerate(cluster.get("representative_component_ids", [])[: int(max_per_cluster)]):
            component_id = str(component_id)
            wanted_by_component[component_id] = cluster_id
            wanted_order[component_id] = order

    examples = {int(cluster.get("cluster_id", -1)): [] for cluster in clusters}
    if not wanted_by_component:
        return examples

    for row in _read_jsonl(catalog_dir / "components.jsonl"):
        component_id = str(row.get("component_id", ""))
        cluster_id = wanted_by_component.get(component_id)
        if cluster_id is None:
            continue
        examples[cluster_id].append(row)

    for cluster_id, rows in examples.items():
        rows.sort(key=lambda row: wanted_order.get(str(row.get("component_id", "")), 999999))
        examples[cluster_id] = rows[: int(max_per_cluster)]
    return examples


class _ZarrCache:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self._stores: dict[str, zarr.storage.LocalStore] = {}
        self._roots: dict[str, zarr.Group] = {}

    def alpha(self, build: str):
        if build not in self._roots:
            zarr_path = self.dataset_dir / f"{build}.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"Missing Zarr store: {zarr_path}")
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            self._stores[build] = store
            self._roots[build] = root
        return self._roots[build]["alpha_256"]

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()
        self._roots.clear()


def _render_entry(cache: _ZarrCache, row: dict, cell_size: int, threshold_override: float | None) -> Image.Image:
    build = str(row.get("build", ""))
    map_name = str(row.get("map_name", row.get("map", "?")))
    tile_id = int(row.get("tile_id", -1))
    tile_x = int(row.get("tile_x", -1))
    tile_y = int(row.get("tile_y", -1))
    layer_idx = int(row.get("layer_idx", 0))
    x, y, w, h = [int(value) for value in row.get("bbox_xywh", [0, 0, 1, 1])]
    threshold = float(row.get("threshold", 0.05) if threshold_override is None else threshold_override)

    alpha = cache.alpha(build)
    tile = np.asarray(alpha[tile_id], dtype=np.float32)
    layer = np.clip(tile[:, :, layer_idx], 0.0, 1.0)
    crop = layer[max(0, y) : min(layer.shape[0], y + h), max(0, x) : min(layer.shape[1], x + w)]
    if crop.size == 0:
        crop = np.zeros((1, 1), dtype=np.float32)
    crop = np.where(crop > threshold, crop, 0.0)

    pad = max(2, int(cell_size * 0.08))
    box = np.pad(crop, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    src_h, src_w = box.shape
    max_draw = max(8, int(cell_size) - 14)
    scale = min(max_draw / max(1, src_w), max_draw / max(1, src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))

    gray = Image.fromarray((box * 255.0).astype(np.uint8), mode="L")
    gray = gray.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
    tile_img = Image.new("RGB", (int(cell_size), int(cell_size)), color=(18, 18, 20))
    rgb = Image.merge("RGB", (gray, gray, gray))
    tile_img.paste(rgb, ((int(cell_size) - dst_w) // 2, (int(cell_size) - dst_h) // 2))

    draw = ImageDraw.Draw(tile_img)
    color = _LAYER_COLORS.get(layer_idx, (255, 255, 255))
    draw.rectangle((0, 0, int(cell_size) - 1, int(cell_size) - 1), outline=color, width=3)
    map_short = map_name[:12]
    label_lines = [
        build,
        f"{map_short} {tile_x},{tile_y}",
        f"box {x},{y} {w}x{h}",
        f"L{layer_idx} area {int(row.get('area', 0))}",
    ]
    text_h = 13 * len(label_lines) + 4
    draw.rectangle((2, int(cell_size) - text_h - 2, int(cell_size) - 3, int(cell_size) - 3), fill=(0, 0, 0))
    for line_idx, label in enumerate(label_lines):
        draw.text((5, int(cell_size) - text_h + line_idx * 13), label, fill=color if line_idx == 2 else (235, 235, 235))
    return tile_img


def _draw_contact_sheet(
    clusters: list[dict],
    examples: dict[int, list[dict]],
    cache: _ZarrCache,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    row_h = int(args.cell_size) + int(args.padding)
    width = int(args.label_width) + (int(args.cell_size) + int(args.padding)) * int(args.max_per_cluster)
    legend_h = 74
    height = legend_h + max(1, len(clusters)) * row_h + int(args.padding)
    image = Image.new("RGB", (width, height), color=(10, 10, 12))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    _draw_legend(draw, width, font)

    for row_idx, cluster in enumerate(clusters):
        y = legend_h + int(args.padding) + row_idx * row_h
        cluster_id = int(cluster.get("cluster_id", -1))
        member_count = int(cluster.get("member_count", 0))
        dominant_map = str(cluster.get("dominant_map", ""))
        dominant_layer = cluster.get("dominant_layer", None)
        label = f"C{cluster_id}\n{member_count} comps\n{dominant_map}\nL{dominant_layer}"
        draw.text((8, y + 8), label, fill=(230, 230, 230), font=font)

        for col_idx, entry in enumerate(examples.get(cluster_id, [])):
            x = int(args.label_width) + col_idx * (int(args.cell_size) + int(args.padding))
            try:
                component_img = _render_entry(cache, entry, int(args.cell_size), args.threshold)
            except Exception as exc:  # Keep one bad entry from killing a whole sheet.
                component_img = Image.new("RGB", (int(args.cell_size), int(args.cell_size)), color=(80, 20, 20))
                ImageDraw.Draw(component_img).text((6, 6), f"ERR\n{type(exc).__name__}", fill=(255, 220, 220))
            image.paste(component_img, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_legend(draw: ImageDraw.ImageDraw, width: int, font: ImageFont.ImageFont) -> None:
    draw.rectangle((0, 0, width - 1, 68), fill=(20, 20, 24), outline=(58, 58, 64), width=1)
    draw.text((10, 8), "Alpha Brush Library", fill=(245, 245, 245), font=font)
    draw.text(
        (10, 24),
        "Each row is a cluster. Each cell is one alpha-mask component crop from that cluster.",
        fill=(190, 190, 195),
        font=font,
    )
    draw.text(
        (10, 40),
        "Cell label shows map tileX,tileY, tile-local bbox x,y w,h, source layer, area, and build.",
        fill=(190, 190, 195),
        font=font,
    )
    x = 430
    for layer_idx in range(4):
        color = _LAYER_COLORS[layer_idx]
        draw.rectangle((x, 10, x + 18, 28), fill=color, outline=(255, 255, 255))
        draw.text((x + 24, 13), _LAYER_LABELS[layer_idx], fill=(230, 230, 230), font=font)
        x += 155


def _write_cluster_pages(
    clusters: list[dict],
    examples: dict[int, list[dict]],
    cache: _ZarrCache,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    page_dir = output_dir / "clusters"
    page_dir.mkdir(parents=True, exist_ok=True)
    for cluster in clusters:
        cluster_id = int(cluster.get("cluster_id", -1))
        _draw_contact_sheet([cluster], examples, cache, page_dir / f"cluster_{cluster_id:04d}.png", args)


def _write_html_index(output_dir: Path, sheet_paths: list[Path], clusters: list[dict]) -> None:
    parts = ["<html><body><h1>Alpha Brush Library Contact Sheets</h1>"]
    parts.append(f"<p>{len(clusters)} clusters rendered.</p>")
    parts.append("<h2>Legend</h2>")
    parts.append("<ul>")
    for layer_idx in range(4):
        color = _LAYER_COLORS[layer_idx]
        rgb = f"rgb({color[0]},{color[1]},{color[2]})"
        parts.append(
            f'<li><span style="display:inline-block;width:1em;height:1em;background:{rgb};border:1px solid #333"></span> '
            f'L{layer_idx}: {html.escape(_LAYER_LABELS[layer_idx])}</li>'
        )
    parts.append("</ul>")
    parts.append(
        "<p>Rows are embedding clusters. Cells are representative alpha-mask component crops. "
        "These are atomic components, not yet multi-tile prefab/paste assemblies.</p>"
    )
    for sheet in sheet_paths:
        rel = html.escape(sheet.name)
        parts.append(f'<h2>{rel}</h2><img src="{rel}" style="max-width:100%;height:auto">')
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    catalog_dir = Path(args.catalog_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else catalog_dir / "montages"
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = _load_top_clusters(catalog_dir, int(args.max_clusters))
    cluster_ids = {int(row.get("cluster_id", -1)) for row in clusters}
    if args.example_source == "representatives":
        examples = _collect_representative_examples(catalog_dir, clusters, int(args.max_per_cluster))
        missing = [int(row.get("cluster_id", -1)) for row in clusters if not examples.get(int(row.get("cluster_id", -1)))]
        if missing:
            print(f"Representative lookup missed {len(missing)} clusters; falling back to catalog-first for those.", flush=True)
            fallback = _collect_examples(catalog_dir, set(missing), int(args.max_per_cluster))
            examples.update({cluster_id: rows for cluster_id, rows in fallback.items() if rows})
    else:
        examples = _collect_examples(catalog_dir, cluster_ids, int(args.max_per_cluster))
    cache = _ZarrCache(Path(args.dataset_dir))
    try:
        if args.split_by_layer:
            for layer_idx in range(4):
                layer_clusters = [
                    cluster for cluster in clusters if int(cluster.get("dominant_layer", -1)) == layer_idx
                ]
                layer_dir = output_dir / f"layer_{layer_idx}"
                sheet_paths = _write_sheet_set(layer_clusters, examples, cache, layer_dir, args)
                _write_html_index(layer_dir, sheet_paths, layer_clusters)
                print(f"Layer {layer_idx}: rendered {len(layer_clusters)} clusters to {layer_dir}", flush=True)
        else:
            sheet_paths = _write_sheet_set(clusters, examples, cache, output_dir, args)
            _write_html_index(output_dir, sheet_paths, clusters)
            if args.write_cluster_pages:
                _write_cluster_pages(clusters, examples, cache, output_dir, args)
        summary = {
            "catalog_dir": str(catalog_dir),
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(output_dir),
            "cluster_count_rendered": len(clusters),
            "max_per_cluster": int(args.max_per_cluster),
            "example_source": str(args.example_source),
            "split_by_layer": bool(args.split_by_layer),
        }
        (output_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Index: {output_dir / 'index.html'}", flush=True)
    finally:
        cache.close()


def _write_sheet_set(
    clusters: list[dict],
    examples: dict[int, list[dict]],
    cache: _ZarrCache,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths: list[Path] = []
    per_page = max(1, int(args.clusters_per_page))
    for page_idx, start in enumerate(range(0, len(clusters), per_page)):
        page_clusters = clusters[start : start + per_page]
        sheet_path = output_dir / f"cluster_contact_sheet_{page_idx:03d}.png"
        _draw_contact_sheet(page_clusters, examples, cache, sheet_path, args)
        sheet_paths.append(sheet_path)
        print(f"Wrote {sheet_path}", flush=True)
    return sheet_paths


if __name__ == "__main__":
    main()
