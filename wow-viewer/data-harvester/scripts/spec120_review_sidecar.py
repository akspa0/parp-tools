#!/usr/bin/env python3
"""Review & Visual Overlay Tool for Spec 120 Metadata Sidecars.

Reads sidecar_output_ek.parquet / sidecar_output_ek.json, generates class & asset breakdowns,
and renders an OBB detection overlay preview over sample minimap tiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and visualize Spec 120 sidecar results.")
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("../output/spec120/sidecar_output_real.parquet"),
        help="Path to sidecar parquet file (default: sidecar_output_real.parquet).",
    )
    parser.add_argument(
        "--zarr-store",
        type=Path,
        default=Path("../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr"),
        help="Path to Zarr dataset store for real minimap tile images.",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=None,
        help="Path to split tiles directory (optional fallback for loose PNGs).",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("../output/spec120/real_minimap_detections_preview.png"),
        help="Path to save visual overlay preview PNG.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top retrieved assets to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"=== Spec 120 Sidecar Review & Visualization Tool ===")
    print(f"Sidecar File: {args.sidecar.resolve()}")

    if not args.sidecar.exists():
        json_alt = args.sidecar.with_suffix(".json")
        if json_alt.exists():
            records = json.loads(json_alt.read_text(encoding="utf-8"))
        else:
            print(f"[ERROR] Sidecar file not found at {args.sidecar.resolve()}")
            sys.exit(1)
    else:
        table = pq.read_table(args.sidecar)
        records = table.to_pylist()

    print(f"Total Detections: {len(records):,}")

    if not records:
        print("No records to summarize.")
        return

    # 1. Class Breakdown
    class_counts: dict[str, int] = {}
    asset_counts: dict[str, int] = {}
    confs = []

    for r in records:
        cls = r.get("coarse_class", "unknown")
        asset = r.get("retrieved_asset", "unknown")
        conf = r.get("confidence", 0.0)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        asset_counts[asset] = asset_counts.get(asset, 0) + 1
        confs.append(conf)

    print("\n--- Coarse Class Breakdown ---")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(records)) * 100.0
        print(f"  {cls.upper():<10}: {count:>8,} ({pct:.1f}%)")

    print(f"\n--- Confidence Statistics ---")
    print(f"  Min: {min(confs):.4f} | Max: {max(confs):.4f} | Mean: {np.mean(confs):.4f} | Median: {np.median(confs):.4f}")

    print(f"\n--- Top {args.top_k} Retrieved Assets ---")
    for asset, count in sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:args.top_k]:
        print(f"  {count:>6,}x : {asset}")

    # 2. Render Overlay Preview Sheet from Zarr Store or Loose Tile PNGs
    records_by_tile: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in records:
        tx = int(r.get("tile_x", 32))
        ty = int(r.get("tile_y", 32))
        records_by_tile.setdefault((tx, ty), []).append(r)

    canvas_w, canvas_h = 512, 512
    grid_img = Image.new("RGB", (canvas_w * 2, canvas_h * 2), (20, 20, 20))
    draw = ImageDraw.Draw(grid_img)
    colors = {"wmo": (255, 60, 60), "mdx": (60, 220, 60), "m2": (60, 160, 255), "doodad": (255, 200, 40)}

    rendered = False

    if args.zarr_store and args.zarr_store.exists():
        import zarr
        zarr_grp = zarr.open_group(args.zarr_store, mode="r")
        if "minimap_rgb_authored" in zarr_grp:
            img_arr = zarr_grp["minimap_rgb_authored"]
            index_path = args.zarr_store / "index.parquet"
            if index_path.exists():
                idx_tbl = pq.read_table(index_path)
                xs = idx_tbl["tile_x"].to_pylist()
                ys = idx_tbl["tile_y"].to_pylist()
                tile_map = {(x, y): i for i, (x, y) in enumerate(zip(xs, ys))}

                # Find top 4 active tiles with highest object counts
                active_list = [(tx, ty, len(recs)) for (tx, ty), recs in records_by_tile.items() if (tx, ty) in tile_map]
                active_list.sort(key=lambda item: item[2], reverse=True)

                if active_list:
                    print(f"\nRendering bounding box overlay preview on top 4 real game minimap RGB tiles from Zarr store...")
                    for i, (tx, ty, num_obj) in enumerate(active_list[:4]):
                        row, col = i // 2, i % 2
                        x_offset, y_offset = col * canvas_w, row * canvas_h

                        zarr_row = tile_map[(tx, ty)]
                        tile_np = np.asarray(img_arr[zarr_row])  # (256, 256, 3)
                        tile_pil = Image.fromarray(tile_np).resize((canvas_w, canvas_h))
                        grid_img.paste(tile_pil, (x_offset, y_offset))

                        recs = records_by_tile.get((tx, ty), [])
                        for r in recs[:30]:
                            px, py = r["position_px"]
                            sw, sh = r["scale_px"]
                            cls = r.get("coarse_class", "mdx")
                            color = colors.get(cls, (60, 220, 60))

                            cx, cy = x_offset + (px / 256.0) * canvas_w, y_offset + (py / 256.0) * canvas_h
                            bw, bh = max(8.0, (sw / 256.0) * canvas_w), max(8.0, (sh / 256.0) * canvas_h)

                            draw.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline=color, width=2)

                        draw.rectangle([x_offset, y_offset, x_offset + canvas_w, y_offset + canvas_h], outline=(180, 180, 180), width=2)
                        draw.text((x_offset + 10, y_offset + 10), f"Tile ({tx}, {ty}) — {len(recs)} objects", fill=(255, 255, 255))

                    rendered = True

    if not rendered and args.tiles_dir and args.tiles_dir.exists():
        tile_files = sorted(list(args.tiles_dir.glob("*.png")) + list(args.tiles_dir.glob("*.jpg")))
        if tile_files:
            print(f"\nRendering bounding box overlay preview on sample tiles from {args.tiles_dir.name}...")
            active_tiles = []
            import re
            tile_re = re.compile(r"tile_(\d+)_(\d+)", re.IGNORECASE)

            for tf in tile_files:
                m = tile_re.search(tf.name)
                if m:
                    tx, ty = int(m.group(1)), int(m.group(2))
                    recs = records_by_tile.get((tx, ty), [])
                    if len(recs) > 0:
                        img_tmp = Image.open(tf).convert("RGB")
                        extrema = img_tmp.getextrema()
                        if extrema[0][1] - extrema[0][0] > 20:
                            active_tiles.append((tf, tx, ty, len(recs)))

            active_tiles.sort(key=lambda x: x[3], reverse=True)
            sample_tiles = [(t[0], t[1], t[2]) for t in active_tiles[:4]] if active_tiles else [(tf, 32, 32) for tf in tile_files[:4]]

            for i, (tf, tx, ty) in enumerate(sample_tiles):
                row, col = i // 2, i % 2
                x_offset, y_offset = col * canvas_w, row * canvas_h

                tile_img = Image.open(tf).convert("RGB").resize((canvas_w, canvas_h))
                grid_img.paste(tile_img, (x_offset, y_offset))

                tile_recs = records_by_tile.get((tx, ty), [])
                for r in tile_recs[:30]:
                    px, py = r["position_px"]
                    sw, sh = r["scale_px"]
                    cls = r.get("coarse_class", "mdx")
                    color = colors.get(cls, (60, 220, 60))

                    cx, cy = x_offset + (px / 256.0) * canvas_w, y_offset + (py / 256.0) * canvas_h
                    bw, bh = max(8.0, (sw / 256.0) * canvas_w), max(8.0, (sh / 256.0) * canvas_h)

                    draw.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline=color, width=2)

                draw.rectangle([x_offset, y_offset, x_offset + canvas_w, y_offset + canvas_h], outline=(180, 180, 180), width=2)
                draw.text((x_offset + 10, y_offset + 10), f"{tf.stem} ({len(tile_recs)} objects)", fill=(255, 255, 255))
            rendered = True

    if rendered:
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(args.output_image)
        print(f"[PREVIEW CREATED] Saved visual detections preview to {args.output_image.resolve()}")


if __name__ == "__main__":
    main()
