"""Visualize PM4 match results as minimap overlay.

Reads a PM4 match report JSON and overlays segment footprints
on the tile minimap, colored by match status.

Usage:
    uv run python visualize_pm4_matches.py <report.json> [--tile X_Y] [--output overlay.png] [--minimap-dir <path>]

Examples:
    uv run python visualize_pm4_matches.py ../output/tmp/pm4-synthesize-placements-seeded-smoke.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TILE_SIZE = 533.3333
CHUNK_SIZE = TILE_SIZE / 16  # ADT has 16x16 chunks per tile


def _g(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _load_minimap(tile_x: int, tile_y: int, minimap_dir: str) -> np.ndarray:
    path = os.path.join(minimap_dir, f"development_{tile_x}_{tile_y}.png")
    if not os.path.isfile(path):
        print(f"  [warning] minimap not found: {path}", file=sys.stderr)
        return np.ones((256, 256, 3), dtype=np.uint8) * 128
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (16, 16, 16, 255))
    bg.paste(img, mask=img.split()[3])
    return np.asarray(bg.convert("RGB"))


def _world_to_pixel(wx: float, wy: float, tile_x: int, tile_y: int):
    px = (wx - tile_x * TILE_SIZE) / TILE_SIZE * 256
    py = (wy - tile_y * TILE_SIZE) / TILE_SIZE * 256
    return px, py


def _draw_footprint(ax, seg: dict, tile_x: int, tile_y: int, show_labels: bool):
    status = _g(seg, "Status", "status") or ""
    color_map = {
        "matched": (0.0, 0.7, 0.0, 0.45),
        "ambiguous": (1.0, 0.7, 0.0, 0.45),
        "unresolved": (0.5, 0.5, 0.5, 0.08),
        "ineligible": (0.3, 0.3, 0.3, 0.04),
    }
    edge_map = {
        "matched": (0.0, 0.5, 0.0, 0.7),
        "ambiguous": (0.8, 0.5, 0.0, 0.7),
        "unresolved": (0.4, 0.4, 0.4, 0.15),
        "ineligible": (0.2, 0.2, 0.2, 0.08),
    }
    face_color = color_map.get(status, (0.8, 0.3, 0.3, 0.3))
    edge_color = edge_map.get(status, (0.5, 0.0, 0.0, 0.4))

    # Draw footprint hull (more accurate than bounding box)
    hull = _g(seg, "FootprintHull", "footprintHull")
    if hull and len(hull) >= 3:
        fpx, fpy = _world_to_pixel(
            np.array([p["X"] for p in hull]),
            np.array([p["Y"] for p in hull]),
            tile_x, tile_y,
        )
        ax.fill(fpx, fpy, color=face_color, edgecolor=edge_color, linewidth=0.5)

    # Draw placement proposal position marker
    prop = _g(seg, "PlacementProposal", "placementProposal")
    if prop:
        wp = _g(prop, "WorldPosition", "worldPosition")
        if wp:
            mpx, mpy = _world_to_pixel(wp["X"], wp["Y"], tile_x, tile_y)
            marker_color = "lime" if status == "matched" else "gold"
            ax.plot(mpx, mpy, marker="*", color=marker_color,
                    markersize=8, markeredgecolor="black", markeredgewidth=0.5)

            # Draw direction arrow from yaw
            rot = _g(prop, "WorldRotation", "worldRotation")
            if rot and _g(rot, "Yaw", "yaw") is not None:
                yaw_deg = _g(rot, "Yaw", "yaw")
                yaw_rad = np.deg2rad(yaw_deg)
                dx = np.cos(yaw_rad) * 8
                dy = -np.sin(yaw_rad) * 8  # negate because image Y is flipped
                ax.arrow(mpx, mpy, dx, dy, head_width=3, head_length=2,
                         fc=marker_color, ec="black", linewidth=0.5)

    # Draw asset name label for matched / ambiguous
    if show_labels and status in ("matched", "ambiguous"):
        candidates = _g(seg, "Candidates", "candidates") or []
        if candidates:
            top = candidates[0]
            path = (top.get("AssetPath") or top.get("assetPath") or "?")
            label = os.path.splitext(os.path.basename(path))[0]
            score = top.get("OverallScore") or top.get("overallScore") or 0
            bounds = _g(seg, "Bounds", "bounds")
            if bounds:
                bmin = _g(bounds, "Min", "min")
                bmax = _g(bounds, "Max", "max")
                if bmin and bmax:
                    cx = (bmin["X"] + bmax["X"]) / 2
                    cy = (bmin["Y"] + bmax["Y"]) / 2
                    lx, ly = _world_to_pixel(cx, cy, tile_x, tile_y)
                    ax.text(lx, ly, f"{label}\n{score:.3f}",
                            fontsize=4, color="white",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="black", alpha=0.7),
                            ha="center", va="center")


def visualize_report(report_path: str, tile_key: str | None,
                     output_path: str, minimap_dir: str,
                     show_all: bool):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    segments = _g(report, "Segments", "segments") or []
    if not segments:
        print("No segments found in report.", file=sys.stderr)
        print(f"  Available keys: {list(report.keys())[:10]}", file=sys.stderr)
        return

    all_tiles: set[str] = set()
    for seg in segments:
        for tc in (_g(seg, "TileCoordinates", "tileCoordinates") or []):
            all_tiles.add(tc)

    target_tiles = [tile_key] if tile_key else sorted(all_tiles)

    for tile in target_tiles:
        if "_" not in tile:
            print(f"  [skip] invalid tile key: {tile}", file=sys.stderr)
            continue

        parts = tile.split("_")
        tile_x, tile_y = int(parts[0]), int(parts[1])
        print(f"Rendering tile {tile} ...")

        minimap = _load_minimap(tile_x, tile_y, minimap_dir)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(minimap, extent=[0, 256, 256, 0])
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_aspect("equal")
        ax.set_title(f"Tile {tile} — PM4 Asset Match Overlay", fontsize=13)

        # Chunk grid (16x16)
        for i in range(17):
            ch = i * 256 / 16
            ax.axvline(ch, color="white", linewidth=0.15, alpha=0.3)
            ax.axhline(ch, color="white", linewidth=0.15, alpha=0.3)

        tile_segments = [
            s for s in segments
            if tile in (_g(s, "TileCoordinates", "tileCoordinates") or [])
        ]

        # Separate by status
        matched = [s for s in tile_segments
                   if _g(s, "Status", "status") == "matched"]
        ambiguous = [s for s in tile_segments
                     if _g(s, "Status", "status") == "ambiguous"]
        unresolved = [s for s in tile_segments
                      if _g(s, "Status", "status") in ("unresolved", None)]
        ineligible = [s for s in tile_segments
                      if _g(s, "Status", "status") == "ineligible"]

        # Draw in layers: unresolved first (faint), then ineligible, then ambiguous, then matched on top
        if show_all:
            for seg in unresolved:
                _draw_footprint(ax, seg, tile_x, tile_y, show_labels=False)
            for seg in ineligible:
                _draw_footprint(ax, seg, tile_x, tile_y, show_labels=False)
        for seg in ambiguous:
            _draw_footprint(ax, seg, tile_x, tile_y, show_labels=True)
        for seg in matched:
            _draw_footprint(ax, seg, tile_x, tile_y, show_labels=True)

        # Legend
        legend_elements = [
            mpatches.Patch(color=(0.0, 0.7, 0.0, 0.45), label=f"Matched ({len(matched)})"),
            mpatches.Patch(color=(1.0, 0.7, 0.0, 0.45), label=f"Ambiguous ({len(ambiguous)})"),
        ]
        if show_all:
            legend_elements += [
                mpatches.Patch(color=(0.5, 0.5, 0.5, 0.08), label=f"Unresolved ({len(unresolved)})"),
                mpatches.Patch(color=(0.3, 0.3, 0.3, 0.04), label=f"Ineligible ({len(ineligible)})"),
            ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        summary = f"Tile {tile} | Matched={len(matched)} Ambiguous={len(ambiguous)} Unresolved={len(unresolved)} Ineligible={len(ineligible)}"
        ax.text(0.5, 0.98, summary, transform=ax.transAxes, fontsize=8,
                ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PM4 match results as minimap overlay")
    parser.add_argument("report", help="Path to PM4 match report JSON")
    parser.add_argument("--tile", default=None,
                        help="Specific tile to render (e.g. '0_0')")
    parser.add_argument("--output", default=None,
                        help="Output PNG path")
    parser.add_argument("--minimap-dir",
                        default="i:/parp/parp-tools/wow-viewer/test_data/development/World/Textures/Minimap",
                        help="Directory containing minimap PNGs")
    parser.add_argument("--show-all", action="store_true",
                        help="Show unresolved and ineligible segments (faint)")
    args = parser.parse_args()

    if not args.output:
        stem = Path(args.report).stem
        tile_suffix = f"_{args.tile}" if args.tile else ""
        args.output = f"{stem}{tile_suffix}_overlay.png"

    visualize_report(args.report, args.tile, args.output, args.minimap_dir, args.show_all)


if __name__ == "__main__":
    main()
