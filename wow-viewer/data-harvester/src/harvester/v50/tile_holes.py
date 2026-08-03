"""Hole masks as first-class data: what the client is told not to draw, and what is under it.

A hole is a per-chunk ``uint16`` bitmask in the MCNK header at offset 0x40. Each bit disables one
4x4 sub-quad of that chunk, so a tile carries three nested levels of granularity at once:

    tile (1) -> chunk (16x16) -> hole quad (4x4 within each chunk)

The geometry is NOT removed when a quad is holed. ``AlphaWdtReader.TryParseMcnk`` reads MCVT under
``if (mcvtRel >= 0 ...)`` and never consults the hole mask, so the heights behind a hole are already
in every harvested ``height_257``. Measured on the 0.5.3 corpus: 230 tiles carry holes, 709 chunks
are holed, and 91.4% of those chunks hold more than a unit of relief — with a HIGHER median relief
(26.08) than the un-holed chunks around them (19.93). These are cutouts placed over interesting
terrain, not padding.

This module turns that into something queryable rather than something you have to look at:

- per-tile metrics folded into the tile inventory (``hole_chunk_count``, ``hole_quad_count``, ...)
- a per-CHUNK table, one row per holed chunk, with the relief statistics of what is hidden
- a bitmask-pattern census, because a hole mask is 16 bits and if the editor emitted structured
  patterns rather than arbitrary ones, the histogram is where that shows up
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

HOLES_SCHEMA = "v50-tile-holes-v1"

CHUNKS_PER_DIM = 16
CHUNK_STRIDE = 16
CHUNK_SPAN = 17
QUADS_PER_CHUNK = 16  # 4x4 sub-quads, one per bit of the uint16 mask
QUAD_DIM = 4


def load_hole_masks(path: Path) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    """Read an ``extract-holes`` JSON into ``{map: {(x, y): uint16[16, 16]}}``.

    The C# writes ``mcnk_holes_uint16_row_major_yx`` — 256 values in [chunk_y][chunk_x] order,
    which is the same orientation the height grid uses, so no transpose is needed downstream.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    field = str(payload.get("hole_field", ""))
    if "uint16" not in field:
        raise ValueError(f"unexpected hole_field {field!r}; expected a uint16 mask export")
    out: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for map_name, tiles in payload.get("maps", {}).items():
        per_map: dict[tuple[int, int], np.ndarray] = {}
        for tile in tiles:
            mask = np.asarray(tile["holes"], dtype=np.uint16)
            if mask.size != CHUNKS_PER_DIM * CHUNKS_PER_DIM:
                raise ValueError(f"{map_name} {tile['x']},{tile['y']}: expected 256 chunk masks")
            per_map[(int(tile["x"]), int(tile["y"]))] = mask.reshape(CHUNKS_PER_DIM, CHUNKS_PER_DIM)
        out[map_name] = per_map
    return out


def quad_count(mask_value: int) -> int:
    """Number of 4x4 sub-quads disabled in one chunk (popcount of the uint16)."""
    return int(bin(int(mask_value) & 0xFFFF).count("1"))


def quad_grid(mask_value: int) -> np.ndarray:
    """Expand one chunk's uint16 into the 4x4 boolean quad layout it encodes.

    Bit i maps to row i//4, column i%4 — the standard MCNK hole layout. Kept as a named function so
    the bit order is asserted in one place rather than assumed at each use.
    """
    bits = [(int(mask_value) >> i) & 1 for i in range(QUADS_PER_CHUNK)]
    return np.asarray(bits, dtype=bool).reshape(QUAD_DIM, QUAD_DIM)


def tile_hole_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Per-tile summary of a ``uint16[16, 16]`` chunk-mask grid."""
    grid = np.asarray(mask, dtype=np.uint16)
    holed = grid != 0
    quads = int(sum(quad_count(v) for v in grid.ravel()))
    total_quads = CHUNKS_PER_DIM * CHUNKS_PER_DIM * QUADS_PER_CHUNK
    return {
        "hole_chunk_count": int(holed.sum()),
        "hole_quad_count": quads,
        "hole_quad_fraction": quads / total_quads,
        # A chunk whose every quad is disabled is fully hidden; a partial one is a shaped cutout,
        # which is the more interesting case for "what were they hiding".
        "fully_holed_chunk_count": int(sum(1 for v in grid.ravel() if quad_count(v) == QUADS_PER_CHUNK)),
        "partial_holed_chunk_count": int(sum(1 for v in grid.ravel() if 0 < quad_count(v) < QUADS_PER_CHUNK)),
    }


def chunk_relief(height_257: np.ndarray, chunk_x: int, chunk_y: int) -> dict[str, float]:
    """Relief statistics of one chunk's 17x17 block of the height grid."""
    block = np.asarray(height_257, dtype=np.float64)[
        chunk_y * CHUNK_STRIDE : chunk_y * CHUNK_STRIDE + CHUNK_SPAN,
        chunk_x * CHUNK_STRIDE : chunk_x * CHUNK_STRIDE + CHUNK_SPAN,
    ]
    return {
        "height_min": float(block.min()),
        "height_max": float(block.max()),
        "height_range": float(block.max() - block.min()),
        "height_levels": int(np.unique(block.astype(np.float32)).size),
    }


def hidden_chunk_records(
    map_name: str,
    tile_x: int,
    tile_y: int,
    mask: np.ndarray,
    height_257: np.ndarray,
) -> list[dict[str, Any]]:
    """One row per HOLED chunk: where it is, how much is hidden, and what is under it."""
    grid = np.asarray(mask, dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    for chunk_y in range(CHUNKS_PER_DIM):
        for chunk_x in range(CHUNKS_PER_DIM):
            value = int(grid[chunk_y, chunk_x])
            if value == 0:
                continue
            quads = quad_count(value)
            rows.append({
                "tile_key": f"{map_name}_{tile_x:02d}_{tile_y:02d}",
                "map": map_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "chunk_x": chunk_x,
                "chunk_y": chunk_y,
                "hole_mask": value,
                "hole_mask_hex": f"0x{value:04X}",
                "hole_quads": quads,
                "fully_holed": quads == QUADS_PER_CHUNK,
                **chunk_relief(height_257, chunk_x, chunk_y),
            })
    return rows


def bitmask_census(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Histogram of the distinct 16-bit hole patterns actually used.

    The nesting question in measurable form: if the editor painted holes freely, the 65,535 possible
    non-zero masks would be spread thin and irregular. If a small set of patterns dominates, the
    masks are being emitted from structure — a brush, a template, or a nested encoding — and that
    set is the thing to go read.
    """
    counts: dict[int, int] = {}
    for record in records:
        value = int(record["hole_mask"])
        counts[value] = counts.get(value, 0) + 1
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [
        {
            "mask": value,
            "hex": f"0x{value:04X}",
            "quads": quad_count(value),
            "count": count,
            "fraction": count / total if total else 0.0,
            "grid": [[int(b) for b in row] for row in quad_grid(value)],
        }
        for value, count in ordered[:24]
    ]
    return {
        "holed_chunks": total,
        "distinct_masks": len(counts),
        "top_masks": top,
        # If a handful of patterns cover most chunks, the masks are structured, not freehand.
        "coverage_of_top_8": sum(c for _, c in ordered[:8]) / total if total else 0.0,
    }


# --- rendering ---------------------------------------------------------------------------------

HOLE_RGB = (235, 70, 70)
PANEL = 257
HEADER = 58
LABEL = 22
PANEL_TITLES = ("Terrain (autostretched)", "Hole mask", "HIDDEN geometry only", "Minimap")


def hole_pixel_mask(mask: np.ndarray, size: int = PANEL) -> np.ndarray:
    """Expand a ``uint16[16, 16]`` chunk grid to a per-pixel boolean mask on the height grid.

    Each chunk spans 16 pixels and each of its 4x4 quads spans 4, so the quad layout survives the
    expansion — a partially-holed chunk renders as the actual shape of the cutout rather than a
    solid square.
    """
    grid = np.asarray(mask, dtype=np.uint16)
    out = np.zeros((size, size), dtype=bool)
    per_quad = CHUNK_STRIDE // QUAD_DIM
    for chunk_y in range(CHUNKS_PER_DIM):
        for chunk_x in range(CHUNKS_PER_DIM):
            value = int(grid[chunk_y, chunk_x])
            if value == 0:
                continue
            quads = quad_grid(value)
            for qy in range(QUAD_DIM):
                for qx in range(QUAD_DIM):
                    if not quads[qy, qx]:
                        continue
                    y0 = chunk_y * CHUNK_STRIDE + qy * per_quad
                    x0 = chunk_x * CHUNK_STRIDE + qx * per_quad
                    out[y0 : y0 + per_quad, x0 : x0 + per_quad] = True
    return out


def render_hidden_tile(
    *,
    height_257: np.ndarray,
    mask: np.ndarray,
    minimap: np.ndarray | None,
    title: str,
    output: Path,
) -> None:
    """Four panels: the tile, where the holes are, the hidden geometry alone, and the minimap."""
    from PIL import Image, ImageDraw, ImageFont

    from harvester.v50.tile_composite import hillshade_np
    from harvester.v50.tile_synthesis import autostretch

    stretched, _, _, flat = autostretch(height_257)
    shaded = np.full(stretched.shape, 0.5) if flat else hillshade_np(stretched)
    grey = np.repeat(np.rint(np.clip(shaded, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)

    pixels = hole_pixel_mask(mask, size=height_257.shape[0])
    overlay = grey.copy()
    overlay[pixels] = HOLE_RGB

    # Hidden-only: the geometry the client never draws, everything else knocked back to near-black
    # so the shape of what was cut out is the only thing on the panel.
    hidden = (grey * 0.12).astype(np.uint8)
    hidden[pixels] = grey[pixels]

    art = (
        np.asarray(minimap, dtype=np.uint8)
        if minimap is not None and np.asarray(minimap).any()
        else np.full((PANEL, PANEL, 3), 32, dtype=np.uint8)
    )
    panels = (grey, overlay, hidden, art)

    sheet = Image.new("RGB", (len(panels) * PANEL, HEADER + PANEL + LABEL), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    metrics = tile_hole_metrics(mask)
    draw.text((8, 6), title, fill=(240, 242, 246), font=font)
    draw.text((8, 24), f"holed chunks {metrics['hole_chunk_count']}/256   "
                       f"holed quads {metrics['hole_quad_count']}/4096   "
                       f"fully-holed {metrics['fully_holed_chunk_count']}   "
                       f"partial {metrics['partial_holed_chunk_count']}",
              fill=(214, 220, 228), font=font)
    draw.text((8, 40), "Hidden geometry is REAL: the client is told not to draw it, the heights are "
                       "still in the file.", fill=(255, 190, 120), font=font)
    for column, (panel, label) in enumerate(zip(panels, PANEL_TITLES)):
        x = column * PANEL
        image = Image.fromarray(np.asarray(panel, dtype=np.uint8), mode="RGB")
        if image.size != (PANEL, PANEL):
            image = image.resize((PANEL, PANEL), Image.Resampling.NEAREST)
        sheet.paste(image, (x, HEADER))
        draw.rectangle((x, HEADER + PANEL, x + PANEL, HEADER + PANEL + LABEL), fill=(28, 31, 37))
        draw.text((x + 6, HEADER + PANEL + 6), label, fill=(218, 222, 228), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    import argparse
    import csv

    import pyarrow.parquet as pq
    import zarr

    parser = argparse.ArgumentParser(
        description="Hole masks as dataset + imagery: what the client never draws"
    )
    parser.add_argument("--holes", required=True, type=Path,
                        help="extract-holes JSON (WowViewer.Tool.Harvest extract-holes)")
    parser.add_argument("--store", required=True, type=Path, action="append", dest="stores",
                        metavar="STORE", help="a per-map v50 store; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render", type=int, default=40,
                        help="render the N most-holed tiles as 4-panel sheets (0 disables)")
    args = parser.parse_args()

    masks = load_hole_masks(args.holes)
    all_chunks: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    renderable: list[tuple[int, str, Any, int, int, np.ndarray]] = []

    for store in args.stores:
        group = zarr.open_group(str(store), mode="r")
        index_rows = pq.read_table(store / "index.parquet").to_pylist()
        map_name = str(index_rows[0].get("map", store.stem))
        per_map = masks.get(map_name)
        if per_map is None:
            print(f"WARNING: no hole data for {map_name}; skipping", flush=True)
            continue
        holed_tiles = 0
        for row_id, row in enumerate(index_rows):
            key = (int(row["tile_x"]), int(row["tile_y"]))
            mask = per_map.get(key)
            if mask is None or not mask.any():
                continue
            holed_tiles += 1
            height = np.asarray(group["height_257"][row_id], dtype=np.float32)
            chunks = hidden_chunk_records(map_name, key[0], key[1], mask, height)
            all_chunks.extend(chunks)
            metrics = tile_hole_metrics(mask)
            tile_rows.append({
                "tile_key": f"{map_name}_{key[0]:02d}_{key[1]:02d}",
                "map": map_name, "tile_x": key[0], "tile_y": key[1], "row_id": row_id,
                **metrics,
                "hidden_relief_median": float(np.median([c["height_range"] for c in chunks])),
                "hidden_relief_max": float(max(c["height_range"] for c in chunks)),
                "hidden_chunks_with_relief": int(sum(1 for c in chunks if c["height_range"] > 1.0)),
            })
            renderable.append((metrics["hole_quad_count"], map_name, group, row_id, key, mask))
        print(f"{map_name:20s} {holed_tiles:>4} holed tiles", flush=True)

    if not all_chunks:
        raise SystemExit("no holed chunks found in any supplied store")

    args.output.mkdir(parents=True, exist_ok=True)
    chunk_fields = list(all_chunks[0].keys())
    with (args.output / "hidden_chunks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=chunk_fields)
        writer.writeheader(); writer.writerows(all_chunks)
    tile_fields = list(tile_rows[0].keys())
    with (args.output / "holed_tiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tile_fields)
        writer.writeheader(); writer.writerows(tile_rows)

    census = bitmask_census(all_chunks)
    with_relief = sum(1 for c in all_chunks if c["height_range"] > 1.0)
    summary = {
        "schema": HOLES_SCHEMA,
        "holed_tiles": len(tile_rows),
        "holed_chunks": len(all_chunks),
        "holed_quads": int(sum(c["hole_quads"] for c in all_chunks)),
        "chunks_with_real_relief": with_relief,
        "chunks_with_real_relief_fraction": with_relief / len(all_chunks),
        "hidden_relief_median": float(np.median([c["height_range"] for c in all_chunks])),
        "fully_holed_chunks": int(sum(1 for c in all_chunks if c["fully_holed"])),
        "partial_holed_chunks": int(sum(1 for c in all_chunks if not c["fully_holed"])),
        "bitmask_census": census,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "hidden_chunks.json").write_text(
        json.dumps({"schema": HOLES_SCHEMA, "chunks": all_chunks}, indent=2), encoding="utf-8"
    )

    if args.render > 0:
        renderable.sort(key=lambda item: -item[0])
        for _, map_name, group, row_id, key, mask in renderable[: args.render]:
            height = np.asarray(group["height_257"][row_id], dtype=np.float32)
            art = None
            for name in ("minimap_rgb_authored", "minimap_rgb"):
                if name in group:
                    art = np.asarray(group[name][row_id], dtype=np.uint8)
                    if art.any():
                        break
                    art = None
            render_hidden_tile(
                height_257=height, mask=mask, minimap=art,
                title=f"{map_name}_{key[0]:02d}_{key[1]:02d}",
                output=args.output / "tiles" / f"{map_name}_{key[0]:02d}_{key[1]:02d}.png",
            )
        print(f"rendered {min(args.render, len(renderable))} of {len(renderable)} holed tiles",
              flush=True)

    print(f"\n[DONE] {len(tile_rows)} holed tiles, {len(all_chunks)} holed chunks "
          f"({with_relief} with real relief) -> {args.output}", flush=True)
    print(f"       distinct hole patterns: {census['distinct_masks']}, "
          f"top-8 cover {census['coverage_of_top_8']:.1%}", flush=True)
    return 0
