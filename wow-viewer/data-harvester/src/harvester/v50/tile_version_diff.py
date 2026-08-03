"""Compare the same map's tiles across two client builds — what happened to the squeezed terrain.

The 0.5.3 analysis showed the alpha editor compressed terrain rather than erasing it. The obvious
next question is what became of those tiles: did a compressed tile get built out into real terrain
in a later client, stay compressed, or get flattened the rest of the way? Nobody can answer that
from one build, and the alpha side has never been catalogued, so the comparison has not existed.

This pairs two inventories by ``(tile_x, tile_y)`` and reports the transition per tile, plus a
side-by-side render of the pairs that changed information class. Both builds are rendered with the
SAME per-tile autostretch, so a tile that looks flat in one panel and detailed in the other differs
in its data, not in its treatment.

A caution the renders carry in their caption: height decoding is not identical across eras. Alpha
stores absolute world Z; 4.x stores per-chunk deltas summed onto a base at load, which destroys
relief below the float32 ULP at that altitude (~6.1e-05 at |Z|=515). So a later build reading
"bit-exact flat" may be a decode artifact, while any structure it DOES show is real. Transitions
toward flatness are therefore weaker evidence than transitions toward detail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION_DIFF_SCHEMA = "v50-tile-version-diff-v1"

INFORMATION_ORDER = ("bit_exact_flat", "trace", "coarse_terrain", "rich_terrain")
PANEL = 257
HEADER = 74
LABEL = 22


def _rank(information_class: str) -> int:
    return INFORMATION_ORDER.index(information_class) if information_class in INFORMATION_ORDER else -1


def pair_inventories(
    old_tiles: list[dict[str, Any]], new_tiles: list[dict[str, Any]]
) -> dict[str, Any]:
    """Pair two builds' inventories of the same map by tile coordinate."""
    old = {(t["tile_x"], t["tile_y"]): t for t in old_tiles}
    new = {(t["tile_x"], t["tile_y"]): t for t in new_tiles}
    shared = sorted(set(old) & set(new))

    pairs: list[dict[str, Any]] = []
    for key in shared:
        a, b = old[key], new[key]
        a_class = a.get("information_class", "unknown")
        b_class = b.get("information_class", "unknown")
        delta = _rank(b_class) - _rank(a_class)
        pairs.append({
            "tile_x": key[0],
            "tile_y": key[1],
            "old_information_class": a_class,
            "new_information_class": b_class,
            "old_levels": a.get("surviving_height_levels"),
            "new_levels": b.get("surviving_height_levels"),
            "old_range": a.get("height_range"),
            "new_range": b.get("height_range"),
            "old_classification": a.get("classification"),
            "new_classification": b.get("classification"),
            "old_row_id": a.get("row_id"),
            "new_row_id": b.get("row_id"),
            "transition": "unchanged" if delta == 0 else ("gained_detail" if delta > 0 else "lost_detail"),
            "information_rank_delta": delta,
        })
    return {
        "shared_tiles": len(shared),
        "only_in_old": sorted(f"{k[0]:02d}_{k[1]:02d}" for k in set(old) - set(new)),
        "only_in_new": sorted(f"{k[0]:02d}_{k[1]:02d}" for k in set(new) - set(old)),
        "pairs": pairs,
    }


def summarize_transitions(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Transition counts, and the specific question: what became of the degenerate tiles?"""
    counts: dict[str, int] = {}
    matrix: dict[str, dict[str, int]] = {}
    for pair in pairs:
        counts[pair["transition"]] = counts.get(pair["transition"], 0) + 1
        row = matrix.setdefault(pair["old_information_class"], {})
        row[pair["new_information_class"]] = row.get(pair["new_information_class"], 0) + 1

    was_degenerate = [p for p in pairs if p["old_information_class"] in ("bit_exact_flat", "trace")]
    return {
        "by_transition": dict(sorted(counts.items())),
        "transition_matrix": {k: dict(sorted(v.items())) for k, v in sorted(matrix.items())},
        "degenerate_in_old": len(was_degenerate),
        "degenerate_in_old_that_gained_detail": sum(
            1 for p in was_degenerate if p["transition"] == "gained_detail"
        ),
        "degenerate_in_old_still_degenerate": sum(
            1 for p in was_degenerate
            if p["new_information_class"] in ("bit_exact_flat", "trace")
        ),
    }


def _autostretch_panel(height: np.ndarray) -> np.ndarray:
    values = np.asarray(height, dtype=np.float64)
    span = float(values.max() - values.min())
    if span <= 0.0:
        return np.full((*values.shape, 3), 128, dtype=np.uint8)
    from harvester.v50.tile_composite import hillshade_np

    normalized = (values - values.min()) / span
    gray = np.rint(np.clip(hillshade_np(normalized), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def render_pair_sheet(
    old_height: np.ndarray,
    new_height: np.ndarray,
    pair: dict[str, Any],
    *,
    old_build: str,
    new_build: str,
    output: Path,
) -> None:
    """Two builds of one tile, side by side, under an identical autostretch."""
    panels = (_autostretch_panel(old_height), _autostretch_panel(new_height))
    titles = (f"{old_build}  ({pair['old_information_class']}, {pair['old_levels']} levels)",
              f"{new_build}  ({pair['new_information_class']}, {pair['new_levels']} levels)")
    sheet = Image.new("RGB", (2 * PANEL, HEADER + PANEL + LABEL), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 6), f"tile {pair['tile_x']},{pair['tile_y']}   {pair['transition'].upper()}",
              fill=(240, 242, 246), font=font)
    draw.text((8, 24), f"range {pair['old_range']:.6g} -> {pair['new_range']:.6g}   "
                       f"levels {pair['old_levels']} -> {pair['new_levels']}",
              fill=(214, 220, 228), font=font)
    draw.text((8, 42), "Both panels: per-tile autostretch + raking hillshade. Identical treatment;"
                       " only the data differs.", fill=(170, 178, 190), font=font)
    draw.text((8, 58), "Note: later builds decode height as float32(delta+base), losing relief below"
                       " ~6.1e-05 at |Z|=515.", fill=(170, 178, 190), font=font)
    for column, (panel, title) in enumerate(zip(panels, titles)):
        x = column * PANEL
        sheet.paste(Image.fromarray(panel, mode="RGB"), (x, HEADER))
        draw.rectangle((x, HEADER + PANEL, x + PANEL, HEADER + PANEL + LABEL), fill=(28, 31, 37))
        draw.text((x + 6, HEADER + PANEL + 6), title, fill=(218, 222, 228), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    import argparse

    import zarr

    parser = argparse.ArgumentParser(
        description="Compare one map's tiles across two client builds"
    )
    parser.add_argument("--old-inventory", required=True, type=Path)
    parser.add_argument("--new-inventory", required=True, type=Path)
    parser.add_argument("--old-store", required=True, type=Path)
    parser.add_argument("--new-store", required=True, type=Path)
    parser.add_argument("--map", required=True, help="map name to compare (must exist in both)")
    parser.add_argument("--old-build", default="old")
    parser.add_argument("--new-build", default="new")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render", type=int, default=24,
                        help="how many CHANGED pairs to render side by side (0 disables)")
    args = parser.parse_args()

    def _load(path: Path) -> list[dict[str, Any]]:
        tiles = json.loads((path / "tiles.json").read_text(encoding="utf-8"))["tiles"]
        return [t for t in tiles if t["map"] == args.map]

    old_tiles, new_tiles = _load(args.old_inventory), _load(args.new_inventory)
    if not old_tiles or not new_tiles:
        raise SystemExit(f"map {args.map!r} missing from one inventory "
                         f"(old={len(old_tiles)}, new={len(new_tiles)})")

    paired = pair_inventories(old_tiles, new_tiles)
    summary = summarize_transitions(paired["pairs"])
    print(f"{args.map}: {paired['shared_tiles']} shared tiles "
          f"(+{len(paired['only_in_new'])} new, -{len(paired['only_in_old'])} dropped)")
    for name, count in summary["by_transition"].items():
        print(f"   {count:>5}  {name}")
    print(f"   degenerate in {args.old_build}: {summary['degenerate_in_old']}  ->  "
          f"gained detail: {summary['degenerate_in_old_that_gained_detail']}, "
          f"still degenerate: {summary['degenerate_in_old_still_degenerate']}", flush=True)

    if args.render > 0:
        old_group = zarr.open_group(str(args.old_store), mode="r")
        new_group = zarr.open_group(str(args.new_store), mode="r")
        changed = [p for p in paired["pairs"] if p["transition"] != "unchanged"]
        # Biggest information swing first — the pairs that actually carry the story.
        changed.sort(key=lambda p: -abs(p["information_rank_delta"] or 0) * 1e9
                     - abs((p["new_levels"] or 0) - (p["old_levels"] or 0)))
        for pair in changed[: args.render]:
            render_pair_sheet(
                np.asarray(old_group["height_257"][pair["old_row_id"]], dtype=np.float32),
                np.asarray(new_group["height_257"][pair["new_row_id"]], dtype=np.float32),
                pair,
                old_build=args.old_build, new_build=args.new_build,
                output=args.output / "pairs" / f"{args.map}_{pair['tile_x']:02d}_{pair['tile_y']:02d}.png",
            )
        print(f"   rendered {min(args.render, len(changed))} of {len(changed)} changed pairs",
              flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "version-diff.json").write_text(
        json.dumps({"schema": VERSION_DIFF_SCHEMA, "map": args.map,
                    "old_build": args.old_build, "new_build": args.new_build,
                    **paired, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] -> {args.output}", flush=True)
    return 0
