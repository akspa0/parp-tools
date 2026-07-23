"""Render a labeled contact sheet of EVERY per-tile signal in a v50 store, for one tile.

A visual "is the dataset right?" check: opens the store, picks a tile (default: the most
object-heavy occupied tile, or --tile-index / --tile-x/--tile-y), and lays out one panel per
per-tile array with a type-appropriate rendering (RGB minimaps as-is, normals as a normal map,
heights/masks with colormaps, integer instance/class masks colorized, small lattices upscaled).
Placement TABLE arrays (mddf_*/modf_*, indexed by offset, not per-tile images) are summarized as a
text note instead of drawn. Reusable: point --store at any v50 curriculum or per-map store.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


# Arrays that are concatenated placement tables (indexed via *_offset), not per-tile rasters.
_TABLE_ARRAYS = {
    "mddf_count", "mddf_model_ids", "mddf_placement_data", "mddf_placement_offset", "mddf_unique_ids",
    "modf_count", "modf_model_ids", "modf_placement_data", "modf_placement_offset", "modf_unique_ids",
}


def _pick_tile(group, index_rows: list[dict]) -> int:
    """Default tile: the occupied tile with the most doodads (busiest, best for eyeballing)."""
    import numpy as np

    n = group["height_257"].shape[0]
    counts = np.asarray(group["mddf_count"]) if "mddf_count" in group else np.zeros(n)
    # require real relief so we don't land on an empty water tile
    best = -1
    best_i = 0
    for i in range(n):
        if float(np.asarray(group["height_257"][i]).std()) < 2.0:
            continue
        if int(counts[i]) > best:
            best = int(counts[i])
            best_i = i
    return best_i


def _draw(ax, name: str, arr: np.ndarray) -> str:
    """Render one signal into ax; return a short label suffix describing what was shown."""
    a = np.asarray(arr)
    # RGB uint8 minimaps
    if a.ndim == 3 and a.shape[-1] == 3 and a.dtype == np.uint8:
        ax.imshow(a)
        return f"{a.shape[0]}x{a.shape[1]} rgb"
    # normal_xyz float HxWx3 -> normal map
    if a.ndim == 3 and a.shape[-1] == 3 and np.issubdtype(a.dtype, np.floating):
        ax.imshow((np.clip(a, -1, 1) * 0.5 + 0.5))
        return "normal map"
    # 4-layer stacks (alpha_256, mcly_*_4) -> reduce over layers
    if a.ndim == 3 and a.shape[-1] == 4:
        if np.issubdtype(a.dtype, np.integer):
            red = np.where((a != 0).any(-1), a.max(-1), 0).astype(np.float32)
        else:
            red = a.astype(np.float32).max(-1)
        ax.imshow(red, cmap="magma", interpolation="nearest")
        return "max over 4 layers"
    # 2D
    if a.ndim == 2:
        f = a.astype(np.float32)
        if np.issubdtype(a.dtype, np.integer) and f.max() > 1:
            ax.imshow(f, cmap="tab20", interpolation="nearest")
            return f"int ids max={int(f.max())}"
        if np.issubdtype(a.dtype, np.bool_):
            ax.imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            return f"bool {a.shape[0]}x{a.shape[1]}"
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
        cmap = "terrain" if "height" in name else ("gray" if "mask" in name or "shadow" in name else "viridis")
        ax.imshow(f, cmap=cmap, interpolation="nearest")
        return f"[{lo:.2f},{hi:.2f}]" if hi > lo else "empty (all %.2f)" % lo
    ax.text(0.5, 0.5, "unrenderable", ha="center", va="center")
    return "skip"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    import zarr

    ap = argparse.ArgumentParser(description="Render a per-tile validation contact sheet of all v50 signals")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--tile-index", type=int, default=None, help="row index; default = busiest occupied tile")
    ap.add_argument("--tile-x", type=int, default=None)
    ap.add_argument("--tile-y", type=int, default=None)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    index_rows = pq.read_table(args.store / "index.parquet").to_pylist()

    if args.tile_x is not None and args.tile_y is not None:
        row = next((i for i, r in enumerate(index_rows)
                    if int(r.get("tile_x", -1)) == args.tile_x and int(r.get("tile_y", -1)) == args.tile_y), None)
        if row is None:
            raise SystemExit(f"no row for tile ({args.tile_x},{args.tile_y})")
        tile = row
    elif args.tile_index is not None:
        tile = args.tile_index
    else:
        tile = _pick_tile(group, index_rows)

    meta = index_rows[tile]
    tile_label = f"{meta.get('map','?')} ({meta.get('tile_x','?')},{meta.get('tile_y','?')})  row {tile}"

    # per-tile raster signals only, sorted for stable layout
    names = [k for k in sorted(group.array_keys())
             if k not in _TABLE_ARRAYS and group[k].shape[0] == len(index_rows)]
    # count note from the placement tables
    n_doodads = int(np.asarray(group["mddf_count"])[tile]) if "mddf_count" in group else 0
    n_wmos = int(np.asarray(group["modf_count"])[tile]) if "modf_count" in group else 0

    cols = 6
    rows = math.ceil((len(names) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.7))
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(f"v50 signal validation — {tile_label}   |   doodads={n_doodads}  wmos={n_wmos}   |   {args.store.name}",
                 fontsize=11, y=0.997)

    for ax in axes:
        ax.axis("off")
    for ax, name in zip(axes, names, strict=False):
        try:
            suffix = _draw(ax, name, group[name][tile])
        except Exception as exc:  # noqa: BLE001 - one bad signal must not kill the sheet
            ax.text(0.5, 0.5, f"ERR\n{exc}", ha="center", va="center", fontsize=6, color="red")
            suffix = "error"
        ax.set_title(f"{name}\n{suffix}", fontsize=7)

    out = args.output or (args.store.parent / f"validation-sheet-{meta.get('map','tile')}-{meta.get('tile_x','x')}-{meta.get('tile_y','y')}.png")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"tile: {tile_label}  |  {len(names)} signal panels  |  doodads={n_doodads} wmos={n_wmos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
