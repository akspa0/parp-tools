"""Render review artifacts for the spec 077 teacher-prior dataset (T019).

Reads a built teacher-prior Zarr store and emits:

  <output-dir>/
      index.html
      contact_sheet.png   raw / mask / suppressed-RGB tiles in a grid
      summary.json        per-tile coverage histogram + mask source counts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import zarr
import zarr.storage

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _open_zarr_array(store_path: Path, key: str) -> np.ndarray | None:
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    if key not in root:
        return None
    return np.asarray(root[key][:])


def _read_tiles_parquet(path: Path) -> list[dict]:
    if not path.exists():
        return []
    table = pq.read_table(str(path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


def _resize_mask_nearest(mask: np.ndarray, target_h: int = 256, target_w: int = 256) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return mask
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask for resize; got {mask.shape}")
    ys = np.linspace(0, mask.shape[0] - 1, target_h).astype(np.int64)
    xs = np.linspace(0, mask.shape[1] - 1, target_w).astype(np.int64)
    return mask[np.ix_(ys, xs)]


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(np.float32, copy=False)
    if float(arr.max(initial=0.0)) <= 1.0:
        arr = arr * 255.0
    return np.stack([arr.clip(0, 255).astype(np.uint8)] * 3, axis=-1)


def _render_contact_sheet(
    raw: np.ndarray,
    mask: np.ndarray,
    prior: np.ndarray,
    tiles: list[dict],
    indices: list[int],
    output_path: Path,
    *,
    source_masks: dict[str, np.ndarray] | None = None,
    cell_size: int = 128,
    cols: int = 4,
) -> None:
    rows = max(1, (len(indices) + cols - 1) // cols)
    has_sources = bool(source_masks)
    bands = 8 if has_sources else 5
    sheet = Image.new("RGB", (cols * cell_size, rows * cell_size * bands), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        tile = tiles[idx] if idx < len(tiles) else {}
        tile_id = int(tile.get("tile_id", idx))
        mask_rgb = _mask_to_rgb(mask[idx])
        raw_rgb = raw[idx].astype(np.uint8, copy=False)
        prior_rgb = prior[idx][:, :, :3].astype(np.uint8, copy=False)
        overlay = raw_rgb.copy()
        mask_bool = mask[idx] > 0
        overlay[mask_bool, 0] = 255
        overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.25).astype(np.uint8)
        overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.25).astype(np.uint8)
        diff = np.abs(raw_rgb.astype(np.int16) - prior_rgb.astype(np.int16)).max(axis=2).astype(np.uint8)
        diff_rgb = np.stack([diff] * 3, axis=-1)
        panels = [
            ("raw", raw_rgb),
            ("teacher mask", mask_rgb.astype(np.uint8, copy=False)),
        ]
        if source_masks:
            source_tile_id = int(tile.get("tile_id", idx))
            for name in ("object_precise_mask", "object_filtered_mask", "object_mask"):
                source_arr = source_masks.get(name)
                if source_arr is not None and source_tile_id < source_arr.shape[0]:
                    panels.append((name, _mask_to_rgb(_resize_mask_nearest(source_arr[source_tile_id]))))
                else:
                    panels.append((name, np.zeros_like(raw_rgb)))
        panels.extend([
            ("overlay", overlay),
            ("suppressed", prior_rgb),
            ("changed", diff_rgb),
        ])
        for band, (label, arr) in enumerate(panels):
            if band == 0:
                text = f"row {idx} tile {tile_id} raw"
            else:
                text = label
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr).resize((cell_size, cell_size), Image.NEAREST)
            y = row * cell_size * bands + band * cell_size
            sheet.paste(img, (col * cell_size, y))
            draw.rectangle((col * cell_size, y, col * cell_size + cell_size - 1, y + 11), fill=(0, 0, 0))
            draw.text((col * cell_size + 4, y + 1), text, fill=(255, 255, 0))
    sheet.save(output_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render spec 077 teacher-prior review artifacts."
    )
    parser.add_argument("--library", type=Path, required=True,
                        help="Path to a built <build>.zarr teacher-prior store.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory under which index.html / contact_sheet.png are written.")
    parser.add_argument("--max-tiles", type=int, default=16,
                        help="Number of tiles to render in the contact sheet.")
    parser.add_argument("--prefer-mask-source", type=str, default="object_precise_mask",
                        help="Filter rendered tiles to those using the given mask source.")
    parser.add_argument("--tile-id", type=int, nargs="*", default=None,
                        help="Render specific original tile_id values instead of the first preferred rows.")
    parser.add_argument("--row-index", type=int, nargs="*", default=None,
                        help="Render specific compact teacher-prior row indices. Use this to reproduce old contact-sheet labels.")
    parser.add_argument("--v18-path", type=Path, default=None,
                        help="Optional source V18 store. When provided, render object_precise/filter/object masks beside the teacher mask.")
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.library.exists():
        print(f"Library not found: {args.library}", file=sys.stderr)
        return 2

    raw = _open_zarr_array(args.library, "raw_minimap_rgb_256")
    mask = _open_zarr_array(args.library, "teacher_object_mask_256")
    prior = _open_zarr_array(args.library, "processed_minimap_prior_256")
    if raw is None or mask is None or prior is None:
        print(f"Library missing one of raw/mask/prior arrays", file=sys.stderr)
        return 2

    tiles = _read_tiles_parquet(args.library / "tiles.parquet")
    source_counts = Counter(str(t.get("filtered_mask_source", "")) for t in tiles)
    coverage = np.array([t.get("teacher_object_cov", 0.0) for t in tiles], dtype=np.float32)

    if args.row_index:
        preferred_indices = [
            int(row_index) for row_index in args.row_index
            if 0 <= int(row_index) < len(tiles)
        ]
    elif args.tile_id:
        requested = set(int(tile_id) for tile_id in args.tile_id)
        preferred_indices = [
            i for i, t in enumerate(tiles)
            if int(t.get("tile_id", -1)) in requested
        ]
    else:
        preferred_indices = [
            i for i, t in enumerate(tiles)
            if t.get("filtered_mask_source") == args.prefer_mask_source
        ]
    if not preferred_indices:
        preferred_indices = list(range(len(tiles)))
    selected = preferred_indices[: args.max_tiles]

    source_masks: dict[str, np.ndarray] = {}
    if args.v18_path is not None:
        for key in ("object_precise_mask", "object_filtered_mask", "object_mask"):
            arr = _open_zarr_array(args.v18_path, key)
            if arr is not None:
                source_masks[key] = arr

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_path = args.output_dir / "contact_sheet.png"
    _render_contact_sheet(raw, mask, prior, tiles, selected, contact_path, source_masks=source_masks or None)

    summary = {
        "build": args.library.stem.replace(".zarr", ""),
        "tile_count": len(tiles),
        "mask_source_counts": dict(source_counts),
        "coverage_mean": float(coverage.mean()) if coverage.size else 0.0,
        "coverage_median": float(np.median(coverage)) if coverage.size else 0.0,
        "coverage_max": float(coverage.max()) if coverage.size else 0.0,
        "preferred_mask_source": args.prefer_mask_source,
        "requested_row_indices": [int(row_index) for row_index in args.row_index] if args.row_index else None,
        "requested_tile_ids": [int(tile_id) for tile_id in args.tile_id] if args.tile_id else None,
        "selected_tile_ids": [int(tiles[i].get("tile_id", i)) for i in selected],
        "selected_rows": [int(i) for i in selected],
        "source_v18_path": str(args.v18_path) if args.v18_path is not None else None,
        "source_mask_arrays": sorted(source_masks),
        "contact_sheet_path": str(contact_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    rows_html = "".join(
        f"<tr><td>{int(t.get('tile_id', 0))}</td>"
        f"<td>{int(t.get('tile_x', 0))},{int(t.get('tile_y', 0))}</td>"
        f"<td>{t.get('map', '')}</td>"
        f"<td>{t.get('filtered_mask_source', '')}</td>"
        f"<td>{float(t.get('teacher_object_cov', 0.0)):.4f}</td></tr>"
        for t in tiles[:64]
    )
    html_path = args.output_dir / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Teacher Prior Review — {args.library.name}</title>
<style>
body {{ font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #444; padding: 4px 8px; text-align: left; }}
th {{ background: #2a2a2a; }}
img {{ image-rendering: pixelated; }}
</style></head>
<body>
<h1>Teacher Prior Review — {args.library.name}</h1>
<p>Generated {summary['generated_at']}</p>
<h2>Summary</h2>
<ul>
<li>tile_count: {summary['tile_count']}</li>
<li>mask_source_counts: {summary['mask_source_counts']}</li>
<li>coverage mean / median / max: {summary['coverage_mean']:.4f} / {summary['coverage_median']:.4f} / {summary['coverage_max']:.4f}</li>
</ul>
<h2>Contact Sheet</h2>
<p><img src='contact_sheet.png' alt='contact sheet'></p>
<h2>First 64 Tiles</h2>
<table><thead><tr>
<th>tile_id</th><th>tile_x,tile_y</th><th>map</th><th>mask_source</th><th>coverage</th>
</tr></thead><tbody>{rows_html}</tbody></table>
</body></html>
""",
        encoding="utf-8",
    )
    print(f"Wrote teacher-prior review to {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main_with_args())
