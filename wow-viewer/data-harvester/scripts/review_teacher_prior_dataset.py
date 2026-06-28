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


def _render_contact_sheet(
    raw: np.ndarray,
    mask: np.ndarray,
    prior: np.ndarray,
    indices: list[int],
    output_path: Path,
    *,
    cell_size: int = 128,
    cols: int = 4,
) -> None:
    rows = max(1, (len(indices) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_size, rows * cell_size * 3), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        for band in range(3):
            if band == 0:
                arr = raw[idx]
            elif band == 1:
                arr = np.stack([mask[idx]] * 3, axis=-1) * 255
            else:
                arr = prior[idx][:, :, :3]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr).resize((cell_size, cell_size), Image.NEAREST)
            sheet.paste(img, (col * cell_size, row * cell_size * 3 + band * cell_size))
        draw.text(
            (col * cell_size + 4, row * cell_size * 3 + 4),
            f"tile {idx}",
            fill=(255, 255, 0),
        )
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
    parser.add_argument("--prefer-mask-source", type=str, default="object_filtered_mask",
                        help="Filter rendered tiles to those using the given mask source.")
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

    preferred_indices = [
        i for i, t in enumerate(tiles)
        if t.get("filtered_mask_source") == args.prefer_mask_source
    ]
    if not preferred_indices:
        preferred_indices = list(range(len(tiles)))
    selected = preferred_indices[: args.max_tiles]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_path = args.output_dir / "contact_sheet.png"
    _render_contact_sheet(raw, mask, prior, selected, contact_path)

    summary = {
        "build": args.library.stem.replace(".zarr", ""),
        "tile_count": len(tiles),
        "mask_source_counts": dict(source_counts),
        "coverage_mean": float(coverage.mean()) if coverage.size else 0.0,
        "coverage_median": float(np.median(coverage)) if coverage.size else 0.0,
        "coverage_max": float(coverage.max()) if coverage.size else 0.0,
        "preferred_mask_source": args.prefer_mask_source,
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
