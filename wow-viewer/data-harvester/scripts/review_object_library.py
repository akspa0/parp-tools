"""Render review artifacts for the spec 077 per-object capture library.

The script reads a built library Zarr store and emits:

  <output-dir>/
      index.html         top-level summary with library stats
      families/          per-family contact sheets (image + mask)
      assets.json        per-entry metadata snapshot

This is the operator review surface for the first proof slice. It is NOT a
training artifact; use the Zarr store + Parquet tables directly for that.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import zarr
import zarr.storage

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.object_library import library_id_from_asset_path  # noqa: E402


def _read_assets_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    table = pq.read_table(str(path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


def _read_index_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    table = pq.read_table(str(path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


def _open_zarr_array(store_path: Path, key: str) -> np.ndarray | None:
    full = store_path / key
    if not full.exists():
        return None
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    if key not in root:
        return None
    return np.asarray(root[key][:])


def _render_contact_sheet(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    cols: int = 4,
    cell_size: int = 128,
) -> None:
    rows = max(1, (len(images) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_size, rows * cell_size * 2), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (img, mask, label) in enumerate(zip(images, masks, labels)):
        row = i // cols
        col = i % cols
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        cell = Image.fromarray(img).resize((cell_size, cell_size), Image.NEAREST)
        sheet.paste(cell, (col * cell_size, row * cell_size * 2))
        if mask.ndim == 2:
            mask_img = Image.fromarray(mask).convert("RGB").resize((cell_size, cell_size), Image.NEAREST)
        else:
            mask_img = Image.fromarray(mask[:, :, 0]).convert("RGB").resize((cell_size, cell_size), Image.NEAREST)
        sheet.paste(mask_img, (col * cell_size, row * cell_size * 2 + cell_size))
        draw.text((col * cell_size + 4, row * cell_size * 2 + 4), label[:24], fill=(255, 255, 0))
    sheet.save(output_path)


def _render_html_index(
    output_dir: Path,
    store_path: Path,
    assets: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    stats: dict[str, Any],
) -> Path:
    families_dir = output_dir / "families"
    families_dir.mkdir(parents=True, exist_ok=True)

    status_counts = Counter(str(a.get("capture_status", "unknown")) for a in assets)
    asset_type_counts = Counter(str(a.get("asset_type", "unknown")) for a in assets)
    visibility_counts = Counter(str(a.get("visibility_class", "unknown")) for a in assets)

    rgb = _open_zarr_array(store_path, "capture_rgb")
    mask = _open_zarr_array(store_path, "capture_mask")

    # Group variants by library id so each family gets one contact sheet.
    by_library: dict[str, list[dict[str, Any]]] = {}
    for v in variants:
        by_library.setdefault(str(v["library_id"]), []).append(v)

    for library_id, family_variants in by_library.items():
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        labels: list[str] = []
        for v in family_variants:
            try:
                idx = int(str(v["image_key"]).rsplit("/", 1)[-1])
            except (IndexError, ValueError):
                continue
            if rgb is None or mask is None or idx >= rgb.shape[0]:
                continue
            images.append(rgb[idx])
            masks.append(mask[idx])
            labels.append(str(v.get("variant_id", library_id))[:24])
        if not images:
            continue
        sheet_path = families_dir / f"{library_id}.png"
        _render_contact_sheet(images, masks, labels, sheet_path)

    rows_html: list[str] = []
    for asset in assets:
        rows_html.append(
            "<tr>"
            + f"<td>{html.escape(str(asset.get('library_id', '')))}</td>"
            + f"<td>{html.escape(str(asset.get('normalized_asset_path', '')))}</td>"
            + f"<td>{html.escape(str(asset.get('asset_type', '')))}</td>"
            + f"<td>{html.escape(str(asset.get('capture_status', '')))}</td>"
            + f"<td>{html.escape(str(asset.get('visibility_class', '')))}</td>"
            + f"<td>{int(asset.get('placement_observation_count', 0) or 0)}</td>"
            + f"<td>{html.escape(str(asset.get('source_builds', '')))}</td>"
            + f"<td><img src='families/{html.escape(str(asset.get('library_id', '')))}.png' width='128' height='64' loading='lazy'></td>"
            + "</tr>"
        )

    html_path = output_dir / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Object Library Review — {html.escape(store_path.name)}</title>
<style>
body {{ font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #444; padding: 4px 8px; text-align: left; }}
th {{ background: #2a2a2a; }}
img {{ image-rendering: pixelated; }}
</style></head>
<body>
<h1>Object Library Review — {html.escape(store_path.name)}</h1>
<p>Generated {datetime.now(timezone.utc).isoformat()}</p>
<h2>Summary</h2>
<ul>
<li>Entries: {stats.get('entry_count', 0)}</li>
<li>Variants: {stats.get('variant_count', 0)}</li>
<li>Target size: {stats.get('target_size', 0)}</li>
<li>capture_status: {dict(status_counts)}</li>
<li>asset_type: {dict(asset_type_counts)}</li>
<li>visibility_class: {dict(visibility_counts)}</li>
</ul>
<h2>Entries</h2>
<table><thead><tr>
<th>library_id</th><th>asset_path</th><th>asset_type</th><th>capture_status</th>
<th>visibility_class</th><th>observation_count</th><th>source_builds</th><th>preview</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>
</body></html>
""",
        encoding="utf-8",
    )
    return html_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render spec 077 per-object capture library review artifacts."
    )
    parser.add_argument("--library", type=Path, required=True,
                        help="Path to a built <run-name>.zarr library store.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory under which index.html and families/ are written.")
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.library.exists():
        print(f"Library not found: {args.library}", file=sys.stderr)
        return 2
    assets_path = args.library / "assets.parquet"
    index_path = args.library / "index.parquet"

    assets = _read_assets_parquet(assets_path)
    variants = _read_index_parquet(index_path)
    if not assets and not variants:
        print(f"Library is empty (no assets.parquet or index.parquet)", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_json = args.output_dir / "assets.json"
    assets_json.write_text(json.dumps(assets, indent=2, sort_keys=True), encoding="utf-8")

    stats = {
        "entry_count": len(assets),
        "variant_count": len(variants),
        "target_size": 0,
    }
    attrs = {}
    store = zarr.storage.LocalStore(str(args.library), read_only=True)
    root = zarr.open_group(store, mode="r")
    attrs = dict(root.attrs)
    if "target_size" in attrs:
        stats["target_size"] = int(attrs["target_size"])

    html_path = _render_html_index(args.output_dir, args.library, assets, variants, stats)
    print(f"Wrote review HTML to {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main_with_args())
