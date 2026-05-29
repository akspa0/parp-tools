"""Patch object-roof masks into existing V16 Zarr stores.

This script produces a new auxiliary signal for training:

  - object_roof_mask: (N, 256, 256) float32
  - object_roof_confidence: (N, 256, 256) float32

Mask sources are combined in precedence order:
1) placement metadata projection (preferred)
2) learned fallback masks from an external prediction directory (optional)
3) heuristic fallback from existing object mask arrays in the Zarr store

The index parquet is updated with has_object_roof_mask and
object_roof_mask_source fields for auditability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.object_roof import (
    d1_style_bbox_fallback,
    is_probable_roof_asset,
    normalize_asset_path,
    world_bbox_to_tile_bbox_xyxy,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_DEFAULT_REPORT_ROOT = _PROJECT_ROOT / "output" / "tmp" / "object_roof_patch_reports"

_SOURCE_NONE = "none"
_SOURCE_METADATA = "metadata"
_SOURCE_LEARNED = "learned"
_SOURCE_HEURISTIC = "heuristic"


def _label_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "description": "Object-roof mask label contract for minimap inputs",
        "arrays": {
            "object_roof_mask": {
                "shape": ["N", 256, 256],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "semantics": "Object/building roof coverage mask in minimap pixel space",
            },
            "object_roof_confidence": {
                "shape": ["N", 256, 256],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "semantics": "Per-pixel confidence for object_roof_mask",
            },
        },
        "index_fields": [
            "has_object_roof_mask",
            "object_roof_mask_source",
        ],
        "source_precedence": [
            _SOURCE_METADATA,
            _SOURCE_LEARNED,
            _SOURCE_HEURISTIC,
            _SOURCE_NONE,
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch object-roof mask arrays into V16 Zarr stores.")
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--build", type=str, default=None)
    parser.add_argument("--builds", nargs="+", default=None)
    parser.add_argument("--learned-mask-dir", type=Path, default=None)
    parser.add_argument("--include-mddf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--roof-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bbox-padding", type=int, default=3)
    parser.add_argument("--metadata-confidence", type=float, default=0.95)
    parser.add_argument("--learned-confidence", type=float, default=0.70)
    parser.add_argument("--heuristic-confidence", type=float, default=0.45)
    parser.add_argument("--allow-zarr-write", action="store_true")
    parser.add_argument("--write-panels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-panels", type=int, default=24)
    parser.add_argument("--report-root", type=Path, default=_DEFAULT_REPORT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def _require_explicit_write(args: argparse.Namespace) -> None:
    if not args.allow_zarr_write:
        raise RuntimeError(
            "Refusing to mutate Zarr stores without --allow-zarr-write. "
            "This patch writes object-roof arrays in-place."
        )


def _resolve_builds(dataset_root: Path, args: argparse.Namespace) -> list[str]:
    if args.builds:
        return [str(item) for item in args.builds]
    if args.build:
        return [str(args.build)]
    return [path.stem.replace(".zarr", "") for path in sorted(dataset_root.glob("*.zarr"))]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(str(path))
    return [{column: table.column(column)[idx].as_py() for column in table.column_names} for idx in range(table.num_rows)]


def _write_index(index_rows: list[dict[str, Any]], output_path: Path) -> None:
    table = pa.Table.from_pylist(index_rows)
    pq.write_table(table, str(output_path / "index.parquet"))


def _load_tile_maps(index_rows: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    tile_meta: dict[int, dict[str, Any]] = {}
    tile_to_row: dict[int, int] = {}
    for idx, row in enumerate(index_rows):
        tile_id = int(row.get("tile_id", idx))
        tile_meta[tile_id] = {
            "map": str(row.get("map", "")),
            "tile_x": int(row.get("tile_x", -1) if row.get("tile_x") is not None else -1),
            "tile_y": int(row.get("tile_y", -1) if row.get("tile_y") is not None else -1),
        }
        tile_to_row[tile_id] = idx
    return tile_meta, tile_to_row


def _project_bbox(row: dict[str, Any], tile_x: int, tile_y: int, padding: int, build: str = "") -> tuple[int, int, int, int] | None:
    if str(row.get("instance_type", "")).lower() == "modf":
        return world_bbox_to_tile_bbox_xyxy(
            min_x=float(row.get("bbMinX", 0.0) or 0.0),
            min_y=float(row.get("bbMinY", 0.0) or 0.0),
            max_x=float(row.get("bbMaxX", 0.0) or 0.0),
            max_y=float(row.get("bbMaxY", 0.0) or 0.0),
            tile_x=tile_x,
            tile_y=tile_y,
            padding_px=padding,
            build=build,
        )
    return d1_style_bbox_fallback(
        pos_x=float(row.get("posX", 0.0) or 0.0),
        pos_y=float(row.get("posY", 0.0) or 0.0),
        scale=float(row.get("scale", 1.0) or 1.0),
        tile_x=tile_x,
        tile_y=tile_y,
        base_radius_px=6.0 + float(padding),
        build=build,
    )


def _heuristic_mask_for_tile(root: zarr.Group, tile_id: int) -> np.ndarray:
    if "object_precise_mask" in root:
        return np.clip(root["object_precise_mask"][tile_id][:256, :256].astype(np.float32), 0.0, 1.0)
    if "object_filtered_mask" in root:
        return np.clip(root["object_filtered_mask"][tile_id][:256, :256].astype(np.float32), 0.0, 1.0)
    if "object_mask" in root:
        return np.clip(root["object_mask"][tile_id][:256, :256].astype(np.float32), 0.0, 1.0)
    return np.zeros((256, 256), dtype=np.float32)


def _load_learned_mask(
    *,
    learned_mask_dir: Path | None,
    build: str,
    map_name: str,
    tile_x: int,
    tile_y: int,
) -> np.ndarray | None:
    if learned_mask_dir is None:
        return None
    stem = f"{build}_{map_name}_{tile_x}_{tile_y}"
    npy_path = learned_mask_dir / f"{stem}.npy"
    png_path = learned_mask_dir / f"{stem}.png"
    if npy_path.exists():
        arr = np.asarray(np.load(npy_path), dtype=np.float32)
    elif png_path.exists():
        arr = np.asarray(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
    else:
        return None

    if arr.shape != (256, 256):
        ys = np.linspace(0, arr.shape[0] - 1, 256).astype(np.int32)
        xs = np.linspace(0, arr.shape[1] - 1, 256).astype(np.int32)
        arr = arr[np.ix_(ys, xs)]
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _resolve_report_dir(args: argparse.Namespace, build: str) -> Path:
    run_name = str(args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out = Path(args.report_root) / run_name / str(build)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _apply_build(args: argparse.Namespace, build: str) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    zarr_path = dataset_root / f"{build}.zarr"
    if not zarr_path.exists():
        return {"build": build, "status": "missing_store"}

    index_path = zarr_path / "index.parquet"
    placements_path = zarr_path / "placements.parquet"
    if not index_path.exists() or not placements_path.exists():
        return {"build": build, "status": "missing_index_or_placements"}

    index_rows = _read_rows(index_path)
    placements_rows = _read_rows(placements_path)
    tile_meta, tile_to_row = _load_tile_maps(index_rows)
    report_dir = _resolve_report_dir(args, build)

    n_tiles = len(index_rows)
    roof_mask = np.zeros((n_tiles, 256, 256), dtype=np.float32)
    roof_conf = np.zeros((n_tiles, 256, 256), dtype=np.float32)
    source_labels: list[str] = [_SOURCE_NONE for _ in range(n_tiles)]

    metadata_hits = 0
    learned_hits = 0
    heuristic_hits = 0
    rejected_non_roof = 0
    rejected_invalid_bbox = 0

    # Step 1: placement-driven metadata masks.
    for row in placements_rows:
        tile_id = int(row.get("tile_id", -1))
        row_idx = tile_to_row.get(tile_id)
        if row_idx is None:
            continue
        tile = tile_meta[tile_id]
        tile_x = int(tile["tile_x"])
        tile_y = int(tile["tile_y"])
        if tile_x < 0 or tile_y < 0:
            continue

        instance_type = str(row.get("instance_type", "")).lower()
        if instance_type == "mddf" and not bool(args.include_mddf):
            continue

        asset_path = normalize_asset_path(str(row.get("asset_path", "")))
        if bool(args.roof_only) and not is_probable_roof_asset(asset_path):
            rejected_non_roof += 1
            continue

        bbox = _project_bbox(row, tile_x=tile_x, tile_y=tile_y, padding=int(args.bbox_padding), build=build)
        if bbox is None:
            rejected_invalid_bbox += 1
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        if x1 <= x0 or y1 <= y0:
            rejected_invalid_bbox += 1
            continue

        roof_mask[row_idx, y0 : y1 + 1, x0 : x1 + 1] = 1.0
        roof_conf[row_idx, y0 : y1 + 1, x0 : x1 + 1] = float(args.metadata_confidence)
        source_labels[row_idx] = _SOURCE_METADATA
        metadata_hits += 1

    store = zarr.storage.LocalStore(str(zarr_path), read_only=False)
    root = zarr.open_group(store=store, mode="a")
    try:
        # Step 2: learned fallback for metadata-empty tiles.
        for tile_id, tile in tile_meta.items():
            row_idx = tile_to_row[tile_id]
            if float(roof_mask[row_idx].sum()) > 0.0:
                continue

            learned = _load_learned_mask(
                learned_mask_dir=Path(args.learned_mask_dir) if args.learned_mask_dir is not None else None,
                build=build,
                map_name=str(tile["map"]),
                tile_x=int(tile["tile_x"]),
                tile_y=int(tile["tile_y"]),
            )
            if learned is not None and float(learned.sum()) > 0.0:
                roof_mask[row_idx] = np.maximum(roof_mask[row_idx], learned)
                roof_conf[row_idx] = np.maximum(roof_conf[row_idx], learned * float(args.learned_confidence))
                source_labels[row_idx] = _SOURCE_LEARNED
                learned_hits += 1

        # Step 3: heuristic fallback for remaining empty tiles.
        for tile_id in tile_meta:
            row_idx = tile_to_row[tile_id]
            if float(roof_mask[row_idx].sum()) > 0.0:
                continue
            heuristic = _heuristic_mask_for_tile(root, tile_id)
            if float(heuristic.sum()) <= 0.0:
                continue
            roof_mask[row_idx] = np.maximum(roof_mask[row_idx], heuristic)
            roof_conf[row_idx] = np.maximum(roof_conf[row_idx], heuristic * float(args.heuristic_confidence))
            source_labels[row_idx] = _SOURCE_HEURISTIC
            heuristic_hits += 1

        codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
        root.create_array(
            "object_roof_mask",
            data=roof_mask,
            chunks=(1, 256, 256),
            compressors=codec,
            overwrite=True,
        )
        root.create_array(
            "object_roof_confidence",
            data=roof_conf,
            chunks=(1, 256, 256),
            compressors=codec,
            overwrite=True,
        )
        contract = _label_contract()
        root.attrs["object_roof_label_contract_version"] = str(contract["schema_version"])
        root.attrs["object_roof_label_source_precedence"] = list(contract["source_precedence"])
    finally:
        store.close()

    (report_dir / "object_roof_label_contract.json").write_text(json.dumps(_label_contract(), indent=2), encoding="utf-8")

    for idx, row in enumerate(index_rows):
        has_mask = bool(float(roof_mask[idx].sum()) > 0.0)
        row["has_object_roof_mask"] = has_mask
        row["object_roof_mask_source"] = source_labels[idx]

    _write_index(index_rows, zarr_path)

    panel_paths: list[str] = []
    if bool(args.write_panels):
        panel_dir = report_dir / "object_roof_review"
        panel_dir.mkdir(parents=True, exist_ok=True)

        store_r = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root_r = zarr.open_group(store=store_r, mode="r")
        try:
            ranked = sorted(
                range(n_tiles),
                key=lambda idx: float(roof_mask[idx].mean()),
                reverse=True,
            )
            for panel_idx, idx in enumerate(ranked[: int(args.max_panels)]):
                source = source_labels[idx]
                if source == _SOURCE_NONE:
                    continue
                minimap = root_r["minimap_rgb"][idx].astype(np.uint8)
                mask = np.clip(roof_mask[idx], 0.0, 1.0)

                mask_u8 = (mask * 255.0).astype(np.uint8)
                mask_rgb = np.repeat(mask_u8[:, :, None], 3, axis=2)
                overlay = minimap.copy()
                overlay[mask >= 0.2, 0] = 255
                overlay[mask >= 0.2, 1] = np.maximum(overlay[mask >= 0.2, 1] // 2, 32)

                left = Image.fromarray(minimap, mode="RGB")
                mid = Image.fromarray(mask_rgb, mode="RGB")
                right = Image.fromarray(overlay, mode="RGB")

                canvas = Image.new("RGB", (256 * 3, 256 + 18), color=(0, 0, 0))
                canvas.paste(left, (0, 18))
                canvas.paste(mid, (256, 18))
                canvas.paste(right, (512, 18))

                draw = ImageDraw.Draw(canvas)
                draw.rectangle([(0, 0), (canvas.width, 17)], fill=(18, 18, 18))
                row = index_rows[idx]
                text = (
                    f"tile={int(row.get('tile_id', idx))} "
                    f"{row.get('map', '')}_{int(row.get('tile_x', -1))}_{int(row.get('tile_y', -1))} "
                    f"source={source} cov={float(mask.mean()):.4f}"
                )
                draw.text((4, 3), text, fill=(235, 235, 235))

                out_path = panel_dir / f"panel_{panel_idx:03d}.png"
                canvas.save(out_path)
                panel_paths.append(str(out_path))
        finally:
            store_r.close()

    report = {
        "build": build,
        "status": "patched",
        "tiles": n_tiles,
        "metadata_hits": int(metadata_hits),
        "learned_hits": int(learned_hits),
        "heuristic_hits": int(heuristic_hits),
        "non_empty_masks": int(sum(1 for idx in range(n_tiles) if float(roof_mask[idx].sum()) > 0.0)),
        "mean_mask_coverage": float(roof_mask.mean()) if n_tiles > 0 else 0.0,
        "rejected_non_roof": int(rejected_non_roof),
        "rejected_invalid_bbox": int(rejected_invalid_bbox),
        "panel_count": len(panel_paths),
        "panel_paths": panel_paths,
        "report_dir": str(report_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (report_dir / "object_roof_patch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    _require_explicit_write(args)

    builds = _resolve_builds(Path(args.dataset_root), args)
    if not builds:
        raise RuntimeError(f"No V16 stores found in {args.dataset_root}")

    reports: list[dict[str, Any]] = []
    for build in builds:
        report = _apply_build(args, build)
        reports.append(report)
        print(json.dumps(report, indent=2))

    summary = {
        "build_count": len(builds),
        "patched_count": int(sum(1 for report in reports if report.get("status") == "patched")),
        "reports": reports,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
