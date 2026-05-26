"""Validate bounded Spec-025 Phase-1 roof-library outputs.

Checks that a roof-library run emitted:

- roof exemplar catalog with stable IDs and asset metadata
- separate object_visual.zarr store with matching sample count
- family catalog with at least one building-heavy family

This is a bounded validation surface for tasks T006/T007.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq
import zarr
import zarr.storage


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LIBRARY_ROOT = _PROJECT_ROOT / "output" / "datasets" / "object_roof_library"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Spec-025 roof-library run outputs.")
    parser.add_argument("--library-dir", type=Path, default=None)
    parser.add_argument("--library-root", type=Path, default=_DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--min-exemplars", type=int, default=1)
    parser.add_argument("--min-families", type=int, default=1)
    parser.add_argument("--require-building-heavy-family", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emit-report", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_library_dir(args: argparse.Namespace) -> Path:
    if args.library_dir is not None:
        return Path(args.library_dir)
    candidates = sorted(Path(args.library_root).glob("*/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No roof-library run found under {args.library_root}")
    return candidates[0].parent


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing required parquet file: {path}")
    table = pq.read_table(str(path))
    return [{column: table.column(column)[idx].as_py() for column in table.column_names} for idx in range(table.num_rows)]


def _read_object_visual_attrs(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required object-visual datastore: {path}")
    store = zarr.storage.LocalStore(str(path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    try:
        required_arrays = ["roof_rgb", "roof_mask", "pose_vec", "bbox_xyxy"]
        for name in required_arrays:
            if name not in root:
                raise RuntimeError(f"Missing required object_visual array: {name}")

        shape_rgb = tuple(int(v) for v in root["roof_rgb"].shape)
        shape_mask = tuple(int(v) for v in root["roof_mask"].shape)
        shape_pose = tuple(int(v) for v in root["pose_vec"].shape)
        shape_bbox = tuple(int(v) for v in root["bbox_xyxy"].shape)
        attrs = dict(root.attrs)
    finally:
        store.close()

    return {
        "shape_rgb": shape_rgb,
        "shape_mask": shape_mask,
        "shape_pose": shape_pose,
        "shape_bbox": shape_bbox,
        "attrs": attrs,
    }


def _is_building_heavy(asset_path: str) -> bool:
    text = str(asset_path or "").lower()
    return bool(re.search(r"/buildings/|roof|house|inn|tower|city|village", text))


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    library_dir = _resolve_library_dir(args)

    summary_path = library_dir / "summary.json"
    exemplars_path = library_dir / "roof_exemplars.parquet"
    families_path = library_dir / "roof_families.parquet"
    object_visual_path = library_dir / "object_visual.zarr"
    atlas_path = library_dir / "roof_atlas.png"

    if not summary_path.exists():
        raise RuntimeError(f"Missing summary file: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exemplars = _read_parquet_rows(exemplars_path)
    families = _read_parquet_rows(families_path)
    object_visual = _read_object_visual_attrs(object_visual_path)

    issues: list[str] = []

    if len(exemplars) < int(args.min_exemplars):
        issues.append(f"exemplar_count_below_min: {len(exemplars)} < {int(args.min_exemplars)}")
    if len(families) < int(args.min_families):
        issues.append(f"family_count_below_min: {len(families)} < {int(args.min_families)}")

    # Stable-ID/provenance surface checks for T001/T004/T006/T007.
    required_exemplar_fields = [
        "exemplar_id",
        "family_id",
        "asset_path",
        "build",
        "map_name",
        "tile_id",
        "tile_x",
        "tile_y",
        "provenance_key",
        "variant_fingerprint",
    ]
    if exemplars:
        first = exemplars[0]
        missing = [field for field in required_exemplar_fields if field not in first]
        if missing:
            issues.append(f"missing_exemplar_fields: {missing}")

    exemplar_ids = [str(row.get("exemplar_id", "")) for row in exemplars]
    if len(exemplar_ids) != len(set(exemplar_ids)):
        issues.append("duplicate_exemplar_id_detected")

    family_ids = [str(row.get("family_id", "")) for row in families]
    if len(family_ids) != len(set(family_ids)):
        issues.append("duplicate_family_id_detected")

    visual_count = int(object_visual["shape_rgb"][0])
    if visual_count != len(exemplars):
        issues.append(f"object_visual_count_mismatch: visual={visual_count} exemplars={len(exemplars)}")

    if object_visual["shape_mask"][0] != visual_count:
        issues.append("object_visual_mask_count_mismatch")
    if object_visual["shape_pose"][0] != visual_count:
        issues.append("object_visual_pose_count_mismatch")
    if object_visual["shape_bbox"][0] != visual_count:
        issues.append("object_visual_bbox_count_mismatch")

    # Building-heavy family requirement for bounded manual QA proof.
    building_heavy_families: list[str] = []
    if exemplars:
        family_to_assets: dict[str, set[str]] = {}
        for row in exemplars:
            family_id = str(row.get("family_id", ""))
            family_to_assets.setdefault(family_id, set()).add(str(row.get("asset_path", "")))
        for family_id, assets in family_to_assets.items():
            if any(_is_building_heavy(asset) for asset in assets):
                building_heavy_families.append(family_id)

    if bool(args.require_building_heavy_family) and not building_heavy_families:
        issues.append("missing_building_heavy_family")

    if not atlas_path.exists():
        issues.append(f"missing_atlas: {atlas_path}")

    status = "pass" if not issues else "fail"
    report = {
        "status": status,
        "library_dir": str(library_dir),
        "summary": summary,
        "counts": {
            "exemplars": len(exemplars),
            "families": len(families),
            "building_heavy_families": len(building_heavy_families),
            "object_visual_samples": visual_count,
        },
        "object_visual": object_visual,
        "building_heavy_family_ids": sorted(building_heavy_families),
        "issues": issues,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }

    if bool(args.emit_report):
        (library_dir / "roof_library_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    args = _parse_args()
    report = _validate(args)
    print(json.dumps(report, indent=2))
    if report.get("status") != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

