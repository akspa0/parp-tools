from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
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
    RoofExemplarRecord,
    RoofFamilySummary,
    build_map_tile_key,
    exemplar_id_from_parts,
    family_id_from_asset_path,
    is_probable_roof_asset,
    normalize_asset_path,
    pose_vector_from_placement,
    variant_fingerprint_from_rgb,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v16"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "datasets" / "object_roof_library"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object-roof exemplar library from MdxViewer per-asset renders.")
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--build", type=str, default=None)
    parser.add_argument("--builds", nargs="+", default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-canonical-atlas", type=int, default=196)
    parser.add_argument("--emit-asset-list", action="store_true", default=False,
                        help="Only write the asset list JSON for MdxViewer, do not build library")
    parser.add_argument("--captured-roof-dir", type=Path, default=None,
                        help="Directory containing MdxViewer per-asset roof renders (subdirs with roof_topdown.png)")
    parser.add_argument("--captured-roof-metadata", type=Path, default=None,
                        help="Path to roof_capture_metadata.json from batch capture")
    parser.add_argument("--include-mddf", action="store_true", default=False)
    parser.add_argument("--include-modf", action="store_true", default=True)
    parser.add_argument("--roof-only", action="store_true", default=True)
    parser.add_argument("--no-roof-only", action="store_false", dest="roof_only")
    return parser.parse_args()


def _resolve_builds(dataset_dir: Path, args: argparse.Namespace) -> list[str]:
    if args.builds:
        return [str(item) for item in args.builds]
    if args.build:
        return [str(args.build)]
    return [path.stem.replace(".zarr", "") for path in sorted(dataset_dir.glob("*.zarr"))]


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(str(path))
    return [{column: table.column(column)[idx].as_py() for column in table.column_names} for idx in range(table.num_rows)]


def _collect_unique_wmo_assets(
    dataset_dir: Path,
    build: str,
    roof_only: bool,
) -> list[str]:
    """Extract unique MODF asset paths from placements, deduped, sorted."""
    placements_path = dataset_dir / f"{build}.zarr" / "placements.parquet"
    if not placements_path.exists():
        return []

    placements = _read_table_rows(placements_path)
    seen: set[str] = set()
    paths: list[str] = []
    for row in placements:
        instance_type = str(row.get("instance_type", "")).lower()
        if instance_type != "modf":
            continue
        asset_path = normalize_asset_path(str(row.get("asset_path", "")))
        if not asset_path or asset_path in seen:
            continue
        if roof_only and not is_probable_roof_asset(asset_path):
            continue
        seen.add(asset_path)
        paths.append(asset_path)
    return sorted(paths)


def _load_captured_roof(captured_dir: Path, asset_path: str, crop_size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a per-asset roof render from MdxViewer output.
    Returns (rgb_uint8, mask_float32) cropped/resized to crop_size, or (None, None) on failure.
    """
    safe_name = _sanitize_path_component(Path(asset_path).stem)
    asset_dir = captured_dir / safe_name
    png_path = asset_dir / "roof_topdown.png"
    if not png_path.exists():
        return None, None

    try:
        img = Image.open(png_path).convert("RGBA")
    except Exception:
        return None, None

    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 4)
    if arr.size == 0:
        return None, None

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3].astype(np.float32) / 255.0

    # Resize to crop_size using nearest neighbor
    rgb_resized = _nearest_resize(rgb, crop_size, crop_size)
    mask_resized = _nearest_resize(alpha, crop_size, crop_size)
    mask_resized = np.clip(mask_resized, 0.0, 1.0)
    return rgb_resized, mask_resized


def _nearest_resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    in_h, in_w = arr.shape[:2]
    ys = np.linspace(0, in_h - 1, h).astype(np.int64)
    xs = np.linspace(0, in_w - 1, w).astype(np.int64)
    return arr[np.ix_(ys, xs)]


def _sanitize_path_component(name: str) -> str:
    invalid = set(r'<>:"/\|?*')
    sb = []
    for c in name:
        if c in invalid or c == '\0':
            sb.append('_')
        else:
            sb.append(c)
    result = "".join(sb)
    return result[:100] if len(result) > 100 else result


def _build_from_captures(
    captured_dir: Path,
    metadata_path: Path | None,
    build: str,
    crop_size: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build exemplar list from MdxViewer capture output directory."""
    # Load metadata JSON if provided (maps asset path -> success status)
    capture_metadata: list[dict[str, Any]] = []
    if metadata_path and metadata_path.exists():
        with open(metadata_path, "r") as f:
            capture_metadata = json.load(f).get("captures", [])

    rows: list[dict[str, Any]] = []
    roof_rgbs: list[np.ndarray] = []
    roof_masks: list[np.ndarray] = []
    pose_vecs: list[np.ndarray] = []
    bbox_vecs: list[np.ndarray] = []

    # Iterate over capture metadata entries for the ordered list of assets
    for entry in capture_metadata:
        if not entry.get("success", False):
            continue
        asset_path = str(entry.get("asset_path", ""))
        if not asset_path:
            continue

        rgb, mask = _load_captured_roof(captured_dir, asset_path, crop_size)
        if rgb is None or mask is None:
            continue

        family_id = family_id_from_asset_path(asset_path)
        mask_coverage = float((mask >= 0.2).mean())
        variant_fp = variant_fingerprint_from_rgb(rgb)
        exemplar_id = exemplar_id_from_parts([family_id, build, variant_fp])
        review_required = bool(mask_coverage < 0.05)
        review_state = "needs_review" if review_required else "auto"

        row = asdict(
            RoofExemplarRecord(
                exemplar_id=exemplar_id,
                family_id=family_id,
                variant_rank=0,
                is_canonical=False,
                asset_path=asset_path,
                instance_type="modf",
                build=build,
                map_name="",
                tile_id=-1,
                tile_x=-1,
                tile_y=-1,
                instance_idx=-1,
                unique_id=0,
                pose_rot_x=0,
                pose_rot_y=0,
                pose_rot_z=0,
                pose_scale=1.0,
                bbox_xyxy=(0, 0, crop_size - 1, crop_size - 1),
                bbox_wh=(crop_size, crop_size),
                crop_size=int(crop_size),
                mask_coverage=mask_coverage,
                minimap_mean=float(rgb.astype(np.float32).mean() / 255.0),
                minimap_std=float(rgb.astype(np.float32).std() / 255.0),
                provenance_key=f"{build}|{asset_path}",
                review_state=review_state,
                review_required=review_required,
            )
        )
        row["variant_fingerprint"] = variant_fp

        rows.append(row)
        roof_rgbs.append(rgb)
        roof_masks.append(mask.astype(np.float32))
        pose_vecs.append(np.zeros(8, dtype=np.float32))
        bbox_vecs.append(np.array([0, 0, crop_size - 1, crop_size - 1], dtype=np.int32))

    return rows, roof_rgbs, roof_masks, pose_vecs, bbox_vecs


def _dedupe_and_rank(rows: list[dict[str, Any]], arrays: dict[str, list[np.ndarray]]) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], list[dict[str, Any]]]:
    family_groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        family_groups[str(row["family_id"])].append(idx)

    keep_indices: list[int] = []
    family_summaries: list[dict[str, Any]] = []

    for family_id, member_indices in sorted(family_groups.items(), key=lambda item: item[0]):
        ordered = sorted(
            member_indices,
            key=lambda idx: (
                float(rows[idx].get("mask_coverage", 0.0)),
                str(rows[idx].get("exemplar_id", "")),
            ),
            reverse=True,
        )

        seen_fp: set[str] = set()
        deduped_members: list[int] = []
        for idx in ordered:
            fingerprint = str(rows[idx].get("variant_fingerprint", ""))
            if fingerprint and fingerprint in seen_fp:
                continue
            if fingerprint:
                seen_fp.add(fingerprint)
            deduped_members.append(idx)

        if not deduped_members:
            continue

        for rank, idx in enumerate(deduped_members):
            rows[idx]["variant_rank"] = int(rank)
            rows[idx]["is_canonical"] = bool(rank == 0)
            keep_indices.append(idx)

        canonical = rows[deduped_members[0]]
        canonical_asset = str(canonical.get("asset_path", ""))
        review_required = bool(any(bool(rows[idx].get("review_required", False)) for idx in deduped_members))
        review_state = "needs_review" if review_required else "auto"
        family_summaries.append(
            asdict(
                RoofFamilySummary(
                    family_id=family_id,
                    canonical_asset_path=canonical_asset,
                    exemplar_count=len(deduped_members),
                    canonical_exemplar_id=str(canonical.get("exemplar_id", "")),
                    review_state=review_state,
                    review_required=review_required,
                )
            )
        )

    keep_indices = sorted(keep_indices)
    kept_rows = [rows[idx] for idx in keep_indices]
    kept_arrays: dict[str, list[np.ndarray]] = {}
    for key, values in arrays.items():
        kept_arrays[key] = [values[idx] for idx in keep_indices]
    return kept_rows, kept_arrays, family_summaries


def _write_object_visual_store(output_dir: Path, arrays: dict[str, list[np.ndarray]], crop_size: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / "object_visual.zarr"
    if zarr_path.exists():
        if zarr_path.is_dir():
            for child in zarr_path.glob("**/*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
        else:
            zarr_path.unlink(missing_ok=True)

    n = len(arrays["roof_rgb"])
    roof_rgb = np.stack(arrays["roof_rgb"], axis=0).astype(np.uint8) if n else np.zeros((0, crop_size, crop_size, 3), dtype=np.uint8)
    roof_mask = np.stack(arrays["roof_mask"], axis=0).astype(np.float32) if n else np.zeros((0, crop_size, crop_size), dtype=np.float32)
    pose_vec = np.stack(arrays["pose_vec"], axis=0).astype(np.float32) if n else np.zeros((0, 8), dtype=np.float32)
    bbox_xyxy = np.stack(arrays["bbox_xyxy"], axis=0).astype(np.int32) if n else np.zeros((0, 4), dtype=np.int32)

    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    store = zarr.storage.LocalStore(str(zarr_path), read_only=False)
    root = zarr.open_group(store=store, mode="w")
    try:
        root.create_array("roof_rgb", data=roof_rgb, chunks=(max(1, min(64, max(1, n))), crop_size, crop_size, 3), compressors=codec, overwrite=True)
        root.create_array("roof_mask", data=roof_mask, chunks=(max(1, min(64, max(1, n))), crop_size, crop_size), compressors=codec, overwrite=True)
        root.create_array("pose_vec", data=pose_vec, chunks=(max(1, min(256, max(1, n))), 8), compressors=codec, overwrite=True)
        root.create_array("bbox_xyxy", data=bbox_xyxy, chunks=(max(1, min(256, max(1, n))), 4), compressors=codec, overwrite=True)
        root.attrs.update({
            "schema_version": "2.0.0",
            "description": "Object-roof exemplar visual datastore from per-asset MdxViewer renders",
            "crop_size": int(crop_size),
            "sample_count": int(n),
        })
    finally:
        store.close()
    return zarr_path


def _write_catalogs(output_dir: Path, exemplars: list[dict[str, Any]], families: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exemplar_table = pa.Table.from_pylist(exemplars) if exemplars else pa.table({})
    family_table = pa.Table.from_pylist(families) if families else pa.table({})

    if exemplar_table.num_rows > 0:
        pq.write_table(exemplar_table, str(output_dir / "roof_exemplars.parquet"))
        with (output_dir / "roof_exemplars.jsonl").open("w", encoding="utf-8") as handle:
            for row in exemplars:
                handle.write(json.dumps(row) + "\n")
    else:
        (output_dir / "roof_exemplars.jsonl").write_text("", encoding="utf-8")

    if family_table.num_rows > 0:
        pq.write_table(family_table, str(output_dir / "roof_families.parquet"))
        with (output_dir / "roof_families.jsonl").open("w", encoding="utf-8") as handle:
            for row in families:
                handle.write(json.dumps(row) + "\n")
    else:
        (output_dir / "roof_families.jsonl").write_text("", encoding="utf-8")


def _write_roof_atlas(
    output_dir: Path,
    exemplars: list[dict[str, Any]],
    roof_rgbs: list[np.ndarray],
    max_tiles: int,
    crop_size: int,
) -> Path | None:
    canonical_rows = [row for row in exemplars if bool(row.get("is_canonical", False))]
    if not canonical_rows:
        return None
    canonical_rows = sorted(canonical_rows, key=lambda row: str(row.get("family_id", "")))[:max(1, int(max_tiles))]
    exemplar_to_idx = {str(row.get("exemplar_id", "")): idx for idx, row in enumerate(exemplars)}

    cols = int(np.ceil(np.sqrt(len(canonical_rows))))
    rows = int(np.ceil(len(canonical_rows) / max(1, cols)))
    label_height = 16

    atlas = Image.new("RGB", (cols * crop_size, rows * (crop_size + label_height)), color=(0, 0, 0))
    draw = ImageDraw.Draw(atlas)

    for idx, row in enumerate(canonical_rows):
        row_idx = idx // cols
        col_idx = idx % cols
        x = col_idx * crop_size
        y = row_idx * (crop_size + label_height)
        exemplar_id = str(row.get("exemplar_id", ""))
        source_idx = exemplar_to_idx.get(exemplar_id)
        if source_idx is None:
            continue
        tile = Image.fromarray(roof_rgbs[source_idx], mode="RGB")
        atlas.paste(tile, (x, y + label_height))
        draw.rectangle([(x, y), (x + crop_size - 1, y + label_height - 1)], fill=(18, 18, 18))
        short_label = str(row.get("family_id", ""))[-10:]
        draw.text((x + 3, y + 2), short_label, fill=(235, 235, 235))

    atlas_path = output_dir / "roof_atlas.png"
    atlas.save(atlas_path)
    return atlas_path


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir)
    builds = _resolve_builds(dataset_dir, args)
    if not builds:
        raise RuntimeError(f"No V16 stores found under {dataset_dir}")

    run_name = args.run_name or f"roof_library_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_root = args.output_dir if args.output_dir is not None else _DEFAULT_OUTPUT_ROOT
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.emit_asset_list:
        for build in builds:
            paths = _collect_unique_wmo_assets(dataset_dir, str(build), roof_only=bool(args.roof_only))
            asset_list_path = output_dir / f"roof_capture_asset_list_{build}.json"
            asset_list_path.write_text(json.dumps(paths, indent=2), encoding="utf-8")
            print(f"Wrote {len(paths)} asset paths -> {asset_list_path}")
            print(f"\nTo capture roof images, run MdxViewer with:\n"
                  f"  --game-path <client_root> "
                  f"--capture-roof {output_dir} "
                  f"--capture-roof-asset-list {asset_list_path} "
                  f"--exit-after-validation")
        return

    if args.captured_roof_dir is None:
        raise RuntimeError("Must provide --captured-roof-dir (or --emit-asset-list to just write the asset list)")

    captured_dir = Path(args.captured_roof_dir)
    metadata_path = args.captured_roof_metadata or (captured_dir / "roof_capture_metadata.json")
    crop_size = int(args.crop_size)

    all_rows: list[dict[str, Any]] = []
    all_roof_rgbs: list[np.ndarray] = []
    all_roof_masks: list[np.ndarray] = []
    all_pose_vecs: list[np.ndarray] = []
    all_bbox_vecs: list[np.ndarray] = []
    build_stats: dict[str, dict[str, int]] = {}

    for build in builds:
        rows, roof_rgbs, roof_masks, pose_vecs, bbox_vecs = _build_from_captures(
            captured_dir=captured_dir,
            metadata_path=metadata_path,
            build=str(build),
            crop_size=crop_size,
        )
        build_stats[str(build)] = {
            "assets_in_metadata": len(rows),
            "exemplars_kept": len(rows),
        }
        all_rows.extend(rows)
        all_roof_rgbs.extend(roof_rgbs)
        all_roof_masks.extend(roof_masks)
        all_pose_vecs.extend(pose_vecs)
        all_bbox_vecs.extend(bbox_vecs)

    arrays = {
        "roof_rgb": all_roof_rgbs,
        "roof_mask": all_roof_masks,
        "pose_vec": all_pose_vecs,
        "bbox_xyxy": all_bbox_vecs,
    }
    deduped_rows, deduped_arrays, family_summaries = _dedupe_and_rank(all_rows, arrays)

    zarr_path = _write_object_visual_store(output_dir, deduped_arrays, crop_size=crop_size)
    _write_catalogs(output_dir, deduped_rows, family_summaries)
    atlas_path = _write_roof_atlas(
        output_dir,
        exemplars=deduped_rows,
        roof_rgbs=deduped_arrays["roof_rgb"],
        max_tiles=int(args.max_canonical_atlas),
        crop_size=crop_size,
    )

    summary = {
        "run_name": run_name,
        "captured_roof_dir": str(captured_dir),
        "builds": [str(item) for item in builds],
        "build_stats": build_stats,
        "exemplars_total": len(all_rows),
        "exemplars_after_dedupe": len(deduped_rows),
        "families_total": len(family_summaries),
        "canonical_count": int(sum(1 for row in deduped_rows if bool(row.get("is_canonical", False)))),
        "review_required_count": int(sum(1 for row in deduped_rows if bool(row.get("review_required", False)))),
        "object_visual_store": str(zarr_path),
        "roof_atlas": str(atlas_path) if atlas_path is not None else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.snapshot.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()