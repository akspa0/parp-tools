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
    crop_and_resize_mask,
    crop_and_resize_rgb,
    d1_style_bbox_fallback,
    exemplar_id_from_parts,
    family_id_from_asset_path,
    is_probable_roof_asset,
    normalize_asset_path,
    pose_vector_from_placement,
    variant_fingerprint_from_rgb,
    world_bbox_to_tile_bbox_xyxy,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v16"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "datasets" / "object_roof_library"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object-roof exemplar library from V16 stores.")
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--build", type=str, default=None)
    parser.add_argument("--builds", nargs="+", default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-tiles-per-build", type=int, default=0)
    parser.add_argument("--include-mddf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-modf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roof-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--min-bbox-area", type=int, default=64)
    parser.add_argument("--bbox-padding", type=int, default=3)
    parser.add_argument("--max-canonical-atlas", type=int, default=196)
    return parser.parse_args()


def _resolve_builds(dataset_dir: Path, args: argparse.Namespace) -> list[str]:
    if args.builds:
        return [str(item) for item in args.builds]
    if args.build:
        return [str(args.build)]
    return [path.stem.replace(".zarr", "") for path in sorted(dataset_dir.glob("*.zarr"))]


def _open_store(zarr_path: Path) -> tuple[zarr.storage.LocalStore, zarr.Group]:
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    return store, root


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(str(path))
    return [{column: table.column(column)[idx].as_py() for column in table.column_names} for idx in range(table.num_rows)]


def _make_tile_index(index_rows: list[dict[str, Any]], max_tiles_per_build: int) -> dict[int, dict[str, Any]]:
    rows = sorted(index_rows, key=lambda row: int(row.get("tile_id", -1)))
    if max_tiles_per_build > 0:
        rows = rows[:max_tiles_per_build]
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        tile_id = int(row.get("tile_id", -1))
        if tile_id < 0:
            continue
        out[tile_id] = {
            "map": str(row.get("map", "")),
            "tile_x": int(row.get("tile_x", -1) if row.get("tile_x") is not None else -1),
            "tile_y": int(row.get("tile_y", -1) if row.get("tile_y") is not None else -1),
        }
    return out


def _filter_placements(
    placements: list[dict[str, Any]],
    *,
    include_mddf: bool,
    include_modf: bool,
    roof_only: bool,
    valid_tile_ids: set[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in placements:
        tile_id = int(row.get("tile_id", -1))
        if tile_id not in valid_tile_ids:
            continue
        instance_type = str(row.get("instance_type", "")).lower()
        if instance_type == "modf" and not include_modf:
            continue
        if instance_type == "mddf" and not include_mddf:
            continue
        asset_path = normalize_asset_path(str(row.get("asset_path", "")))
        if not asset_path:
            continue
        if roof_only and not is_probable_roof_asset(asset_path):
            continue
        out.append(dict(row))
    return out


def _tile_mask_array(root: zarr.Group, tile_id: int) -> np.ndarray:
    if "object_precise_mask" in root:
        return root["object_precise_mask"][tile_id].astype(np.float32)
    if "object_filtered_mask" in root:
        return root["object_filtered_mask"][tile_id].astype(np.float32)
    if "object_mask" in root:
        return root["object_mask"][tile_id].astype(np.float32)
    return np.zeros((257, 257), dtype=np.float32)


def _placement_bbox(row: dict[str, Any], tile_x: int, tile_y: int, padding: int) -> tuple[int, int, int, int] | None:
    instance_type = str(row.get("instance_type", "")).lower()
    if instance_type == "modf":
        return world_bbox_to_tile_bbox_xyxy(
            min_x=float(row.get("bbMinX", 0.0) or 0.0),
            min_y=float(row.get("bbMinY", 0.0) or 0.0),
            max_x=float(row.get("bbMaxX", 0.0) or 0.0),
            max_y=float(row.get("bbMaxY", 0.0) or 0.0),
            tile_x=tile_x,
            tile_y=tile_y,
            padding_px=int(padding),
        )
    return d1_style_bbox_fallback(
        pos_x=float(row.get("posX", 0.0) or 0.0),
        pos_y=float(row.get("posY", 0.0) or 0.0),
        scale=float(row.get("scale", 1.0) or 1.0),
        tile_x=tile_x,
        tile_y=tile_y,
        base_radius_px=6.0 + float(padding),
    )


def _collect_build_exemplars(
    *,
    dataset_dir: Path,
    build: str,
    max_tiles_per_build: int,
    include_mddf: bool,
    include_modf: bool,
    roof_only: bool,
    crop_size: int,
    min_bbox_area: int,
    bbox_padding: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, int]]:
    zarr_path = dataset_dir / f"{build}.zarr"
    index_path = zarr_path / "index.parquet"
    placements_path = zarr_path / "placements.parquet"
    if not zarr_path.exists() or not index_path.exists() or not placements_path.exists():
        return [], [], [], [], [], {
            "placements_total": 0,
            "placements_selected": 0,
            "exemplars_kept": 0,
            "missing_tile": 0,
            "invalid_bbox": 0,
            "too_small": 0,
        }

    index_rows = _read_table_rows(index_path)
    tile_index = _make_tile_index(index_rows, max_tiles_per_build=max_tiles_per_build)
    valid_tile_ids = set(tile_index.keys())

    placements = _read_table_rows(placements_path)
    placements_filtered = _filter_placements(
        placements,
        include_mddf=include_mddf,
        include_modf=include_modf,
        roof_only=roof_only,
        valid_tile_ids=valid_tile_ids,
    )

    stats = {
        "placements_total": len(placements),
        "placements_selected": len(placements_filtered),
        "exemplars_kept": 0,
        "missing_tile": 0,
        "invalid_bbox": 0,
        "too_small": 0,
    }

    rows: list[dict[str, Any]] = []
    roof_rgbs: list[np.ndarray] = []
    roof_masks: list[np.ndarray] = []
    pose_vecs: list[np.ndarray] = []
    bbox_vecs: list[np.ndarray] = []

    store, root = _open_store(zarr_path)
    try:
        cached_minimap: dict[int, np.ndarray] = {}
        cached_mask: dict[int, np.ndarray] = {}

        for placement in placements_filtered:
            tile_id = int(placement.get("tile_id", -1))
            tile_meta = tile_index.get(tile_id)
            if tile_meta is None:
                stats["missing_tile"] += 1
                continue

            tile_x = int(tile_meta["tile_x"])
            tile_y = int(tile_meta["tile_y"])
            if tile_x < 0 or tile_y < 0:
                stats["missing_tile"] += 1
                continue

            bbox = _placement_bbox(placement, tile_x=tile_x, tile_y=tile_y, padding=bbox_padding)
            if bbox is None:
                stats["invalid_bbox"] += 1
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox]
            width = max(0, x1 - x0 + 1)
            height = max(0, y1 - y0 + 1)
            area = width * height
            if area < int(min_bbox_area):
                stats["too_small"] += 1
                continue

            if tile_id not in cached_minimap:
                cached_minimap[tile_id] = root["minimap_rgb"][tile_id].astype(np.uint8)
                cached_mask[tile_id] = _tile_mask_array(root, tile_id)[:256, :256]

            minimap = cached_minimap[tile_id]
            object_mask = cached_mask[tile_id]

            crop_rgb = crop_and_resize_rgb(minimap, bbox, crop_size)
            crop_mask = crop_and_resize_mask(object_mask, bbox, crop_size)
            crop_mask = np.clip(crop_mask, 0.0, 1.0)

            asset_path = normalize_asset_path(str(placement.get("asset_path", "")))
            family_id = family_id_from_asset_path(asset_path)
            map_name = str(tile_meta["map"])
            instance_idx = int(placement.get("instance_idx", -1) or -1)
            unique_id = int(float(placement.get("uniqueId", 0) or 0))
            provenance_key = build_map_tile_key(build, map_name, tile_x, tile_y)
            variant_fp = variant_fingerprint_from_rgb(crop_rgb)
            exemplar_id = exemplar_id_from_parts(
                [
                    family_id,
                    build,
                    map_name,
                    tile_id,
                    instance_idx,
                    unique_id,
                    variant_fp,
                ]
            )
            mask_coverage = float((crop_mask >= 0.2).mean())
            minimap_mean = float(crop_rgb.astype(np.float32).mean() / 255.0)
            minimap_std = float(crop_rgb.astype(np.float32).std() / 255.0)
            review_required = bool(mask_coverage < 0.12)
            review_state = "needs_review" if review_required else "auto"

            row = asdict(
                RoofExemplarRecord(
                    exemplar_id=exemplar_id,
                    family_id=family_id,
                    variant_rank=0,
                    is_canonical=False,
                    asset_path=asset_path,
                    instance_type=str(placement.get("instance_type", "")),
                    build=build,
                    map_name=map_name,
                    tile_id=tile_id,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    instance_idx=instance_idx,
                    unique_id=unique_id,
                    pose_rot_x=float(placement.get("rotX", 0.0) or 0.0),
                    pose_rot_y=float(placement.get("rotY", 0.0) or 0.0),
                    pose_rot_z=float(placement.get("rotZ", 0.0) or 0.0),
                    pose_scale=float(placement.get("scale", 1.0) or 1.0),
                    bbox_xyxy=(x0, y0, x1, y1),
                    bbox_wh=(width, height),
                    crop_size=int(crop_size),
                    mask_coverage=mask_coverage,
                    minimap_mean=minimap_mean,
                    minimap_std=minimap_std,
                    provenance_key=provenance_key,
                    review_state=review_state,
                    review_required=review_required,
                )
            )
            row["variant_fingerprint"] = variant_fp

            rows.append(row)
            roof_rgbs.append(crop_rgb)
            roof_masks.append(crop_mask.astype(np.float32))
            pose_vecs.append(pose_vector_from_placement(placement))
            bbox_vecs.append(np.asarray([x0, y0, x1, y1], dtype=np.int32))
            stats["exemplars_kept"] += 1
    finally:
        store.close()

    return rows, roof_rgbs, roof_masks, pose_vecs, bbox_vecs, stats


def _dedupe_and_rank(rows: list[dict[str, Any]], arrays: dict[str, list[np.ndarray]]) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], list[dict[str, Any]]]:
    family_groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        family_groups[str(row["family_id"])].append(idx)

    keep_indices: list[int] = []
    family_summaries: list[dict[str, Any]] = []

    for family_id, member_indices in sorted(family_groups.items(), key=lambda item: item[0]):
        # Sort by strongest roof signal, then by larger bbox area.
        ordered = sorted(
            member_indices,
            key=lambda idx: (
                float(rows[idx].get("mask_coverage", 0.0)),
                int(rows[idx].get("bbox_wh", [0, 0])[0]) * int(rows[idx].get("bbox_wh", [0, 0])[1]),
                str(rows[idx].get("exemplar_id", "")),
            ),
            reverse=True,
        )

        # Variant dedupe per family by perceptual fingerprint.
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
        root.attrs.update(
            {
                "schema_version": "1.0.0",
                "description": "Object-roof exemplar visual datastore",
                "crop_size": int(crop_size),
                "sample_count": int(n),
            }
        )
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

    all_rows: list[dict[str, Any]] = []
    all_roof_rgbs: list[np.ndarray] = []
    all_roof_masks: list[np.ndarray] = []
    all_pose_vecs: list[np.ndarray] = []
    all_bbox_vecs: list[np.ndarray] = []
    build_stats: dict[str, dict[str, int]] = {}

    for build in builds:
        rows, roof_rgbs, roof_masks, pose_vecs, bbox_vecs, stats = _collect_build_exemplars(
            dataset_dir=dataset_dir,
            build=str(build),
            max_tiles_per_build=int(args.max_tiles_per_build),
            include_mddf=bool(args.include_mddf),
            include_modf=bool(args.include_modf),
            roof_only=bool(args.roof_only),
            crop_size=int(args.crop_size),
            min_bbox_area=int(args.min_bbox_area),
            bbox_padding=int(args.bbox_padding),
        )
        build_stats[str(build)] = stats

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

    zarr_path = _write_object_visual_store(output_dir, deduped_arrays, crop_size=int(args.crop_size))
    _write_catalogs(output_dir, deduped_rows, family_summaries)
    atlas_path = _write_roof_atlas(
        output_dir,
        exemplars=deduped_rows,
        roof_rgbs=deduped_arrays["roof_rgb"],
        max_tiles=int(args.max_canonical_atlas),
        crop_size=int(args.crop_size),
    )

    summary = {
        "run_name": run_name,
        "dataset_dir": str(dataset_dir),
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
