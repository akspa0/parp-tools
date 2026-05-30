"""Orchestrate per-build object roof capture for all staged clients.

Pipeline per build:
  1. Scan placements → deduped WMO asset list JSON
  2. Launch MdxViewer to render each WMO (top-down + all angles)
  3. Pack renders into per-build object_visual.zarr with metadata

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/build_v18_object_catalog_pipeline.py --allow-zarr-write
    uv run python scripts/build_v18_object_catalog_pipeline.py --allow-zarr-write --builds 3_3_5_12340
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
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
    exemplar_id_from_parts,
    family_id_from_asset_path,
    is_probable_roof_asset,
    normalize_asset_path,
    variant_fingerprint_from_rgb,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v16"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "datasets" / "object_roof_library"
_DEFAULT_MDXVIEWER = (
    _PROJECT_ROOT.parent
    / "gillijimproject_refactor"
    / "src"
    / "MdxViewer"
    / "bin"
    / "Debug"
    / "net10.0-windows"
    / "ParpToolsWoWViewer.exe"
)
_STAGED_CLIENTS = _PROJECT_ROOT.parent / "output" / "tmp" / "wowarchive-clients"

_BUILDS_WITH_GAME = {
    "0_5_3_3368": "0_5_3_3368",
    "0_5_5_3494": "0_5_5_3494",
    "0_7_0_3694": "0_7_0_3694",
    "3_0_1_8303": "3_0_1_8303",
    "3_3_5_12340": "3_3_5_12340",
    "4_0_0_11927": "4_0_0_11927",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate per-build WMO roof capture pipeline."
    )
    parser.add_argument("--builds", nargs="+", default=None)
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--mdxviewer", type=Path, default=_DEFAULT_MDXVIEWER)
    parser.add_argument("--staged-clients", type=Path, default=_STAGED_CLIENTS)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--all-angles", action="store_true", default=True)
    parser.add_argument("--skip-capture", action="store_true", default=False)
    parser.add_argument("--skip-pack", action="store_true", default=False)
    parser.add_argument("--pack-only", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--allow-zarr-write", action="store_true", default=False)
    parser.add_argument("--roof-only", action="store_true", default=False)
    parser.add_argument("--no-roof-only", action="store_false", dest="roof_only")
    return parser.parse_args()


def _resolve_builds(args: argparse.Namespace) -> list[str]:
    if args.builds:
        return list(args.builds)
    return sorted(_BUILDS_WITH_GAME.keys())


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(str(path))
    return [
        {
            column: table.column(column)[idx].as_py()
            for column in table.column_names
        }
        for idx in range(table.num_rows)
    ]


def _collect_unique_assets(
    dataset_dir: Path, build: str, roof_only: bool, include_modf: bool = True, include_mddf: bool = True
) -> list[str]:
    placements_path = dataset_dir / f"{build}.zarr" / "placements.parquet"
    if not placements_path.exists():
        print(f"  [SKIP] No placements: {placements_path}")
        return []

    placements = _read_table_rows(placements_path)
    seen = set()
    paths = []
    for row in placements:
        instance_type = str(row.get("instance_type", "")).lower()
        if instance_type == "modf" and not include_modf:
            continue
        if instance_type == "mddf" and not include_mddf:
            continue
        asset_path = normalize_asset_path(str(row.get("asset_path", "")))
        if not asset_path or asset_path in seen:
            continue
        if roof_only and not is_probable_roof_asset(asset_path):
            continue
        seen.add(asset_path)
        paths.append(asset_path)
    return sorted(paths)


def step1_emit_asset_lists(
    dataset_dir: Path,
    output_dir: Path,
    builds: list[str],
    roof_only: bool,
    include_modf: bool,
    include_mddf: bool,
) -> dict[str, list[str]]:
    """Write per-build asset list JSONs. Returns {build: [paths]}."""
    print("=== Step 1: Emit per-build asset lists ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for build in builds:
        paths = _collect_unique_assets(dataset_dir, build, roof_only, include_modf=include_modf, include_mddf=include_mddf)
        result[build] = paths
        list_path = output_dir / f"roof_capture_asset_list_{build}.json"
        list_path.write_text(json.dumps(paths, indent=2), encoding="utf-8")
        print(f"  {build}: {len(paths)} assets -> {list_path}")
    return result


def step2_mdxviewer_capture(
    mdxviewer: Path,
    staged_clients: Path,
    output_dir: Path,
    builds: list[str],
    resolution: int,
    all_angles: bool,
    asset_lists: dict[str, list[str]],
    dry_run: bool,
) -> None:
    """Run MdxViewer per-build to render each WMO."""
    print(f"\n=== Step 2: MdxViewer roof capture ({'dry-run' if dry_run else 'live'}) ===")
    if not mdxviewer.exists():
        print(f"  [ERROR] MdxViewer not found: {mdxviewer}")
        return

    for build in builds:
        asset_list_path = output_dir / f"roof_capture_asset_list_{build}.json"
        if not asset_list_path.exists():
            print(f"  [SKIP] {build}: no asset list")
            continue

        client_root = staged_clients / build / "World of Warcraft"
        if not client_root.exists():
            print(f"  [SKIP] {build}: client not staged at {client_root}")
            continue

        captured_dir = output_dir / f"captured_roofs_{build}"
        captured_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(mdxviewer),
            "--game-path", str(client_root),
            "--capture-roof", str(captured_dir),
            "--capture-roof-asset-list", str(asset_list_path),
            f"--capture-roof-resolution", str(resolution),
        ]
        if all_angles:
            cmd.append("--capture-roof-all-angles")
        cmd.append("--exit-after-validation")

        print(f"\n  [{build}] {len(asset_lists.get(build, []))} assets")
        print(f"  Client: {client_root}")
        print(f"  Output: {captured_dir}")
        print(f"  Cmd: {' '.join(cmd)}")

        if dry_run:
            continue

        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=False,
            timeout=7200,
        )
        elapsed = time.time() - start
        if result.returncode == 0 or result.returncode == -1:
            print(f"  [{build}] Done in {elapsed:.0f}s")
        else:
            print(f"  [{build}] Failed (rc={result.returncode}) in {elapsed:.0f}s")


def _sanitize_path_component(name: str) -> str:
    invalid = set(r'<>:"/\|?*')
    sb = []
    for c in name:
        if c in invalid or c == "\0":
            sb.append("_")
        else:
            sb.append(c)
    result = "".join(sb)
    return result[:100] if len(result) > 100 else result


def _nearest_resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    in_h, in_w = arr.shape[:2]
    ys = np.linspace(0, in_h - 1, h).astype(np.int64)
    xs = np.linspace(0, in_w - 1, w).astype(np.int64)
    return arr[np.ix_(ys, xs)]


def _load_angle_image(
    captured_dir: Path, asset_path: str, angle_name: str
) -> np.ndarray | None:
    safe_name = _sanitize_path_component(Path(asset_path).stem)
    png_path = captured_dir / safe_name / f"{angle_name}.png"
    if png_path.exists():
        try:
            img = Image.open(png_path).convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        except Exception:
            return None
    jpg_path = captured_dir / safe_name / f"{angle_name}.jpg"
    if not jpg_path.exists():
        return None
    try:
        img = Image.open(jpg_path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def step3_pack_object_store(
    output_dir: Path,
    builds: list[str],
    crop_size: int,
    allow_zarr_write: bool,
) -> None:
    """Pack per-asset renders from ALL builds into ONE unified object_visual.zarr."""
    print(f"\n=== Step 3: Pack unified object_visual.zarr ===")
    if not allow_zarr_write:
        print("  [SKIP] --allow-zarr-write not set")
        return

    all_rows: list[dict[str, Any]] = []
    all_roof_rgbs: list[np.ndarray] = []
    all_roof_masks: list[np.ndarray] = []
    all_builds_vec: list[str] = []
    angles_dict: dict[str, list[np.ndarray]] = {}

    for build in builds:
        captured_dir = output_dir / f"captured_roofs_{build}"
        metadata_path = captured_dir / "roof_capture_metadata.json"
        if not metadata_path.exists():
            print(f"  [SKIP] {build}: no capture metadata at {metadata_path}")
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f).get("captures", [])

        for entry in metadata:
            if not entry.get("success", False):
                continue
            asset_path = str(entry.get("asset_path", ""))
            if not asset_path:
                continue

            roof_img = _load_angle_image(captured_dir, asset_path, "roof_topdown")
            if roof_img is None:
                continue

            rgb_resized = _nearest_resize(roof_img, crop_size, crop_size)
            mask = (rgb_resized.astype(np.float32).mean(axis=2) > 5).astype(np.float32)
            family_id = family_id_from_asset_path(asset_path)
            variant_fp = variant_fingerprint_from_rgb(rgb_resized)
            exemplar_id = exemplar_id_from_parts([family_id, build, variant_fp])

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
                    crop_size=crop_size,
                    mask_coverage=float(mask.mean()),
                    minimap_mean=float(rgb_resized.mean() / 255.0),
                    minimap_std=float(rgb_resized.std() / 255.0),
                    provenance_key=f"{build}|{asset_path}",
                    review_state="auto",
                    review_required=False,
                )
            )
            row["variant_fingerprint"] = variant_fp
            all_rows.append(row)
            all_roof_rgbs.append(rgb_resized)
            all_roof_masks.append(mask)
            all_builds_vec.append(build)

            # Load JPGs for all angles
            entry_angles = entry.get("angles", [])
            if isinstance(entry_angles, list):
                for angle_info in entry_angles:
                    angle_name = (
                        angle_info["name"]
                        if isinstance(angle_info, dict)
                        else angle_info
                    )
                    angle_img = _load_angle_image(captured_dir, asset_path, angle_name)
                    if angle_img is not None:
                        angle_resized = _nearest_resize(angle_img, crop_size, crop_size)
                        angles_dict.setdefault(angle_name, []).append(angle_resized)

    if not all_rows:
        print("  [SKIP] no valid roof images across any build")
        return

    n = len(all_rows)
    store_dir = output_dir / "object_visual.zarr"
    if store_dir.exists():
        import shutil
        shutil.rmtree(store_dir)

    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    store = zarr.storage.LocalStore(str(store_dir), read_only=False)
    root = zarr.open_group(store=store, mode="w")
    chunk_1d = min(1024, n)
    chunk_3d = min(64, n)

    try:
        roof_rgb_arr = np.stack(all_roof_rgbs, axis=0).astype(np.uint8)
        roof_mask_arr = np.stack(all_roof_masks, axis=0).astype(np.float32)
        pose_vec_arr = np.zeros((n, 8), dtype=np.float32)
        bbox_arr = np.tile(np.array([0, 0, crop_size - 1, crop_size - 1], dtype=np.int32), (n, 1))

        # Build dimension — string array indexed by sample
        build_codes = sorted(set(all_builds_vec))
        build_to_code = {b: i for i, b in enumerate(build_codes)}
        build_idx = np.array([build_to_code[b] for b in all_builds_vec], dtype=np.int32)

        root.create_array("roof_rgb", data=roof_rgb_arr, chunks=(chunk_3d, crop_size, crop_size, 3), compressors=codec, overwrite=True)
        root.create_array("roof_mask", data=roof_mask_arr, chunks=(chunk_3d, crop_size, crop_size), compressors=codec, overwrite=True)
        root.create_array("pose_vec", data=pose_vec_arr, chunks=(chunk_1d, 8), compressors=codec, overwrite=True)
        root.create_array("bbox_xyxy", data=bbox_arr, chunks=(chunk_1d, 4), compressors=codec, overwrite=True)

        # Per-angle arrays
        for angle_name, angle_list in angles_dict.items():
            if angle_list and len(angle_list) == n:
                angle_arr = np.stack(angle_list, axis=0).astype(np.uint8)
                root.create_array(f"angle_{angle_name}", data=angle_arr, chunks=(chunk_3d, crop_size, crop_size, 3), compressors=codec, overwrite=True)

        # Build metadata
        angle_meta = {}
        for angle_record in Catalog_ScreenshotRenderer_CameraAngles:
            name, az, el = angle_record
            angle_meta[name] = {"azimuth": az, "elevation": el, "file": f"{name}.jpg", "zarr_array": f"angle_{name}"}

        root.attrs.update({
            "schema_version": "2.0.0",
            "dataset_name": "object_visual",
            "description": "Per-asset WMO roof and multi-angle renders across all builds",
            "crop_size": crop_size,
            "sample_count": n,
            "builds": build_codes,
            "build_code_attr_name": "build_code",
            "background": "black",
            "alpha_channel": "background_transparent",
            "format": "jpg",
            "jpeg_quality": 99,
            "camera_angles": angle_meta,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        })

        # Write build_code as separate zarr array + attr
        bc = root.create_array("build_code", data=build_idx, chunks=(chunk_1d,), dtype="int32", compressors=codec, overwrite=True)
        bc.attrs["build_codes"] = build_codes
    finally:
        store.close()

    # Write catalog parquet alongside
    exemplar_table = pa.Table.from_pylist(all_rows) if all_rows else pa.table({})
    if exemplar_table.num_rows > 0:
        pq.write_table(exemplar_table, str(store_dir.parent / "roof_exemplars.parquet"))
        with (store_dir.parent / "roof_exemplars.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row) + "\n")


def _build_object_visual_attrs_from_angles(angle_meta: dict, n: int, build_codes: list[str], crop_size: int) -> dict:
    return {}


# Camera angles matching ScreenshotRenderer.CameraAngles
Catalog_ScreenshotRenderer_CameraAngles = [
    ("front", 0.0, 15.0),
    ("back", 180.0, 15.0),
    ("left", 90.0, 15.0),
    ("right", 270.0, 15.0),
    ("top", 0.0, 80.0),
    ("three_quarter", 35.0, 25.0),
]


def main() -> None:
    args = _parse_args()
    builds = _resolve_builds(args)
    output_dir = Path(args.output_dir)
    run_name = args.run_name or f"object_catalog_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Object catalog pipeline: {run_name}")
    print(f"Builds: {builds}")
    print(f"Output: {run_dir}")
    print()

    # Step 1: asset lists (always emit unless pack-only)
    if not args.pack_only:
        asset_lists = step1_emit_asset_lists(
            dataset_dir=Path(args.dataset_dir),
            output_dir=run_dir,
            builds=builds,
            roof_only=bool(args.roof_only),
            include_modf=True,
            include_mddf=True,
        )

        if args.dry_run:
            print("\n=== DRY RUN COMPLETE ===")
            return

    if args.pack_only:
        print("\n=== PACK-ONLY MODE ===")
        step3_pack_object_store(
            output_dir=run_dir,
            builds=builds,
            crop_size=128,
            allow_zarr_write=bool(args.allow_zarr_write),
        )
        print("\n=== PACK-ONLY COMPLETE ===")
        return

    # Step 2: MdxViewer capture
    if not args.skip_capture:
        step2_mdxviewer_capture(
            mdxviewer=Path(args.mdxviewer),
            staged_clients=Path(args.staged_clients),
            output_dir=run_dir,
            builds=builds,
            resolution=int(args.resolution),
            all_angles=bool(args.all_angles),
            asset_lists=asset_lists,
            dry_run=False,
        )
    else:
        print("\n=== Step 2: SKIPPED (--skip-capture) ===")

    # Step 3: Pack Zarr stores
    if not args.skip_pack:
        step3_pack_object_store(
            output_dir=run_dir,
            builds=builds,
            crop_size=128,
            allow_zarr_write=bool(args.allow_zarr_write),
        )
    else:
        print("\n=== Step 3: SKIPPED (--skip-pack) ===")

    # Write pipeline summary
    summary = {
        "run_name": run_name,
        "builds": builds,
        "asset_counts": {b: len(asset_lists.get(b, [])) for b in builds},
        "resolution": args.resolution,
        "all_angles": args.all_angles,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nPipeline summary: {run_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()