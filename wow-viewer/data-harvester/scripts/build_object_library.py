"""Build the spec 077 per-object capture library from enumerated jobs.

The builder reads a JSONL capture-job ledger (one row per asset) plus a
flat directory of per-job capture artifacts (image PNG, mask PNG, pose
JSON), and writes:

  <output-root>/<run-name>.zarr/
      capture_rgb/    (N, H, W, 3) uint8
      capture_mask/   (N, H, W)    uint8
      capture_alpha/  (N, H, W)    uint8   [optional, only if present]
      assets.parquet  one row per ObjectLibraryEntry
      index.parquet   one row per ObjectCaptureVariant
      metadata.json   group-level provenance

Jobs whose capture artifacts are missing are emitted as
``capture_status=not_attempted`` entries (per spec 077 FR-026); they are
NOT silently dropped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.object_library import (  # noqa: E402
    ObjectCaptureVariant,
    ObjectLibraryEntry,
    detect_asset_type,
    is_clutter_asset,
    library_id_from_asset_path,
    make_entry_from_path,
    make_variant_id,
    normalize_asset_path,
)

DEFAULT_CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_capture_files(
    captures_dir: Path,
    variant_id: str,
) -> tuple[Path | None, Path | None, Path | None]:
    """Locate the (image, mask, pose) files for a given variant id."""
    image = captures_dir / f"{variant_id}_image.png"
    mask = captures_dir / f"{variant_id}_mask.png"
    pose = captures_dir / f"{variant_id}_pose.json"
    return (
        image if image.exists() else None,
        mask if mask.exists() else None,
        pose if pose.exists() else None,
    )


def _resize_array(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = arr.shape[:2]
    th, tw = target_hw
    if h == th and w == tw:
        return arr
    img = Image.fromarray(arr)
    return np.asarray(img.resize((tw, th), Image.NEAREST))


def _visibility_class(asset_path: str) -> str:
    if is_clutter_asset(asset_path):
        return "clutter_filtered"
    if "wmo" in Path(asset_path).suffix.lower():
        return "roof_visible"
    return "likely_visible"


def _build_entries_and_variants(
    jobs: list[dict[str, Any]],
    captures_dir: Path | None,
    target_size: int,
) -> tuple[list[ObjectLibraryEntry], list[ObjectCaptureVariant], list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
    entries: list[ObjectLibraryEntry] = []
    variants: list[ObjectCaptureVariant] = []
    rgb_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    alpha_list: list[np.ndarray | None] = []

    seen_entries: dict[str, int] = {}

    for job in jobs:
        asset_path = str(job.get("asset_path", ""))
        normalized = normalize_asset_path(asset_path)
        if not normalized:
            continue
        library_id = library_id_from_asset_path(normalized)
        if library_id in seen_entries:
            # Existing entry — accumulate observation count.
            idx = seen_entries[library_id]
            existing_entry = entries[idx]
            entries[idx] = ObjectLibraryEntry(
                library_id=existing_entry.library_id,
                original_asset_path=existing_entry.original_asset_path,
                normalized_asset_path=existing_entry.normalized_asset_path,
                asset_type=existing_entry.asset_type,
                capture_status=existing_entry.capture_status,
                visibility_class=existing_entry.visibility_class,
                review_state=existing_entry.review_state,
                source_builds=tuple(sorted(set(existing_entry.source_builds) | {job.get("build", "")})),
                source_maps=tuple(sorted(set(existing_entry.source_maps) | set(job.get("source_maps", []) or []))),
                placement_observation_count=existing_entry.placement_observation_count + int(job.get("observation_count", 0) or 0),
                preferred_variant_id=existing_entry.preferred_variant_id,
            )
            continue

        entry = make_entry_from_path(asset_path)
        source_builds = tuple(sorted({job.get("build", "")}))
        source_maps = tuple(sorted(job.get("source_maps", []) or []))
        observation_count = int(job.get("observation_count", 0) or 0)

        rot_x = float(job.get("first_rot_x") or 0.0)
        rot_y = float(job.get("first_rot_y") or 0.0)
        rot_z = float(job.get("first_rot_z") or 0.0)
        scale = float(job.get("first_scale") or 1.0)
        capture_build = str(job.get("build", ""))
        capture_mode = "orthographic_topdown"
        variant_id = make_variant_id(
            library_id=library_id,
            capture_build=capture_build,
            capture_mode=capture_mode,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
            scale=scale,
        )

        image_path: Path | None = None
        mask_path: Path | None = None
        pose_path: Path | None = None
        if captures_dir is not None:
            image_path, mask_path, pose_path = _resolve_capture_files(captures_dir, variant_id)

        if image_path is not None and mask_path is not None:
            with Image.open(image_path) as img:
                rgb = np.asarray(img.convert("RGB"))
            with Image.open(mask_path) as msk:
                mask = np.asarray(msk.convert("L"))
            rgb = _resize_array(rgb, (target_size, target_size))
            mask = _resize_array(mask, (target_size, target_size))

            alpha: np.ndarray | None = None
            alpha_path = captures_dir / f"{variant_id}_alpha.png"
            if alpha_path.exists():
                with Image.open(alpha_path) as a:
                    alpha = _resize_array(np.asarray(a.convert("L")), (target_size, target_size))

            capture_status = "captured"
            confidence = 1.0
            notes = ""
            if pose_path is not None:
                try:
                    payload = json.loads(pose_path.read_text(encoding="utf-8"))
                    confidence = float(payload.get("capture_confidence", 1.0) or 1.0)
                    notes = str(payload.get("capture_notes", "") or "")
                except (OSError, ValueError):
                    notes = "pose_unreadable"
        else:
            # No capture artifacts on disk — emit a placeholder so the entry
            # is still visible in the library (spec 077 FR-026).
            rgb = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            mask = np.zeros((target_size, target_size), dtype=np.uint8)
            alpha = None
            capture_status = "not_attempted"
            confidence = 0.0
            notes = "no_capture_artifacts"

        visibility_class = _visibility_class(asset_path)
        entry_with_data = ObjectLibraryEntry(
            library_id=library_id,
            original_asset_path=asset_path,
            normalized_asset_path=normalized,
            asset_type=detect_asset_type(normalized),
            capture_status=capture_status,
            visibility_class=visibility_class,
            review_state="unreviewed",
            source_builds=source_builds,
            source_maps=source_maps,
            placement_observation_count=observation_count,
            preferred_variant_id=variant_id,
        )
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = 0, 0, target_size, target_size
        variant = ObjectCaptureVariant(
            variant_id=variant_id,
            library_id=library_id,
            capture_build=capture_build,
            capture_mode=capture_mode,
            asset_type=detect_asset_type(normalized),
            image_key=f"capture_rgb/{len(variants)}",
            mask_key=f"capture_mask/{len(variants)}",
            bbox_x0=bbox_x0,
            bbox_y0=bbox_y0,
            bbox_x1=bbox_x1,
            bbox_y1=bbox_y1,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
            scale=scale,
            capture_notes=notes,
            capture_confidence=confidence,
        )
        entries.append(entry_with_data)
        variants.append(variant)
        rgb_list.append(rgb)
        mask_list.append(mask)
        alpha_list.append(alpha)
        seen_entries[library_id] = len(entries) - 1

    return entries, variants, rgb_list, mask_list, alpha_list


def _stack_arrays(
    arrays: list[np.ndarray],
    target_size: int,
    channels: int,
) -> np.ndarray:
    if not arrays:
        return np.zeros((0, target_size, target_size, channels), dtype=np.uint8)
    return np.stack(arrays, axis=0).astype(np.uint8, copy=False)


def _stack_masks(masks: list[np.ndarray], target_size: int) -> np.ndarray:
    if not masks:
        return np.zeros((0, target_size, target_size), dtype=np.uint8)
    return np.stack(masks, axis=0).astype(np.uint8, copy=False)


def _write_zarr(
    output_root: Path,
    run_name: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray | None,
    metadata: dict[str, Any],
) -> Path:
    store_path = output_root / f"{run_name}.zarr"
    if store_path.exists():
        shutil.rmtree(store_path)

    store = zarr.storage.LocalStore(str(store_path), read_only=False)
    root = zarr.group(store=store)

    if rgb.size:
        root.create_array(
            "capture_rgb",
            data=rgb,
            chunks=(min(8, rgb.shape[0]), rgb.shape[1], rgb.shape[2], 3),
            compressors=DEFAULT_CODEC,
        )
    if mask.size:
        root.create_array(
            "capture_mask",
            data=mask,
            chunks=(min(8, mask.shape[0]), mask.shape[1], mask.shape[2]),
            compressors=DEFAULT_CODEC,
        )
    if alpha is not None and alpha.size:
        root.create_array(
            "capture_alpha",
            data=alpha,
            chunks=(min(8, alpha.shape[0]), alpha.shape[1], alpha.shape[2]),
            compressors=DEFAULT_CODEC,
        )

    root.attrs.update(dict(metadata.items()))
    return store_path


def _write_assets_parquet(entries: list[ObjectLibraryEntry], path: Path) -> None:
    table = pa.table(
        {
            "library_id": [e.library_id for e in entries],
            "original_asset_path": [e.original_asset_path for e in entries],
            "normalized_asset_path": [e.normalized_asset_path for e in entries],
            "asset_type": [e.asset_type for e in entries],
            "capture_status": [e.capture_status for e in entries],
            "visibility_class": [e.visibility_class for e in entries],
            "review_state": [e.review_state for e in entries],
            "source_builds": [list(e.source_builds) for e in entries],
            "source_maps": [list(e.source_maps) for e in entries],
            "placement_observation_count": [e.placement_observation_count for e in entries],
            "preferred_variant_id": [e.preferred_variant_id for e in entries],
        }
    )
    pq.write_table(table, str(path))


def _write_index_parquet(variants: list[ObjectCaptureVariant], path: Path) -> None:
    table = pa.table(
        {
            "variant_id": [v.variant_id for v in variants],
            "library_id": [v.library_id for v in variants],
            "capture_build": [v.capture_build for v in variants],
            "capture_mode": [v.capture_mode for v in variants],
            "asset_type": [v.asset_type for v in variants],
            "image_key": [v.image_key for v in variants],
            "mask_key": [v.mask_key for v in variants],
            "bbox_x0": [v.bbox_x0 for v in variants],
            "bbox_y0": [v.bbox_y0 for v in variants],
            "bbox_x1": [v.bbox_x1 for v in variants],
            "bbox_y1": [v.bbox_y1 for v in variants],
            "rot_x": [v.rot_x for v in variants],
            "rot_y": [v.rot_y for v in variants],
            "rot_z": [v.rot_z for v in variants],
            "scale": [v.scale for v in variants],
            "capture_notes": [v.capture_notes for v in variants],
            "capture_confidence": [v.capture_confidence for v in variants],
        }
    )
    pq.write_table(table, str(path))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build spec 077 per-object capture library from enumerated jobs + capture artifacts."
    )
    parser.add_argument("--jobs", type=Path, required=True,
                        help="JSONL of capture jobs from enumerate_object_capture_jobs.py.")
    parser.add_argument("--captures-dir", type=Path, default=None,
                        help="Flat directory of <variant_id>_image.png / _mask.png / _pose.json files.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory under which <run-name>.zarr is written.")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the Zarr store (e.g. smoke_3_3_5_12340).")
    parser.add_argument("--target-size", type=int, default=128,
                        help="Per-variant image/mask size; defaults to 128x128.")
    parser.add_argument("--empty-captures-ok", action="store_true", default=True,
                        help="Emit not_attempted entries for jobs with no capture artifacts.")
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    jobs = _read_jsonl(args.jobs)
    if not jobs:
        print(f"No jobs found in {args.jobs}", file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    captures_dir = args.captures_dir if args.captures_dir is not None else None
    if captures_dir is not None and not captures_dir.exists():
        print(f"Captures dir does not exist; treating as no-captures: {captures_dir}", file=sys.stderr)
        captures_dir = None

    entries, variants, rgb_list, mask_list, alpha_list = _build_entries_and_variants(
        jobs, captures_dir, args.target_size
    )
    if not entries:
        print("No entries produced", file=sys.stderr)
        return 2

    rgb = _stack_arrays(rgb_list, args.target_size, 3)
    mask = _stack_masks(mask_list, args.target_size)
    if any(a is not None for a in alpha_list):
        filled = [a if a is not None else np.zeros((args.target_size, args.target_size), dtype=np.uint8) for a in alpha_list]
        alpha = _stack_masks(filled, args.target_size)
    else:
        alpha = None

    metadata = {
        "schema": "spec-077-object-library",
        "schema_version": "1",
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_size": args.target_size,
        "entry_count": len(entries),
        "variant_count": len(variants),
        "captures_dir": str(captures_dir) if captures_dir is not None else "",
        "jobs_source": str(args.jobs),
    }

    store_path = _write_zarr(args.output_root, args.run_name, rgb, mask, alpha, metadata)
    _write_assets_parquet(entries, store_path / "assets.parquet")
    _write_index_parquet(variants, store_path / "index.parquet")
    print(
        f"Wrote {len(entries)} entries / {len(variants)} variants to {store_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main_with_args())
