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
import subprocess
import sys
from datetime import datetime, timezone
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
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
from harvester.v50.build import find_harvest_dll, read_harvest_stream  # noqa: E402

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


def _derive_jobs_from_roof_dir(roof_dir: Path) -> list[dict[str, Any]]:
    """Build a minimal jobs ledger from a viewer roof-capture output directory.

    Each per-asset subdir carries a ``metadata.json`` with the source
    ``asset_path`` (and ``build``); this lets the library be built without the
    (now-removed) V18 placement tables — the capture output is self-describing.
    """
    jobs: list[dict[str, Any]] = []
    for meta_path in sorted(roof_dir.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        asset_path = str(meta.get("asset_path", "") or "")
        if not asset_path:
            continue
        jobs.append({
            "asset_path": asset_path,
            "build": str(meta.get("build", "") or ""),
            "instance_type": "modf" if asset_path.lower().endswith(".wmo") else "mddf",
            "observation_count": 0,
            "source_maps": [],
        })
    return jobs


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


# Windows-invalid filename characters, minus '/'/'\\' which the C# sanitizer
# maps to '_' before this set is applied.
_ROOF_INVALID_CHARS = set('<>:"|?*') | {chr(c) for c in range(32)}


def _sanitize_roof_name(stem: str) -> str:
    """Replicate the C# ``ViewerApp.SanitizeRoofCaptureName`` used to name each
    per-asset roof-capture output directory, so we can join back to it."""
    out: list[str] = []
    for ch in stem:
        if ch in (" ", "/", "\\"):
            out.append("_")
        elif ch not in _ROOF_INVALID_CHARS and ch != "\0":
            out.append(ch)
    result = "".join(out)
    return result[:100] if len(result) > 100 else result


def _resolve_roof_capture_files(
    roof_dir: Path,
    asset_path: str,
) -> tuple[Path | None, Path | None, Path | None]:
    """Locate (roof_topdown.png, roof_mask.png, metadata.json) for an asset in
    the viewer's nested roof-capture output layout (``<dir>/<sanitized-stem>/``)."""
    stem = PurePosixPath(asset_path.replace("\\", "/")).name
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    asset_dir = roof_dir / _sanitize_roof_name(stem)
    image = asset_dir / "roof_topdown.png"
    mask = asset_dir / "roof_mask.png"
    meta = asset_dir / "metadata.json"
    return (
        image if image.exists() else None,
        mask if mask.exists() else None,
        meta if meta.exists() else None,
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
    roof_dir: Path | None = None,
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
        if roof_dir is not None:
            image_path, mask_path, pose_path = _resolve_roof_capture_files(roof_dir, asset_path)
        elif captures_dir is not None:
            image_path, mask_path, pose_path = _resolve_capture_files(captures_dir, variant_id)

        if image_path is not None and mask_path is not None:
            with Image.open(image_path) as img:
                rgb = np.asarray(img.convert("RGB"))
            with Image.open(mask_path) as msk:
                mask = np.asarray(msk.convert("L"))
            rgb = _resize_array(rgb, (target_size, target_size))
            mask = _resize_array(mask, (target_size, target_size))

            alpha: np.ndarray | None = None
            # Roof captures carry no separate alpha file — the mask is the silhouette.
            alpha_path = captures_dir / f"{variant_id}_alpha.png" if captures_dir is not None else None
            if alpha_path is not None and alpha_path.exists():
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


def find_harvest_project_dll(harvest_project_path: Path) -> Path:
    """Thin re-export of ``harvester.v50.build.find_harvest_dll`` under this module's naming, so
    callers of this file don't need to know the DLL lookup lives in the v50 build helper."""
    return find_harvest_dll(harvest_project_path)


def build_capture_objects_command(
    harvest_project_path: Path,
    *,
    client_root: Path,
    asset_list: Path | None = None,
    resolution: int = 256,
    limit: int | None = None,
) -> list[str]:
    """Construct the exact ``dotnet`` invocation for WowViewer.Tool.Harvest's
    ``capture-objects`` command (Spec 118). Mirrors
    ``harvester.v50.build.build_harvest_stream_command``'s shape/precedent exactly -- only
    constructs the command; the caller decides whether to launch it."""
    dll_path = find_harvest_dll(harvest_project_path)
    cmd = [
        "dotnet",
        str(dll_path),
        "capture-objects",
        "--client-root",
        str(client_root),
        "--resolution",
        str(resolution),
    ]
    if asset_list is not None:
        cmd += ["--asset-list", str(asset_list)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def _build_entries_and_variants_from_stream(
    records: Iterator[dict[str, Any]],
    target_size: int,
) -> tuple[list[ObjectLibraryEntry], list[ObjectCaptureVariant], list[np.ndarray], list[np.ndarray]]:
    """Consume decoded ``capture-objects`` stream records (one per object, already parsed by
    ``harvester.v50.build.read_harvest_stream`` -> ``harvester.raw_reader.read_tile_blob``) into
    the same ``ObjectLibraryEntry``/``ObjectCaptureVariant`` shape the PNG-folder path builds, so
    both paths share the same zarr writer below."""
    entries: list[ObjectLibraryEntry] = []
    variants: list[ObjectCaptureVariant] = []
    rgb_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    for record in records:
        meta = record.get("_metadata", {})
        asset_path = str(meta.get("asset_path", "") or "")
        if not asset_path or "image_rgb" not in record or "mask" not in record:
            continue

        normalized = normalize_asset_path(asset_path)
        if not normalized:
            continue
        library_id = library_id_from_asset_path(normalized)
        capture_build = str(meta.get("build", "") or "")
        capture_mode = str(meta.get("capture_mode", "orthographic_topdown") or "orthographic_topdown")
        variant_id = make_variant_id(
            library_id=library_id, capture_build=capture_build, capture_mode=capture_mode,
            rot_x=0.0, rot_y=0.0, rot_z=0.0, scale=1.0,
        )

        rgb = _resize_array(np.asarray(record["image_rgb"], dtype=np.uint8), (target_size, target_size))
        mask = _resize_array(np.asarray(record["mask"], dtype=np.uint8), (target_size, target_size))

        visibility_class = "clutter_filtered" if is_clutter_asset(normalized) else (
            "roof_visible" if str(meta.get("asset_type", "")).lower() == "wmo" else "likely_visible"
        )
        entries.append(ObjectLibraryEntry(
            library_id=library_id,
            original_asset_path=asset_path,
            normalized_asset_path=normalized,
            asset_type=detect_asset_type(normalized),
            capture_status="captured",
            visibility_class=visibility_class,
            review_state="unreviewed",
            source_builds=(capture_build,) if capture_build else (),
            source_maps=(),
            placement_observation_count=0,
            preferred_variant_id=variant_id,
        ))
        variants.append(ObjectCaptureVariant(
            variant_id=variant_id,
            library_id=library_id,
            capture_build=capture_build,
            capture_mode=capture_mode,
            asset_type=detect_asset_type(normalized),
            image_key=f"capture_rgb/{len(variants)}",
            mask_key=f"capture_mask/{len(variants)}",
            bbox_x0=0, bbox_y0=0, bbox_x1=target_size, bbox_y1=target_size,
            rot_x=0.0, rot_y=0.0, rot_z=0.0, scale=1.0,
            capture_notes="",
            capture_confidence=1.0,
        ))
        rgb_list.append(rgb)
        mask_list.append(mask)

    return entries, variants, rgb_list, mask_list


def run_capture_objects_stream(
    command: list[str],
    *,
    target_size: int,
    confirm_run: bool,
) -> tuple[list[ObjectLibraryEntry], list[ObjectCaptureVariant], list[np.ndarray], list[np.ndarray]] | None:
    """Print the constructed command and return None unless ``confirm_run=True`` (mirrors
    ``harvester.v50.build.run_fresh_extraction``'s gate exactly -- preparing/testing this path
    does not authorize spending real capture time against a live client). Only when explicitly
    confirmed does this launch the subprocess and consume its stdout stream."""
    if not confirm_run:
        print(f"Would run object capture: {' '.join(command)}")
        print("NOT executing: pass --run only with explicit user authorization for this run.")
        return None

    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        entries, variants, rgb_list, mask_list = _build_entries_and_variants_from_stream(
            read_harvest_stream(proc.stdout), target_size
        )
    finally:
        proc.stdout.close()
        proc.wait()
    return entries, variants, rgb_list, mask_list


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
    parser.add_argument("--jobs", type=Path, default=None,
                        help="JSONL of capture jobs from enumerate_object_capture_jobs.py. "
                             "Optional when --roof-captures-dir is given (jobs are then derived "
                             "from each asset's metadata.json).")
    parser.add_argument("--captures-dir", type=Path, default=None,
                        help="Flat directory of <variant_id>_image.png / _mask.png / _pose.json files.")
    parser.add_argument("--roof-captures-dir", type=Path, default=None,
                        help="Nested viewer roof-capture output (<dir>/<sanitized-asset-stem>/roof_topdown.png "
                             "+ roof_mask.png). Takes precedence over --captures-dir when set.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory under which <run-name>.zarr is written.")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the Zarr store (e.g. smoke_3_3_5_12340).")
    parser.add_argument("--target-size", type=int, default=128,
                        help="Per-variant image/mask size; defaults to 128x128.")
    parser.add_argument("--empty-captures-ok", action="store_true", default=True,
                        help="Emit not_attempted entries for jobs with no capture artifacts.")

    stream_group = parser.add_argument_group(
        "harvest-stream mode (Spec 118)",
        "Drive WowViewer.Tool.Harvest's capture-objects command directly -- no PNG folder, no "
        "GUI click. Mutually exclusive with --jobs/--captures-dir/--roof-captures-dir.")
    stream_group.add_argument("--from-harvest-stream", action="store_true", default=False,
                        help="Build the library from a live capture-objects stream instead of a PNG folder.")
    stream_group.add_argument("--harvest-project", type=Path, default=None,
                        help="Path to WowViewer.Tool.Harvest's .csproj (or its bin output dir).")
    stream_group.add_argument("--client-root", type=Path, default=None,
                        help="WoW client root directory passed through to capture-objects.")
    stream_group.add_argument("--asset-list", type=Path, default=None,
                        help="Optional JSON array of asset paths; omit to enumerate the whole client listfile.")
    stream_group.add_argument("--resolution", type=int, default=256,
                        help="Per-object capture resolution (square); defaults to 256.")
    stream_group.add_argument("--capture-limit", type=int, default=None,
                        help="Cap the number of objects captured (testing/smoke runs).")
    stream_group.add_argument("--run", action="store_true", default=False,
                        help="Actually launch capture-objects and consume its stream (default: print "
                             "the exact command and exit without running it -- Rule 0 execution gate).")
    return parser.parse_args(argv)


def _main_from_harvest_stream(args: argparse.Namespace) -> int:
    if args.harvest_project is None or args.client_root is None:
        print("--from-harvest-stream requires --harvest-project and --client-root", file=sys.stderr)
        return 2

    command = build_capture_objects_command(
        args.harvest_project,
        client_root=args.client_root,
        asset_list=args.asset_list,
        resolution=args.resolution,
        limit=args.capture_limit,
    )

    result = run_capture_objects_stream(command, target_size=args.target_size, confirm_run=args.run)
    if result is None:
        return 0  # Dry-run: command printed, nothing executed (Rule 0 gate).

    entries, variants, rgb_list, mask_list = result
    if not entries:
        print("No entries produced from capture-objects stream", file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    rgb = _stack_arrays(rgb_list, args.target_size, 3)
    mask = _stack_masks(mask_list, args.target_size)

    metadata = {
        "schema": "spec-077-object-library",
        "schema_version": "1",
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_size": args.target_size,
        "entry_count": len(entries),
        "variant_count": len(variants),
        "source": "capture-objects-stream",
        "harvest_command": " ".join(command),
    }

    store_path = _write_zarr(args.output_root, args.run_name, rgb, mask, None, metadata)
    _write_assets_parquet(entries, store_path / "assets.parquet")
    _write_index_parquet(variants, store_path / "index.parquet")
    print(f"Wrote {len(entries)} entries / {len(variants)} variants to {store_path}")
    return 0


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.from_harvest_stream:
        return _main_from_harvest_stream(args)

    args.output_root.mkdir(parents=True, exist_ok=True)
    captures_dir = args.captures_dir if args.captures_dir is not None else None
    if captures_dir is not None and not captures_dir.exists():
        print(f"Captures dir does not exist; treating as no-captures: {captures_dir}", file=sys.stderr)
        captures_dir = None

    roof_dir = args.roof_captures_dir if args.roof_captures_dir is not None else None
    if roof_dir is not None and not roof_dir.exists():
        print(f"Roof captures dir does not exist; treating as no-captures: {roof_dir}", file=sys.stderr)
        roof_dir = None

    if args.jobs is not None:
        jobs = _read_jsonl(args.jobs)
    elif roof_dir is not None:
        # No explicit jobs ledger — derive it from the roof-capture output.
        jobs = _derive_jobs_from_roof_dir(roof_dir)
    else:
        print("Provide --jobs, or --roof-captures-dir to derive jobs from capture metadata.", file=sys.stderr)
        return 2

    if not jobs:
        src = args.jobs if args.jobs is not None else roof_dir
        print(f"No jobs found ({src})", file=sys.stderr)
        return 2

    entries, variants, rgb_list, mask_list, alpha_list = _build_entries_and_variants(
        jobs, captures_dir, args.target_size, roof_dir=roof_dir
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
        "roof_captures_dir": str(roof_dir) if roof_dir is not None else "",
        "jobs_source": str(args.jobs) if args.jobs is not None else (f"derived:{roof_dir}" if roof_dir is not None else ""),
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
