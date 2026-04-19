#!/usr/bin/env python3
"""
generate_m2_masks.py — Build per-tile M2 object footprint masks for the V7 training dataset.

For each tile JSON that contains M2 (or WMO) objects this script:
  1. Collects unique M2 model paths and calls wow-viewer `m2-bounds` (batched per archive root)
     to retrieve model-local BoundsMin/BoundsMax.
  2. Projects every placed object into the tile's 512×512 UV space using its bounds and scale.
  3. Writes a grayscale mask PNG (255 = object footprint) and stores the relative path back into
     the tile JSON under `object_visibility_mask` so that train_v7.py picks it up automatically.
  4. Skips tiles whose mask already exists when --skip-existing is set (the default).

Usage examples
--------------
    # Process all datasets with default archive roots, skip done tiles:
    python generate_m2_masks.py

    # Limit to one build, write verbose diagnostics:
    python generate_m2_masks.py --build-filter 3_3_5_12340 --verbose

    # Custom archive root overrides (JSON map of build_label → archive_root):
    python generate_m2_masks.py --archive-roots-file my_roots.json

    # Force re-generation of all masks:
    python generate_m2_masks.py --skip-existing=false

Coordinate system
-----------------
WoW tile (tile_x, tile_y) parsed from the tile filename e.g. AhnQiraj_27_49.json → (27, 49).
Object world coordinates: local_u = obj_x / TILE_SIZE - tile_x
                          local_v = obj_z / TILE_SIZE - tile_y
Bounds radius in pixels: half_extent_world / TILE_SIZE * output_size
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants (must match v7_object_masks.py)
# ---------------------------------------------------------------------------
TILE_SIZE = 533.33333
MAP_ORIGIN = 32.0 * TILE_SIZE
OUTPUT_SIZE = 512
MASK_CONTEXT_MARGIN_TILES = 0.20
MASK_DIR = "masks"

# ---------------------------------------------------------------------------
# Known archive roots — inferred from data-paths.md
# ---------------------------------------------------------------------------
KNOWN_ARCHIVE_ROOTS: Dict[str, str] = {
    "3_3_5_12340": r"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft",
    "4_0_0_11927": r"H:\CLIENTS\World of Warcraft Cata beta 11927",
    "3_0_1_8303": r"H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft",
}

# ---------------------------------------------------------------------------
# Locate wow-viewer executable (relative to workspace root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_WOW_VIEWER_ROOT = _SCRIPT_DIR.parent
_WORKSPACE_ROOT = _WOW_VIEWER_ROOT.parent  # parp-tools/

_WOWVIEWER_CANDIDATES = [
    _WORKSPACE_ROOT / "wow-viewer" / "src" / "viewer" / "WowViewer.App" / "bin" / "Debug" / "net10.0-windows" / "WowViewer.App.exe",
    _WORKSPACE_ROOT / "wow-viewer" / "src" / "viewer" / "WowViewer.App" / "bin" / "Debug" / "net10.0" / "WowViewer.App.exe",
    _WORKSPACE_ROOT / "wow-viewer" / "src" / "viewer" / "WowViewer.App" / "bin" / "Release" / "net10.0-windows" / "WowViewer.App.exe",
    _WORKSPACE_ROOT / "wow-viewer" / "src" / "viewer" / "WowViewer.App" / "bin" / "Release" / "net10.0" / "WowViewer.App.exe",
]

_BOUNDS_CACHE_DIR = _WORKSPACE_ROOT / "output" / "tmp" / "m2_bounds_cache"
_WOWARCHIVE_STAGED_ROOT = _WORKSPACE_ROOT / "output" / "tmp" / "wowarchive-clients"
_WOWARCHIVE_MOUNT_ROOT = Path(r"G:\WoW\WoWArchive-0.X-3.X\Mount")


def find_wow_viewer_exe() -> Optional[Path]:
    for candidate in _WOWVIEWER_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Dataset discovery  (mirrors cache_v7_6_data.py)
# ---------------------------------------------------------------------------
DATASETS_ROOT = _WORKSPACE_ROOT / "datasets"


def discover_dataset_roots(search_roots: List[str]) -> List[Path]:
    discovered: List[Path] = []
    seen: set = set()
    for root_text in search_roots:
        root = Path(root_text)
        if not root.exists():
            continue
        if (root / "dataset").exists() and (root / "images").exists():
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(root)
        for manifest_path in sorted(root.rglob("ml_dataset_manifest.json")):
            candidate = manifest_path.parent
            if not (candidate / "dataset").exists():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(candidate)
    return discovered


def build_label_from_path(dataset_root: Path) -> Optional[str]:
    """Return the build label string (e.g. '3_3_5_12340') from the dataset path."""
    # Expected layout: datasets/{build_label}/{MapName}/
    parts = dataset_root.resolve().parts
    datasets_parts = DATASETS_ROOT.resolve().parts
    if len(parts) > len(datasets_parts):
        candidate = parts[len(datasets_parts)]
        if re.match(r"^\d", candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def parse_tile_coords(tile_name: str) -> Optional[Tuple[int, int]]:
    """Parse (tile_x, tile_y) from tile name like AhnQiraj_27_49."""
    m = re.search(r"_(\d+)_(\d+)$", tile_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def choose_local_uv(
    world_x: float,
    world_y: float,
    world_z: float,
    tile_x: int,
    tile_y: int,
) -> Optional[Tuple[float, float]]:
    """Return the best (local_u, local_v) in [0,1] for the given world position.

    Tries (pos_x, pos_z) and (pos_x, pos_y) as candidate horizontal pairs, each
    under two interpretations (direct scale and map-origin-relative), picking
    whichever candidate overflows the [0,1] tile range the least.
    """
    candidates: List[Tuple[float, float]] = []
    for a, b in [(world_x, world_z), (world_x, world_y)]:
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        # Interpretation 1: absolute tile index coordinates
        candidates.append((a / TILE_SIZE - tile_x, b / TILE_SIZE - tile_y))
        # Interpretation 2: WoW map-origin relative
        candidates.append(((MAP_ORIGIN - b) / TILE_SIZE - tile_x, (MAP_ORIGIN - a) / TILE_SIZE - tile_y))

    best: Optional[Tuple[float, float]] = None
    best_overflow = float("inf")
    for u, v in candidates:
        overflow = (max(0.0, -u) + max(0.0, u - 1.0) + max(0.0, -v) + max(0.0, v - 1.0))
        if overflow < best_overflow:
            best_overflow = overflow
            best = (u, v)
            if overflow <= 1e-6:
                break

    if best is None:
        return None
    u, v = best
    if u < -MASK_CONTEXT_MARGIN_TILES or u > 1.0 + MASK_CONTEXT_MARGIN_TILES:
        return None
    if v < -MASK_CONTEXT_MARGIN_TILES or v > 1.0 + MASK_CONTEXT_MARGIN_TILES:
        return None
    return u, v


def bounds_to_pixel_radii(
    bmin: List[float],
    bmax: List[float],
    scale: float,
    output_size: int,
) -> Tuple[int, int]:
    """Return (radius_x, radius_y) in pixels from model-local bounds and placed scale."""
    half_w = abs(bmax[0] - bmin[0]) * 0.5 * scale
    half_d = abs(bmax[2] - bmin[2]) * 0.5 * scale
    ppu = output_size / TILE_SIZE
    rx = max(2, int(round(half_w * ppu)))
    ry = max(2, int(round(half_d * ppu)))
    return rx, ry


def fallback_pixel_radius(scale: float, category: str, output_size: int) -> Tuple[int, int]:
    """Fallback ellipse radius when no bounds are available."""
    base = 3.0 * max(0.1, scale)
    if category == "wmo":
        base *= 2.0
    r = max(2, int(round(base / TILE_SIZE * output_size)))
    return r, r


def parse_bounds_string(value: Any) -> Optional[List[float]]:
    """Parse a bounds value which may be None, a list, or a space-separated string."""
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) >= 3 and all(isinstance(v, (int, float)) for v in value[:3]):
            return [float(value[0]), float(value[1]), float(value[2])]
        return None
    if isinstance(value, str):
        parts = value.strip().split()
        if len(parts) >= 3:
            try:
                return [float(p) for p in parts[:3]]
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# wow-viewer bounds querying
# ---------------------------------------------------------------------------

def _bounds_cache_path(build_label: str) -> Path:
    _BOUNDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _BOUNDS_CACHE_DIR / f"m2_bounds_{build_label}.json"


def load_bounds_cache(build_label: str) -> Dict[str, Any]:
    p = _bounds_cache_path(build_label)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_bounds_cache(build_label: str, cache: Dict[str, Any]) -> None:
    p = _bounds_cache_path(build_label)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))


def query_m2_bounds(
    model_paths: List[str],
    archive_root: str,
    wow_viewer_exe: Path,
    build_label: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Call wow-viewer m2-bounds for a batch of model paths.

    Returns a dict keyed by normalized model path (lowercase / forward-slash).
    Values are dict with keys: bounds_min, bounds_max, bounds_radius, error.
    """
    cache = load_bounds_cache(build_label)
    norm = lambda p: p.replace("\\", "/").lower()  # noqa: E731

    missing = [p for p in model_paths if norm(p) not in cache]
    if not missing:
        return cache

    if verbose:
        print(f"  [m2-bounds] Querying {len(missing)} new models for {build_label} …")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(missing))
        tmp_path = tmp.name

    out_path = tmp_path + "_bounds.json"
    try:
        result = subprocess.run(
            [str(wow_viewer_exe), "m2-bounds",
             "--archive-root", archive_root,
             "--model-list", tmp_path,
             "--output", out_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if verbose and result.stdout.strip():
            print(f"    {result.stdout.strip()}")
        if result.returncode not in (0, 1):
            print(f"  [m2-bounds] wow-viewer exited {result.returncode}: {result.stderr.strip()[:200]}")

        if Path(out_path).exists():
            with open(out_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                key = norm(entry["model_path"])
                cache[key] = {
                    "bounds_min": entry.get("bounds_min"),
                    "bounds_max": entry.get("bounds_max"),
                    "bounds_radius": entry.get("bounds_radius"),
                    "error": entry.get("error"),
                }
        else:
            print("  [m2-bounds] Output file not created; all queried models marked as errored.")
            for path in missing:
                cache[norm(path)] = {"bounds_min": None, "bounds_max": None, "bounds_radius": None, "error": "output missing"}

        save_bounds_cache(build_label, cache)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        try:
            if Path(out_path).exists():
                os.unlink(out_path)
        except OSError:
            pass

    return cache


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_tile_mask(
    objects: List[Dict[str, Any]],
    tile_x: int,
    tile_y: int,
    bounds_cache: Dict[str, Any],
    output_size: int,
    verbose: bool,
) -> Tuple[np.ndarray, int, int]:
    """Return (mask_hwc, drawn_count, fallback_count) where mask is uint8 H×W."""
    img = Image.new("L", (output_size, output_size), 0)
    draw = ImageDraw.Draw(img)
    drawn = 0
    fallback = 0

    for obj in objects:
        category = str(obj.get("category", "") or "").lower()
        scale = float(obj.get("scale") or 1.0)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        world_x = float(obj.get("x") or 0.0)
        world_y = float(obj.get("y") or 0.0)
        world_z = float(obj.get("z") or 0.0)

        uv = choose_local_uv(world_x, world_y, world_z, tile_x, tile_y)
        if uv is None:
            if verbose:
                print(f"    skip off-tile object {obj.get('name')!r}")
            continue

        local_u, local_v = uv
        cx = int(round(local_u * output_size))
        cy = int(round(local_v * output_size))

        # Determine radii from bounds
        bmin = parse_bounds_string(obj.get("bounds_min"))
        bmax = parse_bounds_string(obj.get("bounds_max"))

        # For M2 objects with null bounds, try the wow-viewer cache
        if (bmin is None or bmax is None) and category == "m2":
            model_path = str(obj.get("model_path") or "")
            norm_path = model_path.replace("\\", "/").lower()
            entry = bounds_cache.get(norm_path)
            if entry and entry.get("bounds_min") and entry.get("bounds_max"):
                bmin = entry["bounds_min"]
                bmax = entry["bounds_max"]

        if bmin is not None and bmax is not None:
            rx, ry = bounds_to_pixel_radii(bmin, bmax, scale if category == "m2" else 1.0, output_size)
            used_fallback = False
        else:
            rx, ry = fallback_pixel_radius(scale, category, output_size)
            used_fallback = True
            fallback += 1

        # Draw filled ellipse
        x0, y0 = cx - rx, cy - ry
        x1, y1 = cx + rx, cy + ry
        draw.ellipse([x0, y0, x1, y1], fill=255)
        drawn += 1
        if verbose:
            label = f"fallback" if used_fallback else "bounds"
            print(f"    draw {obj.get('name')!r} ({category}) uv=({local_u:.3f},{local_v:.3f}) px=({cx},{cy}) r=({rx},{ry}) [{label}]")

    return np.asarray(img, dtype=np.uint8), drawn, fallback


# ---------------------------------------------------------------------------
# Tile JSON update
# ---------------------------------------------------------------------------

def update_tile_json(json_path: Path, mask_rel_path: str) -> None:
    """Write object_visibility_mask into the terrain_data block of a tile JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    td = data.get("terrain_data")
    if td is None:
        data["terrain_data"] = {}
        td = data["terrain_data"]

    td["object_visibility_mask"] = mask_rel_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset_root(
    dataset_root: Path,
    archive_root: Optional[str],
    wow_viewer_exe: Optional[Path],
    skip_existing: bool,
    dry_run: bool,
    verbose: bool,
    output_size: int,
    build_label: Optional[str],
) -> Dict[str, int]:
    stats: Dict[str, int] = dict(tiles=0, skipped=0, generated=0, no_objects=0, errors=0)
    json_dir = dataset_root / "dataset"
    if not json_dir.exists():
        return stats

    tile_jsons = sorted(json_dir.glob("*.json"))
    if not tile_jsons:
        return stats

    mask_dir = dataset_root / MASK_DIR
    if not dry_run:
        mask_dir.mkdir(exist_ok=True)

    # First pass: collect unique M2 model paths needing bounds
    unique_m2_paths: set = set()
    tile_data_cache: Dict[Path, Any] = {}

    for json_path in tile_jsons:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [warn] Could not load {json_path.name}: {e}")
            continue

        td = data.get("terrain_data") or {}
        objects = td.get("objects") or []
        tile_data_cache[json_path] = (data, td, objects)

        for obj in objects:
            if str(obj.get("category", "") or "").lower() == "m2":
                bmin = parse_bounds_string(obj.get("bounds_min"))
                bmax = parse_bounds_string(obj.get("bounds_max"))
                if bmin is None or bmax is None:
                    model_path = str(obj.get("model_path") or "")
                    if model_path:
                        unique_m2_paths.add(model_path)

    # Query wow-viewer for missing bounds
    bounds_cache: Dict[str, Any] = {}
    if unique_m2_paths and archive_root and wow_viewer_exe:
        bounds_cache = query_m2_bounds(
            list(unique_m2_paths),
            archive_root,
            wow_viewer_exe,
            build_label or "unknown",
            verbose=verbose,
        )
    elif unique_m2_paths and not archive_root:
        print(f"  [warn] {len(unique_m2_paths)} M2 models need bounds but no archive root is configured for build '{build_label}'. "
              "Pass --archive-roots-file to supply custom mappings or ensure the build is in KNOWN_ARCHIVE_ROOTS.")
        # Still load whatever is cached on disk
        bounds_cache = load_bounds_cache(build_label or "unknown")
    elif unique_m2_paths and not wow_viewer_exe:
        print("  [warn] wow-viewer executable not found; loading bounds from disk cache only.")
        bounds_cache = load_bounds_cache(build_label or "unknown")

    # Second pass: generate masks
    for json_path in tile_jsons:
        stats["tiles"] += 1
        entry = tile_data_cache.get(json_path)
        if entry is None:
            stats["errors"] += 1
            continue

        data, td, objects = entry
        tile_name = json_path.stem
        coords = parse_tile_coords(tile_name)
        if coords is None:
            print(f"  [warn] Cannot parse tile coords from {tile_name!r}; skipping.")
            stats["errors"] += 1
            continue

        tile_x, tile_y = coords
        mask_filename = f"{tile_name}_obj_mask.png"
        mask_path = mask_dir / mask_filename
        mask_rel = f"{MASK_DIR}/{mask_filename}"

        # Skip if mask exists and tile JSON already references it
        if skip_existing:
            existing = str(td.get("object_visibility_mask") or "")
            if existing and (dataset_root / existing).exists():
                stats["skipped"] += 1
                continue
            if mask_path.exists():
                # Mask exists but JSON not updated — still update JSON
                if not dry_run:
                    update_tile_json(json_path, mask_rel)
                stats["skipped"] += 1
                continue

        if not objects:
            stats["no_objects"] += 1
            # Write an empty (all-zero) mask so training treats it as "no objects"
            if not dry_run:
                empty = np.zeros((output_size, output_size), dtype=np.uint8)
                Image.fromarray(empty, mode="L").save(mask_path)
                update_tile_json(json_path, mask_rel)
            continue

        if verbose:
            print(f"  {tile_name}: {len(objects)} objects, tile ({tile_x}, {tile_y})")

        try:
            mask, drawn, fallback = generate_tile_mask(
                objects, tile_x, tile_y, bounds_cache, output_size, verbose
            )
            if verbose:
                print(f"    → drew {drawn} objects ({fallback} fallback radius), coverage={np.mean(mask > 0):.3f}")

            if not dry_run:
                Image.fromarray(mask, mode="L").save(mask_path)
                update_tile_json(json_path, mask_rel)

            stats["generated"] += 1
        except Exception as e:
            print(f"  [error] {tile_name}: {e}")
            stats["errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-tile M2 object footprint masks for V7 training datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset_roots", nargs="*",
                   help="Explicit dataset roots (map directories) to process. Omit to auto-discover from --search-root.")
    p.add_argument("--search-root", action="append", default=None,
                   help="Search root for dataset discovery. Default: datasets/. Repeat to add more.")
    p.add_argument("--build-filter", default=None,
                   help="Only process datasets from this build label (e.g. 3_3_5_12340).")
    p.add_argument("--archive-roots-file", default=None,
                   help="JSON file mapping build_label → archive_root path (overrides built-in defaults).")
    p.add_argument("--archive-root-fallback", default=None,
                   help="Fallback archive root used when a build_label has no direct mapping.")
    p.add_argument("--wow-viewer-exe", default=None,
                   help="Path to WowViewer.App.exe. Auto-detected from workspace if omitted.")
    p.add_argument("--output-size", type=int, default=OUTPUT_SIZE,
                   help=f"Mask output resolution (default {OUTPUT_SIZE}).")
    p.add_argument("--skip-existing", type=lambda v: v.lower() not in ("false", "0", "no"), default=True,
                   help="Skip tiles that already have a mask. Default true.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write any files; only print what would be done.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-object projection details.")
    return p.parse_args()


def resolve_archive_root_for_build(
    build_label: Optional[str],
    effective_archive_roots: Dict[str, str],
    archive_root_fallback: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve archive root for build label and return (root, reason)."""
    if not build_label:
        return None, None

    exact = effective_archive_roots.get(build_label)
    if exact:
        return exact, "mapped"

    staged_candidate = _WOWARCHIVE_STAGED_ROOT / build_label
    if staged_candidate.exists():
        return str(staged_candidate), "staged-wowarchive"

    mounted_candidate = _WOWARCHIVE_MOUNT_ROOT / build_label
    if mounted_candidate.exists():
        return str(mounted_candidate), "mounted-wowarchive"

    # Helpful default for older pre-release labels if the 3.0.1 root is available.
    # This is a fallback, not a parity claim.
    if build_label.startswith("0_"):
        pre_release_root = effective_archive_roots.get("3_0_1_8303")
        if pre_release_root:
            return pre_release_root, "fallback-3_0_1_8303"

    if archive_root_fallback:
        return archive_root_fallback, "user-fallback"

    return None, None


def main() -> int:
    args = parse_args()

    # Resolve archive root overrides
    archive_roots_override: Dict[str, str] = {}
    if args.archive_roots_file:
        with open(args.archive_roots_file, "r", encoding="utf-8") as f:
            archive_roots_override = json.load(f)

    effective_archive_roots = {**KNOWN_ARCHIVE_ROOTS, **archive_roots_override}
    archive_root_fallback = args.archive_root_fallback

    # Resolve wow-viewer exe
    if args.wow_viewer_exe:
        wow_viewer_exe: Optional[Path] = Path(args.wow_viewer_exe)
        if not wow_viewer_exe.exists():
            print(f"[error] wow-viewer exe not found: {wow_viewer_exe}")
            return 1
    else:
        wow_viewer_exe = find_wow_viewer_exe()
        if wow_viewer_exe is None:
            print("[warn] wow-viewer executable not found in default locations. M2 bounds will use disk cache only.")
            print("       Run: dotnet build wow-viewer/WowViewer.slnx -c Debug")
        else:
            if args.verbose:
                print(f"[info] wow-viewer: {wow_viewer_exe}")

    # Resolve dataset roots
    search_roots = args.search_root or [str(DATASETS_ROOT)]
    if args.dataset_roots:
        all_roots = [Path(r) for r in args.dataset_roots]
    else:
        all_roots = discover_dataset_roots(search_roots)

    if not all_roots:
        print("[error] No dataset roots found.")
        return 1

    # Apply build filter
    if args.build_filter:
        filtered = []
        for root in all_roots:
            label = build_label_from_path(root)
            if label == args.build_filter:
                filtered.append(root)
        all_roots = filtered
        if not all_roots:
            print(f"[error] No dataset roots found for build '{args.build_filter}'.")
            return 1

    print(f"Processing {len(all_roots)} dataset root(s) …")
    if args.dry_run:
        print("[DRY RUN] No files will be written.")

    total_stats: Dict[str, int] = dict(tiles=0, skipped=0, generated=0, no_objects=0, errors=0)

    for root in all_roots:
        build_label = build_label_from_path(root)
        archive_root, archive_reason = resolve_archive_root_for_build(build_label, effective_archive_roots, archive_root_fallback)
        if archive_root and not Path(archive_root).exists():
            print(f"  [warn] Archive root does not exist: {archive_root}  (build={build_label})")
            archive_root = None
            archive_reason = None

        map_name = root.name
        if archive_root and archive_reason:
            print(f"\n  {map_name}  [{build_label or 'unknown'}]  archive=yes ({archive_reason})")
        else:
            print(f"\n  {map_name}  [{build_label or 'unknown'}]  archive={'yes' if archive_root else 'none'}")

        stats = process_dataset_root(
            dataset_root=root,
            archive_root=archive_root,
            wow_viewer_exe=wow_viewer_exe if (archive_root and wow_viewer_exe) else None,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            verbose=args.verbose,
            output_size=args.output_size,
            build_label=build_label,
        )

        for k, v in stats.items():
            total_stats[k] = total_stats.get(k, 0) + v

        print(f"    tiles={stats['tiles']} generated={stats['generated']} skipped={stats['skipped']} "
              f"no_objects={stats['no_objects']} errors={stats['errors']}")

    print(
        f"\nDone. Total: tiles={total_stats['tiles']} generated={total_stats['generated']} "
        f"skipped={total_stats['skipped']} no_objects={total_stats['no_objects']} errors={total_stats['errors']}"
    )
    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
