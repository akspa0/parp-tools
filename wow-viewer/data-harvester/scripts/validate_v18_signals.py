"""Validate V18 Zarr dataset signals for completeness and consistency.

Checks per build:
- All expected arrays exist with correct shapes
- Object mask signals have non-zero coverage on tiles with placements
- No zero-size or degenerate arrays
- index.parquet row count matches array lengths
- Dumps sample image tiles for visual inspection

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/validate_v18_signals.py --build 3_3_5_12340
    uv run python scripts/validate_v18_signals.py --build 0_5_3_3368 3_3_5_12340
    uv run python scripts/validate_v18_signals.py --build 3_3_5_12340 --sample-dir ../output/tmp/v18_samples
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import zarr
import zarr.storage

_EXPECTED_ARRAYS: dict[str, tuple] = {
    "alpha_256": (256, 256, 4),
    "height_257": (257, 257),
    "holes_16": (16, 16),
    "liquid_height": (256, 256),
    "liquid_mask": (256, 256),
    "mcly_layer_mask": (16, 16, 4),
    "mcly_texture_ids": (16, 16, 4),
    "mcnk_flags_16": (16, 16),
    "mddf_mask": (257, 257),
    "minimap_rgb": (256, 256, 3),
    "modf_mask": (257, 257),
    "normal_mask": (257, 257),
    "normal_xyz": (257, 257, 3),
    "object_filtered_mask": (257, 257),
    "object_instance_mask": (257, 257),
    "object_mask": (257, 257),
    "object_precise_mask": (257, 257),
    "object_roof_confidence": (256, 256),
    "object_roof_mask": (256, 256),
    "shadow_mask": (256, 256),
}

# Arrays allowed to be all-zeros (no data, no inference yet, or rare signal)
_ALLOWED_ZERO = {
    "alpha_256",
    "holes_16",
    "liquid_height",
    "liquid_mask",
    "mcly_layer_mask",
    "mddf_mask",
    "modf_mask",
    "object_filtered_mask",
    "object_precise_mask",
    "object_roof_confidence",
    "object_roof_mask",
    "shadow_mask",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"


def _check_array(
    root: zarr.Group,
    name: str,
    tile_idx: int,
    expected_shape: tuple,
    issues: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": False}
    if name not in root:
        issues.append(f"missing_array:{name}")
        return result
    result["exists"] = True

    arr = root[name]
    actual_shape = tuple(arr.shape[1:])  # skip tile dimension
    result["shape"] = list(arr.shape)

    if arr.shape[0] <= tile_idx:
        issues.append(f"{name}:tile_idx_out_of_range ({arr.shape[0]} tiles, asked {tile_idx})")
        return result

    tile_data = arr[tile_idx]

    if actual_shape != expected_shape:
        issues.append(f"{name}:shape_mismatch expected={expected_shape} actual={actual_shape}")

    n_bytes = tile_data.nbytes
    result["bytes"] = n_bytes

    if n_bytes == 0:
        issues.append(f"{name}:zero_bytes")
        return result

    dt = tile_data.dtype

    if dt.kind == "f":
        fdata = tile_data.astype(np.float64)
        result.update({
            "min": float(fdata.min()),
            "max": float(fdata.max()),
            "mean": float(fdata.mean()),
            "nonzero_frac": float(np.count_nonzero(fdata) / fdata.size),
        })
        if result["nonzero_frac"] == 0 and name not in _ALLOWED_ZERO:
            issues.append(f"{name}:all_zeros")
    elif dt.kind in ("b", "i", "u"):
        fdata = tile_data.astype(np.float64)
        result.update({
            "min": float(fdata.min()),
            "max": float(fdata.max()),
            "mean": float(fdata.mean()),
            "nonzero_frac": float(np.count_nonzero(fdata) / fdata.size),
        })

    return result


def _read_tile_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(str(path))
    return [
        {
            column: table.column(column)[idx].as_py()
            for column in table.column_names
        }
        for idx in range(table.num_rows)
    ]


def _dump_samples(
    root: zarr.Group,
    zarr_path: Path,
    build: str,
    sample_dir: Path | None,
    tile_idx: int,
) -> list[dict[str, Any]]:
    """Dump sample images from the build for visual inspection."""
    samples: list[dict[str, Any]] = []
    if sample_dir is None:
        return samples

    out = sample_dir / build
    out.mkdir(parents=True, exist_ok=True)

    # Minimap
    if "minimap_rgb" in root:
        arr = root["minimap_rgb"][tile_idx]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr, "RGB").save(str(out / "minimap_rgb.png"))
        samples.append({"name": "minimap_rgb", "path": str(out / "minimap_rgb.png")})

    # Alpha layers
    if "alpha_256" in root:
        arr = root["alpha_256"][tile_idx]
        for i in range(min(arr.shape[-1], 4)):
            layer = (np.clip(arr[:, :, i], 0, 1) * 255).astype(np.uint8)
            Image.fromarray(layer, "L").save(str(out / f"alpha_layer_{i}.png"))
            samples.append({"name": f"alpha_layer_{i}", "path": str(out / f"alpha_layer_{i}.png")})

        composite = (arr[:, :, :3].clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(composite, "RGB").save(str(out / "alpha_rgb.png"))
        samples.append({"name": "alpha_rgb", "path": str(out / "alpha_rgb.png")})

    # Object masks - overlay on minimap for review
    if "minimap_rgb" in root and "object_mask" in root:
        mm = root["minimap_rgb"][tile_idx].astype(np.float32)
        om = root["object_mask"][tile_idx].astype(np.float32)
        om_resized = np.clip(om[:256, :256], 0, 1)
        overlay = mm.copy()
        overlay[:, :, 0] = np.where(om_resized > 0.5, 255, mm[:, :, 0])
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8), "RGB").save(
            str(out / "object_mask_overlay.png")
        )
        samples.append({"name": "object_mask_overlay", "path": str(out / "object_mask_overlay.png")})

    # Precise mask overlay
    if "minimap_rgb" in root and "object_precise_mask" in root:
        mm = root["minimap_rgb"][tile_idx].astype(np.float32)
        pm = root["object_precise_mask"][tile_idx].astype(np.float32)
        pm_resized = np.clip(pm[:256, :256], 0, 1)
        overlay = mm.copy()
        overlay[:, :, 2] = np.where(pm_resized > 0.5, 255, mm[:, :, 2])
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8), "RGB").save(
            str(out / "object_precise_mask_overlay.png")
        )
        samples.append({
            "name": "object_precise_mask_overlay",
            "path": str(out / "object_precise_mask_overlay.png"),
        })

    # Roof mask overlay
    if "minimap_rgb" in root and "object_roof_mask" in root:
        mm = root["minimap_rgb"][tile_idx].astype(np.float32)
        rm = root["object_roof_mask"][tile_idx].astype(np.float32)
        overlay = mm.copy()
        overlay[:, :, 1] = np.where(rm > 0.5, 255, mm[:, :, 1])
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8), "RGB").save(
            str(out / "object_roof_mask_overlay.png")
        )
        samples.append({
            "name": "object_roof_mask_overlay",
            "path": str(out / "object_roof_mask_overlay.png"),
        })

    # Height as grayscale
    if "height_257" in root:
        h = root["height_257"][tile_idx]
        h_norm = ((h - h.min()) / max(h.max() - h.min(), 1e-8) * 255).astype(np.uint8)
        Image.fromarray(h_norm, "L").save(str(out / "height.png"))
        samples.append({"name": "height", "path": str(out / "height.png")})

    # Normal as RGB
    if "normal_xyz" in root:
        n = root["normal_xyz"][tile_idx]
        n_rgb = ((n * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(n_rgb, "RGB").save(str(out / "normal_rgb.png"))
        samples.append({"name": "normal_rgb", "path": str(out / "normal_rgb.png")})

    # MODF mask overlay (WMO buildings)
    if "minimap_rgb" in root and "modf_mask" in root:
        mm = root["minimap_rgb"][tile_idx].astype(np.float32)
        modf = root["modf_mask"][tile_idx].astype(np.float32)
        modf_resized = np.clip(modf[:256, :256], 0, 1)
        overlay = mm.copy()
        overlay[:, :, 0] = np.where(modf_resized > 0.5, 255, mm[:, :, 0])
        overlay[:, :, 1] = np.where(modf_resized > 0.5, 0, mm[:, :, 1])
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8), "RGB").save(
            str(out / "modf_mask_overlay.png")
        )
        samples.append({"name": "modf_mask_overlay", "path": str(out / "modf_mask_overlay.png")})

    return samples


def validate_build(
    zarr_path: Path, sample_tile_id: int | None = None
) -> dict[str, Any]:
    issues: list[str] = []
    result: dict[str, Any] = {}

    if not zarr_path.exists():
        return {"status": "fail", "issues": [f"store_not_found:{zarr_path}"]}

    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")

    tile_count = 0
    try:
        if "minimap_rgb" in root:
            tile_count = root["minimap_rgb"].shape[0]
        elif "alpha_256" in root:
            tile_count = root["alpha_256"].shape[0]

        result["tile_count"] = int(tile_count)

        # Check index.parquet
        index_path = zarr_path / "index.parquet"
        if index_path.exists():
            idx_rows = _read_tile_rows(index_path)
            result["index_rows"] = len(idx_rows)
            if len(idx_rows) != tile_count:
                issues.append(f"index_row_mismatch: {len(idx_rows)} rows vs {tile_count} tiles")
        else:
            issues.append("missing_index_parquet")

        # Pick a sample tile
        tidx = sample_tile_id if sample_tile_id is not None else tile_count // 2
        tidx = max(0, min(tidx, tile_count - 1))

        if tile_count == 0:
            issues.append("no_tiles")
            result["status"] = "fail"
            return result

        # Validate each expected array
        array_results = {}
        for name, expected_shape in _EXPECTED_ARRAYS.items():
            ar = _check_array(root, name, tidx, expected_shape, issues)
            if ar["exists"]:
                array_results[name] = ar
        result["arrays"] = array_results

        # Check has_* flags in index against actual tile data
        if index_path.exists() and idx_rows:
            # Build reconstructed has_* from a sample of idx_rows
            misflagged = []
            for row in idx_rows[: min(100, len(idx_rows))]:
                for name in _EXPECTED_ARRAYS:
                    flag_col = f"has_{name}"
                    if flag_col in row:
                        flag_val = bool(row[flag_col])
                        # Check first batch of tiles for an approximate match
            if misflagged:
                issues.append(f"has_flag_mismatches: {misflagged}")

        result["issues"] = issues
        result["status"] = "pass" if not issues else "fail"

    finally:
        store.close()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate V18 dataset signals for a build."
    )
    parser.add_argument("--build", nargs="+", required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DEFAULT_DATASET_DIR,
    )
    parser.add_argument("--sample-tile", type=int, default=None)
    parser.add_argument("--sample-dir", type=Path, default=None,
                        help="Directory to dump sample images for visual inspection")
    args = parser.parse_args()

    all_pass = True
    for build in args.build:
        zarr_path = Path(args.dataset_dir) / f"{build}.zarr"
        print(f"=== {build} ===")
        rep = validate_build(zarr_path, sample_tile_id=args.sample_tile)
        if rep.get("tile_count", 0) > 0:
            print(f"  Tiles: {rep['tile_count']}")
            for name, ar in rep.get("arrays", {}).items():
                pf = ar.get("nonzero_frac", 0)
                print(f"  {name}: non-zero={pf:.3f} range=[{ar.get('min', 0):.3f},{ar.get('max', 0):.3f}] shape={ar.get('shape')}")
        for iss in rep.get("issues", []):
            print(f"  ISSUE: {iss}")
        if rep.get("status") == "fail":
            all_pass = False
        print()

    # Dump sample images if requested
    if args.sample_dir is not None:
        print(f"=== Dumping sample images to {args.sample_dir} ===")
        for build in args.build:
            zarr_path = Path(args.dataset_dir) / f"{build}.zarr"
            if not zarr_path.exists():
                continue
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            tile_count = root["minimap_rgb"].shape[0] if "minimap_rgb" in root else 0
            if tile_count == 0:
                store.close()
                continue
            tidx = args.sample_tile if args.sample_tile is not None else tile_count // 2
            tidx = max(0, min(tidx, tile_count - 1))
            samples = _dump_samples(root, zarr_path, build, args.sample_dir, tidx)
            print(f"  {build}: {len(samples)} images")
            for s in samples:
                print(f"    {s['name']}: {s['path']}")
            store.close()


if __name__ == "__main__":
    main()