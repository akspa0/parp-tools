#!/usr/bin/env python3
"""Refresh synthesis-owned minimap arrays in an existing v50 Zarr store.

This deliberately does not re-harvest terrain.  It renders the current C# synthetic-minimap
output for the exact tiles already present in a v50 store, validates every tile, copies the store
to a new path, and replaces only ``minimap_rgb`` and ``minimap_rgb_1024``.

The source store is never modified.  A fresh client-backed synthesis run is still an explicit,
user-owned operation because it invokes the C# harvester against a real client root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
from PIL import Image


def _hash_array_rows(array: zarr.Array) -> str:
    """Hash a Zarr array using the v50 content-identity contract without loading it all."""
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    for row_id in range(array.shape[0]):
        digest.update(np.ascontiguousarray(array[row_id]).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _tile_rows(store: Path) -> list[dict]:
    index_path = store / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"missing v50 index: {index_path}")
    rows = pq.read_table(str(index_path)).to_pylist()
    if not rows:
        raise ValueError(f"v50 store has no tile rows: {store}")
    for row in rows:
        if "tile_x" not in row or "tile_y" not in row:
            raise ValueError("v50 index must contain tile_x and tile_y")
    return rows


def _tile_list(rows: list[dict]) -> str:
    coordinates = {(int(row["tile_x"]), int(row["tile_y"])) for row in rows}
    return ";".join(f"{tile_x},{tile_y}" for tile_x, tile_y in sorted(coordinates, key=lambda pair: (pair[1], pair[0])))


def _run_synthesis(
    *,
    dll: Path,
    client_root: Path,
    map_name: str,
    build: str | None,
    era: str | None,
    resolution: int,
    tile_list: str,
    output_dir: Path,
) -> dict:
    command = [
        "dotnet",
        str(dll),
        "synthetic-minimap",
        "--client-root",
        str(client_root),
        "--map",
        map_name,
        "--resolution",
        str(resolution),
        "--per-tile",
        "--tile-list",
        tile_list,
        "--output-dir",
        str(output_dir),
    ]
    if build:
        command.extend(["--build", build])
    if era:
        command.extend(["--era", era])
    if resolution == 1024:
        command.append("--detail")

    print(f"Running fresh {resolution}x{resolution} synthesis for {map_name}...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"synthetic-minimap failed for {resolution}px with exit code {result.returncode}")

    manifest_path = output_dir / "synthesis-manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"synthesis completed without a manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_render_mode = "detail" if resolution == 1024 else "material_average"
    if manifest.get("RenderMode") != expected_render_mode:
        raise RuntimeError(
            f"{resolution}px synthesis reported RenderMode={manifest.get('RenderMode')!r}; "
            f"expected {expected_render_mode!r}"
        )
    return manifest


def _validate_tiles(*, rows: list[dict], output_dir: Path, map_name: str, resolution: int, manifest: dict) -> None:
    expected = {(int(row["tile_x"]), int(row["tile_y"])) for row in rows}
    results = {
        (int(tile["TileX"]), int(tile["TileY"])): tile
        for tile in manifest.get("Tiles", [])
    }
    missing_manifest = sorted(expected - results.keys(), key=lambda pair: (pair[1], pair[0]))
    if missing_manifest:
        raise RuntimeError(f"{resolution}px synthesis omitted tiles: {missing_manifest[:12]}")

    tile_dir = output_dir / "tiles"
    for tile_x, tile_y in sorted(expected, key=lambda pair: (pair[1], pair[0])):
        result = results[(tile_x, tile_y)]
        if result.get("Status") != "written":
            raise RuntimeError(
                f"{resolution}px tile {tile_x},{tile_y} status is {result.get('Status')!r}: "
                f"{result.get('Detail') or 'no detail'}"
            )
        path = tile_dir / f"{map_name}_{tile_x:02d}_{tile_y:02d}_synthesized.png"
        if not path.exists():
            raise RuntimeError(f"{resolution}px tile was reported written but is missing: {path}")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        expected_shape = (resolution, resolution, 3)
        if pixels.shape != expected_shape:
            raise RuntimeError(f"{path} has shape {pixels.shape}; expected {expected_shape}")
        if not np.any(pixels):
            raise RuntimeError(f"{path} is completely black")


def _patch_store(
    *,
    source_store: Path,
    output_store: Path,
    rows: list[dict],
    synthesis_dirs: dict[int, Path],
) -> dict[str, object]:
    if output_store.exists():
        raise FileExistsError(f"refreshed output already exists: {output_store}")
    shutil.copytree(source_store, output_store)

    group = zarr.open_group(str(output_store), mode="r+")
    stats: dict[str, object] = {}
    signal_metadata = list(group.attrs.get("signals", []))
    for resolution, synthesis_dir in synthesis_dirs.items():
        signal_name = "minimap_rgb" if resolution == 256 else "minimap_rgb_1024"
        if signal_name not in group:
            raise RuntimeError(f"source store is missing {signal_name}")
        target = group[signal_name]
        if target.shape[0] != len(rows) or tuple(target.shape[1:]) != (resolution, resolution, 3):
            raise RuntimeError(f"{signal_name} shape {target.shape} does not match the v50 index")

        # Historical v50 arrays use multi-row chunks. Repeated row writes then make Zarr's
        # Windows atomic chunk replacement contend with itself. The refreshed copy is disposable,
        # so use one complete tile per chunk and make each write independent.
        array_attrs = dict(target.attrs)
        target = group.create_array(
            signal_name,
            shape=(len(rows), resolution, resolution, 3),
            chunks=(1, resolution, resolution, 3),
            dtype=np.uint8,
            overwrite=True,
        )
        target.attrs.update(array_attrs)

        zero_rows = 0
        for row_id, row in enumerate(rows):
            tile_x = int(row["tile_x"])
            tile_y = int(row["tile_y"])
            path = synthesis_dir / "tiles" / f"{row.get('map', '') or ''}_{tile_x:02d}_{tile_y:02d}_synthesized.png"
            if not path.exists():
                # The map name is not present in every historical index, so locate the unique tile
                # by its suffix when the direct name is unavailable.
                candidates = list((synthesis_dir / "tiles").glob(f"*_{tile_x:02d}_{tile_y:02d}_synthesized.png"))
                if len(candidates) != 1:
                    raise RuntimeError(f"cannot locate synthesized tile for row {row_id}: {tile_x},{tile_y}")
                path = candidates[0]
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            target[row_id] = pixels
            zero_rows += int(not np.any(pixels))
        content_identity = _hash_array_rows(target)
        for signal in signal_metadata:
            if signal.get("name") == signal_name:
                signal["content_identity"] = content_identity
                signal["coverage_count"] = len(rows)
                break
        stats[signal_name] = {
            "rows": len(rows),
            "zero_rows_after_refresh": zero_rows,
            "content_identity": content_identity,
        }

    group.attrs["signals"] = signal_metadata
    group.attrs["synthetic_minimap_refresh"] = {
        "source_store": str(source_store),
        "signals": sorted(stats),
        "validated_non_black": True,
    }
    (output_store / "synthetic-refresh-report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path, help="Existing v50 Zarr store")
    parser.add_argument("--output-store", required=True, type=Path, help="New refreshed Zarr store")
    parser.add_argument("--client-root", required=True, type=Path, help="Exact WoW client root")
    parser.add_argument("--map", required=True, dest="map_name")
    parser.add_argument("--harvest-dll", required=True, type=Path)
    parser.add_argument("--build", default=None, help="Explicit build identity, e.g. 0.5.3.3368")
    parser.add_argument("--era", default=None, help="Optional synthetic era override")
    parser.add_argument("--synthesis-output", required=True, type=Path)
    args = parser.parse_args()

    if not args.store.is_dir():
        raise SystemExit(f"source v50 store not found: {args.store}")
    if not args.client_root.is_dir():
        raise SystemExit(f"client root not found: {args.client_root}")
    if not args.harvest_dll.is_file():
        raise SystemExit(f"harvest DLL not found: {args.harvest_dll}")
    if args.synthesis_output.exists():
        raise SystemExit(f"synthesis output already exists; choose a new path: {args.synthesis_output}")

    rows = _tile_rows(args.store)
    tile_list = _tile_list(rows)
    args.synthesis_output.mkdir(parents=True)
    manifests: dict[int, dict] = {}
    synthesis_dirs: dict[int, Path] = {}
    try:
        for resolution in (256, 1024):
            output_dir = args.synthesis_output / f"minimap_{resolution}"
            output_dir.mkdir()
            manifest = _run_synthesis(
                dll=args.harvest_dll,
                client_root=args.client_root,
                map_name=args.map_name,
                build=args.build,
                era=args.era,
                resolution=resolution,
                tile_list=tile_list,
                output_dir=output_dir,
            )
            _validate_tiles(
                rows=rows,
                output_dir=output_dir,
                map_name=args.map_name,
                resolution=resolution,
                manifest=manifest,
            )
            manifests[resolution] = manifest
            synthesis_dirs[resolution] = output_dir

        stats = _patch_store(
            source_store=args.store,
            output_store=args.output_store,
            rows=rows,
            synthesis_dirs=synthesis_dirs,
        )
    except Exception:
        if args.output_store.exists():
            shutil.rmtree(args.output_store)
        raise

    print(json.dumps({"output_store": str(args.output_store), "rows": len(rows), "signals": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
