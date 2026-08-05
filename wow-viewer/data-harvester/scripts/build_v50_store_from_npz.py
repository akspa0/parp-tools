#!/usr/bin/env python3
"""Build a V50-format Zarr store from NPZ shards, then run the full archaeology pipeline.

The V50 store format is what v50_tile_inventory.py, v50_synthesize_weak_tiles.py,
and v50_tile_composite.py expect. This script creates one from NPZ shards (the output
of the C# harvest tool), then hands it to the archaeology scripts.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/build_v50_store_from_npz.py \\
        --npz-dir ../output/archaeology/2_0_0_5610/npz/Expansion01 \\
        --store ../output/archaeology/2_0_0_5610/store/Expansion01.zarr \\
        --output ../output/archaeology/2_0_0_5610/archaeo \\
        --map Expansion01 \\
        --near-zero-band inf
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from pathlib import Path


def build_v50_store(npz_dir: Path, store_path: Path, map_name: str) -> int:
    """Read NPZ shards, build a V50-format Zarr store with index.parquet."""
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        print(f"ERROR: no NPZ files found in {npz_dir}")
        return 1

    print(f"Found {len(npz_files)} NPZ shards", flush=True)

    # Load all NPZ data
    all_tiles = []
    for npz_path in npz_files:
        try:
            data = dict(np.load(npz_path))
        except Exception as e:
            print(f"  WARNING: {npz_path.name}: {e}", flush=True)
            continue

        # Parse tile coords from filename: e.g. Expansion01_45_32_harvest.npz
        stem = npz_path.stem.replace("_harvest", "")
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
            tile_x = int(parts[-2])
            tile_y = int(parts[-1])
        else:
            print(f"  WARNING: cannot parse coords from {npz_path.name}", flush=True)
            continue

        data["tile_x"] = tile_x
        data["tile_y"] = tile_y
        data["npz_path"] = str(npz_path)
        all_tiles.append(data)

    if not all_tiles:
        print("ERROR: no tiles loaded")
        return 1

    # Determine all signal names present across all tiles
    all_signal_names = set()
    for tile in all_tiles:
        for key in tile:
            if isinstance(tile[key], np.ndarray) and key not in ("tile_x", "tile_y", "npz_path"):
                all_signal_names.add(key)

    print(f"Signals detected: {sorted(all_signal_names)}", flush=True)
    print(f"Tiles: {len(all_tiles)}", flush=True)

    # Build the V50 store
    if store_path.exists():
        import shutil
        shutil.rmtree(store_path)

    root = zarr.open_group(str(store_path), mode="w")

    # Build index.parquet
    index_rows = []
    for tile_id, tile in enumerate(all_tiles):
        index_rows.append({
            "map": map_name,
            "tile_x": int(tile["tile_x"]),
            "tile_y": int(tile["tile_y"]),
            "tile_id": tile_id,
        })

    index_table = pa.Table.from_pylist(index_rows)
    pq.write_table(index_table, str(store_path / "index.parquet"))
    print(f"  Wrote index.parquet with {len(index_rows)} rows", flush=True)

    # Write each signal as a 3D array [tile_id, ...shape]
    # Skip signals with shape mismatches (e.g. hole_mask_16 missing from some tiles)
    for signal_name in sorted(all_signal_names):
        arrays = []
        missing = 0
        for tile in all_tiles:
            arr = tile.get(signal_name)
            if arr is not None:
                arrays.append(np.asarray(arr))
            else:
                missing += 1

        if missing > 0:
            print(f"  SKIP {signal_name}: missing from {missing}/{len(all_tiles)} tiles", flush=True)
            continue

        # Determine dtype and shape
        dtypes = {a.dtype for a in arrays}
        dtype = np.float32 if np.float32 in dtypes else arrays[0].dtype
        shapes = {a.shape for a in arrays}
        if len(shapes) > 1:
            print(f"  SKIP {signal_name}: inconsistent shapes: {shapes}", flush=True)
            continue

        shape = list(shapes)[0]
        arrays = [a.astype(dtype) for a in arrays]
        stacked = np.stack(arrays, axis=0)
        root.create_dataset(signal_name, data=stacked, shape=stacked.shape, dtype=dtype, overwrite=True)
        print(f"  Wrote {signal_name}: shape={stacked.shape} dtype={dtype}", flush=True)

    # Write manifest
    manifest = {
        "store_id": f"npz_{map_name}",
        "map": map_name,
        "tile_count": len(all_tiles),
        "signals": sorted(all_signal_names),
    }
    with open(store_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nV50 store ready: {store_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build V50 Zarr store from NPZ shards, then run archaeology"
    )
    parser.add_argument("--npz-dir", required=True, type=Path, help="Directory of NPZ shards")
    parser.add_argument("--store", required=True, type=Path, help="Output V50 Zarr store path")
    parser.add_argument("--output", required=True, type=Path, help="Archaeology output directory")
    parser.add_argument("--map", default="unknown", help="Map name for index.parquet")
    parser.add_argument("--near-zero-band", default=None,
                        help="Pass 'inf' for non-alpha clients")
    parser.add_argument("--skip-archaeo", action="store_true",
                        help="Only build the store, skip archaeology scripts")
    args = parser.parse_args()

    # Step 1: Build V50 store
    rc = build_v50_store(args.npz_dir, args.store, args.map)
    if rc != 0:
        return rc

    if args.skip_archaeo:
        return 0

    # Step 2: Run archaeology scripts
    import subprocess
    import tempfile

    _SCRIPTS = Path(__file__).resolve().parent
    _SRC = _SCRIPTS.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    def _run(script: str, script_args: list[str]) -> None:
        cmd = [sys.executable, str(_SCRIPTS / script), *script_args]
        print(f"\n>>> {script} {' '.join(script_args[:4])} ...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "ZarrUserWarning" in line or "warnings.warn" in line:
                continue
            print("    " + line, flush=True)
        if result.returncode != 0:
            print(result.stderr[-2000:], flush=True)
            raise SystemExit(f"{script} failed with exit {result.returncode}")

    images = args.output / "images"
    data = args.output / "data"
    images.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    store_args = ["--store", str(args.store)]
    band = ["--near-zero-band", args.near_zero_band] if args.near_zero_band else []

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        # Tile inventory
        inv = work / "inv"
        _run("v50_tile_inventory.py", [*store_args, "--output", str(inv), *band])

        # Synthesis sheets
        synth = work / "synth"
        _run("v50_synthesize_weak_tiles.py", ["--inventory", str(inv), *store_args, "--output", str(synth)])

        # Composites - renders all modes: absolute, autostretch, restored, liquid, textured
        comp = work / "comp"
        _run("v50_tile_composite.py", ["--inventory", str(inv), *store_args, "--output", str(comp), "--cell", "96"])

        # Signal mismatch
        _run("v50_tile_mismatch.py", [*store_args, "--output", str(data / "signal-mismatch.json")])

        # Three-tier brush-signature classification (Spec 132 US1)
        classify_out = work / "classify"
        _run("v50_tile_classify.py", [*store_args, "--output", str(classify_out)])

        # Collect outputs
        from v50_archaeology import _collect
        print("\nCollecting outputs...", flush=True)
        moved = 0
        moved += _collect(inv, data, "*.csv", lambda p: p.name)
        moved += _collect(inv, data, "*.json", lambda p: p.name)
        moved += _collect(classify_out, data, "*.csv", lambda p: p.name)
        moved += _collect(classify_out, data, "*.json", lambda p: p.name)
        moved += _collect(synth, images, "*.png", lambda p: p.name)
        moved += _collect(comp, images, "*.png", lambda p: p.name)
        print(f"  Collected {moved} files", flush=True)

    # Write README
    readme = f"""ARCHAEOLOGY OUTPUT
Build: {args.store.parent.name}
Map: {args.map}

images/
  <map>-absolute.png      every tile on the map's global height scale
  <map>-autostretch.png   every tile normalized to itself
  <map>-restored.png      compressed tiles scaled toward neighbours
  <map>-liquid.png        terrain flooded to liquid surface
  <map>-degenerate.png    mosaic of weak/blank tiles
  tile-<map>_<x>_<y>.png  per-tile 4-panel sheets

data/
  tiles.csv               per-tile inventory
  tiles.json              full tile data
  summary.json            corpus counts
  signal-mismatch.json    near-universal signal rules
  classify.csv            three-tier signal classification (strong/normal/weak)
  classify.json           full classification with evidence
  summary.json            classification tier counts
"""
    (args.output / "README.txt").write_text(readme)

    print(f"\nArchaeology complete. Results in: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())