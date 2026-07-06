"""FR-006: build the merged WDL prior (Stage 0) into a V24 Zarr store.

build  Reads a V18 store + a staged client, merges real WDL (C# reader) with
       synthetic WDL (C# terrain->WDL path) per tile, and writes the V24 store.
infer  Builds a synthetic-only prior NPZ from a single height NPZ (no client).

All game-data reads go through the C# shim (WowViewer.Tool.WdlRead); Python
only merges and stores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import merged_wdl_prior, store, synth_wdl, wdl_reader  # noqa: E402

SYNTH_BATCH = 256


def _select_rows(
    index: dict[str, list],
    maps: list[str] | None,
    limit: int | None,
    min_height_std: float | None = None,
) -> list[int]:
    rows = list(range(len(index["tile_id"])))
    if maps:
        wanted = {m.lower() for m in maps}
        rows = [r for r in rows if index["map"][r].lower() in wanted]
    if min_height_std is not None and "height_std" in index:
        rows = [r for r in rows if index["height_std"][r] >= min_height_std]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _build(args: argparse.Namespace) -> int:
    v18_path = Path(args.v18_store)
    staged_client = Path(args.staged_client)
    output = Path(args.output)

    v18 = zarr.open_group(str(v18_path), mode="r")
    index = store.read_index(v18_path)
    rows = _select_rows(index, args.maps, args.limit, args.min_height_std)
    if not rows:
        print("No tiles selected.", file=sys.stderr)
        return 1

    maps_in_selection = sorted({index["map"][r] for r in rows})
    print(f"build: {len(rows)} tiles across {len(maps_in_selection)} map(s): {maps_in_selection}")

    real_by_map: dict[str, dict | None] = {}
    for map_name in maps_in_selection:
        started = time.time()
        real_by_map[map_name] = wdl_reader.read_wdl_map_tiles(staged_client, map_name)
        tiles = real_by_map[map_name]
        status = f"{len(tiles)} MARE tiles" if tiles is not None else "no WDL"
        print(f"  real WDL [{map_name}]: {status} ({time.time() - started:.1f}s)")

    group = store.create_v24_store(
        output,
        len(rows),
        {
            "v18_store_path": str(v18_path),
            "staged_client_path": str(staged_client),
            "disagree_threshold": args.disagree_threshold,
        },
    )

    height_arr = v18["height_257"]
    liquid_arr = v18["liquid_mask"] if "liquid_mask" in v18 else None
    has_height = index.get("has_height_257")

    for start in range(0, len(rows), SYNTH_BATCH):
        batch_rows = rows[start : start + SYNTH_BATCH]
        heights = np.stack([height_arr[r] for r in batch_rows]).astype(np.float32)
        liquids = (
            np.stack([liquid_arr[r] for r in batch_rows]).astype(np.float32)
            if liquid_arr is not None
            else None
        )
        synth_outer, synth_inner = synth_wdl.build_synth_wdl_batch(heights, liquids)

        for j, r in enumerate(batch_rows):
            out_row = start + j
            audit_empty = bool(has_height is not None and not has_height[r])
            audit_empty = audit_empty or float(np.abs(heights[j]).max()) == 0.0

            map_name = index["map"][r]
            tile_key = (int(index["tile_x"][r]), int(index["tile_y"][r]))
            real_tiles = real_by_map.get(map_name)
            real = real_tiles.get(tile_key) if real_tiles else None

            merged = merged_wdl_prior.build_merged_wdl_prior(
                heights[j],
                real,
                real_wdl_available=real is not None,
                synth_wdl=(synth_outer[j], synth_inner[j]),
                audit_empty=audit_empty,
                disagree_threshold=args.disagree_threshold,
            )
            group["wdl_prior_outer"][out_row] = merged.outer
            group["wdl_prior_inner"][out_row] = merged.inner
            group["wdl_prior_source_outer"][out_row] = merged.source_outer
            group["wdl_prior_source_inner"][out_row] = merged.source_inner
            group["wdl_prior_confidence_outer"][out_row] = merged.confidence_outer
            group["wdl_prior_confidence_inner"][out_row] = merged.confidence_inner
            group["wdl_prior_disagree_ratio"][out_row] = merged.disagree_ratio
            group["wdl_prior_audit_empty"][out_row] = audit_empty
            group["wdl_prior_real_available"][out_row] = real is not None

        print(f"  merged {min(start + SYNTH_BATCH, len(rows))}/{len(rows)} tiles")

    store.write_index(
        output,
        {
            "tile_id": [index["tile_id"][r] for r in rows],
            "build": [index["build"][r] for r in rows],
            "map": [index["map"][r] for r in rows],
            "tile_x": [index["tile_x"][r] for r in rows],
            "tile_y": [index["tile_y"][r] for r in rows],
            "v18_row": rows,
        },
    )

    stats = store.coverage_stats(group)
    group.attrs.update({f"coverage_{k}": v for k, v in stats.items()})
    print("coverage:", json.dumps(stats, indent=2))
    ok = stats["real_plus_synthetic_ratio_of_non_empty"] >= 0.95
    print(f"SC-001 real+synthetic >= 0.95 of non-empty cells: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _infer(args: argparse.Namespace) -> int:
    output = Path(args.output)
    height = None
    if args.height:
        with np.load(args.height) as data:
            key = "height_257" if "height_257" in data else "height"
            height = np.asarray(data[key], dtype=np.float32)

    if height is None or not np.abs(height).max():
        mean = float(height.mean()) if height is not None else 0.0
        np.savez(
            output,
            outer=np.full((17, 17), mean, np.float32),
            inner=np.full((16, 16), mean, np.float32),
            source_outer=np.full((17, 17), 2, np.uint8),
            source_inner=np.full((16, 16), 2, np.uint8),
            prior_unavailable=np.array(True),
        )
        print("prior_unavailable=True (no usable height input)")
        return 0

    liquid = None
    if args.liquid:
        with np.load(args.liquid) as data:
            key = "liquid_mask" if "liquid_mask" in data else "liquid"
            liquid = np.asarray(data[key], dtype=np.float32)

    outer, inner = synth_wdl.build_synth_wdl(height, liquid)
    np.savez(
        output,
        outer=outer,
        inner=inner,
        source_outer=np.full((17, 17), 1, np.uint8),
        source_inner=np.full((16, 16), 1, np.uint8),
        prior_unavailable=np.array(False),
    )
    print(f"wrote synthetic prior to {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build the V24 store from V18 + staged client")
    build.add_argument("--v18-store", required=True)
    build.add_argument("--staged-client", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--maps", nargs="*", default=None)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--disagree-threshold", type=float, default=1.0)
    build.add_argument("--min-height-std", type=float, default=None,
                       help="skip tiles flatter than this height_std (needs index column)")

    infer = sub.add_parser("infer", help="synthetic-only prior for one tile NPZ")
    infer.add_argument("--minimap", default=None, help="accepted for interface parity; unused")
    infer.add_argument("--height", default=None)
    infer.add_argument("--liquid", default=None)
    infer.add_argument("--output", required=True)

    args = parser.parse_args()
    return _build(args) if args.command == "build" else _infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
