"""Canonical v50 entry point for the three-tier tile signal classifier (Spec 132 US1).

Reads a V50 Zarr store (or NPZ shard directory), classifies every tile as strong/normal/weak with
published criteria, and emits a CSV + JSON of per-tile tiers with evidence, plus a summary.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v50_tile_classify.py \\
        --store ../output/archaeology/2_0_0_5610/store/Expansion01.zarr \\
        --output ../output/archaeology/2_0_0_5610/classify
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from harvester.v50 import classify as tile_classify
from harvester.v50.classify import SignalTier, compute_signal_tier

CLASSIFY_SCHEMA = "v50-tile-classify-v1"


def _tier_rows_from_store(store: Path) -> list[dict]:
    """Classify every tile in a V50 Zarr store into a three-tier row."""
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    index = pq.read_table(store / "index.parquet").to_pylist()
    present = set(group.array_keys())

    rows: list[dict] = []
    for row_id, meta in enumerate(index):
        map_name = str(meta.get("map", "unknown"))
        tile_x = int(meta.get("tile_x", -1))
        tile_y = int(meta.get("tile_y", -1))
        height = np.asarray(group["height_257"][row_id], dtype=np.float32)
        height_range = float(np.max(height) - np.min(height)) if height.size else 0.0
        levels = int(np.unique(height).size) if height.size else 0

        # Optional alpha-texture correlation slot. Phase 3 fills this; until then it stays None and
        # the height/levels criteria decide the tier (FR-007: never fabricate a score).
        correlation: float | None = None

        result = compute_signal_tier(
            height_range=height_range, surviving_levels=levels,
            alpha_texture_correlation=correlation,
        )
        rows.append({
            "tile_key": f"{map_name}_{tile_x:02d}_{tile_y:02d}",
            "map": map_name,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "row_id": row_id,
            "signal_class": result.tier.value,
            "height_range": result.height_range,
            "surviving_height_levels": result.surviving_levels,
            "alpha_texture_correlation": result.alpha_texture_correlation,
            "classification_evidence": result.evidence,
        })
    return rows


def _tier_rows_from_npz(npz_dir: Path) -> list[dict]:
    """Classify every tile in an NPZ shard directory (no V50 store required)."""
    rows: list[dict] = []
    for npz_path in sorted(npz_dir.glob("*.npz")):
        try:
            data = dict(np.load(npz_path))
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: {npz_path.name}: {e}", flush=True)
            continue
        stem = npz_path.stem.replace("_harvest", "")
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
            tile_x, tile_y = int(parts[-2]), int(parts[-1])
            map_name = "_".join(parts[:-2])
        else:
            map_name, tile_x, tile_y = "unknown", 0, 0
        height = data.get("height_257")
        height_range = float(np.max(height) - np.min(height)) if height is not None and height.size else 0.0
        levels = int(np.unique(height).size) if height is not None and height.size else 0
        result = compute_signal_tier(height_range=height_range, surviving_levels=levels)
        rows.append({
            "tile_key": f"{map_name}_{tile_x:02d}_{tile_y:02d}",
            "map": map_name, "tile_x": tile_x, "tile_y": tile_y, "row_id": -1,
            "signal_class": result.tier.value, "height_range": result.height_range,
            "surviving_height_levels": result.surviving_levels,
            "alpha_texture_correlation": result.alpha_texture_correlation,
            "classification_evidence": result.evidence,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    """Per-tier counts plus the tile-key lists, mirroring inventory.summarize style."""
    tiers: dict[str, int] = {}
    for row in rows:
        tiers[row["signal_class"]] = tiers.get(row["signal_class"], 0) + 1
    return {
        "tile_count": len(rows),
        "by_signal_class": dict(sorted(tiers.items())),
        "strong": sorted(r["tile_key"] for r in rows if r["signal_class"] == SignalTier.STRONG.value),
        "normal": sorted(r["tile_key"] for r in rows if r["signal_class"] == SignalTier.NORMAL.value),
        "weak": sorted(r["tile_key"] for r in rows if r["signal_class"] == SignalTier.WEAK.value),
    }


CSV_COLUMNS = (
    "tile_key", "map", "tile_x", "tile_y", "row_id", "signal_class",
    "height_range", "surviving_height_levels", "alpha_texture_correlation",
    "classification_evidence",
)


def write_output(rows: list[dict], summaries: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "classify.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "classify.json").write_text(
        json.dumps({"schema": CLASSIFY_SCHEMA, "tiles": rows}, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps({"schema": CLASSIFY_SCHEMA, **summaries}, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-tier tile signal classification (strong/normal/weak)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--store", type=Path, help="a V50 Zarr store (repeatable)")
    source.add_argument("--npz-dir", type=Path, help="a directory of NPZ shards")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    if args.store is not None:
        rows = _tier_rows_from_store(args.store)
    else:
        rows = _tier_rows_from_npz(args.npz_dir)

    if not rows:
        raise SystemExit("no tiles classified")
    summaries = summarize(rows)
    write_output(rows, summaries, args.output)
    print(f"{summaries['tile_count']} tiles  tiers={summaries['by_signal_class']}  -> {args.output}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())