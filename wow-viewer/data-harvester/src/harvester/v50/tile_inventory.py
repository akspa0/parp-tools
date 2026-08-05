"""Per-tile signal inventory for the 0.5.3 corpus: what every tile HAS, not what survived curation.

Curation partitions; it never silently drops. This module is the record of that partition at the
tile level: every tile in every per-map store gets a row naming which signals it carries, whether
its terrain is weak-signal / blank, and — for the weak ones — the adjacent tiles' real height
ranges, which is the reference the viewer's ``WeakSignalDetector.EstimateFactorFromRanges`` wants
and currently never receives (it is defined but unwired; the viewer falls back to the constant
``ClassicCompressionFactor``).

Nothing here filters. A blank "white plate" tile and a fully authored tile are both rows, tagged
differently, so a later consumer can keep one white-plate example and exclude the rest by query
rather than by having the data thrown away upstream.

The weak-signal thresholds are ported verbatim from
``src/core/WowViewer.Core.Runtime/World/Terrain/WeakSignalDetector.cs`` so the inventory and the
viewer classify a tile identically.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from harvester.v50.classify import compute_signal_tier

INVENTORY_SCHEMA = "v50-tile-inventory-v1"

# --- WeakSignalDetector.cs parity -------------------------------------------------------------
WEAK_MIN_RANGE = 0.001  # hasTerrainData
WEAK_MAX_RANGE = 15.0  # isCompressed
WEAK_NEAR_ZERO_BAND = 50.0  # nearZeroBand
CLASSIC_COMPRESSION_FACTOR = 33.334
MAX_FACTOR = 512.0
CHUNKS_PER_DIM = 16
CHUNK_STRIDE = 16
CHUNK_SPAN = 17

# Straight-up MCNR normals mean the chunk carried no real normal data.
FLAT_NORMAL_Z = 0.999

# Both minimap arrays are checked. `minimap_rgb` is the harvest's SYNTHETIC render;
# `minimap_rgb_authored` is the real client art. Checking only the former reported 220 Kalimdor
# tiles as having "no visual record" when every one of them has authored art — the synthetic
# render just was not produced for them.
SIGNAL_FLAGS = (
    "height_257",
    "minimap_rgb",
    "minimap_rgb_authored",
    "normal_xyz",
    "alpha_256",
    "mcly_layer_mask",
    "mcly_texture_ids",
    "shadow_mask",
    "liquid_mask",
    "mcnk_flags_16",
)

NEIGHBOURS = {"W": (-1, 0), "E": (1, 0), "N": (0, -1), "S": (0, 1)}


def classify_severity(height_range: float, is_candidate: bool) -> str:
    """``WeakSignalDetector.ClassifySeverity`` verbatim."""
    if not is_candidate:
        return "none"
    if height_range < 0.5:
        return "high"
    if height_range < 2.0:
        return "medium"
    return "low"


def is_compressed_range(min_height: float, max_height: float) -> bool:
    """The band-free half of the weak test: is this tile's relief compressed at all?

    Recorded separately and ALWAYS, because the band below is era-calibrated and can silently
    suppress every detection on a client it was not tuned for. Measured on 4.0.0.11927 Kalimdor:
    71 tiles are compressed by this test and 0 survive the band, because that map sits at
    |Z| ~440-520 while the band assumes alpha terrain near sea level. A null result from
    ``is_weak_signal`` alone is therefore not evidence of absence — compare the two.
    """
    height_range = max_height - min_height
    return WEAK_MIN_RANGE < height_range < WEAK_MAX_RANGE


def is_weak_signal(
    min_height: float, max_height: float, *, near_zero_band: float = WEAK_NEAR_ZERO_BAND
) -> bool:
    """``WeakSignalDetector.Analyze``'s candidate test.

    ``near_zero_band`` defaults to the viewer's alpha-calibrated 50.0 so 0.5.3 results stay
    reproducible; pass a larger value (or ``float('inf')`` to disable it) for a client whose
    terrain does not sit near sea level.
    """
    return (
        is_compressed_range(min_height, max_height)
        and abs(min_height) < near_zero_band
        and abs(max_height) < near_zero_band
    )


def estimate_factor_from_ranges(
    observed_min: float, observed_max: float, coarse_min: float, coarse_max: float
) -> float:
    """``WeakSignalDetector.EstimateFactorFromRanges`` verbatim.

    ``coarse_*`` is the reference range the weak tile should be scaled up toward. The viewer has
    only ever fed this a WDL-style coarse range; feeding it the ADJACENT tiles' real ranges is the
    smarter reference, which is why this inventory computes those.
    """
    epsilon = 0.001
    observed_range = max(observed_max - observed_min, 0.0)
    coarse_range = max(coarse_max - coarse_min, 0.0)
    if observed_range <= epsilon or coarse_range <= epsilon:
        return 1.0

    raw = 1.0
    if coarse_range > observed_range * 1.15:
        raw = max(raw, coarse_range / observed_range)

    observed_below = max(0.0, -observed_min)
    coarse_below = max(0.0, -coarse_min)
    if observed_below > epsilon and coarse_below > observed_below * 1.15:
        raw = max(raw, coarse_below / observed_below)

    observed_above = max(0.0, observed_max)
    coarse_above = max(0.0, coarse_max)
    if observed_above > epsilon and coarse_above > observed_above * 1.15:
        raw = max(raw, coarse_above / observed_above)

    return float(min(max(raw, 1.0), MAX_FACTOR))


def analyze_chunks(
    height_257: np.ndarray, *, near_zero_band: float = WEAK_NEAR_ZERO_BAND
) -> dict[str, Any]:
    """``WeakSignalDetector.AnalyzeChunks`` over the 16x16 chunk grid.

    The C# samples the MCNR quincunx (9 outer + 8 inner per row); those samples are exactly the
    17x17 block's vertices, so a block min/max is identical and vectorizes.
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(height_257, dtype=np.float32), (CHUNK_SPAN, CHUNK_SPAN)
    )[::CHUNK_STRIDE, ::CHUNK_STRIDE]
    lo = windows.min(axis=(2, 3))
    hi = windows.max(axis=(2, 3))
    ranges = hi - lo
    weak = (
        (ranges > WEAK_MIN_RANGE)
        & (ranges < WEAK_MAX_RANGE)
        & (np.abs(lo) < near_zero_band)
        & (np.abs(hi) < near_zero_band)
    )
    return {
        "weak_chunk_count": int(weak.sum()),
        "blank_chunk_count": int(((ranges <= WEAK_MIN_RANGE) & ~weak).sum()),
        "chunk_range_p50": float(np.percentile(ranges, 50)),
        "chunk_range_max": float(ranges.max()),
    }


def surviving_height_levels(height_257: np.ndarray) -> int:
    """Count distinct height values — how much vertical information a tile still carries.

    Range alone cannot tell a squeezed landscape from a squeezed nothing. Two tiles measured on the
    0.5.3 corpus both have a range near 0.5 world units: one holds 2 distinct values (a one-bit
    plateau) and another holds 27,132 (fully detailed terrain whose AMPLITUDE was compressed but
    whose SHAPE is intact). Only the level count separates them, and only the second is real
    terrain worth recovering.
    """
    return int(np.unique(np.asarray(height_257, dtype=np.float32)).size)


def classify_information(levels: int) -> str:
    """Bucket a tile by surviving vertical information, independent of its amplitude."""
    if levels <= 1:
        return "bit_exact_flat"
    if levels <= 8:
        return "trace"
    if levels <= 64:
        return "coarse_terrain"
    return "rich_terrain"


def mcnr_tilted_fraction(normal_xyz: np.ndarray, mcnr_mask: np.ndarray) -> float:
    """Fraction of REAL MCNR vertices whose normal is not straight-up.

    Masking matters: unwritten vertices hold (0, 0, 0), whose |z| is also != 1, so an unmasked test
    measures mask holes instead of terrain tilt and reports ~100% on flat tiles.
    """
    mask = np.asarray(mcnr_mask).astype(bool)
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(np.asarray(normal_xyz, dtype=np.float32)[mask][:, 2]) < FLAT_NORMAL_Z))


def classify_tile(*, has_height: bool, has_minimap: bool, height_range: float, weak: bool) -> str:
    """One label per tile. Orthogonal to the raw presence flags, which stay queryable on their own."""
    if not has_height or height_range <= WEAK_MIN_RANGE:
        # No relief at all: a "white plate". Keep a sample, exclude the bulk — by query, not by
        # deletion.
        return "white_plate_with_minimap" if has_minimap else "white_plate"
    if weak:
        return "weak_signal_with_minimap" if has_minimap else "weak_signal"
    if not has_minimap:
        return "terrain_no_minimap"
    return "usable"


def inventory_store(
    store_path: Path, *, near_zero_band: float = WEAK_NEAR_ZERO_BAND
) -> list[dict[str, Any]]:
    """Build the per-tile inventory rows for one per-map v50 store."""
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store_path), mode="r")
    index = pq.read_table(store_path / "index.parquet").to_pylist()
    present = set(group.array_keys())

    rows: list[dict[str, Any]] = []
    for row_id, meta in enumerate(index):
        map_name = str(meta.get("map", "unknown"))
        tile_x = int(meta.get("tile_x", -1))
        tile_y = int(meta.get("tile_y", -1))

        height = np.asarray(group["height_257"][row_id], dtype=np.float32)
        has_height = bool(height.any())
        lo, hi = float(height.min()), float(height.max())
        height_range = hi - lo
        weak = is_weak_signal(lo, hi, near_zero_band=near_zero_band)
        compressed = is_compressed_range(lo, hi)
        levels = surviving_height_levels(height)

        flags = {
            f"has_{name}": bool(np.asarray(group[name][row_id]).any())
            for name in SIGNAL_FLAGS
            if name in present
        }
        record: dict[str, Any] = {
            "tile_key": f"{map_name}_{tile_x:02d}_{tile_y:02d}",
            "map": map_name,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "row_id": row_id,
            "build": str(meta.get("build", "")),
            **flags,
            "height_min": lo,
            "height_max": hi,
            "height_range": height_range,
            "is_weak_signal": weak,
            "is_compressed_range": compressed,
            "near_zero_band": near_zero_band,
            "weak_severity": classify_severity(height_range, weak),
            "surviving_height_levels": levels,
            "information_class": classify_information(levels),
            **analyze_chunks(height, near_zero_band=near_zero_band),
        }
        if "normal_xyz" in present and "mcnr_mask_257" in present:
            record["mcnr_tilted_fraction"] = mcnr_tilted_fraction(
                group["normal_xyz"][row_id], group["mcnr_mask_257"][row_id]
            )
        record["classification"] = classify_tile(
            has_height=has_height,
            # ANY minimap counts as a visual record, authored or synthetic.
            has_minimap=bool(flags.get("has_minimap_rgb", False)
                             or flags.get("has_minimap_rgb_authored", False)),
            height_range=height_range,
            weak=weak,
        )
        # Three-tier brush-signature classification (Spec 132 US1). The alpha<->height correlation
        # slot is Phase 3; until then it is None and the height/levels criteria decide the tier
        # (FR-007: never fabricate a score for a tile with no alpha data to measure).
        tier = compute_signal_tier(
            height_range=height_range,
            surviving_levels=levels,
            alpha_texture_correlation=None,
        )
        record["signal_class"] = tier.tier.value
        record["signal_class_evidence"] = tier.evidence
        rows.append(record)

    _attach_neighbour_reference(rows)
    return rows


def _attach_neighbour_reference(rows: list[dict[str, Any]]) -> None:
    """For every weak/blank tile, the adjacent NON-weak tiles' real height range + the factor it implies.

    This is the input the viewer's amplifier is missing: it currently scales a weak tile by a
    constant era factor with no idea what its neighbours actually stand at.
    """
    by_tile = {(r["tile_x"], r["tile_y"]): r for r in rows}
    for record in rows:
        refs = {}
        mins: list[float] = []
        maxs: list[float] = []
        for name, (dx, dy) in NEIGHBOURS.items():
            neighbour = by_tile.get((record["tile_x"] + dx, record["tile_y"] + dy))
            if neighbour is None:
                refs[name] = None
                continue
            refs[name] = neighbour["tile_key"]
            # Only a STRONG neighbour is a valid scale reference; amplifying toward another weak
            # tile just propagates the compression.
            if not neighbour["is_weak_signal"] and neighbour["height_range"] > WEAK_MAX_RANGE:
                mins.append(neighbour["height_min"])
                maxs.append(neighbour["height_max"])
        record["neighbours"] = refs
        record["strong_neighbour_count"] = len(mins)
        if mins:
            record["neighbour_height_min"] = min(mins)
            record["neighbour_height_max"] = max(maxs)
            record["suggested_amplification_factor"] = estimate_factor_from_ranges(
                record["height_min"], record["height_max"], min(mins), max(maxs)
            )
        else:
            record["neighbour_height_min"] = None
            record["neighbour_height_max"] = None
            record["suggested_amplification_factor"] = None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per classification and per signal, plus the tile-key lists that matter downstream."""
    def keys(predicate) -> list[str]:
        return sorted(r["tile_key"] for r in rows if predicate(r))

    classes: dict[str, int] = {}
    for record in rows:
        classes[record["classification"]] = classes.get(record["classification"], 0) + 1
    return {
        "tile_count": len(rows),
        "by_classification": dict(sorted(classes.items())),
        "no_minimap": keys(lambda r: not (r.get("has_minimap_rgb", False)
                                          or r.get("has_minimap_rgb_authored", False))),
        "no_synthetic_minimap": keys(lambda r: not r.get("has_minimap_rgb", False)),
        "no_authored_minimap": keys(lambda r: not r.get("has_minimap_rgb_authored", False)),
        "no_height": keys(lambda r: not r.get("has_height_257", False)),
        "minimap_without_height": keys(
            lambda r: (r.get("has_minimap_rgb", False) or r.get("has_minimap_rgb_authored", False))
            and not r.get("has_height_257", False)
        ),
        "weak_signal": keys(lambda r: r["is_weak_signal"]),
        # Band-free. If this is large while weak_signal is empty, the band is era-wrong, not the map.
        "compressed_range": keys(lambda r: r.get("is_compressed_range", False)),
        "weak_signal_with_strong_neighbours": keys(
            lambda r: r["is_weak_signal"] and r["strong_neighbour_count"] > 0
        ),
        "white_plates": keys(lambda r: r["classification"].startswith("white_plate")),
        # Degenerate by amplitude but INTACT in shape: the population where recovery is a real
        # prospect rather than an aspiration.
        "compressed_rich_terrain": keys(
            lambda r: r.get("information_class") == "rich_terrain"
            and r["classification"] not in ("usable", "terrain_no_minimap")
        ),
        "by_information_class": {
            name: sum(1 for r in rows if r.get("information_class") == name)
            for name in ("bit_exact_flat", "trace", "coarse_terrain", "rich_terrain")
        },
        # Three-tier brush-signature classification (Spec 132 US1).
        "by_signal_class": {
            name: sum(1 for r in rows if r.get("signal_class") == name)
            for name in ("strong", "normal", "weak", "na")
        },
    }


CSV_COLUMNS = (
    "tile_key", "map", "tile_x", "tile_y", "row_id", "classification",
    "has_height_257", "has_minimap_rgb", "has_minimap_rgb_authored", "has_alpha_256",
    "has_mcly_layer_mask",
    "has_mcly_texture_ids", "has_shadow_mask", "has_liquid_mask", "has_mcnk_flags_16",
    "height_min", "height_max", "height_range", "is_weak_signal", "is_compressed_range",
    "weak_severity",
    "surviving_height_levels", "information_class",
    "weak_chunk_count", "blank_chunk_count", "chunk_range_p50", "chunk_range_max",
    "mcnr_tilted_fraction", "strong_neighbour_count", "neighbour_height_min",
    "neighbour_height_max", "suggested_amplification_factor",
    "signal_class", "signal_class_evidence",
)


def write_inventory(rows: list[dict[str, Any]], summaries: dict[str, Any], output: Path) -> None:
    """Emit ``tiles.csv`` (flat, one row per tile) + ``tiles.json`` (full, with neighbour links)."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "tiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "tiles.json").write_text(
        json.dumps({"schema": INVENTORY_SCHEMA, "tiles": rows}, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps({"schema": INVENTORY_SCHEMA, **summaries}, indent=2), encoding="utf-8"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="v50 per-tile signal inventory (records every tile; filters nothing)"
    )
    parser.add_argument("--store", required=True, type=Path, action="append", dest="stores",
                        metavar="STORE", help="a per-map v50 store; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--near-zero-band", type=float, default=WEAK_NEAR_ZERO_BAND,
                        help="Weak-signal |Z| band. The default is the viewer's alpha-calibrated "
                             "50.0; a client whose terrain sits far from sea level (4.0.0 Kalimdor "
                             "measures |Z| p50=441) needs a larger value or every detection is "
                             "suppressed. 'inf' disables the band entirely.")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    per_map: dict[str, Any] = {}
    for store in args.stores:
        rows = inventory_store(store, near_zero_band=args.near_zero_band)
        if not rows:
            print(f"WARNING: {store} produced zero tiles", flush=True)
            continue
        map_name = rows[0]["map"]
        per_map[map_name] = summarize(rows)
        all_rows.extend(rows)
        counts = per_map[map_name]["by_classification"]
        print(f"{map_name:12s} {len(rows):>4} tiles  {counts}", flush=True)

    if not all_rows:
        raise SystemExit("no tiles inventoried")
    summaries = {"per_map": per_map, "corpus": summarize(all_rows)}
    write_inventory(all_rows, summaries, args.output)
    print(f"\n[DONE] {len(all_rows)} tiles -> {args.output}", flush=True)
    return 0
