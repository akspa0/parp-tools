"""V22/V18 dataset signal audit (Spec 094 amendment A8).

C#-grounded: the existing, working C# harvester (WowViewer.Tool.Harvest
extract-unified) re-extracts reference signals for a sampled tile set directly
from the staged client; this script only compares. Checks per signal:

  1. presence + shape + dtype of every store array;
  2. `has_*` index-flag truthfulness (flag True but tile content all-zero);
  3. per-tile agreement between the store and the C# reference shard
     (mean/max abs diff, zero-fill, classification OK / MISMATCH / MISSING);
  4. V22 placement-array internal consistency (offsets/counts vs data rows).

Emits <output>/report.json plus a console summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import store as v24_store  # noqa: E402

DATA_HARVESTER_ROOT = Path(__file__).resolve().parents[1]
WOW_VIEWER_ROOT = DATA_HARVESTER_ROOT.parent
DEFAULT_HARVEST_DLL = (
    WOW_VIEWER_ROOT
    / "tools" / "harvest" / "WowViewer.Tool.Harvest"
    / "bin" / "Debug" / "net10.0" / "WowViewer.Tool.Harvest.dll"
)

# store array name -> C# shard key (harvest NpzTileSerializer names)
SIGNAL_MAP: dict[str, str] = {
    "height_257": "height_257",
    "minimap_rgb": "minimap_rgb_256",
    "alpha_256": "mcal_alpha_pack_256",
    "normal_xyz": "mcnr_normal_xyz",
    "mcnr_mask_257": "mcnr_mask_257",
    "object_precise_mask": "object_precise_mask_257",
    "object_mask": "object_mask_257",
    "liquid_mask": "unified_liquid_mask",
    "holes_16": "hole_mask_16",
    "mcly_texture_ids": "mcly_texture_ids",
    "mcnk_flags_16": "mcnk_flags_16",
    "mddf_mask": "mddf_mask_257",
    "modf_mask": "modf_mask_257",
    "object_filtered_mask": "object_filtered_mask_257",
    "shadow_mask": "mcsh_shadow_mask_256",
}


def _run_harvest_shard(
    harvest_dll: Path, client_root: Path, map_name: str, tile_x: int, tile_y: int, output: Path
) -> dict[str, np.ndarray] | None:
    cmd = [
        "dotnet", str(harvest_dll),
        "extract-unified",
        "--client-root", str(client_root),
        "--map", map_name,
        "--tile-x", str(tile_x),
        "--tile-y", str(tile_y),
        "--export-placements",
        "--output", str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not output.exists():
        return None
    with np.load(output, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files if not k.startswith("raw_")}


def _aligned(store_arr: np.ndarray, shard_arr: np.ndarray) -> np.ndarray | None:
    """Return the shard array reshaped/transposed to the store layout, or None."""
    if shard_arr.shape == store_arr.shape:
        return shard_arr
    if shard_arr.ndim == 3 and shard_arr.shape[::-1] == store_arr.shape:
        return shard_arr.transpose(2, 1, 0)
    if (
        shard_arr.ndim == 3
        and store_arr.ndim == 3
        and shard_arr.shape[0] == store_arr.shape[2]
        and shard_arr.shape[1:] == store_arr.shape[:2]
    ):
        return shard_arr.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    return None


def _compare_signal(store_tile: np.ndarray, shard_tile: np.ndarray) -> dict:
    aligned = _aligned(store_tile, shard_tile)
    if aligned is None:
        return {
            "status": "SHAPE_MISMATCH",
            "store_shape": list(store_tile.shape),
            "shard_shape": list(shard_tile.shape),
        }

    a = store_tile.astype(np.float64)
    b = aligned.astype(np.float64)
    # uint8 RGB in the store vs float [0..255] or [0..1] in the shard.
    if a.max() > 2.0 and 0 < b.max() <= 1.0:
        b = b * 255.0
    elif b.max() > 2.0 and 0 < a.max() <= 1.0:
        a = a * 255.0

    diff = np.abs(a - b)
    scale = max(1e-9, float(np.abs(a).max()))
    rel_max = float(diff.max()) / scale
    status = "OK" if rel_max <= 0.02 else ("CLOSE" if rel_max <= 0.10 else "MISMATCH")
    return {
        "status": status,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "store_zero": bool(not store_tile.any()),
        "shard_zero": bool(not shard_tile.any()),
    }


def _audit_flags(group: zarr.Group, index: dict[str, list], rows: list[int]) -> list[dict]:
    problems = []
    for column in index:
        if not column.startswith("has_"):
            continue
        array_name = column[4:]
        if array_name not in group:
            if any(index[column][r] for r in rows):
                problems.append({"flag": column, "problem": "flag set but array absent from store"})
            continue
        for r in rows:
            content = np.asarray(group[array_name][r])
            has_content = bool(content.any())
            flagged = bool(index[column][r])
            if flagged and not has_content:
                problems.append(
                    {"flag": column, "row": r, "problem": "flag True but tile is all-zero"}
                )
    return problems


def _audit_placements(group: zarr.Group) -> dict:
    result = {}
    for kind in ("mddf", "modf"):
        data_name = f"{kind}_placement_data"
        if data_name not in group:
            result[kind] = {"status": "ABSENT"}
            continue
        data_rows = group[data_name].shape[0]
        offsets = np.asarray(group[f"{kind}_placement_offset"][:])
        counts = np.asarray(group[f"{kind}_count"][:]).reshape(-1)
        checks = {
            "data_rows": int(data_rows),
            "count_sum": int(counts.sum()),
            "counts_match_rows": bool(counts.sum() == data_rows),
            "offsets_monotonic": bool(np.all(np.diff(offsets[offsets >= 0]) >= 0)),
            "offsets_in_range": bool(
                offsets.max(initial=0) <= data_rows and offsets.min(initial=0) >= -1
            ),
            "data_finite": bool(np.isfinite(np.asarray(group[data_name][:1000])).all()),
        }
        checks["status"] = "OK" if all(
            v for k, v in checks.items() if isinstance(v, bool)
        ) else "MISMATCH"
        result[kind] = checks
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="V22 (or V18) Zarr store path")
    parser.add_argument("--staged-client", required=True)
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--output", required=True, help="report directory")
    parser.add_argument("--harvest-dll", default=str(DEFAULT_HARVEST_DLL))
    args = parser.parse_args()

    harvest_dll = Path(args.harvest_dll)
    if not harvest_dll.exists():
        print(
            f"Harvest tool not built: {harvest_dll}\n"
            f"  dotnet build {WOW_VIEWER_ROOT / 'tools' / 'harvest' / 'WowViewer.Tool.Harvest'} -c Debug",
            file=sys.stderr,
        )
        return 1

    group = zarr.open_group(str(args.store), mode="r")
    index = v24_store.read_index(args.store)
    n = len(index["tile_id"])

    rows = list(range(n))
    if args.maps:
        wanted = {m.lower() for m in args.maps}
        rows = [r for r in rows if index["map"][r].lower() in wanted]
    rng = np.random.default_rng(args.seed)
    sampled = sorted(rng.choice(rows, size=min(args.sample, len(rows)), replace=False).tolist())

    arrays = {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)} for name, arr in group.arrays()}
    print(f"store: {args.store} ({n} tiles, {len(arrays)} arrays); sampling {len(sampled)} tiles")

    flag_problems = _audit_flags(group, index, sampled)
    placements = _audit_placements(group)

    tile_reports = []
    with tempfile.TemporaryDirectory(prefix="v22audit_") as tmp:
        for r in sampled:
            map_name = index["map"][r]
            tx, ty = int(index["tile_x"][r]), int(index["tile_y"][r])
            shard_path = Path(tmp) / f"{map_name}_{tx}_{ty}.npz"
            shard = _run_harvest_shard(
                harvest_dll, Path(args.staged_client), map_name, tx, ty, shard_path
            )
            if shard is None:
                tile_reports.append(
                    {"row": r, "map": map_name, "tile": [tx, ty], "status": "HARVEST_FAILED"}
                )
                print(f"  [{map_name} {tx},{ty}] C# harvest FAILED")
                continue

            signals = {}
            for store_name, shard_name in SIGNAL_MAP.items():
                if store_name not in group:
                    continue
                if shard_name not in shard:
                    signals[store_name] = {"status": "NOT_IN_CSHARP_SHARD"}
                    continue
                signals[store_name] = _compare_signal(
                    np.asarray(group[store_name][r]), shard[shard_name]
                )

            bad = [k for k, v in signals.items() if v["status"] in ("MISMATCH", "SHAPE_MISMATCH")]
            tile_reports.append(
                {"row": r, "map": map_name, "tile": [tx, ty], "signals": signals}
            )
            print(f"  [{map_name} {tx},{ty}] {len(signals)} signals; problems: {bad or 'none'}")

    signal_summary: dict[str, dict[str, int]] = {}
    for report in tile_reports:
        for name, result in report.get("signals", {}).items():
            bucket = signal_summary.setdefault(name, {})
            bucket[result["status"]] = bucket.get(result["status"], 0) + 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "store": str(args.store),
        "staged_client": str(args.staged_client),
        "sampled_tiles": len(sampled),
        "arrays": arrays,
        "signal_summary": signal_summary,
        "flag_problems": flag_problems,
        "placement_consistency": placements,
        "tiles": tile_reports,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nsignal summary:")
    for name, buckets in sorted(signal_summary.items()):
        print(f"  {name}: {buckets}")
    print(f"flag problems: {len(flag_problems)}")
    print(f"placements: { {k: v.get('status') for k, v in placements.items()} }")
    print(f"report: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
