"""Spec 103 T012 — point at (and validate) the real clean training store.

The existing V18 store already pairs minimap_rgb with height_257, normal_xyz, liquid signals,
and object_precise_mask (spec FR-012) — no copy or reharvest is needed. This script verifies a
store carries everything the spec103 trainer reads, reports per-signal shapes and holdout-map
counts, and writes a small store-contract JSON the training run records as its data identity.

Run from wow-viewer/data-harvester/ (fast, read-only):

    uv run python scripts/spec103_build_real_store.py \
        --store ../output/datasets/v18/3_3_5_12340.zarr \
        --output ../output/spec103/real_store_contract.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

REQUIRED = {
    "minimap_rgb": (np.uint8, (256, 256, 3)),
    "height_257": (np.float32, (257, 257)),
}
OPTIONAL = ("normal_xyz", "liquid_mask", "liquid_height", "object_precise_mask")
PROHIBITED = ("wdl_height_33",)


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 real-store validator")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("../output/spec103/real_store_contract.json"))
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    problems: list[str] = []
    signals: dict[str, dict] = {}

    row_count = None
    for name, (dtype, shape) in REQUIRED.items():
        if name not in group:
            problems.append(f"missing required array {name}")
            continue
        arr = group[name]
        row_count = arr.shape[0] if row_count is None else row_count
        if tuple(arr.shape[1:]) != shape:
            problems.append(f"{name} has per-tile shape {tuple(arr.shape[1:])}, expected {shape}")
        if not np.can_cast(arr.dtype, np.dtype(dtype), casting="same_kind"):
            problems.append(f"{name} dtype {arr.dtype} cannot safely convert to {np.dtype(dtype)}")
        signals[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "required": True}
    for name in OPTIONAL:
        if name in group:
            signals[name] = {"shape": list(group[name].shape), "dtype": str(group[name].dtype), "required": False}
        else:
            signals[name] = {"present": False, "required": False,
                             "note": "assembler substitutes v7 fallback (flat normals / zero masks)"}
    for name in PROHIBITED:
        if name in group:
            problems.append(f"prohibited array {name} present (spec FR-005)")

    index_path = args.store / "index.parquet"
    if not index_path.exists():
        problems.append("missing index.parquet")
        maps: Counter = Counter()
    else:
        index = pq.read_table(index_path, columns=["map"]).to_pylist()
        maps = Counter(str(r["map"]) for r in index)
        if row_count is not None and len(index) != row_count:
            problems.append(f"index rows ({len(index)}) != array rows ({row_count})")

    contract = {
        "schema": "spec103-real-store-contract-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "store": str(args.store.resolve()),
        "index_sha256": sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None,
        "row_count": row_count,
        "maps": dict(maps.most_common()),
        "signals": signals,
        "wdl_prior_policy": "derived at batch time: outer = height_257[::16, ::16]; wdl_height_33 prohibited",
        "problems": problems,
        "passed": not problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    print(f"[spec103] store: {args.store}")
    print(f"[spec103] rows: {row_count}, maps: {len(maps)} (largest: {maps.most_common(3)})")
    for name, info in signals.items():
        print(f"  {name}: {info}")
    if problems:
        print("[FAIL] " + "; ".join(problems))
        return 1
    print(f"[PASS] contract -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
