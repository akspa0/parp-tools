"""Audit V18 liquid signals against the Spec 102 numeric copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr


def stats(value: np.ndarray) -> dict[str, float | int | list[int]]:
    data = np.asarray(value)
    return {
        "shape": list(data.shape),
        "min": float(data.min(initial=0)),
        "max": float(data.max(initial=0)),
        "mean": float(data.mean()),
        "nonzero": int(np.count_nonzero(data)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Spec 102 liquid copy from V18")
    parser.add_argument("--numeric-store", required=True, type=Path)
    parser.add_argument("--v18-store", required=True, nargs="+", type=Path)
    parser.add_argument("--rows", required=True, nargs="+", type=int)
    args = parser.parse_args()
    numeric = zarr.open_group(str(args.numeric_store), mode="r")
    numeric_index = pq.read_table(args.numeric_store / "index.parquet").to_pylist()
    sources = {}
    for path in args.v18_store:
        group = zarr.open_group(str(path), mode="r")
        index = pq.read_table(path / "index.parquet").to_pylist()
        build = str(index[0]["build"])
        sources[build] = (group, index, path)
    reports = []
    for row in args.rows:
        selected = numeric_index[row]
        build = str(selected["build"])
        source, source_index, source_path = sources[build]
        v18_row = int(selected["v18_row"])
        origin = source_index[v18_row]
        source_mask = np.asarray(source["liquid_mask"][v18_row])
        copied_mask = np.asarray(numeric["liquid_mask_256"][row])
        report = {
            "row": row,
            "build": build,
            "map": selected["map"],
            "tile_x": int(selected["tile_x"]),
            "tile_y": int(selected["tile_y"]),
            "v18_store": str(source_path),
            "v18_row": v18_row,
            "v18_has_liquid_mask": origin.get("has_liquid_mask"),
            "v18_has_liquid_height": origin.get("has_liquid_height"),
            "v18_liquid_source": origin.get("liquid_source"),
            "v18_liquid_mask": stats(source_mask),
            "numeric_liquid_mask": stats(copied_mask),
            "copy_equal_after_u8": bool(np.array_equal(copied_mask, np.rint(np.clip(source_mask, 0, 1) * 255).astype(np.uint8))),
            "v18_mcnk_flags": stats(source["mcnk_flags_16"][v18_row]),
        }
        if "liquid_type_256" in source:
            report["v18_liquid_type_256"] = stats(source["liquid_type_256"][v18_row])
        if "liquid_height" in source:
            report["v18_liquid_height"] = stats(source["liquid_height"][v18_row])
        reports.append(report)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
