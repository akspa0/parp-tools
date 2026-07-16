"""Build the capped, mixed Spec 108 WDL-prior store (USER runs; CPU I/O)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.spec108_mixed_curriculum import assign_group_splits, real_brush_descriptor, select_real_rows, select_synthetic_rows  # noqa: E402, I001

FIELDS = ("minimap_rgb", "height_257", "normal_xyz", "liquid_mask", "liquid_height", "object_precise_mask")


def _records(store: Path) -> list[dict]:
    return pq.read_table(store / "index.parquet").to_pylist()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build capped mixed real/synthetic Spec 108 curriculum")
    ap.add_argument("--real-store", required=True, type=Path)
    ap.add_argument("--synthetic-store", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--real-rows", type=int, default=144)
    ap.add_argument("--synthetic-rows", type=int, default=96)
    ap.add_argument("--max-rows", type=int, default=240)
    args = ap.parse_args()
    if args.real_rows + args.synthetic_rows > args.max_rows or args.max_rows >= 256:
        raise SystemExit("real + synthetic rows must be <= max-rows and max-rows must be < 256")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    real_group, synthetic_group = zarr.open_group(str(args.real_store), mode="r"), zarr.open_group(str(args.synthetic_store), mode="r")
    real_candidates = []
    for row_number, row in enumerate(_records(args.real_store)):
        if not (row.get("has_alpha_256") and row.get("has_height_257") and row.get("has_minimap_rgb")):
            continue
        descriptor = real_brush_descriptor(real_group["alpha_256"][row_number], real_group["height_257"][row_number])
        real_candidates.append({**row, "store_row": row_number, "descriptor": descriptor, "source_kind": "real_053", "source_group_id": f"real:{row['build']}:{row['map']}:{row['tile_id']}"})
    synthetic_candidates = [{**row, "store_row": row_number, "source_kind": "synthetic"} for row_number, row in enumerate(_records(args.synthetic_store))]
    selected = select_real_rows(real_candidates, total=args.real_rows) + select_synthetic_rows(synthetic_candidates, total=args.synthetic_rows)
    if len(selected) != args.real_rows + args.synthetic_rows:
        raise SystemExit(f"insufficient selected rows: {len(selected)}")
    assign_group_splits(selected)
    output = zarr.open_group(str(args.output), mode="w")
    reference = {field: (real_group[field] if field in real_group else synthetic_group[field]) for field in FIELDS if field in real_group or field in synthetic_group}
    for field, array in reference.items():
        output.create_array(field, shape=(len(selected), *array.shape[1:]), dtype=array.dtype, chunks=(1, *array.shape[1:]))
    index_rows = []
    for target_row, row in enumerate(selected):
        source = real_group if row["source_kind"] == "real_053" else synthetic_group
        for field, target in reference.items():
            if field in source:
                output[field][target_row] = source[field][int(row["store_row"])]
            else:
                output[field][target_row] = np.zeros(target.shape[1:], dtype=target.dtype)
        index_rows.append({**row, "tile_id": target_row, "source_tile_id": int(row["tile_id"]), "source_store": str((args.real_store if row["source_kind"] == "real_053" else args.synthetic_store).resolve())})
    pq.write_table(pa.Table.from_pylist(index_rows), args.output / "index.parquet")
    summary = {"schema": "spec108-mixed-curriculum-v1", "total_rows": len(selected), "real_rows": sum(row["source_kind"] == "real_053" for row in selected), "synthetic_rows": sum(row["source_kind"] == "synthetic" for row in selected), "maps": sorted({str(row["map"]) for row in selected if row["source_kind"] == "real_053"}), "splits": {name: sum(row["split"] == name for row in selected) for name in ("train", "val")}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[spec108] mixed curriculum rows={summary['total_rows']} real={summary['real_rows']} synthetic={summary['synthetic_rows']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
