"""Build the Spec 125 residual curriculum store.

Pairs each textureless terrain-shadow residual PNG (from ``synthetic-minimap --textureless-residuals``)
with the same tile's real height ground truth (``height_257`` from an existing v50 store), and writes
a Zarr store with ``residual_256`` (single grayscale channel) and ``height_257`` arrays, row-aligned
with an index.parquet.

The residual PNGs are named ``<map>_<tx>_<ty>_residual.png`` under the residual output's ``tiles/``
directory. The height source is an existing v50 store that already carries ``height_257`` for the same
map/tile coordinates (the dual-source curriculum store).

Usage (USER runs):
  uv run python scripts/v50_build_residual_curriculum.py \
      --residual-dir <residual-output>/tiles \
      --height-store <v50-store> \
      --output <residual-curriculum-store> \
      --map Azeroth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.contracts import release_identity  # noqa: E402

CURRICULUM_SCHEMA = "v125-residual-curriculum-v1"
RESIDUAL_RE = re.compile(r"^(?P<map>.+?)_(?P<tx>\d{2})_(?P<ty>\d{2})_residual\.png$")


def _load_residual_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32)  # single grayscale channel
    if arr.ndim != 2:
        raise ValueError(f"residual {path} is not 2D grayscale")
    return arr


def _build_row_lookup(store_path: Path) -> dict[tuple[str, int, int], int]:
    """Map (map, tile_x, tile_y) -> store row, read once rather than per residual tile."""
    index_path = store_path / "index.parquet"
    if not index_path.exists():
        raise SystemExit(f"store has no index.parquet: {store_path}")
    lookup: dict[tuple[str, int, int], int] = {}
    for i, row in enumerate(pq.read_table(index_path).to_pylist()):
        lookup.setdefault((str(row.get("map", "")), int(row.get("tile_x", -1)), int(row.get("tile_y", -1))), i)
    return lookup


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Spec 125 residual curriculum store")
    ap.add_argument("--residual-dir", required=True, type=Path, help="dir of *_residual.png tiles")
    ap.add_argument("--height-store", required=True, type=Path, help="existing v50 store with height_257")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--map", required=True, help="map name (Azeroth or Kalimdor)")
    ap.add_argument("--release", default="v50.1")
    args = ap.parse_args()

    if not args.residual_dir.is_dir():
        raise SystemExit(f"residual dir not found: {args.residual_dir}")
    if not args.height_store.is_dir():
        raise SystemExit(f"height store not found: {args.height_store}")

    height_group = zarr.open_group(str(args.height_store), mode="r")
    if "height_257" not in height_group:
        raise SystemExit(f"height store has no height_257 array: {args.height_store}")

    residual_paths = sorted(args.residual_dir.glob("*_residual.png"))
    if not residual_paths:
        raise SystemExit(f"no *_residual.png tiles found in {args.residual_dir}")

    row_lookup = _build_row_lookup(args.height_store)

    residuals: list[np.ndarray] = []
    heights: list[np.ndarray] = []
    index_rows: list[dict] = []
    skipped = {"unparsed_name": 0, "no_store_row": 0, "shape_mismatch": 0}
    matched = 0
    for path in residual_paths:
        m = RESIDUAL_RE.match(path.name)
        if not m:
            skipped["unparsed_name"] += 1
            continue
        tx, ty = int(m.group("tx")), int(m.group("ty"))
        row = row_lookup.get((args.map, tx, ty))
        if row is None:
            skipped["no_store_row"] += 1
            continue
        residual = _load_residual_png(path)
        height = np.asarray(height_group["height_257"][row], dtype=np.float32)
        # Residual is 256x256; height_257 is 257x257. Crop the height to the residual's grid so the
        # two fields are pixel-aligned (the 257th row/col is the shared tile-edge vertex).
        if height.shape[0] == residual.shape[0] + 1 and height.shape[1] == residual.shape[1] + 1:
            height = height[: residual.shape[0], : residual.shape[1]]
        if height.shape != residual.shape:
            skipped["shape_mismatch"] += 1
            continue
        residuals.append(residual)
        heights.append(height)
        index_rows.append({"map": args.map, "tile_x": tx, "tile_y": ty, "source_group_id": f"{args.map}_{tx}_{ty}", "split": "train"})
        matched += 1

    if matched < 40:
        raise SystemExit(
            f"only {matched} residual/height pairs matched (need >= 40) from "
            f"{len(residual_paths)} residual PNGs; skipped={skipped}"
        )

    # Held-out split by source group: hold out the last 10% of tiles as val, but never fewer than 8 —
    # the trainer's gate requires train >= 32 and val >= 8, so a 10%-of-40 split would build a store
    # that the trainer then refuses.
    val_count = max(8, matched // 10)
    for row in index_rows[-val_count:]:
        row["split"] = "val"

    args.output.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(args.output), mode="w")
    group.attrs["schema"] = CURRICULUM_SCHEMA
    group.attrs["map"] = args.map
    group.attrs["split_mode"] = "source_group_holdout"
    # model_family/release/schema must be top-level attrs: that triple is what the trainer's
    # require_store_release gate reads (a nested release_identity dict is invisible to it).
    identity = release_identity(args.release)
    group.attrs["model_family"] = identity["model_family"]
    group.attrs["release"] = identity["release"]
    group.attrs["release_identity"] = identity

    # zarr v3: create_array(name, data=...) infers shape/dtype; create_dataset requires an explicit shape.
    # Heights were cropped to the residual's 256 grid above, so chunk from the real shape, not 257.
    residual_stack = np.stack(residuals).astype(np.float32)
    height_stack = np.stack(heights).astype(np.float32)
    group.create_array("residual_256", data=residual_stack, chunks=(1, *residual_stack.shape[1:]))
    group.create_array("height_257", data=height_stack, chunks=(1, *height_stack.shape[1:]))

    table = pa.table(
        {
            "map": [r["map"] for r in index_rows],
            "tile_x": [r["tile_x"] for r in index_rows],
            "tile_y": [r["tile_y"] for r in index_rows],
            "source_group_id": [r["source_group_id"] for r in index_rows],
            "split": [r["split"] for r in index_rows],
        }
    )
    pq.write_table(table, args.output / "index.parquet")

    summary = {
        "schema": CURRICULUM_SCHEMA,
        "map": args.map,
        "rows": matched,
        "train": sum(1 for r in index_rows if r["split"] == "train"),
        "val": sum(1 for r in index_rows if r["split"] == "val"),
        "residual_dir": str(args.residual_dir),
        "height_store": str(args.height_store),
        "residual_pngs_seen": len(residual_paths),
        "skipped": skipped,
        "target_grid": list(height_stack.shape[1:]),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Residual curriculum written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
