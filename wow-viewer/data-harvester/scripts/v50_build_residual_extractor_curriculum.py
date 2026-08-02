"""Build the Spec 125 residual-extractor curriculum store.

Pairs each minimap RGB tile (from an existing v50 store's ``minimap_rgb``) with the same tile's
textureless terrain-shadow residual (from the ``--textureless-residuals`` output PNGs), and writes a
Zarr store with ``minimap_rgb`` (256x256x3) and ``residual_256`` (256x256 single channel) arrays,
row-aligned with an index.parquet.

The residual PNGs are named ``<map>_<tx>_<ty>_residual.png`` under the residual output's ``tiles/``
directory. The minimap source is an existing v50 store that carries ``minimap_rgb`` for the same
map/tile coordinates.

Usage (USER runs):
  uv run python scripts/v50_build_residual_extractor_curriculum.py \
      --residual-dir <residual-output>/tiles \
      --minimap-store <v50-store> \
      --output <extractor-curriculum-store> \
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

CURRICULUM_SCHEMA = "v125-residual-extractor-curriculum-v1"
RESIDUAL_RE = re.compile(r"^(?P<map>.+?)_(?P<tx>\d{2})_(?P<ty>\d{2})_residual\.png$")


def _load_residual_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32)  # single grayscale channel
    if arr.ndim != 2:
        raise ValueError(f"residual {path} is not 2D grayscale")
    return arr


def _load_minimap_rgb(store_path: Path, store: zarr.Group, map_name: str, tx: int, ty: int) -> np.ndarray | None:
    index_path = store_path / "index.parquet"
    if not index_path.exists():
        return None
    table = pq.read_table(index_path)
    rows = table.to_pylist()
    for i, row in enumerate(rows):
        if str(row.get("map", "")) == map_name and int(row.get("tile_x", -1)) == tx and int(row.get("tile_y", -1)) == ty:
            return np.asarray(store["minimap_rgb"][i], dtype=np.float32)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Spec 125 residual-extractor curriculum store")
    ap.add_argument("--residual-dir", required=True, type=Path, help="dir of *_residual.png tiles")
    ap.add_argument("--minimap-store", required=True, type=Path, help="existing v50 store with minimap_rgb")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--map", required=True, help="map name (Azeroth or Kalimdor)")
    ap.add_argument("--release", default="v50.1")
    args = ap.parse_args()

    if not args.residual_dir.is_dir():
        raise SystemExit(f"residual dir not found: {args.residual_dir}")
    if not args.minimap_store.is_dir():
        raise SystemExit(f"minimap store not found: {args.minimap_store}")

    minimap_group = zarr.open_group(str(args.minimap_store), mode="r")
    if "minimap_rgb" not in minimap_group:
        raise SystemExit(f"minimap store has no minimap_rgb array: {args.minimap_store}")

    residual_paths = sorted(args.residual_dir.glob("*_residual.png"))
    if not residual_paths:
        raise SystemExit(f"no *_residual.png tiles found in {args.residual_dir}")

    residuals: list[np.ndarray] = []
    minimaps: list[np.ndarray] = []
    index_rows: list[dict] = []
    matched = 0
    for path in residual_paths:
        m = RESIDUAL_RE.match(path.name)
        if not m:
            continue
        tx, ty = int(m.group("tx")), int(m.group("ty"))
        residual = _load_residual_png(path)
        minimap = _load_minimap_rgb(args.minimap_store, minimap_group, args.map, tx, ty)
        if minimap is None:
            continue
        # minimap_rgb is 256x256x3; residual is 256x256. Confirm alignment.
        if minimap.shape[:2] != residual.shape:
            continue
        residuals.append(residual)
        minimaps.append(minimap)
        index_rows.append({"map": args.map, "tile_x": tx, "tile_y": ty, "source_group_id": f"{args.map}_{tx}_{ty}", "split": "train"})
        matched += 1

    if matched < 40:
        raise SystemExit(f"only {matched} residual/minimap pairs matched; need >= 40")

    # Held-out split by source group: hold out the last 10% of tiles as val.
    val_count = max(1, matched // 10)
    for row in index_rows[-val_count:]:
        row["split"] = "val"

    args.output.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(args.output), mode="w")
    group.attrs["schema"] = CURRICULUM_SCHEMA
    group.attrs["map"] = args.map
    group.attrs["split_mode"] = "source_group_holdout"
    group.attrs["release"] = args.release
    group.attrs["release_identity"] = release_identity(args.release)

    group.create_dataset("minimap_rgb", data=np.stack(minimaps), chunks=(1, 256, 256, 3), dtype="f4")
    group.create_dataset("residual_256", data=np.stack(residuals), chunks=(1, 256, 256), dtype="f4")

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
        "minimap_store": str(args.minimap_store),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Residual-extractor curriculum written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
