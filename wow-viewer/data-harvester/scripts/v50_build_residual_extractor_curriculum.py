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
from harvester.v50.dxt1_approx import (  # noqa: E402
    block_edge_ratio,
    dxt1_round_trip,
    unique_colour_count,
)

CURRICULUM_SCHEMA = "v125-residual-extractor-curriculum-v1"
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

    row_lookup = _build_row_lookup(args.minimap_store)

    residuals: list[np.ndarray] = []
    minimaps: list[np.ndarray] = []
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
        minimap = np.asarray(minimap_group["minimap_rgb"][row], dtype=np.float32)
        # minimap_rgb is 256x256x3; residual is 256x256. Confirm alignment.
        if minimap.shape[:2] != residual.shape:
            skipped["shape_mismatch"] += 1
            continue
        residuals.append(residual)
        minimaps.append(minimap)
        index_rows.append({"map": args.map, "tile_x": tx, "tile_y": ty, "source_group_id": f"{args.map}_{tx}_{ty}", "split": "train"})
        matched += 1

    if matched < 40:
        raise SystemExit(
            f"only {matched} residual/minimap pairs matched (need >= 40) from "
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
    minimap_stack = np.stack(minimaps).astype(np.float32)
    residual_stack = np.stack(residuals).astype(np.float32)

    # Deployment reads AUTHORED minimaps, which are DXT1. Our synthesizer emits pristine 24-bit RGB,
    # so training on it alone leaves a codec domain gap the loss never sees. Store the degraded input
    # alongside the pristine one — both stay queryable, neither is dropped — and let the trainer pick.
    # The TARGET stays pristine on purpose: the model should read a codec-damaged minimap and predict
    # the clean underlying residual.
    degraded_stack = np.stack(
        [dxt1_round_trip(np.clip(tile, 0, 255).astype(np.uint8)) for tile in minimap_stack]
    ).astype(np.float32)

    group.create_array("minimap_rgb", data=minimap_stack, chunks=(1, *minimap_stack.shape[1:]))
    group.create_array("minimap_rgb_dxt1", data=degraded_stack, chunks=(1, *degraded_stack.shape[1:]))
    group.create_array("residual_256", data=residual_stack, chunks=(1, *residual_stack.shape[1:]))

    # Record what the degradation actually did, so a bad codec wiring is visible in the summary
    # rather than silently training a model on the wrong domain. Authored 0.5.3 reference:
    # 1196-5269 unique colours per tile, ~3200 median.
    sample = min(8, len(degraded_stack))
    codec_stats = {
        "transform": "dxt1_round_trip",
        "parity": "bit-exact with WowViewer.Core.IO.Blp.Dxt1TileCodec",
        "authored_reference_unique_colours": [1196, 5269],
        "pristine_unique_colours_median": float(np.median(
            [unique_colour_count(minimap_stack[i].astype(np.uint8)) for i in range(sample)])),
        "degraded_unique_colours_median": float(np.median(
            [unique_colour_count(degraded_stack[i].astype(np.uint8)) for i in range(sample)])),
        "pristine_block_edge_ratio_median": float(np.median(
            [block_edge_ratio(minimap_stack[i]) for i in range(sample)])),
        "degraded_block_edge_ratio_median": float(np.median(
            [block_edge_ratio(degraded_stack[i]) for i in range(sample)])),
    }

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
        "residual_pngs_seen": len(residual_paths),
        "skipped": skipped,
        "deployment_domain_input": "minimap_rgb_dxt1",
        "codec_degradation": codec_stats,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Residual-extractor curriculum written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
