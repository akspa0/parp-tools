"""Spec 103 T014 — export predicted terrain as OBJ + MTL for eyeball review.

Consumes an infer_spec103_v7.py predictions directory and writes one OBJ per tile (the
existing export_terrain_obj.py mesh convention: 257×257 vertices, two triangles per cell,
minimap as the texture when a store is given).

Run from wow-viewer/data-harvester/ (fast, CPU; open the OBJs in any viewer):

    uv run python scripts/spec103_export_mesh.py \
        --predictions ../output/spec103_v7_synth_v1/predictions \
        --store ../output/datasets/spec103/synthetic_v1.zarr \
        --output ../output/spec103_v7_synth_v1/meshes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
from PIL import Image

TILE_SIZE = 533.333


def height_to_obj(height: np.ndarray, obj_path: Path, texture_name: str | None) -> None:
    h, w = height.shape
    rows, cols = h - 1, w - 1
    lines: list[str] = []
    if texture_name:
        mtl_name = obj_path.stem + ".mtl"
        lines += [f"mtllib {mtl_name}", "usemtl terrain", ""]
        (obj_path.parent / mtl_name).write_text(
            "\n".join(["newmtl terrain", f"map_Kd {texture_name}", "Ka 1.0 1.0 1.0",
                       "Kd 1.0 1.0 1.0", "Ns 0.0", "d 1.0"]), encoding="utf-8")
    for y in range(h):
        for x in range(w):
            lines.append(f"v {(x / cols) * TILE_SIZE:.6f} {(y / rows) * TILE_SIZE:.6f} {float(height[y, x]):.6f}")
    lines.append("")
    for y in range(h):
        for x in range(w):
            lines.append(f"vt {x / cols:.6f} {1.0 - (y / rows):.6f}")
    lines.append("")
    for y in range(rows):
        for x in range(cols):
            v00 = y * w + x + 1
            v10 = y * w + (x + 1) + 1
            v01 = (y + 1) * w + x + 1
            v11 = (y + 1) * w + (x + 1) + 1
            lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
            lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")
    obj_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 predicted-terrain OBJ export")
    ap.add_argument("--predictions", required=True, type=Path, help="infer_spec103_v7.py output dir")
    ap.add_argument("--store", type=Path, default=None, help="optional: source store for minimap textures")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads((args.predictions / "predictions_manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    minimap_lookup: dict[str, int] = {}
    group = None
    if args.store is not None:
        group = zarr.open_group(str(args.store), mode="r")
        index = pq.read_table(args.store / "index.parquet").to_pylist()
        minimap_lookup = {f"{r['map']}_{r['tile_x']}_{r['tile_y']}": i for i, r in enumerate(index)}

    for tile in manifest["tiles"]:
        tile_name = tile["tile_name"]
        height = np.load(args.predictions / tile["prediction_dir"] / "predicted_height_257.npy")
        texture_name = None
        if group is not None and tile_name in minimap_lookup:
            texture_name = f"{tile_name}.png"
            Image.fromarray(np.asarray(group["minimap_rgb"][minimap_lookup[tile_name]], dtype=np.uint8)).save(
                args.output / texture_name)
        height_to_obj(height, args.output / f"{tile_name}.obj", texture_name)
        print(f"[mesh] {tile_name}.obj", flush=True)

    print(f"[DONE] {len(manifest['tiles'])} meshes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
