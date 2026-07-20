"""Spec 114: turn composed relative-relief PNGs into a flat, drag-and-drop folder of
world-positioned OBJ + MTL + texture PNG per tile.

Reads the ``inference_manifest.json`` written by ``v50_infer_geometry_detailer.py`` (or
``v50_infer_direct_geometry.py``) plus the ``*_relief16.png`` outputs it references, and writes
one ``.obj`` + ``.mtl`` + ``.png`` per tile into a flat output folder positioned in world X/Y by
tile coordinates parsed from the input filename (``..._<tx>_<ty>.png``). Mirrors the proven
``v24_quilt_objs.py`` "quilt" pattern: drag the whole output folder into MeshLab/Blender/Windows 3D
Viewer and the tiles line up horizontally. No re-inference, no single merged master mesh.

IMPORTANT caveat, printed on every run: the Spec 114 relative-height contract (``v112.1``) is
PER-TILE min-max normalized (``harvester.v50.height_relative_model.encode_relative_height``) --
absolute altitude is not identifiable from one minimap, so Z is NOT on a consistent scale across
tiles. Adjacent tiles in the quilt will show a vertical step at their shared border; that is the
model's documented relative-only contract, not a bug in this exporter. ``--center`` (default on)
recenters each tile on its own mean height so the quilt does not read as a one-directional
ascending staircase, but it does not and cannot restore true cross-tile altitude continuity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TILE_SIZE = 533.333
RELIEF_DTYPE_MAX = 65535.0

_TILE_RE = re.compile(r"^(?P<prefix>.+)_(?P<tx>\d+)_(?P<ty>\d+)$")


def _parse_tile_xy(stem: str) -> tuple[int, int] | None:
    match = _TILE_RE.match(stem)
    if match is None:
        return None
    return int(match.group("tx")), int(match.group("ty"))


def _load_relief(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("I", "I;16"):
            raise ValueError(f"{path} is not a 16-bit grayscale relief PNG (mode={image.mode})")
        arr = np.asarray(image, dtype=np.float64)
    if arr.shape != (257, 257):
        raise ValueError(f"{path} must be 257x257, got {arr.shape}")
    return (arr / RELIEF_DTYPE_MAX).astype(np.float32)


def _write_tile_obj(
    height: np.ndarray,
    texture_name: str,
    obj_path: Path,
    world_x: float,
    world_y: float,
    tile_size: float,
) -> None:
    h, w = height.shape
    rows, cols = h - 1, w - 1
    lines: list[str] = [f"mtllib {obj_path.stem}.mtl", "usemtl terrain", ""]
    # Wavefront/Windows 3D Viewer/MeshLab convention: Y is up, Z is depth (not the Z-up
    # convention common in game engines). Height goes in the v-y field; world "north/south"
    # tile position goes in v-z. Swapping which field carries height flips mesh handedness,
    # so the face winding below is ALSO reversed vs. a naive row-major quad split -- verified
    # by computing the actual cross-product normal for a flat patch (must point +Y).
    for y in range(h):
        for x in range(w):
            ox = world_x + (x / cols) * tile_size
            oy = float(height[y, x])
            oz = world_y + (y / rows) * tile_size
            lines.append(f"v {ox:.4f} {oy:.4f} {oz:.4f}")
    lines.append("")
    for y in range(h):
        for x in range(w):
            u = x / cols
            v = 1.0 - (y / rows)
            lines.append(f"vt {u:.6f} {v:.6f}")
    lines.append("")
    for y in range(rows):
        for x in range(cols):
            v00 = y * w + x + 1
            v10 = y * w + (x + 1) + 1
            v01 = (y + 1) * w + x + 1
            v11 = (y + 1) * w + (x + 1) + 1
            lines.append(f"f {v00}/{v00} {v01}/{v01} {v10}/{v10}")
            lines.append(f"f {v10}/{v10} {v01}/{v01} {v11}/{v11}")
    obj_path.write_text("\n".join(lines), encoding="utf-8")

    mtl_path = obj_path.parent / f"{obj_path.stem}.mtl"
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl terrain",
                f"map_Kd {texture_name}",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ns 0.0",
                "d 1.0",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="inference_manifest.json from v50_infer_geometry_detailer.py "
                             "or v50_infer_direct_geometry.py")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="flat output folder for the quilt OBJs + PNGs")
    parser.add_argument("--tile-size", type=float, default=TILE_SIZE,
                         help="world-space tile size (default 533.333 = one WoW ADT tile)")
    parser.add_argument("--height-scale", type=float, default=100.0,
                         help="vertex-Z multiplier applied to the [0,1] relative relief "
                              "(default 100.0; purely a visualization exaggeration factor)")
    parser.add_argument("--no-center", action="store_true",
                         help="disable per-tile mean-centering (default: each tile's Z is "
                              "recentered on its own mean before scaling, so the quilt does not "
                              "read as a one-directional staircase; raw per-tile [0,1] relief "
                              "is still NOT cross-tile comparable either way)")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tiles = manifest.get("tiles", [])
    if not tiles:
        raise ValueError(f"manifest {args.manifest} has no tiles (was it run with --write?)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "NOTE: Spec 114 relief is PER-TILE relative height (contract v112.1) -- Z is NOT on a "
        "consistent scale across tiles. Expect a visible step at every tile boundary; this "
        "reflects the model's relative-only contract, not an exporter bug.",
        flush=True,
    )

    written = 0
    unplaced: list[str] = []
    for entry in tiles:
        if "output" not in entry:
            unplaced.append(entry.get("input", "<unknown>"))
            continue
        input_path = Path(entry["input"])
        relief_path = Path(entry["output"])
        stem = input_path.stem
        parsed = _parse_tile_xy(stem)
        if parsed is None:
            unplaced.append(stem)
            continue
        tx, ty = parsed

        height = _load_relief(relief_path)
        if not args.no_center:
            height = height - float(height.mean())
        height = height * args.height_scale

        with Image.open(input_path) as src:
            tex = src.convert("RGB")
            if tex.size != (256, 256):
                tex = tex.resize((256, 256), Image.Resampling.LANCZOS)
        texture_name = f"{stem}.png"
        tex.save(args.output_dir / texture_name)

        obj_path = args.output_dir / f"{stem}.obj"
        _write_tile_obj(
            height,
            texture_name,
            obj_path,
            world_x=tx * args.tile_size,
            world_y=ty * args.tile_size,
            tile_size=args.tile_size,
        )
        written += 1

    print(f"Wrote {written} OBJ+MTL+PNG tiles to {args.output_dir}")
    if unplaced:
        print(f"Skipped {len(unplaced)} tile(s) with no --write output or unparseable "
              f"'..._<tx>_<ty>' filename (first 5): {unplaced[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
