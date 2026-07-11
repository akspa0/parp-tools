"""Spec 097 (follow-on) — quilt mode: a flat folder of world-positioned OBJs.

For each PNG in an input folder, this script writes ONE .obj + ONE .png
(texture) into a single flat output folder, with the OBJ vertices
positioned in WORLD SPACE so the user can drag the entire folder's
contents into any 3D viewer (MeshLab, Blender, Windows 3D Viewer) and
see them line up as a quilted map. No subdirectories. No stitched
master OBJ.

File names:
  input : <name>.png  (e.g. tile_31_27.png)
  output: <name>.obj, <name>.png  in the flat output folder

World position is parsed from the filename. Supported name patterns:
  - tile_X_Y.png        -> world (X * 533, Y * 533, 0)
  - Y_X.png             -> world (X * 533, Y * 533, 0)  (legacy mdxviewer)
  - mapname_X_Y.png     -> world (X * 533, Y * 533, 0)
  - any.png             -> world (0, 0, 0)  (not in the quilt; user can drag it)

Defaults: V24 minimap-only Stage A checkpoint auto-discovered, world
tile size = 533.333 (one WoW tile). All flags have sensible defaults so
the typical call is just one path.

Usage:
    uv run python scripts/v24_quilt_objs.py \\
        --input-dir path/to/tiles/ \\
        --output-dir path/to/quilt/
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import lattice, stage_a, train_common  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
V24_VALIDATION_ROOT = SCRIPT_DIR.parent.parent / "output" / "v24_validation"
TILE_SIZE = 533.333
HEIGHT_SCALE = 100.0

# Filename patterns. The first group is X, the second is Y.
_TILE_RE = [
    re.compile(r"tile[_-](?P<x>\d+)[_-](?P<y>\d+)$", re.IGNORECASE),
    re.compile(r"(?P<y>\d+)[_-](?P<x>\d+)$"),
    re.compile(r"^(?P<mapname>[^_]+)_(?P<x>\d+)_(?P<y>\d+)$", re.IGNORECASE),
]


def _parse_world_xy(stem: str, naming: str = "xy") -> tuple[int, int] | None:
    """Return (tile_x, tile_y) parsed from the file stem, or None if unknown.

    ``naming`` selects the convention:
      "xy"  -> file stem is ...X_Y, e.g. tile_31_27.png  (X first, Y second)
      "yx"  -> file stem is ...Y_X, e.g. tile_27_31.png  (Y first, X second;
                                                              common in legacy
                                                              mdxviewer captures
                                                              and the user's
                                                              aligned-grid folders)
    """
    if naming == "yx":
        # Reorder: first group is Y, second is X.
        m = re.match(r"^tile[_-](?P<y>\d+)[_-](?P<x>\d+)$", stem, re.IGNORECASE)
        if m:
            return int(m.group("x")), int(m.group("y"))
        m = re.match(r"^(?P<x>\d+)[_-](?P<y>\d+)$", stem)
        if m:
            return int(m.group("x")), int(m.group("y"))
        m = re.match(r"^(?P<mapname>[^_]+)_(?P<x>\d+)_(?P<y>\d+)$", stem, re.IGNORECASE)
        if m:
            return int(m.group("x")), int(m.group("y"))
        return None
    # Default: XY.
    for pat in _TILE_RE:
        m = pat.match(stem)
        if m:
            return int(m.group("x")), int(m.group("y"))
    return None


def _discover_checkpoint() -> Path:
    candidates: list[tuple[float, Path]] = []
    for run_dir in V24_VALIDATION_ROOT.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("v24_minimap_only"):
            continue
        ckpt = run_dir / "stage_a.pt"
        if ckpt.exists():
            candidates.append((ckpt.stat().st_mtime, ckpt))
    if not candidates:
        raise FileNotFoundError(
            f"no minimap-only checkpoint found under {V24_VALIDATION_ROOT}. "
            f"Train one first."
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def _load_minimap(path: Path) -> np.ndarray:
    with Image.open(path) as src:
        rgb = src.convert("RGB")
        if rgb.size != (256, 256):
            rgb = rgb.resize((256, 256), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _write_quilt_obj(
    height: np.ndarray,
    tex_rgb: np.ndarray,
    obj_path: Path,
    world_x: float,
    world_y: float,
    tile_size: float = TILE_SIZE,
) -> None:
    """Write one OBJ in world space, textured, with face indices that match
    the per-vertex UV grid.

    Both the heightmap and the texture stay in the original (un-flipped)
    image orientation so they line up. The OBJ writer's V-flip handles
    the standard image-Y vs world-Y convention.
    """
    h, w = height.shape
    rows, cols = h - 1, w - 1
    lines: list[str] = ["mtllib terrain.mtl", "usemtl terrain", ""]
    # Vertices: world-space position, Z = prior height.
    for y in range(h):
        for x in range(w):
            wx = world_x + (x / cols) * tile_size
            wy = world_y + (y / rows) * tile_size
            wz = float(height[y, x])
            lines.append(f"v {wx:.4f} {wy:.4f} {wz:.4f}")
    lines.append("")
    # Texture coordinates: image-Y top-down, OBJ V bottom-up -> flip.
    for y in range(h):
        for x in range(w):
            u = x / cols
            v = 1.0 - (y / rows)
            lines.append(f"vt {u:.4f} {v:.4f}")
    lines.append("")
    # Faces: two triangles per cell.
    for y in range(rows):
        for x in range(cols):
            v00 = y * w + x + 1
            v10 = y * w + (x + 1) + 1
            v01 = (y + 1) * w + x + 1
            v11 = (y + 1) * w + (x + 1) + 1
            lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
            lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")
    obj_path.write_text("\n".join(lines), encoding="utf-8")
    # Per-OBJ .mtl in the same flat folder as the OBJ. Each tile gets
    # its own mtl so the viewer's texture lookup does not collide
    # across the quilt. (Writing all mtls to a single `terrain.mtl`
    # made every tile point at the last-written tile's texture.)
    mtl_path = obj_path.parent / f"{obj_path.stem}.mtl"
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl terrain",
                f"map_Kd {obj_path.stem}.png",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ns 0.0",
                "d 1.0",
            ]
        ),
        encoding="utf-8",
    )


def _align_tile_boundaries(
    per_tile_heights: dict[tuple[int, int], np.ndarray],
    seam_width: int = 0,
) -> dict[tuple[int, int], np.ndarray]:
    """Align each tile's right/bottom **1-pixel shared border** with its
    east/south neighbour, so the seam vertices match exactly when the
    tiles are placed side-by-side in a 3D viewer.

    Tile (ty, tx) and tile (ty, tx+1) share exactly the 1-pixel column at
    world X = tx+1 * tile_size — A's ``heightmap[:, 256]`` and B's
    ``heightmap[:, 0]`` are the same point in world space. Setting
    both to their mean produces a perfectly continuous surface at the
    seam, with no visible wide "border band" anywhere on the tiles.

    ``seam_width`` (default 0) optionally widens the averaging to
    ``seam_width`` interior columns/rows on each side of the seam,
    producing a softer local transition at the cost of a slightly
    visible smoothing band. Most users want ``seam_width=0`` for a
    clean tile-by-tile quilt.
    """
    if not per_tile_heights:
        return {}
    aligned = {k: v.copy() for k, v in per_tile_heights.items()}
    # 1-pixel shared border alignment (always on).
    for (ty, tx), h in aligned.items():
        east = per_tile_heights.get((ty, tx + 1))
        south = per_tile_heights.get((ty + 1, tx))
        if east is not None:
            shared = (h[:, -1] + east[:, 0]) * 0.5
            h[:, -1] = shared
            aligned[(ty, tx + 1)][:, 0] = shared
        if south is not None:
            shared = (h[-1, :] + south[0, :]) * 0.5
            h[-1, :] = shared
            aligned[(ty + 1, tx)][0, :] = shared
    # Optional wider seam smoothing (off by default — produces a visible
    # soft band, which is what the user reported as "weird borders").
    if seam_width > 0:
        for (ty, tx), h in aligned.items():
            east = per_tile_heights.get((ty, tx + 1))
            south = per_tile_heights.get((ty + 1, tx))
            if east is not None:
                h_east = east
                # Average interior columns only (the shared 1-pixel
                # column is already aligned; don't widen the seam).
                lo_a = max(0, 256 - seam_width)
                lo_b = min(256, seam_width)
                h[:, lo_a:256] = (h[:, lo_a:256] + h_east[:, 1:lo_b + 1]) * 0.5
                aligned[(ty, tx + 1)][:, 1:lo_b + 1] = h[:, lo_a:256]
            if south is not None:
                h_south = south
                lo_a = max(0, 256 - seam_width)
                lo_b = min(256, seam_width)
                h[lo_a:256, :] = (h[lo_a:256, :] + h_south[1:lo_b + 1, :]) * 0.5
                aligned[(ty + 1, tx)][1:lo_b + 1, :] = h[lo_a:256, :]
    return aligned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="folder of PNG minimaps (NEVER written to)")
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="flat folder for the quilt OBJs + PNGs. Default: "
                             "the repo root's output/v24_quilt/<input-basename>/ "
                             "(always outside the input). The script refuses to "
                             "run if you pass a path inside the input.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="override the auto-discovered minimap-only Stage A checkpoint")
    parser.add_argument("--tile-size", type=float, default=TILE_SIZE,
                        help="world-space tile size (default 533.333 = one WoW tile)")
    parser.add_argument("--height-scale", type=float, default=1.0,
                        help="OBJ vertex-Z multiplier (default 1.0)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--naming", choices=["xy", "yx"], default="xy",
                        help="filename convention: 'xy' (default) treats the "
                             "two numbers in the file stem as X then Y; 'yx' "
                             "treats them as Y then X (common in legacy "
                             "mdxviewer captures and aligned-grid folders)")
    parser.add_argument("--flip-x", action="store_true",
                        help="flip the heightmap along the X axis at load time. "
                             "Use this if the mesh opens X-mirrored in your viewer "
                             "vs the source minimap texture.")
    parser.add_argument("--flip-y", action="store_true",
                        help="flip the heightmap along the Y axis at load time. "
                             "Use this if the mesh opens Y-mirrored in your viewer "
                             "vs the source minimap texture.")
    parser.add_argument("--no-align", action="store_true",
                        help="disable the 1-pixel shared-border edge alignment. "
                             "Each tile is written with its raw predicted heightmap. "
                             "Use this if you see a visible 'weird border' at every "
                             "256-pixel tile boundary; the trade-off is that adjacent "
                             "tiles' boundary vertices will not be welded.")
    parser.add_argument("--flip-z", action="store_true",
                        help="invert the heightmap (multiply by -1). Use this if "
                             "the model's prior is upside-down vs your viewer's "
                             "Y-up convention; the texture is unaffected, only the "
                             "Z values are inverted.")
    parser.add_argument("--seed", type=int, default=94)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"input dir not found: {args.input_dir}")
    # Default output lands in the repo-root output/ folder under
    # `v24_quilt/<input_dir_basename>/`. The repo root is two parents
    # up from the script dir (data-harvester/scripts/.. -> wow-viewer/).
    if args.output_dir is None:
        repo_root = SCRIPT_DIR.parent.parent
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", args.input_dir.name)
        args.output_dir = (repo_root / "output" / "v24_quilt" / safe_stem).resolve()
    # Hard safety: never write inside the input dir.
    input_resolved = args.input_dir.resolve()
    output_resolved = args.output_dir.resolve()
    try:
        output_resolved.relative_to(input_resolved)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"refusing to run: --output-dir {args.output_dir} is inside "
            f"--input-dir {args.input_dir}. Pick an --output-dir outside "
            f"the input."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(p for p in args.input_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not pngs:
        raise FileNotFoundError(f"no PNG/JPG files in {args.input_dir}")
    print(f"quilt: {len(pngs)} PNGs from {args.input_dir}")
    print(f"quilt output: {args.output_dir} (flat)")

    ckpt_path = args.checkpoint or _discover_checkpoint()
    print(f"checkpoint: {ckpt_path}")
    train_common.set_determinism(args.seed, strict=True)
    device = train_common.pick_device(args.device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model = stage_a.StageAMinimapOnly(base=ckpt["config"]["base"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    started = time.time()
    placed = 0
    unplaced: list[Path] = []
    # Two-pass: predict all heights, then align seams across the quilt,
    # then write the OBJs. This way the boundary vertices of every
    # adjacent pair of tiles match (the spiky-bits-along-boundaries bug
    # the user reported).
    parsed_pngs: list[tuple[Path, int, int]] = []
    heights: dict[tuple[int, int], np.ndarray] = {}
    for i, png in enumerate(pngs, 1):
        parsed = _parse_world_xy(png.stem, naming=args.naming)
        if parsed is None:
            unplaced.append(png)
            continue
        tx, ty = parsed
        parsed_pngs.append((png, tx, ty))

        minimap = _load_minimap(png)
        x = stage_a.build_minimap_only_input(minimap)
        with torch.no_grad():
            outer, inner = model(torch.from_numpy(x)[None].to(device))
        outer = (outer[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
        inner = (inner[0].cpu().numpy() * HEIGHT_SCALE).astype(np.float32)
        height = lattice.upsample_prior_257(outer, inner) * args.height_scale
        # Optional per-axis flip. The default is no flip (heightmap stays
        # in the source PNG's orientation); pass --flip-x / --flip-y if
        # the mesh opens mirrored in your viewer.
        if args.flip_x:
            height = np.fliplr(height).copy()
        if args.flip_y:
            height = np.flipud(height).copy()
        if args.flip_z:
            height = -height
        heights[(ty, tx)] = height
        if i % 50 == 0 or i == len(pngs):
            elapsed = time.time() - started
            eta = elapsed / i * (len(pngs) - i)
            print(f"  [predict] {i}/{len(pngs)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                  flush=True)

    # Edge alignment across the quilt. After this, every adjacent pair
    # of tiles shares an exact-equal boundary column/row, so the
    # spiky-bits problem disappears. The user-reported "weird border"
    # complaint is real for some view conventions; pass --no-align to
    # skip this and write each tile with its raw predicted heightmap.
    if args.no_align:
        print("edge alignment: DISABLED (--no-align); tiles written as-is",
              flush=True)
    else:
        print(f"aligning 1-pixel shared borders across {len(heights)} placed tiles ...",
              flush=True)
        heights = _align_tile_boundaries(heights, seam_width=0)

    for png, tx, ty in parsed_pngs:
        world_x = tx * args.tile_size
        world_y = ty * args.tile_size
        height = heights[(ty, tx)]
        out_obj = args.output_dir / f"{png.stem}.obj"
        out_png = args.output_dir / f"{png.stem}.png"
        with Image.open(png) as src:
            tex = src.convert("RGB")
            if args.flip_x:
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
            if args.flip_y:
                tex = tex.transpose(Image.FLIP_TOP_BOTTOM)
            tex = tex.resize((256, 256), Image.Resampling.LANCZOS)
        tex.save(str(out_png))
        _write_quilt_obj(height, np.asarray(tex), out_obj, world_x, world_y,
                         tile_size=args.tile_size)
        placed += 1

    print()
    print(f"done: {placed} OBJs in {args.output_dir} "
          f"(unplaced={len(unplaced)}, could not parse world XY from stem)")
    if unplaced:
        print("unplaced files (first 5):")
        for p in unplaced[:5]:
            print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
