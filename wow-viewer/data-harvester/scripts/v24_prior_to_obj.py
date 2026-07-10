"""Spec 096 helper: turn a WDL prior NPZ (Spec 096 output) into an OBJ + MTL + texture PNG.

Loads the ``outer`` (17,17) and ``inner`` (16,16) prior grids from a prior NPZ
written by ``infer_v24_stage_a_png.py`` (or the v24 wrapper), up-samples them
to a 257x257 heightmap via the same quincunx bilinear the WDL lattice uses,
and writes a textured OBJ mesh the user can open in any 3D viewer to see
whether the prior is a sensible terrain surface.

The texture is the source PNG (bilinear-resized to 256x256) so the OBJ
displays both the predicted height field and the input minimap together.

Usage:
    uv run python scripts/v24_prior_to_obj.py \\
        --prior path/to/prior.npz \\
        --image path/to/source.png \\
        --output-dir path/to/mesh/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import lattice  # noqa: E402


def _load_height_from_prior(npz_path: Path) -> np.ndarray:
    """Read outer/inner from a Spec 096 prior NPZ and upsample to 257x257."""
    with np.load(str(npz_path)) as data:
        if "outer" not in data.files or "inner" not in data.files:
            raise ValueError(
                f"{npz_path} is not a V24 prior NPZ (missing outer/inner arrays)"
            )
        outer = np.asarray(data["outer"], dtype=np.float32)
        inner = np.asarray(data["inner"], dtype=np.float32)
    if outer.shape != (17, 17) or inner.shape != (16, 16):
        raise ValueError(
            f"unexpected shapes: outer={outer.shape}, inner={inner.shape}; "
            f"expected (17,17) and (16,16)"
        )
    return lattice.upsample_prior_257(outer, inner)


def _height_to_obj(
    height: np.ndarray,
    texture_path: Path,
    obj_path: Path,
    tile_size: float = 533.333,
    height_scale: float = 1.0,
) -> None:
    """Write an OBJ + MTL pair from a 257x257 heightmap with texture coords.

    Mirrors the helper in export_terrain_obj.py so the meshes load in the
    same viewers.
    """
    h, w = height.shape
    rows, cols = h - 1, w - 1

    mtl_name = obj_path.stem + ".mtl"
    obj_lines: list[str] = [
        f"mtllib {mtl_name}",
        "usemtl terrain",
        "",
    ]

    for y in range(h):
        for x in range(w):
            wx = (x / cols) * tile_size
            wy = (y / rows) * tile_size
            wz = float(height[y, x]) * height_scale
            obj_lines.append(f"v {wx:.6f} {wy:.6f} {wz:.6f}")

    obj_lines.append("")
    for y in range(h):
        for x in range(w):
            u = x / cols
            v = 1.0 - (y / rows)
            obj_lines.append(f"vt {u:.6f} {v:.6f}")

    obj_lines.append("")
    for y in range(rows):
        for x in range(cols):
            v00 = y * w + x + 1
            v10 = y * w + (x + 1) + 1
            v01 = (y + 1) * w + x + 1
            v11 = (y + 1) * w + (x + 1) + 1
            obj_lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
            obj_lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")

    obj_path.write_text("\n".join(obj_lines), encoding="utf-8")

    # Per-OBJ .mtl. Sharing a single `terrain.mtl` across multiple
    # OBJs in a flat folder made every tile point at the last-written
    # texture; per-OBJ mtls fix that.
    mtl_path = obj_path.parent / f"{obj_path.stem}.mtl"
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl terrain",
                f"map_Kd {texture_path.name}",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ns 0.0",
                "d 1.0",
            ]
        ),
        encoding="utf-8",
    )


def _grid_stitch_obj(
    priors: list[Path],
    images: list[Path],
    out_dir: Path,
    tile_size: float = 533.333,
    height_scale: float = 1.0,
    cols: int | None = None,
) -> dict:
    """Stitch a list of (prior, image) pairs into a single OBJ + atlas texture.

    Each tile becomes a 257x257 patch; patches are arranged in a grid whose
    column count is auto-picked unless the caller overrides. The texture
    atlas is a single PNG with each tile's 256x256 minimap placed in a row
    (or column) at the same grid layout. The OBJ references the atlas by
    per-vertex texture coordinates.
    """
    if len(priors) != len(images):
        raise ValueError(
            f"priors ({len(priors)}) and images ({len(images)}) must match in length"
        )
    n = len(priors)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Load all heights and textures.
    heights: list[np.ndarray] = []
    tile_textures: list[np.ndarray] = []
    for prior_path, image_path in zip(priors, images, strict=True):
        h = _load_height_from_prior(prior_path)
        h = np.fliplr(h).copy()  # image-X -> world-X (see single-tile path)
        heights.append(h)
        with Image.open(image_path) as src:
            tex = src.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        tile_textures.append(np.asarray(tex))

    tile_w, tile_h = 256, 256
    atlas_w = cols * tile_w
    atlas_h = rows * tile_h
    atlas = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)
    for idx, tex in enumerate(tile_textures):
        r, c = divmod(idx, cols)
        atlas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tex

    out_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = out_dir / "atlas.png"
    Image.fromarray(atlas, mode="RGB").save(str(atlas_path))

    obj_lines: list[str] = [
        f"mtllib terrain.mtl",
        "usemtl terrain",
        "",
    ]
    # Vertices: row-major across all tiles, world space (X, Y) for placement.
    v_offset = 0
    for idx, h in enumerate(heights):
        r, c = divmod(idx, cols)
        for y in range(h.shape[0]):
            for x in range(h.shape[1]):
                wx = (c * tile_size) + (x / (h.shape[1] - 1)) * tile_size
                wy = (r * tile_size) + (y / (h.shape[0] - 1)) * tile_size
                wz = float(h[y, x]) * height_scale
                obj_lines.append(f"v {wx:.6f} {wy:.6f} {wz:.6f}")

        # Texture coords (UV): each tile's 256x256 in the atlas, V flipped to
        # match OBJ's bottom-up convention.
        for y in range(256):
            for x in range(256):
                u = (c * tile_w + x) / atlas_w
                v = 1.0 - ((r * tile_h + y) / atlas_h)
                obj_lines.append(f"vt {u:.6f} {v:.6f}")

        # Faces: two triangles per grid cell.
        for y in range(256):
            for x in range(256):
                v00 = v_offset + y * 257 + x + 1
                v10 = v_offset + y * 257 + (x + 1) + 1
                v01 = v_offset + (y + 1) * 257 + x + 1
                v11 = v_offset + (y + 1) * 257 + (x + 1) + 1
                obj_lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
                obj_lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")
        v_offset += 257 * 257
        obj_lines.append("")

    (out_dir / "terrain.obj").write_text("\n".join(obj_lines), encoding="utf-8")
    (out_dir / "terrain.mtl").write_text(
        "\n".join(
            [
                "newmtl terrain",
                f"map_Kd {atlas_path.name}",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ns 0.0",
                "d 1.0",
            ]
        ),
        encoding="utf-8",
    )

    n_verts = n * 257 * 257
    n_faces = n * 256 * 256 * 2
    return {
        "n_tiles": n,
        "cols": cols,
        "rows": rows,
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "atlas": str(atlas_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=None,
                        help="path to a prior NPZ (outer/inner arrays). "
                             "Mutually exclusive with --grid-from-priors.")
    parser.add_argument("--image", type=Path, default=None,
                        help="path to the source PNG (used as the mesh texture)")
    parser.add_argument("--grid-from-priors", type=Path, default=None,
                        help="directory of prior NPZ + matching source PNG; "
                             "stitches all tiles into a single OBJ + texture atlas")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="single-tile output dir OR grid-stitch output dir")
    parser.add_argument("--tile-size", type=float, default=533.333,
                        help="world-space tile size for X/Y (default 533.333)")
    parser.add_argument("--height-scale", type=float, default=1.0,
                        help="height multiplier for vertex Z (default 1.0)")
    parser.add_argument("--grid-cols", type=int, default=None,
                        help="column count for the grid stitch (default: ceil(sqrt(n)))")
    args = parser.parse_args()

    if args.grid_from_priors is not None:
        # Grid mode: discover (prior.npz, *.png) pairs by stem.
        prior_dir = args.grid_from_priors
        if not prior_dir.is_dir():
            raise NotADirectoryError(f"--grid-from-priors is not a directory: {prior_dir}")
        priors: list[Path] = []
        images: list[Path] = []
        for prior_path in sorted(prior_dir.glob("*.prior.npz")):
            stem = prior_path.name[: -len(".prior.npz")]
            candidates = [
                prior_dir / f"{stem}.png",
                prior_dir / f"{stem}.jpg",
                prior_dir / f"{stem}.jpeg",
            ]
            img = next((p for p in candidates if p.exists()), None)
            if img is None:
                print(f"warning: no source image for {prior_path.name}, skipping")
                continue
            priors.append(prior_path)
            images.append(img)
        if not priors:
            raise FileNotFoundError(f"no usable (prior, image) pairs in {prior_dir}")
        out_dir = args.output_dir or (prior_dir / "stitched_mesh")
        info = _grid_stitch_obj(
            priors, images, out_dir,
            tile_size=args.tile_size,
            height_scale=args.height_scale,
            cols=args.grid_cols,
        )
        print(f"Exported grid mesh to {out_dir}")
        print(f"  {info['n_tiles']} tiles ({info['rows']} rows x {info['cols']} cols)")
        print(f"  terrain.obj  ({info['n_vertices']} vertices, {info['n_faces']} faces)")
        print(f"  terrain.mtl")
        print(f"  atlas.png    ({info['atlas']})")
        return 0

    # Single-tile mode.
    if not args.prior or not args.image or not args.output_dir:
        parser.error("single-tile mode requires --prior, --image, and --output-dir")
    if not args.prior.exists():
        raise FileNotFoundError(f"prior NPZ not found: {args.prior}")
    if not args.image.exists():
        raise FileNotFoundError(f"image not found: {args.image}")

    height = _load_height_from_prior(args.prior)
    # No X-flip: the heightmap and the source-PNG texture are both kept in
    # the original (un-flipped) image orientation so they line up. The OBJ
    # writer's existing V-flip handles the Y axis consistently.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save the heightmap as a PNG for downstream tools that prefer images.
    h_min, h_max = float(height.min()), float(height.max())
    h_vis = ((height - h_min) / max(h_max - h_min, 1e-6) * 255.0).astype(np.uint8)
    Image.fromarray(h_vis, mode="L").save(str(args.output_dir / "height.png"))

    # Texture = the source PNG, resized to 256x256.
    with Image.open(args.image) as src:
        tex = src.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    tex_path = args.output_dir / "texture.png"
    tex.save(str(tex_path))

    # OBJ.
    obj_path = args.output_dir / "terrain.obj"
    _height_to_obj(
        height,
        tex_path.relative_to(args.output_dir),
        obj_path,
        tile_size=args.tile_size,
        height_scale=args.height_scale,
    )

    print(f"Exported to {args.output_dir}")
    print(f"  terrain.obj  ({(257 * 257)} vertices, {(256 * 256 * 2)} faces)")
    print(f"  terrain.mtl")
    print(f"  texture.png  (256x256 source minimap)")
    print(f"  height.png   (257x257 prior upsample, world_min={h_min:.2f}, world_max={h_max:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
