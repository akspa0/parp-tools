"""Spec 097 — split a single composite image into named tile PNGs.

The user's pipeline gives a single big image aligned to the 64x64 tile
grid (e.g. a hand-rendered pre-alpha minimap quilt). This tool splits
that image into the individual 256x256 tile PNGs the V24 model expects
as input, with the ``tile_X_Y.png`` naming the inference scripts parse.

Output names default to ``tile_X_Y.png`` (XY convention); pass
``--naming yx`` for the legacy ``tile_Y_X.png`` convention.

Usage:
    uv run python scripts/v24_split_image.py \\
        --image path/to/composite.png \\
        [--output-dir wow-viewer/output/v24_tiles/<basename>] \\
        [--grid-cols 64] [--grid-rows 64] \\
        [--tile-size 256] \\
        [--naming xy] \\
        [--x-offset 0] [--y-offset 0]

The composite is split at (col*tile_size, row*tile_size) for
(col, row) in [0, grid_cols) x [0, grid_rows). The output_dir is the
repo root's ``output/v24_tiles/<image-basename>/`` by default; the
script refuses to write inside the image's directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCRIPT_DIR = Path(__file__).resolve().parent


def _refuse_inside(image_path: Path, output_path: Path) -> None:
    """Hard safety: refuse to write outputs inside the image's directory."""
    image_resolved = image_path.resolve().parent
    output_resolved = output_path.resolve()
    try:
        output_resolved.relative_to(image_resolved)
    except ValueError:
        return
    raise ValueError(
        f"refusing to run: --output-dir {output_path} is inside the image's "
        f"directory {image_path.parent}. Pick an --output-dir outside."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path,
                        help="path to the composite image (e.g. a 16384x16384 minimap quilt)")
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="output folder for the split tile PNGs. Default: "
                             "the repo root's output/v24_tiles/<image-basename>/. "
                             "The script refuses to write inside the image's dir.")
    parser.add_argument("--grid-cols", type=int, default=None,
                        help="number of tile columns in the composite. "
                             "Default: auto-computed from the image width and "
                             "--tile-size. Pass explicitly if the composite is "
                             "padded (e.g. has a border) and you want fewer tiles.")
    parser.add_argument("--grid-rows", type=int, default=None,
                        help="number of tile rows in the composite. Default: "
                             "same as the auto-computed --grid-cols.")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="size of one tile in pixels (default 256). For a "
                             "low-res 8192x8192 composite, the default gives a "
                             "32x32 grid; for 16384x16384 it gives 64x64.")
    parser.add_argument("--naming", choices=["xy", "yx"], default="xy",
                        help="filename convention: 'xy' (default) -> tile_X_Y.png; "
                             "'yx' -> tile_Y_X.png")
    parser.add_argument("--x-offset", type=int, default=0,
                        help="x pixel offset of the first tile in the composite (default 0)")
    parser.add_argument("--y-offset", type=int, default=0,
                        help="y pixel offset of the first tile in the composite (default 0)")
    parser.add_argument("--max-black-fraction", type=float, default=1.0,
                        help="skip tiles whose fraction of pixels below --black-threshold "
                             "exceeds this (default 1.0 = no skip; 0.5 = skip if more "
                             "than half the tile is black). Saves disk + downstream "
                             "inference time on partial maps.")
    parser.add_argument("--black-threshold", type=int, default=8,
                        help="per-channel pixel value below this counts as 'black' "
                             "for the --max-black-fraction check (default 8). A pixel is "
                             "black only if ALL three RGB channels are below the threshold, "
                             "so JPEG-style 1-channel noise doesn't trigger a skip.")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"image not found: {args.image}")

    # Auto-compute grid from image size when not given. This is the
    # common case for low-res / high-res composites: the user only
    # needs to pass the image path and the tile size, and the script
    # figures out how many tiles fit.
    img = Image.open(args.image)
    width, height = img.size
    img.close()
    if args.grid_cols is None:
        args.grid_cols = width // args.tile_size
    if args.grid_rows is None:
        args.grid_rows = args.grid_rows or args.grid_cols
        # If the user gave only --grid-cols, default rows to the same.
        if args.grid_rows == 0:
            args.grid_rows = height // args.tile_size

    if args.output_dir is None:
        repo_root = SCRIPT_DIR.parent.parent
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", args.image.stem)
        args.output_dir = (repo_root / "output" / "v24_tiles" / safe_stem).resolve()
    else:
        args.output_dir = args.output_dir.resolve()
    _refuse_inside(args.image, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    width, height = img.size
    expected_w = args.x_offset + args.grid_cols * args.tile_size
    expected_h = args.y_offset + args.grid_rows * args.tile_size
    # If the composite is smaller than the requested grid, upsample with
    # nearest-neighbor to fit. A 8192x8192 lo-res image still produces
    # a valid quilt, just at half the spatial resolution. The user's
    # pipeline is calibrated for 16384x16384 and we want the splitter
    # to "just work" rather than refuse.
    if width != expected_w or height != expected_h:
        if width < expected_w or height < expected_h:
            print(f"composite {width}x{height} is smaller than the requested "
                  f"grid {expected_w}x{expected_h}; upsampling with nearest-neighbor")
        else:
            print(f"composite {width}x{height} does not exactly match the grid "
                  f"{expected_w}x{expected_h}; resizing to fit")
        img = img.resize((expected_w, expected_h), Image.Resampling.NEAREST)
    arr = np.asarray(img)

    print(f"composite: {width}x{height} from {args.image}")
    print(f"grid: {args.grid_rows} rows x {args.grid_cols} cols of {args.tile_size}x{args.tile_size} tiles")
    print(f"output: {args.output_dir}")
    print(f"naming: {args.naming} (--naming)")
    print()

    n_written = 0
    n_skipped = 0
    for row in range(args.grid_rows):
        for col in range(args.grid_cols):
            x0 = args.x_offset + col * args.tile_size
            y0 = args.y_offset + row * args.tile_size
            x1, y1 = x0 + args.tile_size, y0 + args.tile_size
            tile_arr = arr[y0:y1, x0:x1]
            # Black-skip check. A pixel is 'black' if all three channels
            # are below --black-threshold; the fraction of such pixels
            # in the tile is the black-fraction.
            if args.max_black_fraction < 1.0:
                black_mask = (tile_arr < args.black_threshold).all(axis=-1)
                black_frac = float(black_mask.mean())
                if black_frac > args.max_black_fraction:
                    n_skipped += 1
                    continue
            tile = Image.fromarray(tile_arr, mode="RGB")
            stem = f"tile_{col}_{row}" if args.naming == "xy" else f"tile_{row}_{col}"
            tile.save(str(args.output_dir / f"{stem}.png"))
            n_written += 1
    print(f"done: {n_written} tiles in {args.output_dir}"
          + (f" (skipped {n_skipped} black tiles)" if n_skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
