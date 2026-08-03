"""v50 deployment: minimap tile -> MiT-B0 (HF SegFormer) -> relative height -> OBJ mesh.

Chains the existing Spec 114 direct-geometry inference pipeline
(``direct_geometry_infer.py``) with the OBJ/MTL mesh writer from
``export_terrain_obj.py``.  The MiT-B0 model is a HuggingFace SegFormer
(``nvidia/mit-b0``), a real pre-trained architecture, not custom code.

Usage (from wow-viewer/data-harvester)::

    uv run python scripts/v50_deploy_height_to_mesh.py ^
        --checkpoint "path/to/mit_b0-authored-v3-deconfounded/checkpoint_best.pt" ^
        --input "path/to/minimap.png" ^
        --output-dir "./out" ^
        --device cuda

Outputs per tile::

    {tile}_relief16.png   16-bit grayscale relief (257x257, [0,65535])
    {tile}_texture.png    RGB texture tile (256x256, from minimap)
    {tile}.obj            3D mesh (world-space, 257x257 vertices)
    {tile}.mtl            Material file referencing the texture
    review_sheet.png      Side-by-side [minimap | relief] sheet
    deploy_manifest.json  Per-tile manifest (input hash, output hash, checkpoint identity)
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from harvester.v50.direct_geometry_infer import (
    InferenceContractError,
    discover_tiles,
    load_geometry_checkpoint,
    load_tile_rgb,
    predict_relief,
    relief_to_uint16,
    sha256_file,
)

TILE_SIZE = 533.333  # world units per ADT tile
HEIGHT_GRID = 257


def _height_to_obj(
    height: np.ndarray,
    texture_path: Path,
    obj_path: Path,
    tile_size: float = TILE_SIZE,
    height_scale: float = 1.0,
) -> None:
    """Write an OBJ file from a 257x257 heightmap with texture coordinates.

    ``height_scale`` converts the [0,1] relative height to world units.
    The default of 1.0 produces a terrain with ~1 world-unit relief per
    tile; adjust to match the real elevation range of the map.
    """
    h, w = height.shape
    rows = h - 1
    cols = w - 1

    mtl_name = obj_path.stem + ".mtl"
    obj_lines: list[str] = []
    obj_lines.append(f"mtllib {mtl_name}")
    obj_lines.append("usemtl terrain")
    obj_lines.append("")

    # vertices (ADT convention: X decreases with row, Y decreases with column, Z up)
    for y in range(h):
        for x in range(w):
            wx = -float(y) / rows * tile_size
            wy = -float(x) / cols * tile_size
            wz = float(height[y, x]) * height_scale
            obj_lines.append(f"v {wx:.6f} {wy:.6f} {wz:.6f}")

    obj_lines.append("")

    # texture coordinates
    for y in range(h):
        for x in range(w):
            u = x / cols
            v = 1.0 - (y / rows)
            obj_lines.append(f"vt {u:.6f} {v:.6f}")

    obj_lines.append("")

    # faces (two triangles per grid cell)
    for y in range(rows):
        for x in range(cols):
            v00 = y * w + x + 1
            v10 = y * w + (x + 1) + 1
            v01 = (y + 1) * w + x + 1
            v11 = (y + 1) * w + (x + 1) + 1
            obj_lines.append(f"f {v00}/{v00} {v10}/{v10} {v01}/{v01}")
            obj_lines.append(f"f {v10}/{v10} {v11}/{v11} {v01}/{v01}")

    obj_path.write_text("\n".join(obj_lines), encoding="utf-8")

    # MTL file
    mtl_path = obj_path.parent / mtl_name
    mtl_lines = [
        "newmtl terrain",
        f"map_Kd {texture_path.name}",
        "Ka 1.0 1.0 1.0",
        "Kd 1.0 1.0 1.0",
        "Ns 0.0",
        "d 1.0",
    ]
    mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")


def _render_review_sheet(rows: list[dict], output: Path, *, title: str) -> None:
    """Fixed-scale [input | relief] + [OBJ preview] sheet."""
    from PIL import ImageDraw, ImageFont

    panel = 256
    header = 40
    canvas = Image.new("RGB", (panel * 3 + 18, header + panel * len(rows) + 4 * len(rows)), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 3), title, fill=(20, 20, 20), font=font)
    draw.text((5, 22), "Minimap", fill=(30, 30, 30), font=font)
    draw.text((panel + 17, 22), "Relief [0,1]", fill=(30, 30, 30), font=font)
    draw.text((panel * 2 + 22, 22), "Relief colour", fill=(30, 30, 30), font=font)
    for index, row in enumerate(rows):
        y = header + index * (panel + 4)
        rgb_image = Image.fromarray(row["rgb"], mode="RGB").resize((panel, panel), Image.Resampling.NEAREST)
        relief8 = np.rint(row["relief"] * 255.0).astype(np.uint8)
        relief_image = Image.fromarray(np.repeat(relief8[:, :, None], 3, axis=2), mode="RGB")
        # Colour relief (dark blue -> tan -> white)
        values = row["relief"]
        lo, hi = float(values.min()), float(values.max())
        scale = max(hi - lo, 1.0)
        t = np.clip((values - lo) / scale, 0.0, 1.0)
        stops = np.asarray([[18, 34, 70], [42, 112, 80], [156, 145, 86], [238, 236, 220]], dtype=np.float32)
        pos = t * (len(stops) - 1)
        lower = np.floor(pos).astype(np.int32)
        upper = np.minimum(lower + 1, len(stops) - 1)
        blend = (pos - lower)[..., None]
        colour8 = np.clip(stops[lower] * (1.0 - blend) + stops[upper] * blend, 0, 255).astype(np.uint8)
        colour_image = Image.fromarray(np.repeat(relief8[:, :, None], 3, axis=2), mode="RGB")
        # Actually just use the colour map
        colour_image = Image.fromarray(colour8, mode="RGB")

        canvas.paste(rgb_image, (0, y))
        canvas.paste(relief_image, (panel + 12, y))
        canvas.paste(colour_image, (panel * 2 + 18, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="v50 minimap-to-height-to-mesh deploy (HF SegFormer MiT-B0)"
    )
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="mit_b0_regression checkpoint (checkpoint_best.pt)")
    ap.add_argument("--input", required=True, type=Path, action="append",
                    help="256x256 minimap tile or folder (repeatable)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--height-scale", type=float, default=50.0,
                    help="multiply [0,1] relative height by this to get world Z (default 50.0)")
    ap.add_argument("--tile-size", type=float, default=TILE_SIZE,
                    help="world-space tile extent (default 533.333)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--in-channels", type=int, default=3,
                    help="model input channels (3 for RGB-only, 8 for deconfounded)")
    args = ap.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable; use --device cpu.")

    # Resolve tiles.
    tiles = discover_tiles(args.input)
    if not tiles:
        raise SystemExit("no valid 256x256 minimap tiles found in inputs")

    # Load model.
    print(f"loading checkpoint {args.checkpoint}...", flush=True)
    model, checkpoint, identity = load_geometry_checkpoint(
        args.checkpoint, device=args.device, in_channels=args.in_channels
    )
    variant = checkpoint.get("model_variant", "?")
    print(f"model: {variant} | params: {identity.get('parameter_count', '?')} | "
          f"epoch: {checkpoint.get('epoch', '?')} | val_mae: {checkpoint.get('val_mae', '?'):.6f}",
          flush=True)

    # Run inference and export.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    rows: list[dict] = []

    for tile_path in tiles:
        rgb = load_tile_rgb(tile_path)
        relief = predict_relief(model, rgb, device=args.device)
        # shape: (257, 257) float32 in [0, 1]

        # Save relief PNG.
        relief_path = args.output_dir / f"{tile_path.stem}_relief16.png"
        Image.fromarray(relief_to_uint16(relief)).save(relief_path)

        # Save texture PNG (256x256 from minimap).
        tex_path = args.output_dir / f"{tile_path.stem}_texture.png"
        Image.fromarray(rgb, mode="RGB").save(tex_path)

        # Export OBJ + MTL.
        obj_path = args.output_dir / f"{tile_path.stem}.obj"
        _height_to_obj(relief, tex_path, obj_path,
                       tile_size=args.tile_size, height_scale=args.height_scale)

        entry = {
            "input": str(tile_path),
            "input_sha256": sha256_file(tile_path),
            "relief_png": str(relief_path),
            "relief_sha256": sha256_file(relief_path),
            "texture_png": str(tex_path),
            "obj": str(obj_path),
            "relief_min": float(relief.min()),
            "relief_max": float(relief.max()),
            "relief_mean": float(relief.mean()),
            "height_scale": args.height_scale,
            "tile_size": args.tile_size,
        }
        manifest.append(entry)
        rows.append({"rgb": rgb, "relief": relief})
        print(f"  {tile_path.name} -> {obj_path.name}  "
              f"relief [{relief.min():.3f}, {relief.max():.3f}]", flush=True)

    # Review sheet.
    review_path = args.output_dir / "review_sheet.png"
    _render_review_sheet(rows, review_path, title=(
        f"v50 direct geometry deploy | {variant} epoch {checkpoint.get('epoch')} | "
        f"{len(tiles)} tiles | height_scale={args.height_scale}"
    ))

    # Manifest.
    deploy_manifest = {
        "schema": "v50-height-to-mesh-deploy-v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "model_variant": variant,
            "architecture": identity.get("architecture", variant),
            "epoch": int(checkpoint.get("epoch", 0)),
            "val_mae": float(checkpoint.get("val_mae", float("nan"))),
        },
        "device": args.device,
        "height_scale": args.height_scale,
        "tile_size": args.tile_size,
        "tiles": manifest,
    }
    (args.output_dir / "deploy_manifest.json").write_text(
        json.dumps(deploy_manifest, indent=2), encoding="utf-8"
    )

    print(f"[DONE] {len(tiles)} tiles -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())