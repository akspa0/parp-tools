"""Render inspection-only raw/liquid/precise panels from a curated split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw

from harvester.spec102.m0 import PRECISE_MASK_KEY, precise_object_target_256


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Spec 102 curated rows")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="validation_map")
    parser.add_argument("--stage", choices=("m0", "h2"), default="m0")
    parser.add_argument("--liquid-source", choices=("mcnk", "mh2o", "mclq", "unified", "wl"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        row for row in manifest["rows"]
        if row["split"] == args.split and row.get(f"eligible_{args.stage}") is True
        and (args.liquid_source is None or row.get("liquid_source") == args.liquid_source)
    ][: args.count]
    group = zarr.open_group(str(args.store), mode="r")
    canvas = Image.new("RGB", (256 * 3, 256 * len(rows)), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, metadata in enumerate(rows):
        row = int(metadata["row"])
        rgb = np.asarray(group["minimap_rgb"][row], dtype=np.uint8)
        liquid = np.asarray(group["liquid_mask_256"][row], dtype=np.uint8)
        precise = precise_object_target_256(np.asarray(group[PRECISE_MASK_KEY][row]))
        canvas.paste(Image.fromarray(rgb, "RGB"), (0, index * 256))
        canvas.paste(Image.fromarray(liquid, "L").convert("RGB"), (256, index * 256))
        canvas.paste(Image.fromarray(np.clip(precise * 255.0, 0, 255).astype(np.uint8), "L").convert("RGB"), (512, index * 256))
        draw.text((4, index * 256 + 4), f"row {row} {metadata['tile_x']},{metadata['tile_y']}", fill=(255, 255, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} curated {args.split}/{args.stage} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
