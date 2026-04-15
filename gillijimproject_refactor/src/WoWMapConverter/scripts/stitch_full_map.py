from __future__ import annotations

import argparse
import torch
from pathlib import Path
from PIL import Image
import re
from v76_prediction_utils import (
    DEFAULT_MODEL_PATH,
    ensure_unique_sample_id,
    finalize_prediction_dataset,
    load_source_image,
    load_v76_model,
    predict_height_and_albedo,
    prepare_prediction_layout,
    write_prediction_sample,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TILE_SIZE = 512

# Regex to parse MapName_X_Y.png or MapName_X_Y_vcol.png
PATTERN = re.compile(r"(.+?)_(\d+)_(\d+)(?:_vcol)?\.png$")


def parse_args():
    parser = argparse.ArgumentParser(description="Run V7.6 inference over tiled map inputs and emit a structured predicted dataset.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Checkpoint path.")
    parser.add_argument("--input-dir", default="inference_input", help="Directory containing Map_X_Y.png tiles.")
    parser.add_argument("--output-dir", default="stitched_output_v7_restore", help="Prediction dataset root.")
    parser.add_argument("--no-mesh", action="store_true", help="Skip per-tile OBJ/MTL export.")
    parser.add_argument("--no-stitch", action="store_true", help="Skip stitched full-map quilt outputs.")
    return parser.parse_args()


def collect_map_groups(input_dir: Path):
    map_groups = {}
    for path in sorted(input_dir.glob("*.png")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        map_name, tile_x, tile_y = match.groups()
        map_groups.setdefault(map_name, []).append((int(tile_x), int(tile_y), path))
    return map_groups


def stitch_maps(args):
    model_path = Path(args.model_path)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    layout = prepare_prediction_layout(output_dir)

    print(f"Loading model from {model_path}...")
    model = load_v76_model(model_path, DEVICE)
    map_groups = collect_map_groups(input_dir)

    if not map_groups:
        print("No valid MapName_X_Y.png files were found.")
        return

    records = []
    stitched_outputs = []
    used_ids: set[str] = set()

    for map_name, tiles in map_groups.items():
        print(f"Stitching Map: {map_name} ({len(tiles)} tiles)")
        min_x = min(tile_x for tile_x, _, _ in tiles)
        max_x = max(tile_x for tile_x, _, _ in tiles)
        min_y = min(tile_y for _, tile_y, _ in tiles)
        max_y = max(tile_y for _, tile_y, _ in tiles)
        stitched_width = (max_x - min_x + 1) * TILE_SIZE
        stitched_height = (max_y - min_y + 1) * TILE_SIZE

        full_albedo = None if args.no_stitch else Image.new("RGB", (stitched_width, stitched_height), (0, 0, 0))
        full_height = None if args.no_stitch else Image.new("I;16", (stitched_width, stitched_height), 0)

        for tile_x, tile_y, source_path in sorted(tiles, key=lambda item: (item[1], item[0])):
            source_image, input_tensor, source_meta = load_source_image(source_path)
            height_image, albedo_image, height_map = predict_height_and_albedo(model, input_tensor, DEVICE)
            tile_name = f"{map_name}_{tile_x}_{tile_y}"
            sample_id = ensure_unique_sample_id(tile_name, used_ids)
            record = write_prediction_sample(
                layout=layout,
                sample_id=sample_id,
                source_path=source_path,
                source_kind="map_tile_batch",
                source_meta=source_meta,
                model_path=model_path,
                height_image=height_image,
                albedo_image=albedo_image,
                height_map=height_map,
                write_mesh=not args.no_mesh,
                source_tile_name=tile_name,
                source_map_name=map_name,
            )
            records.append(record)
            _ = source_image

            if full_albedo is not None and full_height is not None:
                paste_x = (tile_x - min_x) * TILE_SIZE
                paste_y = (tile_y - min_y) * TILE_SIZE
                full_albedo.paste(albedo_image, (paste_x, paste_y))
                full_height.paste(height_image, (paste_x, paste_y))

        if full_albedo is not None and full_height is not None:
            stitched_albedo_path = layout["stitched"] / f"{map_name}_full_albedo_pred.png"
            stitched_height_path = layout["stitched"] / f"{map_name}_full_height_pred.png"
            full_albedo.save(stitched_albedo_path)
            full_height.save(stitched_height_path)
            stitched_outputs.append(
                {
                    "map_name": map_name,
                    "albedo_prediction_path": str(stitched_albedo_path.relative_to(layout["root"])).replace("\\", "/"),
                    "height_prediction_path": str(stitched_height_path.relative_to(layout["root"])).replace("\\", "/"),
                    "min_tile_x": min_x,
                    "max_tile_x": max_x,
                    "min_tile_y": min_y,
                    "max_tile_y": max_y,
                }
            )

    if not records:
        print("No predictions were written.")
        return

    manifest_path = finalize_prediction_dataset(
        layout=layout,
        model_path=model_path,
        records=records,
        stitched_outputs=stitched_outputs,
    )
    print(f"Prediction dataset complete: samples={len(records)} maps={len(map_groups)}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    stitch_maps(parse_args())
