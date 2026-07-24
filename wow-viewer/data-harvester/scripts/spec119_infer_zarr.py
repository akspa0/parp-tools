#!/usr/bin/env python3
"""Spec 119 — Direct Zarr Minimap Segmenter Inference Tool.

Loads frozen ObjectSegmenter checkpoint (99.21% IoU) and runs per-pixel object segmentation
directly on the `minimap_rgb_authored` array inside 0_5_3_3368-Azeroth.zarr.
Outputs the predicted 256x256 object segmentation masks directly to a Zarr dataset array.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec119.infer import infer_segmenter, load_segmenter_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Zarr Minimap Object Segmenter Inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../output/spec119/segmenter_v1/segmenter.pt"),
        help="Path to trained ObjectSegmenter checkpoint.",
    )
    parser.add_argument(
        "--zarr-store",
        type=Path,
        default=Path("../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr"),
        help="Path to input/output Zarr store containing minimap_rgb_authored.",
    )
    parser.add_argument(
        "--output-array",
        type=str,
        default="object_segmentation_mask",
        help="Zarr array name to write predicted object masks (default: object_segmentation_mask).",
    )
    parser.add_argument(
        "--output-preview",
        type=Path,
        default=Path("../output/spec119/zarr_segmentation_visual_preview.png"),
        help="Path to save visual overlay comparison PNG.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Spec 119 Direct Zarr Minimap Object Segmenter Inference ===")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Zarr Store: {args.zarr_store.resolve()}")

    if not args.checkpoint.exists() or not args.zarr_store.exists():
        print("[ERROR] Input checkpoint or Zarr store does not exist.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading segmenter model on {device}...")
    model = load_segmenter_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    zarr_grp = zarr.open_group(args.zarr_store, mode="r+")
    if "minimap_rgb_authored" not in zarr_grp:
        print("[ERROR] 'minimap_rgb_authored' array missing from Zarr store.")
        sys.exit(1)

    rgb_arr = zarr_grp["minimap_rgb_authored"]
    num_tiles, height, width, channels = rgb_arr.shape
    print(f"Loaded 'minimap_rgb_authored' array: {num_tiles} tiles of shape ({height}, {width}, {channels}).")

    # Create or overwrite object_segmentation_mask Zarr array
    print(f"Preparing output Zarr array '{args.output_array}' ({num_tiles}, {height}, {width}) uint8...")
    mask_arr = zarr_grp.require_array(
        args.output_array,
        shape=(num_tiles, height, width),
        chunks=(1, height, width),
        dtype=np.uint8,
        overwrite=True,
    )

    print(f"Running GPU segmentation inference in batches of {args.batch_size}...")

    with torch.no_grad():
        for start_idx in range(0, num_tiles, args.batch_size):
            end_idx = min(num_tiles, start_idx + args.batch_size)
            batch_np = np.asarray(rgb_arr[start_idx:end_idx])  # (B, 256, 256, 3)

            # Convert to torch tensor (B, 3, 256, 256) normalized to [0, 1]
            batch_tensor = torch.from_numpy(batch_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)

            # Forward pass
            logits = model(batch_tensor)  # (B, 1, 256, 256) or (B, 256, 256)
            if logits.dim() == 4 and logits.shape[1] == 1:
                logits = logits.squeeze(1)

            probs = logits  # ObjectSegmenter.forward() already applies sigmoid internally
            masks_np = (probs > 0.5).byte().cpu().numpy() * 255  # (B, 256, 256) uint8

            mask_arr[start_idx:end_idx] = masks_np

    print(f"\n[ZARR INFERENCE SUCCESS] Saved predicted object segmentation masks to Zarr array '{args.output_array}'.")

    # Render Visual Preview Grid comparing Minimap RGB vs Predicted Segmentation Mask
    if args.output_preview:
        from PIL import Image, ImageDraw

        idx_path = args.zarr_store / "index.parquet"
        placements_path = args.zarr_store / "placements.parquet"

        active_indices = []
        if idx_path.exists() and placements_path.exists():
            idx_tbl = pq.read_table(idx_path)
            p_tbl = pq.read_table(placements_path)
            xs = idx_tbl["tile_x"].to_pylist()
            ys = idx_tbl["tile_y"].to_pylist()
            pos_x = p_tbl["posX"].to_pylist()
            pos_y = p_tbl["posY"].to_pylist()

            placements_by_tile = {}
            import math
            for i in range(p_tbl.num_rows):
                tx = int(math.floor((17066.666666666668 - pos_x[i]) / 533.3333333333333))
                ty = int(math.floor((17066.666666666668 - pos_y[i]) / 533.3333333333333))
                placements_by_tile.setdefault((tx, ty), []).append(i)

            for row_idx, (tx, ty) in enumerate(zip(xs, ys)):
                if (tx, ty) in placements_by_tile and len(placements_by_tile[(tx, ty)]) >= 10:
                    active_indices.append((row_idx, tx, ty))

        if not active_indices:
            active_indices = [(i, 32, 32) for i in range(min(4, num_tiles))]

        canvas_w, canvas_h = 256, 256
        # 4 sample tiles, each showing: [Minimap RGB | Red Segmentation Overlay]
        grid_img = Image.new("RGB", (canvas_w * 4, canvas_h * 2), (20, 20, 20))
        draw = ImageDraw.Draw(grid_img)

        for i, (row_idx, tx, ty) in enumerate(active_indices[:4]):
            col = i
            x_rgb = col * canvas_w
            y_rgb = 0
            x_mask = col * canvas_w
            y_mask = canvas_h

            rgb_np = np.asarray(rgb_arr[row_idx])
            mask_np = np.asarray(mask_arr[row_idx])

            # Paste RGB
            rgb_pil = Image.fromarray(rgb_np)
            grid_img.paste(rgb_pil, (x_rgb, y_rgb))

            # Create Red Mask Overlay on RGB
            overlay_np = rgb_np.copy()
            red_mask = mask_np > 128
            overlay_np[red_mask, 0] = np.clip(overlay_np[red_mask, 0].astype(int) + 150, 0, 255).astype(np.uint8)
            overlay_np[red_mask, 1] = (overlay_np[red_mask, 1] * 0.4).astype(np.uint8)
            overlay_np[red_mask, 2] = (overlay_np[red_mask, 2] * 0.4).astype(np.uint8)

            overlay_pil = Image.fromarray(overlay_np)
            grid_img.paste(overlay_pil, (x_mask, y_mask))

            draw.rectangle([x_rgb, y_rgb, x_rgb + canvas_w, y_rgb + canvas_h], outline=(180, 180, 180), width=1)
            draw.rectangle([x_mask, y_mask, x_mask + canvas_w, y_mask + canvas_h], outline=(180, 180, 180), width=1)
            draw.text((x_rgb + 5, y_rgb + 5), f"Tile ({tx}, {ty}) RGB", fill=(255, 255, 255))
            draw.text((x_mask + 5, y_mask + 5), f"Tile ({tx}, {ty}) Mask", fill=(255, 255, 255))

        args.output_preview.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(args.output_preview)
        print(f"[PREVIEW SAVED] Visual segmentation comparison PNG saved to {args.output_preview.resolve()}")


if __name__ == "__main__":
    main()
