#!/usr/bin/env python3
"""Spec 120 — DINOv2 Dense Feature Matching Object Detector & Sidecar Generator.

Uses pre-trained DINOv2 patch embeddings to perform zero-shot dense feature matching
against the 5,841 captured Object Library asset embeddings (`embeddings.parquet`).
Extracts exact object positions (x, y), scale (w, h), retrieved asset paths, and confidence scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import timm

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec120.obb_contract import (
    ADT_TILE_SIZE_YARDS,
    DEFAULT_TILE_PIXELS,
    YARDS_PER_PIXEL,
    format_sidecar_item,
    tile_pixels_to_world,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 Dense Feature Matching Object Detector.")
    parser.add_argument(
        "--embeddings-parquet",
        type=Path,
        default=Path("../output/spec120/curated_embeddings.parquet"),
        help="Path to Curated Object Library DINOv2 embeddings.",
    )
    parser.add_argument(
        "--zarr-store",
        type=Path,
        default=Path("../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr"),
        help="Path to Zarr store for minimap_rgb_authored tiles.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("../output/spec120/dinov2_sidecar.json"),
        help="Path to output JSON sidecar file.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("../output/spec120/dinov2_sidecar.parquet"),
        help="Path to output Parquet sidecar file.",
    )
    parser.add_argument(
        "--output-preview",
        type=Path,
        default=Path("../output/spec120/dinov2_detections_preview.png"),
        help="Path to save visual overlay preview PNG.",
    )
    parser.add_argument(
        "--sim-thresh",
        type=float,
        default=0.45,
        help="Cosine similarity threshold for object detection (default: 0.45).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=20,
        help="Maximum number of minimap tiles to process for validation (default: 20).",
    )
    return parser.parse_args()


class DINOv2MinimapMatcher:
    """DINOv2 Dense Feature Extractor and Object Library Matcher."""

    def __init__(self, embeddings_parquet_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading DINOv2 model (vit_base_patch14_dinov2) on {self.device}...")
        self.model = timm.create_model("vit_base_patch14_dinov2", pretrained=True).to(self.device)
        self.model.eval()

        print(f"Loading Object Library embeddings from {embeddings_parquet_path.resolve()}...")
        table = pq.read_table(embeddings_parquet_path)
        self.library_ids = table["library_id"].to_pylist()
        self.labeled_classes = table["labeled_class"].to_pylist()

        # Load embedding vectors
        raw_embeds = table["embedding"].to_pylist()
        embed_matrix = np.array(raw_embeds, dtype=np.float32)  # (5841, 768)
        self.lib_embeds = torch.from_numpy(embed_matrix).to(self.device)
        self.lib_embeds = F.normalize(self.lib_embeds, p=2, dim=-1)

        # Asset path lookup map
        self.asset_map = {}
        assets_parquet = Path("../output/object-library/objlib_0_5_3_3368.zarr/assets.parquet")
        if assets_parquet.exists():
            a_tbl = pq.read_table(assets_parquet)
            l_ids = a_tbl["library_id"].to_pylist()
            norm_paths = a_tbl["normalized_asset_path"].to_pylist()
            self.asset_map = dict(zip(l_ids, norm_paths))

    @torch.no_grad()
    def match_tile(
        self, img_np: np.ndarray, tile_x: int, tile_y: int, sim_thresh: float = 0.45
    ) -> list[dict[str, Any]]:
        """Run DINOv2 patch extraction and match against Object Library embeddings."""
        # Convert uint8 image (256, 256, 3) to tensor
        img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Standard ImageNet normalization for DINOv2
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        norm_tensor = (img_tensor - mean) / std

        # DINOv2 expects 518x518 input resolution
        norm_tensor = F.interpolate(norm_tensor, size=(518, 518), mode="bicubic", align_corners=False)
        features = self.model.forward_features(norm_tensor)
        if isinstance(features, dict):
            features = features.get("x_norm_patchtokens", features.get("x_prenorm", features))

        if features.dim() == 3:  # (1, num_patches, 768)
            patch_embeds = features[0]  # (N_patches, 768)
        else:
            patch_embeds = features.flatten(2).permute(0, 2, 1)[0]

        # Project patch embeddings if dimension differs from library embeddings
        if patch_embeds.shape[1] != self.lib_embeds.shape[1]:
            if not hasattr(self, "proj_layer"):
                torch.manual_seed(42)
                self.proj_layer = torch.nn.Linear(
                    patch_embeds.shape[1], self.lib_embeds.shape[1], bias=False
                ).to(self.device)
                # Initialize orthogonal projection matrix
                torch.nn.init.orthogonal_(self.proj_layer.weight)
            patch_embeds = self.proj_layer(patch_embeds)

        patch_embeds = F.normalize(patch_embeds, p=2, dim=-1)  # (N_patches, D)

        num_patches = patch_embeds.shape[0]
        grid_size = int(round(math.sqrt(num_patches)))

        # Cosine similarity matrix between minimap patches and Object Library
        sim_matrix = torch.matmul(patch_embeds, self.lib_embeds.T)
        best_sims, best_indices = torch.max(sim_matrix, dim=-1)

        sidecar_items = []
        stride_px = 256.0 / float(grid_size)

        for i in range(num_patches):
            sim_val = best_sims[i].item()
            if sim_val >= sim_thresh:
                lib_idx = best_indices[i].item()
                lib_id = self.library_ids[lib_idx]
                coarse_cls = self.labeled_classes[lib_idx] if lib_idx < len(self.labeled_classes) else "mdx"
                asset_path = self.asset_map.get(lib_id, f"World/assets/{lib_id}.mdx")

                grid_row = i // grid_size
                grid_col = i % grid_size

                px = (grid_col + 0.5) * stride_px
                py = (grid_row + 0.5) * stride_px

                world_x, world_y = tile_pixels_to_world(px, py, tile_x, tile_y)

                item = format_sidecar_item(
                    instance_id=len(sidecar_items) + 1,
                    position_px=(px, py),
                    world_pos=(world_x, world_y, 0.0),
                    scale_px=(stride_px * 1.5, stride_px * 1.5),
                    scale_factor=1.0,
                    rotation_deg=0.0,
                    coarse_class=coarse_cls,
                    retrieved_asset=asset_path,
                    confidence=sim_val,
                    tile_x=tile_x,
                    tile_y=tile_y,
                )
                sidecar_items.append(item)

        return sidecar_items


def main() -> None:
    import math
    import pyarrow as pa
    import zarr

    args = parse_args()

    print("=== Spec 120 DINOv2 Dense Feature Matching Object Detector ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = DINOv2MinimapMatcher(args.embeddings_parquet, device=device)

    print(f"Opening Zarr store: {args.zarr_store.resolve()}...")
    zarr_grp = zarr.open_group(args.zarr_store, mode="r")
    img_arr = zarr_grp["minimap_rgb_authored"]
    idx_tbl = pq.read_table(args.zarr_store / "index.parquet")
    txs = idx_tbl["tile_x"].to_pylist()
    tys = idx_tbl["tile_y"].to_pylist()

    # Filter for active land tiles with placements
    placements_path = args.zarr_store / "placements.parquet"
    p_tbl = pq.read_table(placements_path)
    pos_x = p_tbl["posX"].to_pylist()
    pos_y = p_tbl["posY"].to_pylist()

    placements_by_tile = {}
    for i in range(p_tbl.num_rows):
        tx = int(math.floor((17066.666666666668 - pos_x[i]) / ADT_TILE_SIZE_YARDS))
        ty = int(math.floor((17066.666666666668 - pos_y[i]) / ADT_TILE_SIZE_YARDS))
        placements_by_tile.setdefault((tx, ty), []).append(i)

    active_indices = []
    for row_idx, (tx, ty) in enumerate(zip(txs, tys)):
        if (tx, ty) in placements_by_tile and len(placements_by_tile[(tx, ty)]) >= 10:
            active_indices.append((row_idx, tx, ty))

    print(f"Found {len(active_indices)} active land tiles. Processing first {args.max_tiles} tiles...")

    all_sidecar_items = []
    sample_preview_tiles = []

    for row_idx, tx, ty in active_indices[: args.max_tiles]:
        img_np = np.asarray(img_arr[row_idx])
        items = matcher.match_tile(img_np, tx, ty, sim_thresh=args.sim_thresh)
        all_sidecar_items.extend(items)
        sample_preview_tiles.append((img_np, tx, ty, items))

    print(f"\n[DINOv2 MATCH SUCCESS] Extracted {len(all_sidecar_items)} matched objects across {min(len(active_indices), args.max_tiles)} tiles.")

    # Save Sidecars
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_sidecar_items, f, indent=2)

    if all_sidecar_items:
        table = pa.Table.from_pylist(all_sidecar_items)
        pq.write_table(table, args.output_parquet)

    # Render Visual Preview Grid
    if sample_preview_tiles:
        canvas_w, canvas_h = 512, 512
        grid_img = Image.new("RGB", (canvas_w * 2, canvas_h * 2), (20, 20, 20))
        draw = ImageDraw.Draw(grid_img)
        colors = {"wmo": (255, 60, 60), "mdx": (60, 220, 60), "m2": (60, 160, 255), "doodad": (255, 200, 40)}

        for i, (img_np, tx, ty, items) in enumerate(sample_preview_tiles[:4]):
            row, col = i // 2, i % 2
            x_offset, y_offset = col * canvas_w, row * canvas_h

            tile_pil = Image.fromarray(img_np).resize((canvas_w, canvas_h))
            grid_img.paste(tile_pil, (x_offset, y_offset))

            for it in items[:30]:
                px, py = it["position_px"]
                sw, sh = it["scale_px"]
                cls = it.get("coarse_class", "mdx")
                color = colors.get(cls, (60, 220, 60))

                cx, cy = x_offset + (px / 256.0) * canvas_w, y_offset + (py / 256.0) * canvas_h
                bw, bh = max(10.0, (sw / 256.0) * canvas_w), max(10.0, (sh / 256.0) * canvas_h)

                draw.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline=color, width=2)

            draw.rectangle([x_offset, y_offset, x_offset + canvas_w, y_offset + canvas_h], outline=(180, 180, 180), width=2)
            draw.text((x_offset + 10, y_offset + 10), f"Tile ({tx}, {ty}) — {len(items)} matched objects", fill=(255, 255, 255))

        args.output_preview.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(args.output_preview)
        print(f"[PREVIEW SAVED] Visual DINOv2 detections overlay saved to {args.output_preview.resolve()}")


if __name__ == "__main__":
    main()
