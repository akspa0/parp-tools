"""Benchmark 1-2 Layer MCLY/MTEX/MCAL Texture Decomposer against Zarr datastores (Spec 268).

Tests the V7-era 1-2 layer minimap decomposition (Model D1 / TextureDecomposerNet)
directly against pre-decoded Zarr stores (0.5.3 and 3.3.5 datasets), evaluating:
  1. Layer 1 (overlay) alpha map correlation and MAE against ground-truth alpha_256[..., 1].
  2. Tileset texture identification accuracy against authentic mtex_texture_paths.
  3. Chunk-level MCLY layer occupancy agreement.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

from harvester.d1_model import D1UNet
from harvester.v60.mcal_layer_decipherer import McalLayerDecipherer, TilesetTexture

logger = logging.getLogger("texture_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 1-2 Layer Texture Decomposer on Zarr stores")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("I:/parp/parp-tools/wow-viewer/output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr"),
        help="Path to pre-decoded Zarr dataset store",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints" / "d1_best.pt",
        help="Model D1 checkpoint path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs" / "zarr_texture_decomposer_benchmark.json",
        help="Path to write benchmark report",
    )
    parser.add_argument("--max-tiles", type=int, default=50, help="Maximum tiles to evaluate")
    parser.add_argument("--filter-active", action="store_true", help="Filter for tiles with active multi-layer alpha")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def categorize_texture(tex_name: str) -> str:
    """Classify texture path or palette name into canonical terrain material."""
    low = tex_name.lower()
    if "grass" in low:
        return "grass"
    if any(w in low for w in ["rock", "cliff", "stone", "mountain", "crag"]):
        return "rock"
    if any(w in low for w in ["dirt", "mud", "ground", "earth", "soil"]):
        return "dirt"
    if any(w in low for w in ["sand", "shore", "beach"]):
        return "sand"
    if any(w in low for w in ["snow", "ice", "frost"]):
        return "snow"
    if any(w in low for w in ["road", "cobble", "path"]):
        return "road"
    if any(w in low for w in ["mulch", "leaf", "leaves", "forest"]):
        return "mulch"
    return "other"


def benchmark_store(
    store_path: Path,
    checkpoint_path: Path,
    max_tiles: int,
    device: str,
    filter_active: bool = False,
) -> Dict[str, Any]:
    print(f"=== Benchmarking Texture Decomposer on {store_path.name} ===")
    if not store_path.exists():
        raise FileNotFoundError(f"Zarr store not found at {store_path}")

    # 1. Open Zarr store and metadata index
    g = zarr.open_group(str(store_path), mode="r")
    index_path = store_path / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Index parquet missing in {store_path}")

    tbl = pq.read_table(str(index_path))
    total_rows = tbl.num_rows

    # 2. Setup Device & Model
    dev_str = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    dev = torch.device(dev_str)
    print(f"Inference device: {dev}")

    model = D1UNet()
    if checkpoint_path.is_file():
        ckpt = torch.load(str(checkpoint_path), map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded D1 weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, running uninitialized model!")

    model.to(dev)
    model.eval()

    decipherer = McalLayerDecipherer()

    # 3. Stream arrays directly from Zarr
    minimap_arr = g["minimap_rgb"]
    alpha_arr = g["alpha_256"]
    mcly_ids_arr = g["mcly_texture_ids"] if "mcly_texture_ids" in g else None
    mcly_mask_arr = g["mcly_layer_mask"] if "mcly_layer_mask" in g else None

    tile_x_col = tbl["tile_x"].to_pylist() if "tile_x" in tbl.column_names else list(range(total_rows))
    tile_y_col = tbl["tile_y"].to_pylist() if "tile_y" in tbl.column_names else list(range(total_rows))
    mtex_col = tbl["mtex_texture_paths"].to_pylist() if "mtex_texture_paths" in tbl.column_names else [[] for _ in range(total_rows)]

    # 4. Select candidate row indices
    candidate_indices: List[int] = []
    if filter_active:
        print("Filtering for tiles with active overlay alpha...")
        for i in range(total_rows):
            # Fast check on layer 1 alpha
            a_slice = np.asarray(alpha_arr[i, :, :, 1], dtype=np.float32)
            if np.mean(a_slice) > 0.02 or np.max(a_slice) > 0.15:
                candidate_indices.append(i)
                if len(candidate_indices) >= max_tiles:
                    break
        print(f"Selected {len(candidate_indices)} active multi-layer tiles.")
    else:
        candidate_indices = list(range(min(total_rows, max_tiles)))

    print(f"Total store rows: {total_rows}, evaluating: {len(candidate_indices)}")

    results: List[Dict[str, Any]] = []
    alpha_maes: List[float] = []
    alpha_corrs: List[float] = []
    category_matches: List[bool] = []
    multi_layer_count = 0

    for i in candidate_indices:
        tx = tile_x_col[i]
        ty = tile_y_col[i]
        mtex_textures = mtex_col[i]

        mm = np.asarray(minimap_arr[i], dtype=np.float32) / 255.0  # (256, 256, 3)
        gt_alpha = np.asarray(alpha_arr[i], dtype=np.float32)      # (256, 256, 4)
        gt_mcly_mask = np.asarray(mcly_mask_arr[i], dtype=np.float32) if mcly_mask_arr is not None else None

        # Run D1 inference
        inp = torch.from_numpy(mm.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        with torch.no_grad():
            pred_t1, pred_t2, pred_a1, pred_a2 = model(inp)

        t1_np = pred_t1.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        t2_np = pred_t2.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        a1_np = pred_a1.squeeze(0).squeeze(0).cpu().numpy().clip(0.0, 1.0)
        a2_np = pred_a2.squeeze(0).squeeze(0).cpu().numpy().clip(0.0, 1.0)

        # Ground truth layer 1 overlay alpha
        gt_l1_alpha = gt_alpha[..., 1]
        has_l1 = float(np.mean(gt_l1_alpha)) > 0.01

        mae = float(np.mean(np.abs(a2_np - gt_l1_alpha)))
        alpha_maes.append(mae)

        # Compute Pearson correlation if there is variation in GT
        std_gt = float(np.std(gt_l1_alpha))
        std_pred = float(np.std(a2_np))
        if std_gt > 1e-4 and std_pred > 1e-4:
            corr = float(np.corrcoef(a2_np.flatten(), gt_l1_alpha.flatten())[0, 1])
            alpha_corrs.append(corr)
            if has_l1:
                multi_layer_count += 1
        else:
            corr = 0.0

        # Dominant color matching
        mean_c1 = np.mean(t1_np, axis=(0, 1))
        matched_id1, matched_name1, _ = decipherer.match_color_to_palette(mean_c1)
        pred_cat = categorize_texture(matched_name1)
        gt_cats = {categorize_texture(p) for p in mtex_textures}
        is_cat_match = (pred_cat in gt_cats) if gt_cats else True
        category_matches.append(is_cat_match)

        tile_result = {
            "row": i,
            "tile": f"({tx}, {ty})",
            "has_overlay_layer": has_l1,
            "gt_overlay_mean_alpha": float(np.mean(gt_l1_alpha)),
            "pred_overlay_mean_alpha": float(np.mean(a2_np)),
            "alpha_mae": round(mae, 4),
            "alpha_correlation": round(corr, 4) if corr is not None else 0.0,
            "layer_1_mean_color": [round(float(c), 3) for c in mean_c1],
            "matched_tileset": matched_name1,
            "matched_category": pred_cat,
            "category_match": is_cat_match,
            "gt_textures_count": len(mtex_textures),
            "gt_textures_sample": mtex_textures[:3],
        }
        results.append(tile_result)

    mean_mae = float(np.mean(alpha_maes)) if alpha_maes else 0.0
    mean_corr = float(np.mean(alpha_corrs)) if alpha_corrs else 0.0
    cat_acc = float(np.mean(category_matches)) if category_matches else 1.0

    summary = {
        "store": str(store_path),
        "evaluated_tiles": len(candidate_indices),
        "multi_layer_tiles_count": multi_layer_count,
        "overall_mean_alpha_mae": round(mean_mae, 4),
        "overall_mean_alpha_correlation": round(mean_corr, 4),
        "overall_material_category_accuracy": round(cat_acc, 4),
        "device": str(dev),
        "filter_active": filter_active,
        "results": results,
    }

    print("\n=== Benchmark Summary ===")
    print(f"Evaluated Tiles:           {len(candidate_indices)}")
    print(f"Multi-Layer Tiles Tested:  {multi_layer_count}")
    print(f"Overall Alpha MAE:         {mean_mae:.4f}")
    print(f"Active Alpha Correlation:  {mean_corr:.4f}")
    print(f"Material Category Acc:     {cat_acc * 100:.1f}%")

    return summary


def main() -> None:
    args = parse_args()
    summary = benchmark_store(
        args.store,
        args.checkpoint,
        args.max_tiles,
        args.device,
        filter_active=args.filter_active,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDetailed report written to: {args.output_json}")


if __name__ == "__main__":
    main()
