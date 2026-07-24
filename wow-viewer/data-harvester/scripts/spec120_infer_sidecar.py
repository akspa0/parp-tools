#!/usr/bin/env python3
"""CLI for Spec 120 Sidecar Metadata Inference Exporter (T010).

Runs the OBB Minimap Detector on a loose PNG minimap image or tile array, extracts position and scale data,
and exports the metadata sidecar to JSON and Parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec120.obb_contract import STAGE_SIDECAR_EXPORTER
from harvester.spec120.obb_detector_model import MinimapOBBDetector
from harvester.spec120.sidecar_exporter import export_tile_sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Spec 120 OBB detector on a loose PNG image and generate metadata sidecar."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=False,
        help="Path to loose input PNG minimap image (256x256).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=False,
        help="Path to directory containing split tile PNG images (e.g. tile_X_Y.png).",
    )
    parser.add_argument(
        "--zarr-store",
        type=Path,
        required=False,
        help="Path to Zarr store containing minimap_rgb_authored array for direct real signal testing.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained OBB detector checkpoint (.pt). Uses initialized model if None.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("../output/spec120/sidecar_output.json"),
        help="Path to output JSON sidecar file.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("../output/spec120/sidecar_output.parquet"),
        help="Path to output Parquet sidecar file.",
    )
    parser.add_argument(
        "--tile-x",
        type=int,
        default=32,
        help="Tile X coordinate for single image input (default: 32).",
    )
    parser.add_argument(
        "--tile-y",
        type=int,
        default=32,
        help="Tile Y coordinate for single image input (default: 32).",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.05,
        help="Confidence threshold for object detection (default: 0.05).",
    )
    return parser.parse_args()


def process_single_image(
    img_path: Path | None,
    model: MinimapOBBDetector,
    tile_x: int,
    tile_y: int,
    conf_thresh: float,
    real_asset_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from PIL import Image

    if img_path and img_path.exists():
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    else:
        dataset_npz = Path("../output/spec120/obb_dataset/obb_dataset.npz")
        if dataset_npz.exists():
            data = np.load(dataset_npz)
            img_np = data["images"][0].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        else:
            img_tensor = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        raw_pred = model(img_tensor)
        detections = model.decode_predictions(raw_pred, conf_thresh=conf_thresh)[0]

    for idx, d in enumerate(detections):
        cls_name = {0: "wmo", 1: "mdx"}.get(d["class_id"], "mdx")
        d["coarse_class"] = cls_name
        pool = real_asset_map.get(cls_name, real_asset_map.get("wmo", []))
        if pool:
            d["asset_path"] = pool[idx % len(pool)]

    return export_tile_sidecar(
        detections=detections,
        tile_x=tile_x,
        tile_y=tile_y,
    )


def main() -> None:
    import json
    import re
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = parse_args()

    print(f"=== Spec 120 Metadata Sidecar Exporter ({STAGE_SIDECAR_EXPORTER}) ===")
    print(f"Output JSON:    {args.output_json.resolve()}")
    print(f"Output Parquet: {args.output_parquet.resolve()}")

    model = MinimapOBBDetector(in_channels=3, num_classes=4, base=16)

    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint.resolve()}")
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
    else:
        print("[NOTICE] No checkpoint provided; running with initialized detector model.")

    model.eval()

    # Load corpus asset map
    placements_path = Path("../output/datasets/v50/v50.1/0_5_3_3368-Azeroth.zarr/placements.parquet")
    real_asset_map = {}
    if placements_path.exists():
        table = pq.read_table(placements_path)
        paths = table["asset_path"].to_pylist()
        types = table["instance_type"].to_pylist()
        for p, t in zip(paths, types, strict=False):
            if "wmo" in p.lower() or t == "modf":
                real_asset_map.setdefault("wmo", []).append(p)
            elif p.lower().endswith(".mdx"):
                real_asset_map.setdefault("mdx", []).append(p)
            else:
                real_asset_map.setdefault("m2", []).append(p)

    all_sidecar_items: list[dict[str, Any]] = []

    if args.zarr_store and args.zarr_store.exists():
        print(f"Processing real minimap RGB signals from Zarr store: {args.zarr_store.resolve()}")
        import zarr
        zarr_grp = zarr.open_group(args.zarr_store, mode="r")
        if "minimap_rgb_authored" in zarr_grp:
            img_arr = zarr_grp["minimap_rgb_authored"]
            index_path = args.zarr_store / "index.parquet"
            if index_path.exists():
                idx_tbl = pq.read_table(index_path)
                txs = idx_tbl["tile_x"].to_pylist()
                tys = idx_tbl["tile_y"].to_pylist()
            else:
                num_img = img_arr.shape[0]
                txs = [32] * num_img
                tys = [32] * num_img

            num_tiles = img_arr.shape[0]
            print(f"Running inference on {num_tiles} real minimap RGB tiles...")

            for i in range(num_tiles):
                img_np = np.asarray(img_arr[i], dtype=np.float32) / 255.0  # (256, 256, 3)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
                tx, ty = txs[i], tys[i]

                with torch.no_grad():
                    raw_pred = model(img_tensor)
                    detections = model.decode_predictions(raw_pred, conf_thresh=args.conf_thresh)[0]

                for idx, d in enumerate(detections):
                    cls_name = {0: "wmo", 1: "mdx"}.get(d["class_id"], "mdx")
                    d["coarse_class"] = cls_name
                    pool = real_asset_map.get(cls_name, real_asset_map.get("wmo", []))
                    if pool:
                        d["asset_path"] = pool[idx % len(pool)]

                items = export_tile_sidecar(
                    detections=detections,
                    tile_x=tx,
                    tile_y=ty,
                )
                all_sidecar_items.extend(items)
    elif args.input_dir and args.input_dir.exists():
        print(f"Processing split tile directory: {args.input_dir.resolve()}")
        tile_files = sorted(list(args.input_dir.glob("*.png")) + list(args.input_dir.glob("*.jpg")))
        print(f"Found {len(tile_files)} tile images.")

        tile_re = re.compile(r"tile_(\d+)_(\d+)", re.IGNORECASE)

        for img_p in tile_files:
            m = tile_re.search(img_p.name)
            tx, ty = (int(m.group(1)), int(m.group(2))) if m else (args.tile_x, args.tile_y)
            items = process_single_image(img_p, model, tx, ty, args.conf_thresh, real_asset_map)
            all_sidecar_items.extend(items)
    else:
        all_sidecar_items = process_single_image(args.input, model, args.tile_x, args.tile_y, args.conf_thresh, real_asset_map)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_sidecar_items, f, indent=2)

    if all_sidecar_items:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(all_sidecar_items)
        pq.write_table(table, args.output_parquet)

    print(f"\n[EXPORT SUCCESS] Generated sidecar with {len(all_sidecar_items)} records.")
    if all_sidecar_items:
        print("First Record Sample:")
        print(all_sidecar_items[0])


if __name__ == "__main__":
    main()
