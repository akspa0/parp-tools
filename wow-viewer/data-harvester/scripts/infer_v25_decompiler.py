"""V25 single-image inference CLI (Spec 102, Slice 10).

Runs the universal decompiler on one raw RGB minimap tile — a bare PNG or a
row of a V25 store — with **no** PM4 input required (User Story 1), and writes
the predictions to a structured Zarr group store with Blosc LZ4 level-1
compression (no loose array files).

Output store arrays:

* ``minimap_rgb``       (1, 256, 256, 3) uint8 — the input tile
* ``clean_minimap_256`` (1, 256, 256, 3) uint8 — inpainted terrain-shadow map
* ``object_mask_256``   (1, 256, 256) float32 — footprint probabilities
* ``height_257``        (1, 257, 257) float32 — solved terrain heights
* ``wdl_height_33``     (1, 33, 33) float32 — WDL prior (downsampled from the
  predicted 257 mesh, so both are mathematically aligned — SC-102-004)
* ``alpha_256``         (1, 4, 256, 256) uint8 — fractal-generated MCAL maps
* ``mcly_labels``       (1, 16, 16, 4) int16 — argmax MCLY assignments
* ``mtex_probs``        (1, vocab) float32 — multi-hot texture probabilities

plus ``placements.parquet`` (predicted objects, denormalized to world units).

PM4-guided post-processing (User Story 2) is a separate, optional step: pass
``--pm4-records <dir>`` pointing at a store/directory carrying
``pm4_segments.parquet`` (pre-parsed by the C# tooling and attached via
``build_v25_dataset.py --attach-pm4-segments``).  Python never opens raw
``.pm4`` files (FR-102-402).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

DATA_HARVESTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_HARVESTER_ROOT / "src"))

from harvester.v24.train_common import pick_device  # noqa: E402
from harvester.v25.dataset import (  # noqa: E402
    V25TileSource,
    load_pm4_segment_records,
    write_prediction_store,
)
from harvester.v25.pm4_guide import V25Pm4GuideHandler  # noqa: E402
from harvester.v25.prior import WdlDownsampler  # noqa: E402


def _load_minimap_png(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _build_pipeline(checkpoint: dict, device: torch.device):
    from train_v25_decompiler import V25Pipeline

    config = checkpoint["config"]
    pipeline = V25Pipeline(
        vocab_size=int(config["vocab_size"]),
        num_classes=int(config["num_classes"]),
        max_objects=int(config["max_objects"]),
        device=device.type,
    )
    pipeline.load_state_dict(checkpoint["pipeline"])
    pipeline.to(device)
    pipeline.eval()
    return pipeline, config


def main() -> int:
    parser = argparse.ArgumentParser(description="V25 universal single-image inference (Spec 102)")
    parser.add_argument("--checkpoint", required=True, type=Path, help="trained V25 checkpoint (.pt)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--minimap-png", type=Path, help="raw RGB minimap tile PNG")
    src.add_argument("--v25-store", type=Path, help="V25 store to read the input tile from")
    parser.add_argument("--row", type=int, default=0, help="store row when using --v25-store")
    parser.add_argument("--output", required=True, type=Path, help="output Zarr prediction store")
    parser.add_argument("--device", default=None, help="cpu/cuda override")
    parser.add_argument("--exist-threshold", type=float, default=0.5,
                        help="existence probability cutoff for emitted placements")
    parser.add_argument("--pm4-records", type=Path, default=None,
                        help="directory carrying pm4_segments.parquet (pre-parsed records) "
                             "to snap placements against (optional post-processing)")
    parser.add_argument("--pm4-tile", default=None,
                        help="tile coordinate filter for PM4 records, e.g. 27_29")
    parser.add_argument("--pm4-snap-distance", type=float, default=15.0)
    args = parser.parse_args()

    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pipeline, config = _build_pipeline(checkpoint, device)
    coord_scale = float(config.get("coord_scale", 17066.0))
    rot_scale = float(config.get("rot_scale", 180.0))
    classes = list(config.get("placement_classes", ["m2", "wmo"]))

    if args.minimap_png is not None:
        minimap_u8 = _load_minimap_png(args.minimap_png)
        source_desc = str(args.minimap_png)
    else:
        source = V25TileSource(args.v25_store)
        minimap_u8 = np.asarray(source.root["minimap_rgb"][args.row])
        source_desc = f"{args.v25_store}[{args.row}]"
    print(f"input: {source_desc}", flush=True)

    minimap = torch.from_numpy(minimap_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = pipeline(minimap, prior_33=None)  # fully universal path
        h_257 = preds["h_257"].float()
        # SC-102-004: the exported WDL prior is the mathematical downsample of
        # the exported high-res mesh — aligned by construction.
        wdl_33 = WdlDownsampler()(h_257)
        obj_mask = torch.sigmoid(preds["mask_logits"]).squeeze(1).float()
        clean = preds["clean_rgb"].float().clamp(0, 1)
        alpha = preds["alpha_256"].float().clamp(0, 1)
        mcly_labels = preds["mcly_logits"].float().argmax(dim=1).to(torch.int16)
        mtex_probs = torch.sigmoid(preds["mtex_logits"]).float()

        placements = []
        p = preds["placements"]
        exist_probs = torch.sigmoid(p["exist_logits"]).squeeze(-1)[0]
        class_pred = p["class_logits"][0].argmax(dim=-1)
        for i in range(exist_probs.shape[0]):
            prob = float(exist_probs[i])
            if prob < args.exist_threshold:
                continue
            placements.append(
                {
                    "index": i,
                    "class_id": int(class_pred[i]),
                    "kind": classes[int(class_pred[i])] if int(class_pred[i]) < len(classes) else "m2",
                    "exist_prob": prob,
                    "pos_x": float(p["coords"][0, i, 0]) * coord_scale,
                    "pos_y": float(p["coords"][0, i, 1]) * coord_scale,
                    "pos_z": float(p["coords"][0, i, 2]) * coord_scale,
                    "rot_x": float(p["rotations"][0, i, 0]) * rot_scale,
                    "rot_y": float(p["rotations"][0, i, 1]) * rot_scale,
                    "rot_z": float(p["rotations"][0, i, 2]) * rot_scale,
                    "pm4_snapped": False,
                }
            )

    # Optional decoupled PM4 post-processing (never part of the network).
    if args.pm4_records is not None:
        records = load_pm4_segment_records(args.pm4_records, tile_coordinate=args.pm4_tile)
        print(f"PM4 records loaded: {len(records)}", flush=True)
        if records:
            handler = V25Pm4GuideHandler()
            guide_input = [
                {
                    "coords": [pl["pos_x"], pl["pos_y"], pl["pos_z"]],
                    "rotations": [pl["rot_x"], pl["rot_y"], pl["rot_z"]],
                    "class_id": pl["class_id"],
                    "exist_prob": pl["exist_prob"],
                    "index": pl["index"],
                }
                for pl in placements
            ]
            guided = handler.guide_placements(
                guide_input, records, snap_distance=args.pm4_snap_distance
            )
            by_index = {g["index"]: g for g in guided}
            for pl in placements:
                g = by_index.get(pl["index"])
                if g is None or "pm4_segment_idx" not in g:
                    continue
                pl["pos_x"], pl["pos_y"], pl["pos_z"] = g["coords"]
                pl["pm4_snapped"] = True
                pl["pm4_segment_idx"] = g["pm4_segment_idx"]
                if "resolved_asset_name" in g:
                    pl["resolved_asset_name"] = g["resolved_asset_name"]
                    pl["match_confidence"] = g["match_confidence"]

    out = write_prediction_store(
        args.output,
        predictions={
            "minimap_rgb": minimap_u8[None, ...],
            "clean_minimap_256": (clean[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)[None, ...],
            "object_mask_256": obj_mask.cpu().numpy(),
            "height_257": h_257.cpu().numpy(),
            "wdl_height_33": wdl_33.cpu().numpy(),
            "alpha_256": (alpha.cpu().numpy() * 255.0).round().astype(np.uint8),
            "mcly_labels": mcly_labels.cpu().numpy(),
            "mtex_probs": mtex_probs.cpu().numpy(),
        },
        placements=placements,
        attrs={
            "source": source_desc,
            "checkpoint": str(args.checkpoint),
            "pm4_guided": args.pm4_records is not None,
        },
    )
    print(f"prediction store written: {out} ({len(placements)} placements)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
