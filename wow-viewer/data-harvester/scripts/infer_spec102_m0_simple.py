"""Spec 102 M0 inference (simple trainer) — one minimap image in, object mask out.

Proves the deployment contract: the model regenerates the object mask from a
SINGLE minimap image alone. It reads no object mask anywhere. Point it at any
minimap PNG (which obviously has no mask) or a store row (only minimap_rgb is
read) and it still produces a predicted mask + cleaned minimap.

Run from wow-viewer/data-harvester/:

    # from any minimap PNG that has no mask whatsoever:
    uv run python scripts/infer_spec102_m0_simple.py \
        --checkpoint ../output/spec102_m0_precise_full_v1.2/checkpoint_best.pt \
        --minimap some_minimap.png --output-dir ../output/m0_infer_demo

    # or from a store row (only minimap_rgb is read, never the mask):
    uv run python scripts/infer_spec102_m0_simple.py \
        --checkpoint ../output/spec102_m0_precise_full_v1.2/checkpoint_best.pt \
        --store ../output/datasets/spec102/numeric_3_3_5_full_raw_v1.zarr --row 0 \
        --output-dir ../output/m0_infer_demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image

from harvester.spec102.m0 import M0ObjectMask, clean_minimap_with_mask


def edge_channel(rgb_hw3: np.ndarray) -> np.ndarray:
    gray = rgb_hw3.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    return np.clip(np.hypot(gx, gy) / 128.0, 0.0, 1.0).astype(np.float32)


def load_rgb(args) -> tuple[np.ndarray, str]:
    if args.minimap is not None:
        img = Image.open(args.minimap).convert("RGB").resize((256, 256))
        return np.asarray(img, dtype=np.uint8), str(args.minimap)
    g = zarr.open_group(str(args.store), mode="r")
    return np.asarray(g["minimap_rgb"][int(args.row)], dtype=np.uint8), f"{args.store} row {args.row}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 102 M0 RGB-only inference (simple trainer)")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--minimap", type=Path, default=None, help="minimap PNG (no mask needed)")
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    if args.minimap is None and args.store is None:
        raise SystemExit("give either --minimap <png> or --store <zarr> --row <n>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    in_ch = int(ck.get("in_channels", 3))
    edge = bool(ck.get("edge", False))
    thr = float(ck.get("threshold", 0.5))
    model = M0ObjectMask(in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    rgb, src = load_rgb(args)  # (256,256,3) uint8 — the ONLY thing read from the source
    x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    if edge:
        x = np.concatenate([x, edge_channel(rgb)[None]], axis=0)  # still derived from RGB alone
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x))[None].to(device)).float())[0, 0].cpu().numpy()
    pred = prob >= thr

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output_dir / "input_minimap.png")
    Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8), "L").save(args.output_dir / "predicted_prob.png")
    Image.fromarray((pred * 255).astype(np.uint8), "L").save(args.output_dir / "predicted_mask.png")
    Image.fromarray(clean_minimap_with_mask(rgb, pred), "RGB").save(args.output_dir / "cleaned_minimap.png")
    print(f"[infer] source={src}")
    print(f"[infer] model in_channels={in_ch} edge={edge} threshold={thr:.2f}")
    print(f"[infer] predicted object pixels: {int(pred.sum())}/{pred.size}")
    print(f"[infer] wrote input/prob/mask/cleaned -> {args.output_dir}")
    print("[infer] only the RGB minimap image was read; no object mask was used as input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
