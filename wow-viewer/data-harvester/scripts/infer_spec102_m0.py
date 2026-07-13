"""Run frozen Spec 102 M0 on a PNG and deterministically clean the minimap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from harvester.spec102.m0 import M0ObjectMask, clean_minimap_with_mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer Spec 102 M0 object mask")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M0 inference is CUDA-only; CPU fallback is prohibited")

    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if checkpoint.get("schema") != "spec102-m0-checkpoint-v1" or not checkpoint["config"].get("single_output"):
        raise RuntimeError("Checkpoint is not a frozen single-output Spec 102 M0 checkpoint")
    model = M0ObjectMask(base_channels=int(checkpoint["config"]["base_channels"])).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    rgb = np.asarray(Image.open(args.input).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS))
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None].cuda() / 255.0
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
    mask = probability >= args.threshold
    cleaned = clean_minimap_with_mask(rgb, mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), "L").save(args.output_dir / "object_mask_probability.png")
    Image.fromarray((mask.astype(np.uint8) * 255), "L").save(args.output_dir / "object_mask.png")
    Image.fromarray(cleaned.astype(np.uint8), "RGB").save(args.output_dir / "cleaned_minimap.png")
    manifest = {
        "schema": "spec102-m0-inference-v1",
        "deployment_input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "forward_inputs": ["raw minimap RGB"],
        "numeric_outputs": ["object mask probability"],
        "deterministic_transform": "nearest-unmasked-pixel fill",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
