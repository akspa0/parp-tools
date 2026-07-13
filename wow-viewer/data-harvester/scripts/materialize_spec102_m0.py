"""Materialize frozen M0 probabilities and deterministic cleaned minimaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import zarr
import zarr.codecs

from harvester.spec102.m0 import M0ObjectMask, clean_minimap_with_mask


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Spec 102 M0 outputs")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M0 materialization is CUDA-only; CPU fallback is prohibited")

    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if checkpoint.get("schema") != "spec102-m0-checkpoint-v1" or not checkpoint["config"].get("single_output"):
        raise RuntimeError("Checkpoint is not a frozen single-output Spec 102 M0 checkpoint")
    model = M0ObjectMask(base_channels=int(checkpoint["config"]["base_channels"])).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    source = zarr.open_group(str(args.store), mode="r")
    rgb_array = source["minimap_rgb"]
    count = rgb_array.shape[0]
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if count != len(manifest["rows"]):
        raise RuntimeError("Source row count does not match the frozen split")

    if args.output.exists():
        shutil.rmtree(args.output)
    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    output = zarr.open_group(str(args.output), mode="w")
    probability_array = output.create_array(
        "predicted_object_mask_probability_256",
        shape=(count, 256, 256),
        dtype=np.uint8,
        chunks=(8, 256, 256),
        compressors=codec,
    )
    cleaned_array = output.create_array(
        "clean_minimap_256",
        shape=(count, 256, 256, 3),
        dtype=np.uint8,
        chunks=(8, 256, 256, 3),
        compressors=codec,
    )

    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        rgb = np.asarray(rgb_array[start:stop], dtype=np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))).float().cuda() / 255.0
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probability = torch.sigmoid(model(tensor))[:, 0].float().cpu().numpy()
        encoded = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
        cleaned = np.empty_like(rgb)
        for offset in range(stop - start):
            cleaned[offset] = clean_minimap_with_mask(rgb[offset], probability[offset] >= args.threshold)
        probability_array[start:stop] = encoded
        cleaned_array[start:stop] = cleaned
        print(f"materialized {stop}/{count}", flush=True)

    output.attrs.update(
        {
            "schema": "spec102-m0-materialized-v1",
            "source_store": str(args.store.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "threshold": args.threshold,
            "cleaner": "deterministic_nearest_unmasked_pixel",
            "deployment_inputs": ["raw minimap RGB"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
