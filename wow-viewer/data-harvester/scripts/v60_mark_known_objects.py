#!/usr/bin/env python3
"""Mark known objects in one minimap using explicit candidate footprints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.object_marker import (  # noqa: E402
    IMAGE_SIGNAL,
    MARKER_SIGNAL,
    ObjectMarkerError,
    ObjectMarkerNet,
    build_library_gallery_inputs,
    build_marker_map,
    marker_input_tensor,
    retrieve_library_identity,
)


def _gallery_embeddings(model, images, masks, device, batch_size: int = 64) -> np.ndarray:
    import torch

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            inputs = torch.cat(
                [marker_input_tensor(image, mask) for image, mask in zip(images[start : start + batch_size], masks[start : start + batch_size], strict=True)],
                dim=0,
            ).to(device)
            outputs.append(model(inputs)["embedding"].cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark known library objects in a minimap")
    parser.add_argument("--minimap-npz", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--object-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimap-key", default=IMAGE_SIGNAL)
    parser.add_argument("--candidate-key", default="candidate_masks_256")
    parser.add_argument("--known-threshold", type=float, default=0.55)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    try:
        import torch

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        architecture = checkpoint.get("architecture", {})
        model = ObjectMarkerNet(
            base=int(architecture.get("base", 16)),
            embedding_dim=int(architecture.get("embedding_dim", 64)),
        )
        model.load_state_dict(checkpoint["model"])
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ObjectMarkerError("CUDA requested but unavailable")
        model.to(device)
        with np.load(args.minimap_npz, allow_pickle=False) as payload:
            image = np.asarray(payload[args.minimap_key], dtype=np.float32)
            candidates = np.asarray(payload[args.candidate_key], dtype=np.float32)
        if candidates.shape == (256, 256):
            candidates = candidates[None]
        if candidates.ndim != 3 or candidates.shape[1:] != (256, 256):
            raise ObjectMarkerError(f"candidate masks must have shape (N,256,256), got {candidates.shape}")
        gallery_images, gallery_masks, gallery_ids = build_library_gallery_inputs(args.object_library)
        gallery_embeddings = _gallery_embeddings(model, gallery_images, gallery_masks, device)
        candidate_records: list[dict] = []
        decisions: list[dict] = []
        with torch.no_grad():
            for index, mask in enumerate(candidates):
                outputs = model(marker_input_tensor(image, mask).to(device))
                confidence = float(torch.sigmoid(outputs["known_logit"])[0].item())
                decision = retrieve_library_identity(
                    outputs["embedding"][0].cpu().numpy(),
                    gallery_embeddings,
                    gallery_ids,
                    known_confidence=confidence,
                    known_threshold=args.known_threshold,
                    top_k=5,
                )
                decisions.append(decision)
                candidate_records.append({"candidate_id": f"candidate-{index:04d}", "mask": mask})
        marker, identity_rows = build_marker_map((256, 256), candidate_records, decisions)
        args.output.mkdir(parents=True)
        np.savez_compressed(
            args.output / "marker_map.npz",
            **{MARKER_SIGNAL: marker, "input_image": image.astype(np.float32)},
        )
        from PIL import Image

        overlay = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        for marker_id in range(1, len(identity_rows) + 1):
            color = np.asarray(
                [(marker_id * 67) % 255, (marker_id * 131) % 255, (marker_id * 193) % 255],
                dtype=np.float32,
            )
            pixels = marker == marker_id
            overlay[pixels] = np.clip(0.35 * overlay[pixels] + 0.65 * color, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(overlay, mode="RGB").save(args.output / "marker_overlay.png")
        (args.output / "identity_table.json").write_text(json.dumps(identity_rows, indent=2), encoding="utf-8")
        report = {
            "schema": "v60-object-marker-export-v1",
            "input": str(args.minimap_npz.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "gallery": {"source_library": str(args.object_library.resolve()), "count": len(gallery_ids)},
            "candidate_count": len(candidate_records),
            "accepted_count": len(identity_rows),
            "known_object_marker": MARKER_SIGNAL,
            "identity_table": "identity_table.json",
            "visual_overlay": "marker_overlay.png",
            "decisions": decisions,
        }
        (args.output / "marker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    except (FileNotFoundError, KeyError, OSError, ObjectMarkerError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
