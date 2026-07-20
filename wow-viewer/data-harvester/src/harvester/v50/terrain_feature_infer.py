"""Spec 115 deployment inference: minimap tile(s) -> generated terrain-feature map.

Mirrors ``direct_geometry_infer.py``'s contract exactly: loose 256x256 RGB tiles in, generated
signal out, ``--write`` gate, and a manifest binding every input hash to the checkpoint hash and
output hash (FR-010 auditability).

This is the model's whole deployment story: RGB in, feature map out, no ground truth anywhere. It
therefore runs unchanged on arbitrary images with no client backing -- which is exactly how the
out-of-distribution gate (the `ek.jpg` tiles that exposed roads-being-read-as-hills) is evaluated.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.v50.direct_geometry_infer import (
    InferenceContractError,
    discover_tiles,
    load_tile_rgb,
)
from harvester.v50.model_stage_contract import sha256_file
from harvester.v50.terrain_feature_labels import CLASS_COUNT, FAMILY_NAMES, TAXONOMY_REVISION
from harvester.v50.terrain_feature_model import (
    TERRAIN_FEATURE_ARCHITECTURE_ID,
    build_terrain_feature_model,
)

INPUT_SIZE = 256

# Stable per-family review colours (unknown grey, terrain green, road orange, water blue,
# structure purple). Orange is deliberately the most conspicuous: road is the class under test.
FAMILY_COLORS = np.asarray(
    [
        [128, 128, 128],
        [64, 148, 74],
        [244, 140, 30],
        [56, 116, 214],
        [148, 92, 196],
    ],
    dtype=np.uint8,
)


def load_terrain_feature_checkpoint(checkpoint_path: Path, *, device: str):
    """Load a classifier checkpoint; refuse architecture or taxonomy drift."""
    import torch

    if not checkpoint_path.is_file():
        raise InferenceContractError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = checkpoint.get("model_variant")
    if variant != TERRAIN_FEATURE_ARCHITECTURE_ID:
        raise InferenceContractError(
            f"checkpoint model_variant {variant!r} != {TERRAIN_FEATURE_ARCHITECTURE_ID!r}"
        )
    recorded_revision = checkpoint.get("taxonomy_revision")
    if recorded_revision != TAXONOMY_REVISION:
        raise InferenceContractError(
            f"checkpoint taxonomy revision {recorded_revision!r} != code {TAXONOMY_REVISION!r}; "
            "the class ordinals would not mean the same thing"
        )
    model, identity = build_terrain_feature_model(
        base=int(checkpoint.get("base", 32)),
        num_classes=int(checkpoint.get("num_classes", CLASS_COUNT)),
    )
    try:
        model.load_state_dict(checkpoint["model"])
    except (KeyError, RuntimeError) as exc:
        raise InferenceContractError(
            f"checkpoint weights do not match architecture {TERRAIN_FEATURE_ARCHITECTURE_ID!r}: {exc}"
        ) from exc
    model.eval()
    model.to(torch.device(device))
    return model, checkpoint, identity


def predict_feature_map(model, rgb: np.ndarray, *, device: str) -> np.ndarray:
    """One RGB tile -> (CLASS_COUNT, 256, 256) float32 class probabilities."""
    import torch

    tensor = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].float().cpu().numpy()
    if probabilities.shape[1:] != (INPUT_SIZE, INPUT_SIZE) or not np.isfinite(probabilities).all():
        raise InferenceContractError(
            f"model emitted an invalid feature map: shape {probabilities.shape}"
        )
    return probabilities.astype(np.float32)


def colorize(class_indices: np.ndarray) -> np.ndarray:
    return FAMILY_COLORS[np.clip(class_indices, 0, len(FAMILY_COLORS) - 1)]


def render_review_sheet(rows: list[dict], output: Path, *, title: str) -> None:
    """Fixed-scale [input | predicted classes | road probability] sheet."""
    from PIL import Image, ImageDraw, ImageFont

    if not rows:
        raise InferenceContractError("cannot render a review sheet over zero tiles")
    panel = INPUT_SIZE
    header = 44
    canvas = Image.new(
        "RGB", (panel * 3 + 16, header + panel * len(rows) + 4 * len(rows)), (245, 245, 245)
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 3), title, fill=(20, 20, 20), font=font)
    legend = "  ".join(f"{name}" for name in FAMILY_NAMES)
    draw.text((5, 20), f"classes: {legend}", fill=(60, 60, 60), font=font)
    for index, row in enumerate(rows):
        y = header + index * (panel + 4)
        canvas.paste(Image.fromarray(row["rgb"], mode="RGB"), (0, y))
        canvas.paste(Image.fromarray(colorize(row["classes"]), mode="RGB"), (panel + 8, y))
        road = np.rint(row["road_probability"] * 255.0).astype(np.uint8)
        canvas.paste(
            Image.fromarray(np.repeat(road[:, :, None], 3, axis=2), mode="RGB"), (panel * 2 + 16, y)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_inference(
    *,
    checkpoint_path: Path,
    inputs: list[Path],
    output: Path,
    device: str,
    write: bool,
    sheet_limit: int = 12,
) -> dict:
    tiles = discover_tiles(inputs)
    model, checkpoint, identity = load_terrain_feature_checkpoint(checkpoint_path, device=device)
    checkpoint_sha = sha256_file(checkpoint_path)

    manifest = {
        "schema": "v115-terrain-feature-inference-v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deployment_contract": {
            "input": "minimap RGB 256x256",
            "output_signal": "terrain_feature_map_256",
            "taxonomy_revision": TAXONOMY_REVISION,
            "families": list(FAMILY_NAMES),
            "ground_truth_inputs": "none",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "model_variant": checkpoint.get("model_variant"),
            "architecture": identity["architecture"],
            "epoch": int(checkpoint.get("epoch", 0)),
        },
        "device": device,
        "tile_count": len(tiles),
        "tiles": [],
    }

    rows: list[dict] = []
    for tile_path in tiles:
        rgb = load_tile_rgb(tile_path)
        probabilities = predict_feature_map(model, rgb, device=device)
        classes = probabilities.argmax(axis=0).astype(np.uint8)
        fractions = {
            FAMILY_NAMES[family]: float(np.count_nonzero(classes == family) / classes.size)
            for family in range(CLASS_COUNT)
        }
        entry = {
            "input": str(tile_path),
            "input_sha256": sha256_file(tile_path),
            "class_fractions": fractions,
        }
        if write:
            output.mkdir(parents=True, exist_ok=True)
            npy_path = output / f"{tile_path.stem}_features.npy"
            np.save(npy_path, probabilities)
            entry["output"] = str(npy_path)
            entry["output_sha256"] = sha256_file(npy_path)
        manifest["tiles"].append(entry)
        if len(rows) < sheet_limit:
            rows.append(
                {"rgb": rgb, "classes": classes, "road_probability": probabilities[2]}
            )

    if write:
        render_review_sheet(
            rows,
            output / "review_sheet.png",
            title=(
                f"terrain-feature inference | {checkpoint.get('model_variant')} "
                f"epoch {checkpoint.get('epoch')} | {len(tiles)} tiles "
                f"(showing {len(rows)})"
            ),
        )
        (output / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 115 terrain-feature deployment inference (dry run by default)"
    )
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path, action="append",
                    help="256x256 minimap tile or folder of tiles (repeatable)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--sheet-limit", type=int, default=12)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable; use --device cpu.")
    try:
        manifest = run_inference(
            checkpoint_path=args.checkpoint,
            inputs=args.input,
            output=args.output,
            device=args.device,
            write=args.write,
            sheet_limit=args.sheet_limit,
        )
    except InferenceContractError as exc:
        raise SystemExit(str(exc)) from exc

    summary = {k: v for k, v in manifest.items() if k != "tiles"}
    print(json.dumps(summary, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to persist feature maps, review sheet, and manifest.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
