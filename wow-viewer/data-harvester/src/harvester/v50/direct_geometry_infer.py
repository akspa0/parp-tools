"""Spec 114 T056: deployment inference for direct geometry (FR-015).

The deployment contract: one authored minimap tile (256x256 RGB) in, one relative-relief field
(257x257, contract ``v112.1``) out — for tiles the model never trained on, with no store, no
ground truth, and no tile identity involved. This CLI is that contract made executable:

- loads an immutable training checkpoint and refuses architecture/target-contract mismatch;
- accepts loose PNG/JPEG tiles (exactly 256x256; anything else is refused loudly, never silently
  resampled into a different contract);
- writes one 16-bit grayscale relief PNG per tile, a fixed-scale side-by-side review sheet, and a
  per-tile manifest binding input hash -> checkpoint hash -> output hash (FR-015 auditability).

Caveat recorded in every manifest: the output is RELATIVE relief per tile. Absolute world altitude
is not identifiable from one minimap (spec edge case); a future independent offset model would be
its own stage with its own proof.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.v50.direct_geometry_model import build_geometry_model
from harvester.v50.height_relative_model import TARGET_CONTRACT_VERSION
from harvester.v50.model_stage_contract import sha256_file

SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
INPUT_SIZE = 256
RELIEF_DTYPE_MAX = 65535.0


class InferenceContractError(ValueError):
    """Raised when deployment inference would violate the Spec 114 contract."""


def discover_tiles(inputs: list[Path]) -> list[Path]:
    """Resolve files/folders into a stable sorted tile list; refuse unsupported or missing paths."""
    tiles: list[Path] = []
    for item in inputs:
        if item.is_dir():
            tiles.extend(
                sorted(
                    p for p in item.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        elif item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES:
            tiles.append(item)
        else:
            raise InferenceContractError(f"unsupported or missing input: {item}")
    unique = sorted(set(tiles))
    if not unique:
        raise InferenceContractError("no decodable 256x256 minimap tiles found in the given inputs")
    return unique


def load_tile_rgb(path: Path) -> np.ndarray:
    """Load one tile as uint8 RGB, enforcing the exact 256x256 deployment contract."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (INPUT_SIZE, INPUT_SIZE, 3):
        raise InferenceContractError(
            f"deployment tiles must be exactly {INPUT_SIZE}x{INPUT_SIZE} RGB; "
            f"{path.name} decodes to {rgb.shape}"
        )
    return rgb


def load_geometry_checkpoint(checkpoint_path: Path, *, device: str, in_channels: int = 3):
    """Load a training checkpoint into its architecture; refuse identity/contract drift.

    ``in_channels`` defaults to 3 (RGB-only), matching the true deployment contract this module
    exists for. A checkpoint trained with a Spec 115/117-style ``--feature-store`` needs its real
    ``in_channels`` (3 + class_count) passed explicitly by the caller (e.g.
    ``direct_geometry_materialize.py``, which has the feature store available); this function does
    not infer it from the checkpoint on its own, so a mismatched value still fails loudly at
    ``load_state_dict`` rather than silently reshaping.
    """
    import torch

    if not checkpoint_path.is_file():
        raise InferenceContractError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = checkpoint.get("model_variant")
    if not variant:
        raise InferenceContractError("checkpoint does not record model_variant")
    if checkpoint.get("target_contract_version") != TARGET_CONTRACT_VERSION:
        raise InferenceContractError(
            f"checkpoint target contract {checkpoint.get('target_contract_version')!r} "
            f"!= {TARGET_CONTRACT_VERSION!r}"
        )
    model, identity = build_geometry_model(variant, in_channels=in_channels)
    try:
        model.load_state_dict(checkpoint["model"])
    except (KeyError, RuntimeError) as exc:
        raise InferenceContractError(
            f"checkpoint weights do not match architecture {variant!r} with in_channels="
            f"{in_channels}: {exc}"
        ) from exc
    model.eval()
    model.to(torch.device(device))
    return model, checkpoint, identity


def _run_model(model, channels_hwc: np.ndarray, *, device: str) -> np.ndarray:
    """Shared core: an already-assembled (H, W, C) float array -> one relief field in [0, 1]."""
    import torch

    tensor = torch.from_numpy(channels_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(tensor)[0].float().cpu().numpy()
    if predicted.shape != (257, 257) or not np.isfinite(predicted).all():
        raise InferenceContractError(
            f"model emitted an invalid relief field: shape {predicted.shape}"
        )
    return np.clip(predicted, 0.0, 1.0).astype(np.float32)


def predict_relief(model, rgb: np.ndarray, *, device: str) -> np.ndarray:
    """One RGB tile -> one float32 relief field in [0, 1] at the 257x257 world grid."""
    return _run_model(model, rgb.astype(np.float32) / 255.0, device=device)


def predict_relief_with_feature_map(
    model, rgb: np.ndarray, feature_map: np.ndarray, *, device: str
) -> np.ndarray:
    """RGB tile + a GENERATED (class_count, H, W) feature map, concatenated exactly as
    ``direct_geometry_train.py``'s ``RowDataset`` does, -> one relief field in [0, 1].

    Only for materialization against a store where the feature map already exists (never for the
    true RGB-only deployment contract, which stays untouched — see ``predict_relief``).
    """
    rgb_hwc = rgb.astype(np.float32) / 255.0
    feature_hwc = np.asarray(feature_map, dtype=np.float32).transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    channels = np.concatenate([rgb_hwc, feature_hwc], axis=2)
    return _run_model(model, channels, device=device)


def relief_to_uint16(relief: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(relief, 0.0, 1.0) * RELIEF_DTYPE_MAX).astype(np.uint16)


def render_review_sheet(rows: list[dict], output: Path, *, title: str) -> None:
    """Fixed-scale [input | relief] sheet; relief always stretched over the full [0,1] contract."""
    from PIL import Image, ImageDraw, ImageFont

    if not rows:
        raise InferenceContractError("cannot render a review sheet over zero tiles")
    panel = 256
    header = 40
    canvas = Image.new("RGB", (panel * 2 + 12, header + panel * len(rows) + 4 * len(rows)), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 3), title, fill=(20, 20, 20), font=font)
    draw.text((5, 22), "Minimap RGB", fill=(30, 30, 30), font=font)
    draw.text((panel + 17, 22), "Predicted relative relief [0,1]", fill=(30, 30, 30), font=font)
    for index, row in enumerate(rows):
        y = header + index * (panel + 4)
        rgb_image = Image.fromarray(row["rgb"], mode="RGB").resize((panel, panel), Image.Resampling.NEAREST)
        relief8 = np.rint(row["relief"] * 255.0).astype(np.uint8)
        relief_image = Image.fromarray(np.repeat(relief8[:, :, None], 3, axis=2), mode="RGB")
        canvas.paste(rgb_image, (0, y))
        canvas.paste(relief_image, (panel + 12, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_inference(
    *,
    checkpoint_path: Path,
    inputs: list[Path],
    output: Path,
    device: str,
    write: bool,
) -> dict:
    """Shared CLI/test path: validate, predict, optionally persist, always return the manifest."""
    tiles = discover_tiles(inputs)
    model, checkpoint, identity = load_geometry_checkpoint(checkpoint_path, device=device)
    checkpoint_sha = sha256_file(checkpoint_path)

    manifest = {
        "schema": "v114-direct-geometry-inference-v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deployment_contract": {
            "input": "authored minimap RGB 256x256",
            "output_signal": "relative_height_257",
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "relative_only": True,
            "absolute_altitude": "not identifiable from one minimap (spec edge case)",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "model_variant": checkpoint.get("model_variant"),
            "architecture": identity["architecture"],
            "epoch": int(checkpoint.get("epoch", 0)),
            "val_mae": float(checkpoint.get("val_mae", float("nan"))),
        },
        "device": device,
        "tile_count": len(tiles),
        "tiles": [],
    }

    rows: list[dict] = []
    for tile_path in tiles:
        rgb = load_tile_rgb(tile_path)
        relief = predict_relief(model, rgb, device=device)
        entry = {
            "input": str(tile_path),
            "input_sha256": sha256_file(tile_path),
            "relief_min": float(relief.min()),
            "relief_max": float(relief.max()),
            "relief_mean": float(relief.mean()),
        }
        if write:
            from PIL import Image

            output.mkdir(parents=True, exist_ok=True)
            relief_path = output / f"{tile_path.stem}_relief16.png"
            # uint16 arrays map to PIL mode "I", which PNG persists as 16-bit grayscale.
            Image.fromarray(relief_to_uint16(relief)).save(relief_path)
            entry["output"] = str(relief_path)
            entry["output_sha256"] = sha256_file(relief_path)
        manifest["tiles"].append(entry)
        rows.append({"rgb": rgb, "relief": relief})

    if write:
        render_review_sheet(
            rows,
            output / "review_sheet.png",
            title=(
                f"direct geometry inference | {checkpoint.get('model_variant')} "
                f"epoch {checkpoint.get('epoch')} | {len(tiles)} tiles"
            ),
        )
        (output / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 114 direct geometry deployment inference (dry run by default)"
    )
    ap.add_argument("--checkpoint", required=True, type=Path, help="checkpoint_best.pt from a training run")
    ap.add_argument("--input", required=True, type=Path, action="append",
                    help="256x256 minimap tile or folder of tiles (repeatable)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--write", action="store_true",
                    help="persist relief PNGs, review sheet, and manifest; default prints only")
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
        )
    except InferenceContractError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(manifest, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to persist relief PNGs, the review sheet, and the manifest.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
