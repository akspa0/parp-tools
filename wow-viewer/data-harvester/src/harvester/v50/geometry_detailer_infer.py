"""Spec 114 deployment inference for the composed coarse+detailer geometry stage (FR-015).

Extends the T056 coarse-only deployment contract (``direct_geometry_infer.py``) to the promoted
residual detailer: one authored minimap tile (256x256 RGB) in, one composed relative-relief field
(257x257, contract ``v112.1``) out, where the composition is exactly ``clamp(coarse + residual, 0,
1)`` per ``geometry_detailer_model.compose_final`` — never a retrained/merged single model
(constitution IV: the two stages stay independently replaceable).

Both checkpoints are loaded and their identities recorded. The coarse checkpoint the caller
supplies is cross-checked by sha256 against the detailer checkpoint's own
``upstream_coarse_checkpoint`` provenance; a mismatch refuses to run rather than silently pairing
stages that were never trained together.
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
    load_geometry_checkpoint,
    load_tile_rgb,
    relief_to_uint16,
)
from harvester.v50.height_relative_model import TARGET_CONTRACT_VERSION
from harvester.v50.model_stage_contract import sha256_file

INPUT_SIZE = 256


def load_detailer_checkpoint(checkpoint_path: Path, *, device: str):
    """Load a detailer training checkpoint; refuse architecture/target-contract drift."""
    import torch

    from harvester.v50.geometry_detailer_model import (
        DETAILER_ARCHITECTURE_ID,
        GeometryDetailerNet,
        detailer_identity,
    )

    if not checkpoint_path.is_file():
        raise InferenceContractError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = checkpoint.get("model_variant")
    if variant != DETAILER_ARCHITECTURE_ID:
        raise InferenceContractError(
            f"checkpoint model_variant {variant!r} != {DETAILER_ARCHITECTURE_ID!r}"
        )
    if checkpoint.get("target_contract_version") != TARGET_CONTRACT_VERSION:
        raise InferenceContractError(
            f"checkpoint target contract {checkpoint.get('target_contract_version')!r} "
            f"!= {TARGET_CONTRACT_VERSION!r}"
        )
    model = GeometryDetailerNet()
    try:
        model.load_state_dict(checkpoint["model"])
    except (KeyError, RuntimeError) as exc:
        raise InferenceContractError(
            f"checkpoint weights do not match architecture {DETAILER_ARCHITECTURE_ID!r}: {exc}"
        ) from exc
    model.eval()
    model.to(torch.device(device))
    return model, checkpoint, detailer_identity(model)


def verify_coarse_pairing(detailer_checkpoint: dict, coarse_checkpoint_path: Path, coarse_sha256: str) -> None:
    """Refuse to compose stages that were never trained together (FR-015 auditability)."""
    recorded = detailer_checkpoint.get("upstream_coarse_checkpoint")
    if not recorded or not recorded.get("sha256"):
        raise InferenceContractError(
            "detailer checkpoint does not record its upstream_coarse_checkpoint provenance"
        )
    if recorded["sha256"] != coarse_sha256:
        raise InferenceContractError(
            f"coarse checkpoint {coarse_checkpoint_path} (sha256 {coarse_sha256}) does not match "
            f"the detailer's recorded upstream coarse checkpoint "
            f"({recorded.get('path')}, sha256 {recorded['sha256']}); "
            "this detailer was trained against a different coarse checkpoint"
        )


def predict_composed_relief(
    coarse_model, detailer_model, rgb: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """One RGB tile -> (coarse relief, final relief = clamp(coarse + residual, 0, 1))."""
    import torch

    from harvester.v50.geometry_detailer_model import compose_final

    tensor = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        coarse = coarse_model(tensor)
        residual = detailer_model(tensor, coarse)
        final = compose_final(coarse, residual, clamp=True)
    coarse_np = coarse[0].float().cpu().numpy()
    final_np = final[0].float().cpu().numpy()
    if final_np.shape != (257, 257) or not np.isfinite(final_np).all():
        raise InferenceContractError(
            f"composed model emitted an invalid relief field: shape {final_np.shape}"
        )
    return np.clip(coarse_np, 0.0, 1.0).astype(np.float32), np.clip(final_np, 0.0, 1.0).astype(
        np.float32
    )


def render_review_sheet(rows: list[dict], output: Path, *, title: str) -> None:
    """Fixed-scale [input | coarse | final] sheet; every panel stretched over the full [0,1] contract."""
    from PIL import Image, ImageDraw, ImageFont

    if not rows:
        raise InferenceContractError("cannot render a review sheet over zero tiles")
    panel = 256
    header = 40
    canvas = Image.new(
        "RGB", (panel * 3 + 16, header + panel * len(rows) + 4 * len(rows)), (245, 245, 245)
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 3), title, fill=(20, 20, 20), font=font)
    draw.text((5, 22), "Minimap RGB", fill=(30, 30, 30), font=font)
    draw.text((panel + 12, 22), "Coarse relief [0,1]", fill=(30, 30, 30), font=font)
    draw.text((panel * 2 + 20, 22), "Coarse + residual [0,1]", fill=(30, 30, 30), font=font)
    for index, row in enumerate(rows):
        y = header + index * (panel + 4)
        rgb_image = Image.fromarray(row["rgb"], mode="RGB").resize(
            (panel, panel), Image.Resampling.NEAREST
        )
        coarse8 = np.rint(row["coarse"] * 255.0).astype(np.uint8)
        coarse_image = Image.fromarray(np.repeat(coarse8[:, :, None], 3, axis=2), mode="RGB")
        final8 = np.rint(row["final"] * 255.0).astype(np.uint8)
        final_image = Image.fromarray(np.repeat(final8[:, :, None], 3, axis=2), mode="RGB")
        canvas.paste(rgb_image, (0, y))
        canvas.paste(coarse_image, (panel + 8, y))
        canvas.paste(final_image, (panel * 2 + 16, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_inference(
    *,
    coarse_checkpoint_path: Path,
    detailer_checkpoint_path: Path,
    inputs: list[Path],
    output: Path,
    device: str,
    write: bool,
) -> dict:
    """Shared CLI/test path: validate, pair-check, predict, optionally persist, always return the manifest."""
    tiles = discover_tiles(inputs)
    coarse_model, coarse_checkpoint, coarse_identity = load_geometry_checkpoint(
        coarse_checkpoint_path, device=device
    )
    coarse_sha256 = sha256_file(coarse_checkpoint_path)
    detailer_model, detailer_checkpoint, detailer_identity = load_detailer_checkpoint(
        detailer_checkpoint_path, device=device
    )
    verify_coarse_pairing(detailer_checkpoint, coarse_checkpoint_path, coarse_sha256)
    detailer_sha256 = sha256_file(detailer_checkpoint_path)

    manifest = {
        "schema": "v114-detailer-geometry-inference-v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deployment_contract": {
            "input": "authored minimap RGB 256x256",
            "output_signal": "relative_height_257",
            "composition": "clamp(coarse + residual, 0, 1)",
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "relative_only": True,
            "absolute_altitude": "not identifiable from one minimap (spec edge case)",
        },
        "coarse_checkpoint": {
            "path": str(coarse_checkpoint_path),
            "sha256": coarse_sha256,
            "model_variant": coarse_checkpoint.get("model_variant"),
            "architecture": coarse_identity["architecture"],
            "epoch": int(coarse_checkpoint.get("epoch", 0)),
            "val_mae": float(coarse_checkpoint.get("val_mae", float("nan"))),
        },
        "detailer_checkpoint": {
            "path": str(detailer_checkpoint_path),
            "sha256": detailer_sha256,
            "model_variant": detailer_checkpoint.get("model_variant"),
            "architecture": detailer_identity["id"],
            "epoch": int(detailer_checkpoint.get("epoch", 0)),
            "val_mae": float(detailer_checkpoint.get("val_mae", float("nan"))),
        },
        "device": device,
        "tile_count": len(tiles),
        "tiles": [],
    }

    rows: list[dict] = []
    for tile_path in tiles:
        rgb = load_tile_rgb(tile_path)
        coarse, final = predict_composed_relief(coarse_model, detailer_model, rgb, device=device)
        entry = {
            "input": str(tile_path),
            "input_sha256": sha256_file(tile_path),
            "relief_min": float(final.min()),
            "relief_max": float(final.max()),
            "relief_mean": float(final.mean()),
        }
        if write:
            from PIL import Image

            output.mkdir(parents=True, exist_ok=True)
            relief_path = output / f"{tile_path.stem}_relief16.png"
            Image.fromarray(relief_to_uint16(final)).save(relief_path)
            entry["output"] = str(relief_path)
            entry["output_sha256"] = sha256_file(relief_path)
        manifest["tiles"].append(entry)
        rows.append({"rgb": rgb, "coarse": coarse, "final": final})

    if write:
        render_review_sheet(
            rows,
            output / "review_sheet.png",
            title=(
                f"composed geometry inference | coarse={coarse_checkpoint.get('model_variant')} "
                f"detailer={detailer_checkpoint.get('model_variant')} epoch "
                f"{detailer_checkpoint.get('epoch')} | {len(tiles)} tiles"
            ),
        )
        (output / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 114 composed coarse+detailer geometry deployment inference (dry run by default)"
    )
    ap.add_argument("--coarse-checkpoint", required=True, type=Path, help="coarse stage checkpoint_best.pt")
    ap.add_argument(
        "--detailer-checkpoint", required=True, type=Path, help="detailer stage checkpoint_best.pt"
    )
    ap.add_argument(
        "--input", required=True, type=Path, action="append",
        help="256x256 minimap tile or folder of tiles (repeatable)",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument(
        "--write", action="store_true",
        help="persist relief PNGs, review sheet, and manifest; default prints only",
    )
    args = ap.parse_args(argv)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable; use --device cpu.")
    try:
        manifest = run_inference(
            coarse_checkpoint_path=args.coarse_checkpoint,
            detailer_checkpoint_path=args.detailer_checkpoint,
            inputs=args.input,
            output=args.output,
            device=args.device,
            write=args.write,
        )
    except InferenceContractError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(manifest, indent=2), flush=True)
    if not args.write:
        print(
            "DRY RUN ONLY: add --write to persist relief PNGs, the review sheet, and the manifest.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
