"""Any-raster inference and textured terrain OBJ export for Spec 114."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from harvester.v50.universal_relief_contract import (
    build_terrain_mesh,
    load_raster_rgb,
    prepare_raster,
    stitch_relief,
    write_obj,
)
from harvester.v50.universal_relief_model import (
    UniversalReliefNet,
    download_pinned_student_backbone,
    sha256_file,
    student_identity_dict,
)


class UniversalInferenceError(ValueError):
    """Raised when checkpoint/source/output violates universal inference identity."""


def resize_relief_for_mesh(relief: np.ndarray, max_resolution: int) -> np.ndarray:
    values = np.asarray(relief, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise UniversalInferenceError("relief must be a non-empty 2D array")
    if max_resolution < 2:
        raise UniversalInferenceError("mesh max resolution must be at least 2")
    height, width = values.shape
    if max(height, width) <= max_resolution:
        return values.copy()
    scale = max_resolution / max(height, width)
    target_height = max(2, round(height * scale))
    target_width = max(2, round(width * scale))
    tensor = torch.from_numpy(values)[None, None]
    return (
        torch.nn.functional.interpolate(
            tensor, size=(target_height, target_width), mode="bilinear", align_corners=False
        )[0, 0]
        .numpy()
        .astype(np.float32)
    )


def _load_checkpoint_identity(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "v114-universal-relief-checkpoint-v1":
        raise UniversalInferenceError("checkpoint is not a Spec 114 universal relief checkpoint")
    if checkpoint.get("student") != student_identity_dict():
        raise UniversalInferenceError("checkpoint student identity does not match the pinned model")
    if "model" not in checkpoint:
        raise UniversalInferenceError("checkpoint has no EMA deploy model state")
    return checkpoint


def build_inference_plan(
    *,
    image_path: str | Path,
    checkpoint_path: str | Path,
    output: str | Path,
    overlap: int,
    mesh_max_resolution: int,
    extent_x: float,
    vertical_scale: float,
) -> dict[str, Any]:
    source = Path(image_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    output_path = Path(output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if output_path.is_file() or (output_path.is_dir() and any(output_path.iterdir())):
        raise UniversalInferenceError(f"refusing to overwrite occupied output {output_path}")
    rgb, mode = load_raster_rgb(source)
    if not 0 <= overlap < 224:
        raise UniversalInferenceError("overlap must be in [0,224)")
    if mesh_max_resolution < 2 or extent_x <= 0.0 or not np.isfinite(vertical_scale):
        raise UniversalInferenceError("mesh resolution/extent/vertical scale are invalid")
    checkpoint = _load_checkpoint_identity(checkpoint)
    prepared = prepare_raster(Image.fromarray(rgb, mode="RGB"), tile_size=224, overlap=overlap)
    return {
        "schema": "v114-universal-inference-plan-v1",
        "source": str(source),
        "source_mode": mode,
        "source_width": int(rgb.shape[1]),
        "source_height": int(rgb.shape[0]),
        "tile_count": len(prepared.tiles),
        "overlap": overlap,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "student": student_identity_dict(),
        "output": str(output_path),
        "mesh_max_resolution": mesh_max_resolution,
        "extent_x": extent_x,
        "vertical_scale": vertical_scale,
        "semantics": "view_axis_relief",
    }


def _save_preview(source_rgb: np.ndarray, relief: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    dy, dx = np.gradient(relief)
    hillshade = np.clip(0.5 + (-dx - dy) * 2.0, 0.0, 1.0)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(source_rgb)
    axes[0].set_title("source raster")
    axes[1].imshow(relief, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("normalized view-axis relief")
    axes[2].imshow(hillshade, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("relief hillshade")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_inference(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UniversalInferenceError("CUDA inference requested but CUDA is unavailable")
    if args.batch < 1:
        raise UniversalInferenceError("inference batch must be positive")
    checkpoint = _load_checkpoint_identity(Path(plan["checkpoint"]))
    backbone = download_pinned_student_backbone(cache_dir=args.hf_cache)
    model = UniversalReliefNet(
        backbone,
        freeze_backbone=bool(checkpoint.get("freeze_backbone", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    source_rgb, _ = load_raster_rgb(plan["source"])
    prepared = prepare_raster(
        Image.fromarray(source_rgb, mode="RGB"), tile_size=224, overlap=plan["overlap"]
    )
    predictions = []
    use_amp = device.type == "cuda" and not args.no_amp
    for start in range(0, len(prepared.tiles), args.batch):
        tiles = prepared.tiles[start : start + args.batch]
        rgb = torch.from_numpy(np.stack([tile.rgb_chw for tile in tiles])).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            predicted = model(rgb)
        predictions.extend(predicted.float().cpu().numpy())
    relief = stitch_relief(predictions, prepared.transform, normalize=True)
    mesh_relief = resize_relief_for_mesh(relief, plan["mesh_max_resolution"])
    mesh = build_terrain_mesh(
        mesh_relief,
        extent_x=plan["extent_x"],
        vertical_scale=plan["vertical_scale"],
    )

    output = Path(plan["output"])
    if output.is_file() or (output.is_dir() and any(output.iterdir())):
        raise UniversalInferenceError(f"refusing to overwrite occupied output {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_output = output / "source.png"
    Image.fromarray(source_rgb, mode="RGB").save(source_output)
    relief_u16 = np.round(np.clip(relief, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(relief_u16, mode="I;16").save(output / "relief_16.png")
    write_obj(mesh, output / "terrain.obj", texture_filename=source_output.name)
    _save_preview(source_rgb, relief, output / "validation.png")
    manifest = {
        **plan,
        "relief_shape": list(relief.shape),
        "mesh_grid": [mesh.grid_height, mesh.grid_width],
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "artifacts": {
            "source_texture": "source.png",
            "relief_16": "relief_16.png",
            "mesh": "terrain.obj",
            "material": "terrain.mtl",
            "validation": "validation.png",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert any raster to normalized relief and textured terrain OBJ."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=28)
    parser.add_argument("--mesh-max-resolution", type=int, default=257)
    parser.add_argument("--extent-x", type=float, default=533.3333333333)
    parser.add_argument("--vertical-scale", type=float, default=128.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_inference_plan(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output=args.output,
        overlap=args.overlap,
        mesh_max_resolution=args.mesh_max_resolution,
        extent_x=args.extent_x,
        vertical_scale=args.vertical_scale,
    )
    print(json.dumps(plan, indent=2))
    if not args.confirm_run:
        print("DRY RUN: add --confirm-run to load the model and write terrain artifacts.")
        return 0
    manifest = run_inference(args, plan)
    print(json.dumps(manifest, indent=2))
    return 0
