from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from train_v9 import (
    V9TerrainModel,
    _build_v9_input_channels,
    build_predictions,
    load_npz_arrays,
    resolve_amp_dtype,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(path)


def sanitize_output_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "tile"


def export_heightmap_obj(
    *,
    heightmap: np.ndarray,
    obj_path: Path,
    texture_name: str,
    tile_world_size: float,
    center_mesh: bool,
    height_offset: float,
) -> tuple[Path, Path]:
    if heightmap.ndim != 2:
        raise ValueError(f"Expected a 2D heightmap for OBJ export, got shape {heightmap.shape!r}")

    height, width = heightmap.shape
    if height < 2 or width < 2:
        raise ValueError("OBJ export requires at least a 2x2 heightmap.")

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = obj_path.with_suffix(".mtl")
    material_name = obj_path.stem + "_mat"
    spacing_x = tile_world_size / max(width - 1, 1)
    spacing_y = tile_world_size / max(height - 1, 1)
    origin_x = tile_world_size * 0.5 if center_mesh else 0.0
    origin_y = tile_world_size * 0.5 if center_mesh else 0.0

    with mtl_path.open("w", encoding="utf-8") as handle:
        handle.write("# v9 predicted terrain material\n")
        handle.write(f"newmtl {material_name}\n")
        handle.write("Ka 1.0 1.0 1.0\n")
        handle.write("Kd 1.0 1.0 1.0\n")
        handle.write("Ks 0.0 0.0 0.0\n")
        handle.write("d 1.0\n")
        handle.write("illum 1\n")
        handle.write(f"map_Kd {texture_name}\n")

    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write("# v9 predicted terrain mesh\n")
        handle.write(f"# grid {width}x{height}\n")
        handle.write(f"mtllib {mtl_path.name}\n")
        handle.write(f"usemtl {material_name}\n")

        for row in range(height):
            for column in range(width):
                world_x = column * spacing_x - origin_x
                world_z = row * spacing_y - origin_y
                world_y = float(heightmap[row, column] + height_offset)
                handle.write(f"v {world_x:.6f} {world_y:.6f} {world_z:.6f}\n")

        texture_width = max(width - 1, 1)
        texture_height = max(height - 1, 1)
        half_u = 0.5 / max(width, 1)
        half_v = 0.5 / max(height, 1)
        for row in range(height):
            for column in range(width):
                u = half_u + (1.0 - (column / texture_width)) * (1.0 - 2.0 * half_u)
                v = half_v + (1.0 - (row / texture_height)) * (1.0 - 2.0 * half_v)
                handle.write(f"vt {u:.6f} {v:.6f}\n")

        for row in range(height - 1):
            for column in range(width - 1):
                vertex_index = row * width + column + 1
                v0 = vertex_index
                v1 = vertex_index + 1
                v2 = vertex_index + width + 1
                v3 = vertex_index + width
                handle.write(f"f {v0}/{v0} {v3}/{v3} {v2}/{v2}\n")
                handle.write(f"f {v0}/{v0} {v2}/{v2} {v1}/{v1}\n")

    return obj_path, mtl_path


def write_mesh_bundle(
    *,
    output_dir: Path,
    stem: str,
    heightmap: np.ndarray,
    texture_rgb: np.ndarray,
    tile_world_size: float,
    center_mesh: bool,
    height_offset: float,
) -> dict[str, str]:
    texture_path = output_dir / f"{stem}_texture.png"
    write_rgb_png(texture_path, texture_rgb)
    obj_path, mtl_path = export_heightmap_obj(
        heightmap=heightmap,
        obj_path=output_dir / f"{stem}.obj",
        texture_name=texture_path.name,
        tile_world_size=tile_world_size,
        center_mesh=center_mesh,
        height_offset=height_offset,
    )
    return {
        "obj_path": str(obj_path),
        "mtl_path": str(mtl_path),
        "texture_path": str(texture_path),
    }


def load_manifest_entries(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"Manifest '{manifest_path}' does not contain an 'entries' list.")
    return [entry for entry in entries if isinstance(entry, dict)]


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[V9TerrainModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint.get("config", {}))
    hidden_channels = int(config.get("hidden_channels", 32))
    blocks_per_stage = int(config.get("blocks_per_stage", 2))
    state_dict = checkpoint["model"]
    in_channels = int(state_dict["stem.0.weight"].shape[1])
    model = V9TerrainModel(in_channels=in_channels, hidden_channels=hidden_channels, blocks_per_stage=blocks_per_stage).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def build_input_batch(
    shard_path: Path,
    height_scale: float,
    *,
    include_brush_mask: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, np.ndarray]]:
    arrays = load_npz_arrays(shard_path)
    required = ["height_17", "hole_mask_16x16"]
    missing = [name for name in required if name not in arrays]
    if missing:
        raise SystemExit(f"Missing required arrays in shard: {', '.join(missing)}")

    height_17 = torch.from_numpy(arrays["height_17"].astype(np.float32)).unsqueeze(0)
    if "wdl_17" in arrays:
        base_17 = torch.from_numpy(arrays["wdl_17"].astype(np.float32)).unsqueeze(0)
    else:
        base_17 = height_17.clone()

    base_65 = torch.nn.functional.interpolate(base_17.unsqueeze(0), size=(65, 65), mode="bilinear", align_corners=True).squeeze(0)
    base_257 = torch.nn.functional.interpolate(base_17.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=True).squeeze(0)

    if "minimap_rgb_256" not in arrays:
        raise SystemExit("Shard does not contain minimap_rgb_256, so this checkpoint cannot run minimap-driven inference on it.")

    inputs, _ = _build_v9_input_channels(
        arrays=arrays,
        base_257_scaled=base_257 / height_scale,
        include_brush_mask=include_brush_mask,
    )

    metadata = {
        "height_min": float(arrays.get("height_257", base_257.squeeze(0).numpy()).min()),
        "height_max": float(arrays.get("height_257", base_257.squeeze(0).numpy()).max()),
        "has_ground_truth_height_257": "height_257" in arrays,
    }
    outputs = {
        "inputs": inputs.unsqueeze(0),
        "base_height_17": (base_17 / height_scale).unsqueeze(0),
        "base_height_65": (base_65 / height_scale).unsqueeze(0),
        "base_height_257": (base_257 / height_scale).unsqueeze(0),
    }

    minimap_rgb_256 = arrays.get("minimap_rgb_256")
    if minimap_rgb_256 is None:
        texture_rgb = np.zeros((257, 257, 3), dtype=np.uint8)
    else:
        texture_rgb = np.asarray(
            Image.fromarray(minimap_rgb_256.astype(np.uint8)).resize((257, 257), resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    export_assets = {
        "texture_rgb_257": texture_rgb,
        "wdl_height_257": base_257.squeeze(0).numpy().astype(np.float32),
    }
    return outputs, metadata, export_assets


def run_single_inference(
    *,
    model: V9TerrainModel,
    shard_path: Path,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    height_scale: float,
    residual_scale: float,
    channels_last: bool,
    tile_world_size: float,
    center_mesh: bool,
    height_offset: float,
    export_wdl_baseline: bool,
    include_brush_mask: bool,
    checkpoint_path: Path,
    tile_name: str | None = None,
    flat_mesh_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, metadata, export_assets = build_input_batch(
        shard_path,
        height_scale=height_scale,
        include_brush_mask=include_brush_mask,
    )

    moved_batch: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = value.to(device)
        if channels_last and tensor.ndim == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        moved_batch[key] = tensor

    autocast_enabled = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}
    with torch.no_grad():
        with (torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled else torch.no_grad()):
            coarse_delta_17, mid_delta_65, detail_delta_257 = model(moved_batch["inputs"])
            coarse_height_17, mid_height_65, full_height_257 = build_predictions(
                coarse_delta_17=coarse_delta_17,
                mid_delta_65=mid_delta_65,
                detail_delta_257=detail_delta_257,
                base_height_17=moved_batch["base_height_17"],
                base_height_65=moved_batch["base_height_65"],
                base_height_257=moved_batch["base_height_257"],
                residual_scale=residual_scale,
                height_scale=height_scale,
            )

    predicted_height_257 = full_height_257.squeeze(0).squeeze(0).detach().cpu().numpy() * height_scale
    predicted_height_65 = mid_height_65.squeeze(0).squeeze(0).detach().cpu().numpy() * height_scale
    predicted_height_17 = coarse_height_17.squeeze(0).squeeze(0).detach().cpu().numpy() * height_scale

    np.save(output_dir / "predicted_height_257.npy", predicted_height_257)
    np.save(output_dir / "predicted_height_65.npy", predicted_height_65)
    np.save(output_dir / "predicted_height_17.npy", predicted_height_17)

    predicted_mesh = write_mesh_bundle(
        output_dir=output_dir,
        stem="predicted_terrain",
        heightmap=predicted_height_257,
        texture_rgb=export_assets["texture_rgb_257"],
        tile_world_size=tile_world_size,
        center_mesh=center_mesh,
        height_offset=height_offset,
    )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "shard": str(shard_path),
        "disable_brush_mask": not include_brush_mask,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "height_scale": height_scale,
        "residual_scale": residual_scale,
        "predicted_height_257_path": str(output_dir / "predicted_height_257.npy"),
        "predicted_height_65_path": str(output_dir / "predicted_height_65.npy"),
        "predicted_height_17_path": str(output_dir / "predicted_height_17.npy"),
        "predicted_mesh_obj_path": predicted_mesh["obj_path"],
        "predicted_mesh_mtl_path": predicted_mesh["mtl_path"],
        "predicted_mesh_texture_path": predicted_mesh["texture_path"],
        "predicted_mesh_tile_world_size": tile_world_size,
        "predicted_mesh_centered": center_mesh,
        "predicted_mesh_height_offset": height_offset,
        "predicted_height_257_min": float(predicted_height_257.min()),
        "predicted_height_257_max": float(predicted_height_257.max()),
        **metadata,
    }

    if export_wdl_baseline:
        np.save(output_dir / "wdl_baseline_height_257.npy", export_assets["wdl_height_257"])
        baseline_mesh = write_mesh_bundle(
            output_dir=output_dir,
            stem="wdl_baseline_terrain",
            heightmap=export_assets["wdl_height_257"],
            texture_rgb=export_assets["texture_rgb_257"],
            tile_world_size=tile_world_size,
            center_mesh=center_mesh,
            height_offset=height_offset,
        )
        summary.update(
            {
                "wdl_baseline_height_257_path": str(output_dir / "wdl_baseline_height_257.npy"),
                "wdl_baseline_mesh_obj_path": baseline_mesh["obj_path"],
                "wdl_baseline_mesh_mtl_path": baseline_mesh["mtl_path"],
                "wdl_baseline_mesh_texture_path": baseline_mesh["texture_path"],
                "wdl_baseline_height_257_min": float(export_assets["wdl_height_257"].min()),
                "wdl_baseline_height_257_max": float(export_assets["wdl_height_257"].max()),
            }
        )

    if flat_mesh_dir is not None:
        flat_mesh_dir.mkdir(parents=True, exist_ok=True)
        safe_tile_name = sanitize_output_name(tile_name or shard_path.stem)
        flat_predicted = write_mesh_bundle(
            output_dir=flat_mesh_dir,
            stem=f"{safe_tile_name}_predicted_terrain",
            heightmap=predicted_height_257,
            texture_rgb=export_assets["texture_rgb_257"],
            tile_world_size=tile_world_size,
            center_mesh=center_mesh,
            height_offset=height_offset,
        )
        summary.update(
            {
                "flat_predicted_mesh_obj_path": flat_predicted["obj_path"],
                "flat_predicted_mesh_mtl_path": flat_predicted["mtl_path"],
                "flat_predicted_mesh_texture_path": flat_predicted["texture_path"],
            }
        )
        if export_wdl_baseline:
            flat_baseline = write_mesh_bundle(
                output_dir=flat_mesh_dir,
                stem=f"{safe_tile_name}_wdl_baseline_terrain",
                heightmap=export_assets["wdl_height_257"],
                texture_rgb=export_assets["texture_rgb_257"],
                tile_world_size=tile_world_size,
                center_mesh=center_mesh,
                height_offset=height_offset,
            )
            summary.update(
                {
                    "flat_wdl_baseline_mesh_obj_path": flat_baseline["obj_path"],
                    "flat_wdl_baseline_mesh_mtl_path": flat_baseline["mtl_path"],
                    "flat_wdl_baseline_mesh_texture_path": flat_baseline["texture_path"],
                }
            )

    write_json(output_dir / "inference_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 minimap-to-terrain inference from a cached native tensor shard.")
    parser.add_argument("checkpoint", help="Path to v9 best_model.pt")
    parser.add_argument("input_path", help="Path to a cached v9 .npz shard or a v9_tensor_cache_manifest.json manifest")
    parser.add_argument("--output-dir", required=True, help="Directory to write prediction arrays and summary JSON")
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--channels-last", action="store_true", help="Run the model in channels-last memory format")
    parser.add_argument("--tile-world-size", type=float, default=533.3333333333334, help="World-space width/depth assigned to the exported terrain tile OBJ")
    parser.add_argument("--height-offset", type=float, default=0.0, help="Constant vertical offset added to OBJ vertex heights")
    parser.add_argument("--no-center-mesh", action="store_true", help="Export the OBJ in positive tile coordinates instead of centering it around the origin")
    parser.add_argument("--no-export-wdl-baseline", action="store_true", help="Skip writing the WDL baseline mesh alongside the model prediction")
    parser.add_argument("--entry-limit", type=int, default=None, help="Optional cap when input_path is a cache manifest")
    parser.add_argument("--disable-brush-mask", action=argparse.BooleanOptionalAction, default=False, help="Zero the brush imprint mask channel during inference.")
    parser.add_argument("--flat-mesh-dir", default=None, help="Optional directory that also receives all manifest OBJ/MTL/texture bundles in one flat folder.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    model, config = load_checkpoint_model(checkpoint_path, device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    height_scale = float(config.get("height_scale", 1024.0))
    residual_scale = float(config.get("residual_scale", 128.0))
    if input_path.suffix.lower() == ".json":
        manifest_entries = load_manifest_entries(input_path)
        if args.entry_limit is not None:
            manifest_entries = manifest_entries[: max(0, int(args.entry_limit))]
        batch_results: list[dict[str, Any]] = []
        flat_mesh_dir = Path(args.flat_mesh_dir) if args.flat_mesh_dir else (output_dir / "flat_meshes")
        for index, entry in enumerate(manifest_entries, start=1):
            shard_value = entry.get("shard_path")
            tile_name = str(entry.get("tile_name") or f"tile_{index:04d}")
            if not shard_value:
                continue
            shard_path = Path(str(shard_value))
            tile_output_dir = output_dir / sanitize_output_name(tile_name)
            summary = run_single_inference(
                model=model,
                shard_path=shard_path,
                output_dir=tile_output_dir,
                device=device,
                amp_dtype=amp_dtype,
                height_scale=height_scale,
                residual_scale=residual_scale,
                channels_last=bool(args.channels_last),
                tile_world_size=float(args.tile_world_size),
                center_mesh=not args.no_center_mesh,
                height_offset=float(args.height_offset),
                export_wdl_baseline=not args.no_export_wdl_baseline,
                include_brush_mask=not args.disable_brush_mask,
                checkpoint_path=checkpoint_path,
                tile_name=tile_name,
                flat_mesh_dir=flat_mesh_dir,
            )
            summary["tile_name"] = tile_name
            summary["batch_index"] = index
            batch_results.append(summary)
            print(f"[{index}/{len(manifest_entries)}] Saved v9 terrain predictions to {tile_output_dir}")

        batch_summary = {
            "checkpoint": str(checkpoint_path),
            "input_manifest": str(input_path),
            "output_dir": str(output_dir),
            "entry_count": len(batch_results),
            "export_wdl_baseline": not args.no_export_wdl_baseline,
            "disable_brush_mask": bool(args.disable_brush_mask),
            "tile_world_size": float(args.tile_world_size),
            "center_mesh": not args.no_center_mesh,
            "height_offset": float(args.height_offset),
            "flat_mesh_dir": str(flat_mesh_dir),
            "entries": batch_results,
        }
        write_json(output_dir / "batch_inference_summary.json", batch_summary)
        print(f"Saved v9 terrain predictions for {len(batch_results)} tile(s) to {output_dir}")
        return

    summary = run_single_inference(
        model=model,
        shard_path=input_path,
        output_dir=output_dir,
        device=device,
        amp_dtype=amp_dtype,
        height_scale=height_scale,
        residual_scale=residual_scale,
        channels_last=bool(args.channels_last),
        tile_world_size=float(args.tile_world_size),
        center_mesh=not args.no_center_mesh,
        height_offset=float(args.height_offset),
        export_wdl_baseline=not args.no_export_wdl_baseline,
        include_brush_mask=not args.disable_brush_mask,
        checkpoint_path=checkpoint_path,
        tile_name=input_path.stem,
        flat_mesh_dir=Path(args.flat_mesh_dir) if args.flat_mesh_dir else None,
    )
    write_json(output_dir / "inference_summary.json", summary)
    print(f"Saved v9 terrain predictions to {output_dir}")


if __name__ == "__main__":
    main()