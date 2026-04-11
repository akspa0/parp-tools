#!/usr/bin/env python3
"""
WoW Height Regressor V7.1 - inference engine.

Inference restores the original multichannel V7.1 contract:
- minimap RGB
- normal map RGB
- WDL prior
- per-tile bounds hints
- liquid mask
- liquid height prior
- object footprint mask

The primary mesh export still uses the predicted global height channel. The
terrain checkpoint no longer owns alpha-mask prediction.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

try:
    from train_v7 import HEIGHT_GLOBAL_MAX, HEIGHT_GLOBAL_MIN, MultiChannelUNetV7, OUTPUT_SIZE
except ImportError:
    OUTPUT_SIZE = 512
    HEIGHT_GLOBAL_MIN = -1000.0
    HEIGHT_GLOBAL_MAX = 3000.0

    class MultiChannelUNetV7(nn.Module):
        def __init__(self, in_channels: int = 12, out_channels: int = 2):
            super().__init__()
            raise ImportError("train_v7.py is required so the V7.1 architecture matches the checkpoint.")


TILE_SIZE = 533.33333


class V7InferenceEngine:
    def __init__(self, model_path: Path, device: str = "auto") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        print(f"Loading V7.1 model from {model_path} on {self.device}...")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.metadata: Dict[str, object] = dict(checkpoint.get("metadata", {}))
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        in_channels = state_dict["enc1.0.weight"].shape[1]
        out_channels = state_dict["out_conv.weight"].shape[0]
        print(f"Detected input channels: {in_channels}")

        self.model = MultiChannelUNetV7(in_channels=in_channels, out_channels=out_channels).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)

    def prepare_input(self, dataset_root: Path, tile_name: str) -> Tuple[torch.Tensor, Dict[str, float], Path]:
        json_path = dataset_root / "dataset" / f"{tile_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        terrain = payload.get("terrain_data", {})

        minimap_path = dataset_root / "images" / f"{tile_name}.png"
        normalmap_rel = terrain.get("normalmap")
        if not normalmap_rel:
            raise FileNotFoundError(f"Missing normalmap reference for {tile_name}")
        normalmap_path = dataset_root / normalmap_rel

        if not minimap_path.exists() or not normalmap_path.exists():
            raise FileNotFoundError(f"Missing input images for {tile_name}")

        minimap = Image.open(minimap_path).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        minimap = self.blur(minimap)
        normalmap = Image.open(normalmap_path).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)

        minimap_tensor = self.normalize(self.to_tensor(minimap))
        normalmap_tensor = self.normalize(self.to_tensor(normalmap))
        wdl_tensor = self._render_wdl(terrain.get("wdl_heights"))

        height_min = float(terrain.get("height_min", 0.0))
        height_max = float(terrain.get("height_max", 100.0))
        global_min = float(terrain.get("height_global_min", HEIGHT_GLOBAL_MIN))
        global_max = float(terrain.get("height_global_max", HEIGHT_GLOBAL_MAX))
        global_range = max(global_max - global_min, 1e-6)

        height_min_mask = torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), np.clip((height_min - global_min) / global_range, 0.0, 1.0), dtype=torch.float32)
        height_max_mask = torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), np.clip((height_max - global_min) / global_range, 0.0, 1.0), dtype=torch.float32)

        liquid_mask = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        liquid_mask_rel = terrain.get("liquid_mask")
        if liquid_mask_rel:
            liquid_mask_path = dataset_root / liquid_mask_rel
            if liquid_mask_path.exists():
                liquid_image = Image.open(liquid_mask_path).convert("L").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
                liquid_mask = (self.to_tensor(liquid_image) > 0.1).float()

        liquid_height_prior = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        liquid_height_rel = terrain.get("liquid_height")
        if liquid_height_rel:
            liquid_height_path = dataset_root / liquid_height_rel
            if liquid_height_path.exists():
                liquid_height_prior = load_heightmap_16bit(liquid_height_path, OUTPUT_SIZE) * liquid_mask

        object_mask = self._build_object_mask(terrain.get("objects"))
        input_tensor = torch.cat(
            [
                minimap_tensor,
                normalmap_tensor,
                wdl_tensor,
                height_min_mask,
                height_max_mask,
                liquid_mask,
                liquid_height_prior,
                object_mask,
            ],
            dim=0,
        ).unsqueeze(0)

        bounds_info = {
            "h_min": height_min,
            "h_max": height_max,
            "g_min": global_min,
            "g_max": global_max,
        }
        return input_tensor.to(self.device), bounds_info, minimap_path

    def _render_wdl(self, wdl_data: Optional[Dict[str, object]]) -> torch.Tensor:
        if not wdl_data:
            return torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), 0.5)

        outer = np.asarray(wdl_data.get("outer_17", []), dtype=np.float32)
        if len(outer) != 289:
            return torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), 0.5)

        grid = outer.reshape(17, 17)
        minimum = float(grid.min())
        maximum = float(grid.max())
        if maximum - minimum > 1e-6:
            grid = (grid - minimum) / (maximum - minimum)
        else:
            grid[:] = 0.5

        image = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        return self.to_tensor(image)

    def _build_object_mask(self, objects: Optional[Sequence[Dict[str, object]]]) -> torch.Tensor:
        object_mask = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        if not objects:
            return object_mask

        image = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        for obj in objects:
            pos_x = float(obj.get("x", obj.get("pos_x", 0.0)))
            pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
            scale = float(obj.get("scale", 1.0))

            bounds_min = obj.get("bounds_min")
            bounds_max = obj.get("bounds_max")
            if bounds_min and bounds_max and len(bounds_min) >= 2 and len(bounds_max) >= 2:
                half_width = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
                half_depth = abs(float(bounds_max[1]) - float(bounds_min[1])) * 0.5 * scale
                pixels_per_unit = OUTPUT_SIZE / TILE_SIZE
                radius_x = max(1, int(half_width * pixels_per_unit))
                radius_y = max(1, int(half_depth * pixels_per_unit))
            else:
                radius_x = max(1, int(5 * scale))
                radius_y = radius_x

            if abs(pos_x) < 2 and abs(pos_y) < 2:
                normalized_x = int((pos_x + 1) * 0.5 * OUTPUT_SIZE)
                normalized_y = int((pos_y + 1) * 0.5 * OUTPUT_SIZE)
            else:
                normalized_x = int((pos_x / TILE_SIZE) * OUTPUT_SIZE) % OUTPUT_SIZE
                normalized_y = int((pos_y / TILE_SIZE) * OUTPUT_SIZE) % OUTPUT_SIZE

            x1 = max(0, normalized_x - radius_x)
            y1 = max(0, normalized_y - radius_y)
            x2 = min(OUTPUT_SIZE, normalized_x + radius_x)
            y2 = min(OUTPUT_SIZE, normalized_y + radius_y)
            image[y1:y2, x1:x2] = 1.0

        return torch.from_numpy(image).unsqueeze(0)

    def predict(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            predicted_heightmap, predicted_bounds = self.model(input_tensor)
        return predicted_heightmap.cpu().numpy(), predicted_bounds.cpu().numpy()

    @staticmethod
    def denormalize_heightmap(normalized_heightmap: np.ndarray, global_min: float = HEIGHT_GLOBAL_MIN, global_max: float = HEIGHT_GLOBAL_MAX) -> np.ndarray:
        return normalized_heightmap * (global_max - global_min) + global_min

    @staticmethod
    def save_obj(heightmap: np.ndarray, output_path: Path, tile_x: int, tile_y: int, texture_path: Optional[Path] = None) -> None:
        resolution = heightmap.shape[0]
        step = TILE_SIZE / max(resolution - 1, 1)

        vertices = []
        uvs = []
        faces = []

        for grid_y in range(resolution):
            for grid_x in range(resolution):
                world_x = tile_x * TILE_SIZE + grid_x * step
                world_y = float(heightmap[grid_y, grid_x])
                world_z = tile_y * TILE_SIZE + grid_y * step
                vertices.append((world_x, world_y, world_z))
                uvs.append((grid_x / max(resolution - 1, 1), 1.0 - (grid_y / max(resolution - 1, 1))))

        for grid_y in range(resolution - 1):
            for grid_x in range(resolution - 1):
                v0 = grid_y * resolution + grid_x + 1
                v1 = grid_y * resolution + (grid_x + 1) + 1
                v2 = (grid_y + 1) * resolution + (grid_x + 1) + 1
                v3 = (grid_y + 1) * resolution + grid_x + 1
                faces.append((v0, v3, v2))
                faces.append((v0, v2, v1))

        mtl_path = output_path.with_suffix(".mtl")
        texture_name = texture_path.name if texture_path else "default.png"
        if texture_path and texture_path.exists():
            copied_texture = output_path.parent / texture_name
            if not copied_texture.exists():
                shutil.copy(texture_path, copied_texture)

        with open(mtl_path, "w", encoding="utf-8") as handle:
            handle.write("newmtl TerrainMat\n")
            handle.write("Ka 1.0 1.0 1.0\n")
            handle.write("Kd 1.0 1.0 1.0\n")
            handle.write(f"map_Kd {texture_name}\n")

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"mtllib {mtl_path.name}\n")
            handle.write("usemtl TerrainMat\n")
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.2f} {vertex[1]:.2f} {vertex[2]:.2f}\n")
            for uv in uvs:
                handle.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
            for face in faces:
                handle.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")

    def save_debug_image(self, input_tensor: torch.Tensor, predicted_heightmap: torch.Tensor, output_path: Path) -> None:
        minimap = input_tensor[0, 0:3].cpu()
        normalmap = input_tensor[0, 3:6].cpu()
        wdl = input_tensor[0, 6:7].cpu().repeat(3, 1, 1)
        water = input_tensor[0, 9:10].cpu().repeat(3, 1, 1) * torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
        objects = input_tensor[0, 10:11].cpu().repeat(3, 1, 1) * torch.tensor([1.0, 0.5, 0.0]).view(3, 1, 1)
        prediction = predicted_heightmap[0, 0:1].cpu()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        minimap = torch.clamp(minimap * std + mean, 0.0, 1.0)
        normalmap = torch.clamp(normalmap * std + mean, 0.0, 1.0)

        prediction = prediction - prediction.min()
        prediction = prediction / (prediction.max() + 1e-6)
        prediction = prediction.repeat(3, 1, 1)

        grid = torch.cat([minimap, normalmap, wdl, water, objects, prediction], dim=2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transforms.ToPILImage()(grid).save(output_path)


def save_height_outputs(heightmap_world: np.ndarray, output_dir: Path, tile_name: str) -> None:
    normalized = np.clip((heightmap_world - HEIGHT_GLOBAL_MIN) / max(HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN, 1e-6), 0.0, 1.0)
    height_u16 = (normalized * 65535.0).astype(np.uint16)
    Image.fromarray(height_u16, mode="I;16").save(output_dir / f"{tile_name}_height.png")

    metadata = {
        "tile_name": tile_name,
        "height_encoding": "v7_1_global_range",
        "height_global_min": HEIGHT_GLOBAL_MIN,
        "height_global_max": HEIGHT_GLOBAL_MAX,
        "predicted_height_min": float(heightmap_world.min()),
        "predicted_height_max": float(heightmap_world.max()),
    }
    with open(output_dir / f"{tile_name}_height_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_tile_coords(tile_name: str) -> Tuple[int, int]:
    parts = tile_name.split("_")
    if len(parts) < 3:
        return 0, 0
    return int(parts[-2]), int(parts[-1])


def run_batch_inference(
    model_path: Path,
    dataset_root: Path,
    output_dir: Path,
    tile_filter: Optional[str] = None,
    debug: bool = False,
    z_scale: float = 1.0,
    smooth_sigma: float = 0.0,
    out_res: int = OUTPUT_SIZE,
) -> None:
    engine = V7InferenceEngine(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = dataset_root / "dataset"
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    tiles = sorted(dataset_dir.glob("*.json"))
    print(f"Found {len(tiles)} tiles in {dataset_root}")
    success_count = 0

    for json_file in tiles:
        tile_name = json_file.stem
        if tile_filter and tile_filter not in tile_name:
            continue

        try:
            input_tensor, bounds_info, texture_path = engine.prepare_input(dataset_root, tile_name)
            predicted_heightmap, predicted_bounds = engine.predict(input_tensor)

            heightmap_normalized = predicted_heightmap[0, 0]
            heightmap_world = engine.denormalize_heightmap(heightmap_normalized, bounds_info["g_min"], bounds_info["g_max"])

            if abs(z_scale - 1.0) > 1e-6:
                heightmap_world *= z_scale
            if smooth_sigma > 0:
                heightmap_world = gaussian_filter(heightmap_world, sigma=smooth_sigma)
            if out_res != OUTPUT_SIZE:
                resized = Image.fromarray(heightmap_world, mode="F")
                resized = resized.resize((out_res, out_res), Image.BILINEAR)
                heightmap_world = np.asarray(resized, dtype=np.float32)

            tile_x, tile_y = parse_tile_coords(tile_name)
            engine.save_obj(heightmap_world, output_dir / f"{tile_name}.obj", tile_x, tile_y, texture_path)
            save_height_outputs(heightmap_world, output_dir, tile_name)

            if debug:
                engine.save_debug_image(input_tensor, torch.from_numpy(predicted_heightmap), output_dir / f"{tile_name}_debug.png")

            if predicted_bounds.size > 0:
                with open(output_dir / f"{tile_name}_bounds.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "predicted_bounds": predicted_bounds[0].tolist(),
                            "input_bounds": bounds_info,
                        },
                        handle,
                        indent=2,
                    )

            success_count += 1
        except Exception as exc:
            print(f"Failed {tile_name}: {exc}")

    print(f"Processed {success_count} tiles")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V7.1 multichannel inference.")
    parser.add_argument("--model", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to dataset root containing dataset/ and images/")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--filter", help="Optional string filter for tile names")
    parser.add_argument("--debug", action="store_true", help="Save debug composite images")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Optional scale factor for output world heights")
    parser.add_argument("--smooth-output", type=float, default=0.0, help="Optional Gaussian smoothing sigma")
    parser.add_argument("--res", type=int, default=OUTPUT_SIZE, help="Output mesh and heightmap resolution")
    return parser


if __name__ == "__main__":
    arguments = build_arg_parser().parse_args()
    run_batch_inference(
        model_path=Path(arguments.model),
        dataset_root=Path(arguments.dataset),
        output_dir=Path(arguments.out),
        tile_filter=arguments.filter,
        debug=arguments.debug,
        z_scale=arguments.z_scale,
        smooth_sigma=arguments.smooth_output,
        out_res=arguments.res,
    )
