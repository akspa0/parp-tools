#!/usr/bin/env python3
"""
WoW Height Regressor V7 - Inference Engine
Supports:
- 512x512 Resolution
- Super-Resolution UNet (V7)
- Minimap Blurring (Sigma=2.0)
- Dual Heightmap + Bounds Output
- Batch Processing (v30 Dataset Structure)
- Auto-texturing (OBJ + MTL)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter

# Import architecture from train script (assumes in same dir)
# If fails, we define strictly local fallback
try:
    from train_v7 import MultiChannelUNetV7, OUTPUT_SIZE, HEIGHT_GLOBAL_MIN, HEIGHT_GLOBAL_MAX, ALPHA_LAYERS
except ImportError:
    # Fallback definition if train_v7 not importable
    print("Warning: train_v7 not found, using local fallback definitions.")
    OUTPUT_SIZE = 512
    HEIGHT_GLOBAL_MIN = -1000.0
    HEIGHT_GLOBAL_MAX = 3000.0
    ALPHA_LAYERS = 4
    
    class MultiChannelUNetV7(nn.Module):
        """Fallback V7 Definition"""
        def __init__(self, in_channels=9, out_channels=2 + ALPHA_LAYERS):
            super().__init__()
            # ... (Full definition would be needed here if strict decoupling desired)
            raise ImportError("Critical: train_v7 module required for architecture definition.")

TILE_SIZE = 533.33333  # World units
CHUNK_SIZE = TILE_SIZE / 16.0

class V7InferenceEngine:
    def __init__(self, model_path: Path, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        print(f"Loading V7 Model from {model_path} on {self.device}...")
        
        # Load weights to check strict architecture
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Determine Input Channels from first conv layer weight shape
        # weight shape: [out, in, k, k]
        in_channels = state_dict['enc1.0.weight'].shape[1]
        print(f"Detected Input Channels: {in_channels}")
        
        self.model = MultiChannelUNetV7(in_channels=in_channels).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Preprocessing
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
        
    def prepare_input(self, dataset_root: Path, tile_name: str):
        """Load and preprocess inputs for a single tile."""
        json_path = dataset_root / "dataset" / f"{tile_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        terrain = data.get("terrain_data", {})
        
        # Load Images
        mm_path = dataset_root / "images" / f"{tile_name}.png"
        nm_path = dataset_root / terrain.get("normalmap")
        
        if not mm_path.exists() or not nm_path.exists():
            raise FileNotFoundError(f"Missing input images for {tile_name}")
            
        minimap = Image.open(mm_path).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        normalmap = Image.open(nm_path).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        
        # Apply V7 Blurring to Minimap
        minimap = self.blur(minimap)
        
        mm_t = self.normalize(self.to_tensor(minimap))
        nm_t = self.normalize(self.to_tensor(normalmap))
        
        # WDL
        wdl_data = terrain.get("wdl_heights")
        wdl_t = self._render_wdl(wdl_data)
        
        # Bounds Hint
        h_min = terrain.get("height_min", 0.0)
        h_max = terrain.get("height_max", 100.0)
        g_min = terrain.get("height_global_min", HEIGHT_GLOBAL_MIN)
        g_max = terrain.get("height_global_max", HEIGHT_GLOBAL_MAX)
        g_range = g_max - g_min
        
        # Channel 7: H_Min Mask
        h_min_n = np.clip((h_min - g_min) / g_range, 0, 1)
        h_min_mask = torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), h_min_n, dtype=torch.float32)
        
        # Channel 8: H_Max Mask
        h_max_n = np.clip((h_max - g_min) / g_range, 0, 1)
        h_max_mask = torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), h_max_n, dtype=torch.float32)

        # Channel 9: Liquid Mask
        liquid_mask = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        l_path_str = terrain.get("liquid_mask")
        if l_path_str:
            l_path = dataset_root / l_path_str
            if l_path.exists():
                l_img = Image.open(l_path).convert("L").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
                l_t = self.to_tensor(l_img)
                liquid_mask = (l_t > 0.1).float()
        
        # Channel 10: Object Footprint Mask
        object_mask = torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32)
        objects = terrain.get("objects")
        if objects:
            obj_img = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
            tile_size = 533.33333
            for obj in objects:
                px = obj.get("pos_x", 0)
                py = obj.get("pos_y", 0)
                scale = obj.get("scale", 1.0)
                
                # Use actual bounding box if available, otherwise fallback to scale
                bounds_min = obj.get("bounds_min")
                bounds_max = obj.get("bounds_max")
                
                if bounds_min and bounds_max and len(bounds_min) >= 2 and len(bounds_max) >= 2:
                    half_width = abs(bounds_max[0] - bounds_min[0]) * 0.5 * scale
                    half_depth = abs(bounds_max[1] - bounds_min[1]) * 0.5 * scale
                    pixels_per_unit = OUTPUT_SIZE / tile_size
                    radius_x = max(1, int(half_width * pixels_per_unit))
                    radius_y = max(1, int(half_depth * pixels_per_unit))
                else:
                    radius_x = radius_y = max(1, int(5 * scale))
                
                if abs(px) < 2 and abs(py) < 2:
                    nx = int((px + 1) * 0.5 * OUTPUT_SIZE)
                    ny = int((py + 1) * 0.5 * OUTPUT_SIZE)
                else:
                    nx = int((px / tile_size) * OUTPUT_SIZE) % OUTPUT_SIZE
                    ny = int((py / tile_size) * OUTPUT_SIZE) % OUTPUT_SIZE
                
                x1, y1 = max(0, nx - radius_x), max(0, ny - radius_y)
                x2, y2 = min(OUTPUT_SIZE, nx + radius_x), min(OUTPUT_SIZE, ny + radius_y)
                obj_img[y1:y2, x1:x2] = 1.0
                
            object_mask = torch.from_numpy(obj_img).unsqueeze(0)
        
        input_tensor = torch.cat([mm_t, nm_t, wdl_t, h_min_mask, h_max_mask, liquid_mask, object_mask], dim=0).unsqueeze(0) # [1, 11, H, W]
        
        # True bounds for denormalization reference
        true_bounds = {"h_min": h_min, "h_max": h_max, "g_min": g_min, "g_max": g_max}
        
        return input_tensor.to(self.device), true_bounds

    def _render_wdl(self, wdl_data):
        """Render WDL 17x17 to 512x512 tensor (Matches Train Logic)."""
        if not wdl_data: return torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), 0.5)
        outer = np.array(wdl_data.get("outer_17", []), dtype=np.float32)
        if len(outer) != 289: return torch.full((1, OUTPUT_SIZE, OUTPUT_SIZE), 0.5)
        
        grid = outer.reshape(17, 17)
        vmin, vmax = grid.min(), grid.max()
        if vmax - vmin > 1e-6:
            grid = (grid - vmin) / (vmax - vmin)
        else:
            grid[:] = 0.5
            
        img = Image.fromarray((grid * 255).astype(np.uint8), 'L')
        img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        return self.to_tensor(img)

    def predict(self, input_tensor):
        self.last_input_tensor = input_tensor # Store for debug
        with torch.no_grad():
            pred_heightmap, pred_bounds = self.model(input_tensor)
        return pred_heightmap.cpu().numpy(), pred_bounds.cpu().numpy()

    def denormalize_heightmap(self, normalized_heightmap: np.ndarray, g_min: float = HEIGHT_GLOBAL_MIN, g_max: float = HEIGHT_GLOBAL_MAX) -> np.ndarray:
        """Denormalize using GLOBAL bounds (V7 Logic)."""
        return normalized_heightmap * (g_max - g_min) + g_min

    def save_obj(self, heightmap: np.ndarray, output_path: Path, tile_x: int, tile_y: int, texture_path: Path = None):
        """Export textured OBJ mesh."""
        res = heightmap.shape[0]
        step = TILE_SIZE / (res - 1)
        
        vertices = []
        uvs = []
        faces = []
        
        # Vertices & UVs
        for gy in range(res):
            for gx in range(res):
                wx = tile_x * TILE_SIZE + gx * step
                wy = heightmap[gy, gx] # Y-up
                wz = tile_y * TILE_SIZE + gy * step
                vertices.append((wx, wy, wz))
                
                # UVs (Flip V)
                u = gx / (res - 1)
                v = 1.0 - (gy / (res - 1))
                uvs.append((u, v))
                
        # Faces
        for gy in range(res - 1):
            for gx in range(res - 1):
                v0 = gy * res + gx + 1
                v1 = gy * res + (gx + 1) + 1
                v2 = (gy + 1) * res + (gx + 1) + 1
                v3 = (gy + 1) * res + gx + 1
                
                faces.append((v0, v3, v2))
                faces.append((v0, v2, v1))

        # Write Material File
        mtl_path = output_path.with_suffix(".mtl")
        tex_name = texture_path.name if texture_path else "default.png"
        
        if texture_path:
            # Copy texture to output if needed, or reference relative
            # For simplicity, assume texture is alongside obj
            import shutil
            dest_tex = output_path.parent / tex_name
            if not dest_tex.exists():
                shutil.copy(texture_path, dest_tex)
                
        with open(mtl_path, 'w') as f:
            f.write("newmtl TerrainMat\n")
            f.write("Ka 1.0 1.0 1.0\n")
            f.write("Kd 1.0 1.0 1.0\n")
            f.write(f"map_Kd {tex_name}\n")
            
        # Write OBJ
        with open(output_path, 'w') as f:
            f.write(f"mtllib {mtl_path.name}\n")
            f.write(f"usemtl TerrainMat\n")
            
            for v in vertices:
                f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
            
            for vt in uvs:
                f.write(f"vt {vt[0]:.4f} {vt[1]:.4f}\n")
                
            for face in faces:
                f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")

                f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")

        """Save a composite debug image: Minimap | Normal | WDL | Water | Pred"""
        try:
            # Inputs: [1, 9, 512, 512]
            # 0-2: Minimap
            # 3-5: Normal
            # 6: WDL
            # 7: Bounds
            # 8: Water
            
            mm = input_tensor[0, 0:3].cpu()
            nm = input_tensor[0, 3:6].cpu()
            wdl = input_tensor[0, 6:7].cpu()
            water = input_tensor[0, 8:9].cpu() if input_tensor.shape[1] > 8 else torch.zeros_like(wdl)
            pred = pred_heightmap[0, 0:1].cpu() # Global
            
            # Un-normalize RGB
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            mm = torch.clamp(mm * std + mean, 0, 1)
            nm = torch.clamp(nm * std + mean, 0, 1)
            
            # WDL Visualization (Gray)
            wdl = wdl.repeat(3, 1, 1)
            
            # Water Visualization (Blue)
            water_viz = water.repeat(3, 1, 1) * torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
            
            # Pred Visualization (Gray, normalized for display)
            p_min, p_max = pred.min(), pred.max()
            pred_viz = (pred - p_min) / (p_max - p_min + 1e-6)
            pred_viz = pred_viz.repeat(3, 1, 1)
            
            # Stack: MM | NM | WDL | Water | Pred
            grid = torch.cat([mm, nm, wdl, water_viz, pred_viz], dim=2)
            
            transforms.ToPILImage()(grid).save(output_path)
            
        except Exception as e:
            print(f"Debug save failed: {e}")

def run_batch_inference(model_path, dataset_root, output_dir, tile_filter=None, debug=False, z_scale=1.0, smooth_sigma=0.0, out_res=512):
    engine = V7InferenceEngine(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find tiles
    dataset_dir = Path(dataset_root) / "dataset"
    if not dataset_dir.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        return
        
    tiles = sorted(list(dataset_dir.glob("*.json")))
    print(f"Found {len(tiles)} tiles in {dataset_root}")
    
    success_count = 0
    
    for json_file in tiles:
        tile_name = json_file.stem
        
        # Filter
        if tile_filter and tile_filter not in tile_name:
            continue
            
        print(f"Processing {tile_name}...", end="\r")
        
        try:
            # Run Inference
            input_tensor, bounds_info = engine.prepare_input(Path(dataset_root), tile_name)
            pred_hm, pred_bounds = engine.predict(input_tensor)
            
            # Process Output (Global Heightmap only for now)
            # Channel 0 = Global
            hm_norm = pred_hm[0, 0]
            
            # Denormalize
            g_min, g_max = HEIGHT_GLOBAL_MIN, HEIGHT_GLOBAL_MAX
            hm_world = engine.denormalize_heightmap(hm_norm, g_min, g_max)
            
            # 1. Z-Scale
            if abs(z_scale - 1.0) > 1e-6:
                hm_world *= z_scale

            # 2. Smoothing (Gaussian)
            if smooth_sigma > 0:
                hm_world = gaussian_filter(hm_world, sigma=smooth_sigma)

            # 3. Resizing (Downscaling/Upscaling)
            if out_res != OUTPUT_SIZE:
                 # Use PIL for high-quality resize of float32 array
                im = Image.fromarray(hm_world, mode='F')
                im = im.resize((out_res, out_res), Image.BILINEAR)
                hm_world = np.array(im)
            
            # Save
            # Parse Coordinates
            parts = tile_name.split("_")
            tx = int(parts[-2]) if len(parts) >= 3 else 0
            ty = int(parts[-1]) if len(parts) >= 3 else 0
            
            # Save OBJ
            obj_path = output_dir / f"{tile_name}.obj"
            tex_path = Path(dataset_root) / "images" / f"{tile_name}.png"
            
            engine.save_obj(hm_world, obj_path, tx, ty, tex_path)
            
            # Save Heightmap PNG (16-bit)
            # Normalize to 0-65535
            hm_u16 = (hm_norm * 65535).astype(np.uint16)
            Image.fromarray(hm_u16, mode='I;16').save(output_dir / f"{tile_name}_height.png")
            
             if debug:
                 # Pass raw input tensor for debug visualization
                 engine.save_debug_image(engine.last_input_tensor, torch.tensor(pred_hm), output_dir / f"{tile_name}_debug.png")
            
            success_count += 1
            
        except Exception as e:
            print(f"\nFailed {tile_name}: {e}")
            
    print(f"\nBatch Complete. {success_count}/{len(tiles)} processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V7 Inference Tool")
    parser.add_argument("--model", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to v30 dataset root (e.g. .../053_Azeroth_v30)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--filter", help="Optional string filter for tile names (e.g. '32_48')")
    parser.add_argument("--debug", action="store_true", help="Save debug composite images")
    
    # V7 Refinement Args
    parser.add_argument("--z-scale", type=float, default=1.0, help="Scale factor for output Z heights (default: 1.0)")
    parser.add_argument("--smooth-output", type=float, default=0.0, help="Sigma for Gaussian smoothing of output (default: 0.0)")
    parser.add_argument("--res", type=int, default=512, help="Output resolution for mesh/heightmap (default: 512)")

    args = parser.parse_args()
    
    run_batch_inference(args.model, args.dataset, args.out, args.filter, args.debug, args.z_scale, args.smooth_output, args.res)
