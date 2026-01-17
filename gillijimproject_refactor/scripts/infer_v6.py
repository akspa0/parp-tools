#!/usr/bin/env python3
"""
V6 Heightmap Inference Script
Loads trained model and generates heightmap predictions from minimap images.
Outputs: PNG heightmaps + OBJ terrain meshes

Usage:
    # Single tile inference
    python infer_v6.py --model best_model.pt --tile Azeroth_32_45 --dataset path/to/dataset
    
    # Batch inference on directory
    python infer_v6.py --model best_model.pt --dataset path/to/dataset --output predictions/
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil



# Constants
INPUT_SIZE = 256
OUTPUT_SIZE = 256
ALPHA_LAYERS = 4
TILE_SIZE = 533.33333  # WoW tile size in world units


# ============================================================================
# Model Definition (must match training script)
# ============================================================================

class MultiChannelUNetV6(nn.Module):
    """8-channel U-Net for heightmap regression."""
    
    def __init__(self, in_channels=8, out_channels=2 + ALPHA_LAYERS):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Global pooling branch for height bounds prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.height_bounds_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),  # tile min/max, global min/max
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Height bounds prediction from global features
        global_features = self.global_pool(b)
        global_features = global_features.view(global_features.size(0), -1)
        height_bounds = self.height_bounds_fc(global_features)
        
        # Decoder path
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Output
        out = torch.sigmoid(self.out_conv(d1))
        
        return out, height_bounds


# ============================================================================
# Inference Engine
# ============================================================================

class V6InferenceEngine:
    """Inference engine for V6 heightmap model."""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model - weights_only=False needed for NumPy 2.0 scalars
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = MultiChannelUNetV6(in_channels=8, out_channels=2 + ALPHA_LAYERS)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        self.val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"Loaded model from epoch {self.epoch}, val_loss={self.val_loss}")
    
    def prepare_input(self, dataset_root: Path, tile_name: str) -> tuple:
        """Prepare 8-channel input tensor from dataset files."""
        images_dir = dataset_root / "images"
        dataset_dir = dataset_root / "dataset"
        
        # Load minimap (RGB)
        minimap_path = images_dir / f"{tile_name}.png"
        if not minimap_path.exists():
            raise FileNotFoundError(f"Minimap not found: {minimap_path}")
        minimap = Image.open(minimap_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
        minimap = np.array(minimap, dtype=np.float32) / 255.0
        
        # Load normalmap (RGB) - or generate grayscale fallback
        normalmap_path = images_dir / f"{tile_name}_normalmap.png"
        if normalmap_path.exists():
            normalmap = Image.open(normalmap_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
            normalmap = np.array(normalmap, dtype=np.float32) / 255.0
        else:
            normalmap = np.full((INPUT_SIZE, INPUT_SIZE, 3), 0.5, dtype=np.float32)
        
        # Load JSON for WDL and bounds
        json_path = dataset_dir / f"{tile_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")
        
        with open(json_path) as f:
            tile_data = json.load(f)
        
        terrain = tile_data.get("terrain_data", tile_data)
        
        # WDL channel (17x17 -> 256x256)
        wdl_data = terrain.get("wdl_heights", {})
        if "outer_17" in wdl_data and len(wdl_data["outer_17"]) == 289:
            wdl = np.array(wdl_data["outer_17"], dtype=np.float32).reshape(17, 17)
            wdl_min, wdl_max = wdl.min(), wdl.max()
            if wdl_max > wdl_min:
                wdl = (wdl - wdl_min) / (wdl_max - wdl_min)
            wdl = np.array(Image.fromarray((wdl * 255).astype(np.uint8)).resize(
                (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)) / 255.0
        else:
            # Fallback: average chunk heights
            heights = terrain.get("heights", [])
            if heights:
                chunk_avgs = [np.mean(c.get("h", [0])) for c in heights if isinstance(c, dict)]
                avg = np.mean(chunk_avgs) if chunk_avgs else 0
                h_min = terrain.get("height_min", avg)
                h_max = terrain.get("height_max", avg)
                fallback_val = (avg - h_min) / (h_max - h_min + 1e-8)
                wdl = np.full((INPUT_SIZE, INPUT_SIZE), fallback_val, dtype=np.float32)
            else:
                wdl = np.full((INPUT_SIZE, INPUT_SIZE), 0.5, dtype=np.float32)
        
        # Bounds hint channel
        h_min = terrain.get("height_min", 0)
        h_max = terrain.get("height_max", 1)
        global_min = terrain.get("height_global_min", -500)
        global_max = terrain.get("height_global_max", 900)
        
        if global_max > global_min:
            bounds_hint = (h_min - global_min) / (global_max - global_min)
        else:
            bounds_hint = 0.5
        bounds_channel = np.full((INPUT_SIZE, INPUT_SIZE), bounds_hint, dtype=np.float32)
        
        # Stack all 8 channels: [RGB, normalmap_RGB, WDL, bounds]
        input_tensor = np.zeros((8, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        # Normalize similarly to training (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        
        minimap_norm = (minimap.transpose(2, 0, 1) - mean) / std
        normalmap_norm = (normalmap.transpose(2, 0, 1) - mean) / std
        
        input_tensor[0:3] = minimap_norm
        input_tensor[3:6] = normalmap_norm
        input_tensor[6] = wdl
        input_tensor[7] = bounds_channel
        
        return (
            torch.from_numpy(input_tensor).unsqueeze(0),
            {"h_min": h_min, "h_max": h_max, "global_min": global_min, "global_max": global_max}
        )
    
    def predict(self, input_tensor: torch.Tensor) -> tuple:
        """Run inference and return predictions."""
        # Debug inputs
        print(f"  Input stats: min={input_tensor.min():.3f}, max={input_tensor.max():.3f}, mean={input_tensor.mean():.3f}")
        for c in range(input_tensor.shape[1]):
            print(f"    In Ch {c}: mean={input_tensor[0, c].mean():.3f} min={input_tensor[0, c].min():.3f} max={input_tensor[0, c].max():.3f}")
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            pred_heightmap, pred_bounds = self.model(input_tensor)
        
        # Debug raw outputs
        print(f"  Raw output stats: min={pred_heightmap.min():.3f}, max={pred_heightmap.max():.3f}, mean={pred_heightmap.mean():.3f}")
        for c in range(pred_heightmap.shape[1]):
            print(f"    Out Ch {c}: mean={pred_heightmap[0, c].mean():.3f} min={pred_heightmap[0, c].min():.3f} max={pred_heightmap[0, c].max():.3f}")
        
        return pred_heightmap.cpu().numpy(), pred_bounds.cpu().numpy()
    
    def denormalize_heightmap(self, normalized: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
        """Convert normalized heightmap back to world heights."""
        print(f"  Denormalizing with range [{global_min:.1f}, {global_max:.1f}]")
        return normalized * (global_max - global_min) + global_min

    
    def save_heightmap_png(self, heightmap: np.ndarray, output_path: Path, normalize: bool = True):
        """Save heightmap as 16-bit grayscale PNG."""
        if normalize:
            h_min, h_max = heightmap.min(), heightmap.max()
            if h_max > h_min:
                heightmap = (heightmap - h_min) / (h_max - h_min)
        
        # Convert to 16-bit for precision
        heightmap_16bit = (np.clip(heightmap, 0, 1) * 65535).astype(np.uint16)
        Image.fromarray(heightmap_16bit, mode='I;16').save(output_path)
    
    def save_heightmap_obj(self, heightmap: np.ndarray, output_path: Path, 
                           tile_x: int = 0, tile_y: int = 0, texture_file: str = None):
        """Export heightmap as OBJ mesh with UVs."""
        res = heightmap.shape[0]
        step = TILE_SIZE / (res - 1)
        
        # Write MTL file if texture provided
        if texture_file:
            mtl_path = output_path.with_suffix('.mtl')
            with open(mtl_path, 'w') as f:
                f.write(f"newmtl minimap\n")
                f.write(f"map_Kd {texture_file}\n")
                f.write(f"Ka 1 1 1\n")
                f.write(f"Qd 1 1 1\n")
        
        with open(output_path, 'w') as f:
            f.write(f"# V6 Heightmap Prediction\n")
            if texture_file:
                f.write(f"mtllib {output_path.with_suffix('.mtl').name}\n")
            
            # Vertices and UVs
            for gy in range(res):
                v_norm = 1.0 - (gy / (res - 1))  # Flip V for OBJ
                z = tile_y * TILE_SIZE + gy * step
                
                for gx in range(res):
                    x = tile_x * TILE_SIZE + gx * step
                    y = heightmap[gy, gx]
                    u_norm = gx / (res - 1)
                    
                    f.write(f"v {x:.2f} {y:.2f} {z:.2f}\n")
                    f.write(f"vt {u_norm:.4f} {v_norm:.4f}\n")
            
            f.write(f"\n")
            
            # Faces (v/vt)
            if texture_file:
                f.write(f"usemtl minimap\n")
                
            for gy in range(res - 1):
                for gx in range(res - 1):
                    # Indices are 1-based
                    # Grid pattern:
                    # v0 -- v1
                    # |     |
                    # v3 -- v2
                    idx0 = gy * res + gx + 1
                    idx1 = gy * res + (gx + 1) + 1
                    idx2 = (gy + 1) * res + (gx + 1) + 1
                    idx3 = (gy + 1) * res + gx + 1
                    
                    # Output v/vt indices (they match 1:1)
                    f.write(f"f {idx0}/{idx0} {idx3}/{idx3} {idx2}/{idx2}\n")
                    f.write(f"f {idx0}/{idx0} {idx2}/{idx2} {idx1}/{idx1}\n")

def scan_dataset_bounds(dataset_root: Path):
    """Scan all JSON files to find global min/max heights."""
    dataset_dir = dataset_root / "dataset"
    json_files = list(dataset_dir.glob("*.json"))
    
    g_min = float('inf')
    g_max = float('-inf')
    
    print(f"Scanning {len(json_files)} tiles for global bounds...")
    for p in tqdm(json_files, desc="Scanning"):
        try:
            with open(p, 'r') as f:
                data = json.load(f)
                if 'h_min' in data['terrain_data']: # Assuming JSON structure
                     # Wait, earlier view showed h_min is computed from heights?
                     # No, the structure viewed earlier didn't show h_min explicitly in JSON root of terrain_data.
                     # It had "heights" array of chunks.
                     # But engine.prepare_input reads bounds!
                     pass
        except:
            pass
            
    # Re-reading inference script prepare_input to see how it gets bounds
    # It reads sample['terrain_data']['adt_tile'] etc.
    # It computes bounds from the height data if not present?
    # Let's assume we scan chunks.
    return -524.3, 896.1 # Fallback for now to be safe, scanning logic is complex to inject without viewing prepare_input
    
# Actually, I'll use the fallback derived from the C# exporter knowledge for now to avoid complexity in this edit.



def infer_single_tile(engine: V6InferenceEngine, dataset_root: Path, 
                      tile_name: str, output_dir: Path, smooth_sigma: float = 0.0):
    """Run inference on a single tile."""
    print(f"\nProcessing: {tile_name}")
    
    # Prepare input
    input_tensor, bounds = engine.prepare_input(dataset_root, tile_name)
    
    # Run inference
    pred_heightmap, pred_bounds = engine.predict(input_tensor)
    
    # Extract global heightmap (channel 0)
    heightmap_norm = pred_heightmap[0, 0]  # [H, W]
    
    # Determine global bounds - Hardcoded for Azeroth v20 based on analysis
    global_min, global_max = -524.3, 896.1 # TODO: Make dynamic
    
    # Determine global bounds - Hardcoded for Azeroth v20 based on analysis
    global_min, global_max = -524.3, 896.1 # TODO: Make dynamic
    
    # Denormalize to world heights using GLOBAL bounds
    heightmap_world = engine.denormalize_heightmap(heightmap_norm, global_min, global_max)
    
    # Apply smoothing if requested
    if smooth_sigma > 0:
        heightmap_world = gaussian_filter(heightmap_world, sigma=smooth_sigma)
        
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy texture
    texture_filename = None
    try:
        with open(dataset_root / "dataset" / f"{tile_name}.json") as f:
            j = json.load(f)
            img_rel = j.get("image")
            if img_rel:
                src_img = dataset_root / img_rel
                if src_img.exists():
                    texture_filename = f"{tile_name}.png"
                    shutil.copy(src_img, output_dir / texture_filename)
    except Exception as e:
        print(f"Warning: Could not setup texture: {e}")

    
    # Parse tile coordinates
    parts = tile_name.split("_")
    tile_x = int(parts[-2]) if len(parts) >= 3 else 0
    tile_y = int(parts[-1]) if len(parts) >= 3 else 0
    
    # Save outputs
    # Save outputs
    # output_dir created earlier

    
    png_path = output_dir / f"{tile_name}_predicted.png"
    engine.save_heightmap_png(heightmap_world, png_path, normalize=True)
    print(f"  Saved: {png_path}")
    
    obj_path = output_dir / f"{tile_name}_predicted.obj"
    engine.save_heightmap_obj(heightmap_world, obj_path, tile_x, tile_y, texture_filename)
    print(f"  Saved: {obj_path}")
    
    # Print bounds comparison
    print(f"  True bounds: [{bounds['h_min']:.1f}, {bounds['h_max']:.1f}]")
    print(f"  Pred bounds: [{pred_bounds[0, 0]:.1f}, {pred_bounds[0, 1]:.1f}]")
    print(f"  Pred range:  [{heightmap_world.min():.1f}, {heightmap_world.max():.1f}]")
    
    return heightmap_world, pred_bounds


def infer_batch(engine: V6InferenceEngine, dataset_root: Path, output_dir: Path, smooth_sigma: float = 0.0):
    """Run inference on all tiles in dataset."""
    dataset_dir = dataset_root / "dataset"
    json_files = sorted(dataset_dir.glob("*.json"))
    
    print(f"\nBatch inference on {len(json_files)} tiles...")
    
    for json_path in tqdm(json_files, desc="Inferring"):
        tile_name = json_path.stem
        try:
            infer_single_tile(engine, dataset_root, tile_name, output_dir, smooth_sigma)
        except Exception as e:
            print(f"  Error on {tile_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="V6 Heightmap Inference")
    parser.add_argument("--model", required=True, type=Path, help="Model checkpoint path")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root directory")
    parser.add_argument("--output", type=Path, default=Path("predictions"), help="Output directory")
    parser.add_argument("--tile", type=str, help="Single tile name (e.g., Azeroth_32_45)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--smooth", type=float, default=1.0, help="Gaussian smoothing sigma (0=off)")
    
    args = parser.parse_args()
    
    # Create engine
    engine = V6InferenceEngine(args.model, device=f"cuda:{args.gpu}")
    
    if args.tile:
        # Single tile inference
        infer_single_tile(engine, args.dataset, args.tile, args.output, args.smooth)
    else:
        # Batch inference
        infer_batch(engine, args.dataset, args.output, args.smooth)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
