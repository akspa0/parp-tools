"""
WoW Minimap to Mesh V3 - Full ADT Resolution
=============================================
Converts a 256×256 minimap tile (native WoW resolution) to a complete ADT heightmap mesh.

This uses the V3 model which outputs all 256 chunks × 145 heights = 37,120 vertices.

Usage:
    python img2mesh_v3.py <minimap_image> [--output <output.obj>] [--smooth]

Example:
    python img2mesh_v3.py Kalimdor_35_12.png --output terrain.obj
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

# Model path - check local repo first, then external location
MODEL_PATH = Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\wow_height_regressor_v3")
if not MODEL_PATH.exists():
    MODEL_PATH = Path(r"j:\vlm_output\wow_height_regressor_v3")

# Default output directory for inference results
DEFAULT_OUTPUT_ROOT = Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_output\model_outputs")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class HeightmapUNetLite(nn.Module):
    """U-Net for 256×256 native WoW minimap input. Outputs heights and optionally normals."""
    def __init__(self, predict_normals=True, in_channels=5):
        super().__init__()
        self.predict_normals = predict_normals
        self.in_channels = in_channels
        
        # Encoder for 256×256 input
        self.enc1 = ConvBlock(in_channels, 64)  # 256 - accepts variable input channels
        self.enc2 = ConvBlock(64, 128)    # 128
        self.enc3 = ConvBlock(128, 256)   # 64
        self.enc4 = ConvBlock(256, 512)   # 32
        self.enc5 = ConvBlock(512, 512)   # 16
        
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)  # 8
        
        self.dec1 = ConvBlock(1024 + 512, 512)  # 16
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Chunk position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        # Height projection: 512 (decoder) + 64 (position) = 576 channels
        self.height_proj = nn.Sequential(
            nn.Conv2d(512 + 64, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 145, 1),
        )
        
        # Normal projection (matches training script)
        if predict_normals:
            self.normal_proj = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 145 * 3, 1),
                nn.Tanh(),
            )
        
    def forward(self, x, chunk_positions=None):
        B = x.shape[0]
        
        # Ensure 256×256 input
        if x.shape[2] != 256 or x.shape[3] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        
        e1 = self.enc1(x)              # 256
        e2 = self.enc2(self.pool(e1))  # 128
        e3 = self.enc3(self.pool(e2))  # 64
        e4 = self.enc4(self.pool(e3))  # 32
        e5 = self.enc5(self.pool(e4))  # 16
        
        b = self.bottleneck(self.pool(e5))  # 8
        
        d1 = self.dec1(torch.cat([self.up(b), e5], dim=1))  # [B, 512, 16, 16]
        
        # Embed chunk positions
        if chunk_positions is not None:
            pos_flat = chunk_positions.reshape(B * 256, 3)
            pos_embed = self.pos_embed(pos_flat).reshape(B, 16, 16, 64).permute(0, 3, 1, 2)
        else:
            pos_embed = torch.zeros(B, 64, 16, 16, device=x.device)
        
        d1_with_pos = torch.cat([d1, pos_embed], dim=1)  # [B, 576, 16, 16]
        
        heights = self.height_proj(d1_with_pos)  # [B, 145, 16, 16]
        heights = heights.permute(0, 2, 3, 1).reshape(B, 256, 145)
        
        if self.predict_normals:
            normals = self.normal_proj(d1)  # [B, 435, 16, 16]
            normals = normals.permute(0, 2, 3, 1).reshape(B, 256, 145, 3)
            return heights, normals
        
        return heights


def load_model(model_path):
    """Load trained V3 model and normalization stats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load stats
    stats_path = model_path / "normalization_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Check model configuration from stats
    predict_normals = stats.get("predict_normals", True)
    in_channels = stats.get("in_channels", 5)  # Default to 5 for new models
    
    # Load model
    model = HeightmapUNetLite(predict_normals=predict_normals, in_channels=in_channels)
    
    checkpoint_path = model_path / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_path / "final_model.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Model not found in {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model, stats, device


def predict_heights(model, image_path, stats, device):
    """Predict full ADT heights (and normals) from minimap image."""
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Resize to 256×256 (native WoW minimap resolution)
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BILINEAR)
    
    # Convert to tensor
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]
    
    # Check if model expects 5 channels (RGB + Shadow + Alpha)
    in_channels = stats.get("in_channels", 3)
    if in_channels == 5:
        # Add placeholder shadow and alpha channels (zeros)
        # In production, these would come from the dataset
        shadow_tensor = torch.zeros(1, 256, 256)
        alpha_tensor = torch.zeros(1, 256, 256)
        img_tensor = torch.cat([img_tensor, shadow_tensor, alpha_tensor], dim=0)  # [5, H, W]
    
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
    
    # Predict (no chunk positions during inference - use zeros)
    with torch.no_grad():
        output = model(img_tensor, chunk_positions=None)
    
    # Handle model returning heights only or heights + normals
    if isinstance(output, tuple):
        pred_heights_norm, pred_normals = output
        pred_normals = pred_normals.squeeze(0).cpu().numpy()  # [256, 145, 3]
    else:
        pred_heights_norm = output
        pred_normals = None
    
    pred_heights_norm = pred_heights_norm.squeeze(0).cpu().numpy()  # [256, 145]
    
    # Denormalize heights
    global_min = stats["global_min"]
    global_max = stats["global_max"]
    pred_heights = (pred_heights_norm + 1.0) / 2.0 * (global_max - global_min) + global_min
    
    return pred_heights, pred_normals


def heights_to_mesh(heights, output_path, tile_size=533.33333, smooth=False):
    """
    Convert 256×145 heights to OBJ mesh.
    
    WoW terrain layout:
    - 16×16 chunks per ADT
    - Each chunk is 33.33333 units
    - 145 vertices per chunk in 9+8 interleaved pattern
    """
    chunk_size = tile_size / 16  # ~33.33333
    vertex_spacing = chunk_size / 8  # ~4.1667
    
    # Reshape to [16, 16, 145]
    heights_grid = heights.reshape(16, 16, 145)
    
    if smooth:
        # Apply Gaussian smoothing
        try:
            from scipy.ndimage import gaussian_filter
            for cy in range(16):
                for cx in range(16):
                    h = heights_grid[cy, cx].reshape(17, 9)[:9, :9]  # Approximate
                    heights_grid[cy, cx, :81] = gaussian_filter(h, sigma=0.5).flatten()
        except ImportError:
            print("Warning: scipy not available, skipping smoothing")
    
    vertices = []
    faces = []
    uvs = []
    
    # Generate vertices for each chunk
    for chunk_y in range(16):
        for chunk_x in range(16):
            chunk_idx = chunk_y * 16 + chunk_x
            chunk_heights = heights_grid[chunk_y, chunk_x]
            
            # Chunk origin in world space
            origin_x = chunk_x * chunk_size
            origin_y = chunk_y * chunk_size
            
            # WoW's 9+8 interleaved vertex pattern
            # Outer ring: 9×9 = 81 vertices
            # Inner ring: 8×8 = 64 vertices
            # Total: 145 vertices
            
            chunk_verts = []
            chunk_uvs = []
            
            # Generate 9×9 outer grid + 8×8 inner grid
            h_idx = 0
            for row in range(17):  # 9 outer + 8 inner rows
                if row % 2 == 0:
                    # Outer row (9 vertices)
                    for col in range(9):
                        x = origin_x + col * vertex_spacing
                        y = origin_y + (row // 2) * vertex_spacing
                        z = chunk_heights[h_idx] if h_idx < 145 else 0
                        
                        vertices.append((x, y, z))
                        uvs.append((x / tile_size, y / tile_size))
                        chunk_verts.append(len(vertices))
                        h_idx += 1
                else:
                    # Inner row (8 vertices, offset by half spacing)
                    for col in range(8):
                        x = origin_x + (col + 0.5) * vertex_spacing
                        y = origin_y + (row // 2 + 0.5) * vertex_spacing
                        z = chunk_heights[h_idx] if h_idx < 145 else 0
                        
                        vertices.append((x, y, z))
                        uvs.append((x / tile_size, y / tile_size))
                        chunk_verts.append(len(vertices))
                        h_idx += 1
            
            # Generate faces (triangles)
            # Connect outer and inner vertices
            for row in range(8):
                outer_row_start = row * 17  # 9 + 8 per two rows
                inner_row_start = row * 17 + 9
                next_outer_start = (row + 1) * 17
                
                for col in range(8):
                    # Get vertex indices (1-based for OBJ)
                    v_base = chunk_idx * 145
                    
                    # Simplified triangulation
                    tl = v_base + outer_row_start + col + 1
                    tr = v_base + outer_row_start + col + 2
                    bl = v_base + next_outer_start + col + 1
                    br = v_base + next_outer_start + col + 2
                    center = v_base + inner_row_start + col + 1
                    
                    if center <= len(vertices) and br <= len(vertices):
                        # 4 triangles around center
                        faces.append((tl, tr, center))
                        faces.append((tr, br, center))
                        faces.append((br, bl, center))
                        faces.append((bl, tl, center))
    
    # Write OBJ
    with open(output_path, 'w') as f:
        f.write(f"# WoW ADT Heightmap Mesh (V3)\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")
    
    print(f"Wrote {len(vertices)} vertices, {len(faces)} faces to {output_path}")


def create_vlm_dataset_json(heights, normals, tile_name, output_dir):
    """
    Create a VLM-compatible dataset JSON from predicted heights/normals.
    This JSON can be used with MinimapBakeService to regenerate minimaps.
    
    Returns the dataset dict for further processing.
    """
    # Build chunk heights array
    chunk_heights = []
    for chunk_idx in range(256):
        chunk_heights.append({
            "idx": chunk_idx,
            "h": heights[chunk_idx].tolist()
        })
    
    # Build chunk layers with normals (if available)
    chunk_layers = []
    for chunk_idx in range(256):
        layer_data = {
            "idx": chunk_idx,
            "layers": [],  # Would need texture info from original dataset
            "area_id": 0,
            "flags": 0,
        }
        
        # Add normals if predicted
        if normals is not None:
            # Convert normalized floats back to signed bytes
            chunk_normals = normals[chunk_idx]  # [145, 3]
            normals_sbytes = []
            for v in range(145):
                for c in range(3):
                    val = int(chunk_normals[v, c] * 127)
                    val = max(-127, min(127, val))
                    normals_sbytes.append(val)
            layer_data["normals"] = normals_sbytes
        
        chunk_layers.append(layer_data)
    
    # Calculate chunk positions (16x16 grid)
    chunk_size = 533.33333 / 16  # ~33.33333
    chunk_positions = []
    for chunk_y in range(16):
        for chunk_x in range(16):
            x = chunk_x * chunk_size + chunk_size / 2
            y = chunk_y * chunk_size + chunk_size / 2
            chunk_idx = chunk_y * 16 + chunk_x
            z = float(np.mean(heights[chunk_idx]))  # Average height
            chunk_positions.extend([x, y, z])
    
    # Build terrain data structure matching VlmTerrainData
    terrain_data = {
        "adt_tile": tile_name,
        "heights": chunk_heights,
        "chunk_positions": chunk_positions,
        "holes": [0] * 256,  # No holes in predicted data
        "height_min": float(heights.min()),
        "height_max": float(heights.max()),
        "chunk_layers": chunk_layers,
        "textures": [],  # Would need from original
        "objects": [],
    }
    
    # Full sample structure
    dataset = {
        "image": f"{tile_name}_minimap.png",
        "terrain_data": terrain_data,
    }
    
    # Save JSON
    json_path = output_dir / f"{tile_name}.json"
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Wrote VLM dataset JSON: {json_path}")
    return dataset


def save_heightmap_image(heights, output_path):
    """Save heights as a 16-bit grayscale PNG for visualization."""
    # Reshape to 16x16 grid, take center height per chunk
    heights_grid = heights.reshape(16, 16, 145)
    
    # Create a higher-res heightmap by interpolating
    # For now, just use the 9x9 outer vertices per chunk
    full_height = 16 * 9  # 144 pixels
    heightmap = np.zeros((full_height, full_height), dtype=np.float32)
    
    for cy in range(16):
        for cx in range(16):
            chunk_h = heights_grid[cy, cx]
            # Extract 9x9 outer grid (first 81 vertices)
            outer = chunk_h[:81].reshape(9, 9)
            heightmap[cy*9:(cy+1)*9, cx*9:(cx+1)*9] = outer
    
    # Normalize to 0-65535 for 16-bit PNG
    h_min, h_max = heightmap.min(), heightmap.max()
    if h_max > h_min:
        heightmap_norm = (heightmap - h_min) / (h_max - h_min) * 65535
    else:
        heightmap_norm = np.zeros_like(heightmap)
    
    heightmap_16bit = heightmap_norm.astype(np.uint16)
    
    # Save using PIL
    img = Image.fromarray(heightmap_16bit, mode='I;16')
    img.save(output_path)
    print(f"Wrote heightmap image: {output_path}")


def save_normalmap_image(normals, output_path):
    """Save normals as RGB normal map image."""
    if normals is None:
        return
    
    # Reshape to 16x16 grid
    normals_grid = normals.reshape(16, 16, 145, 3)
    
    # Create normal map using 9x9 outer vertices per chunk
    full_size = 16 * 9  # 144 pixels
    normalmap = np.zeros((full_size, full_size, 3), dtype=np.float32)
    
    for cy in range(16):
        for cx in range(16):
            chunk_n = normals_grid[cy, cx]
            # Extract 9x9 outer grid (first 81 vertices)
            outer = chunk_n[:81].reshape(9, 9, 3)
            normalmap[cy*9:(cy+1)*9, cx*9:(cx+1)*9] = outer
    
    # Convert from [-1, 1] to [0, 255]
    normalmap_uint8 = ((normalmap + 1.0) / 2.0 * 255).astype(np.uint8)
    
    # Save as RGB
    img = Image.fromarray(normalmap_uint8, mode='RGB')
    img.save(output_path)
    print(f"Wrote normal map image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert minimap to full ADT mesh (V3)")
    parser.add_argument("image", help="Input minimap image (256×256 or any size)")
    parser.add_argument("--output", "-o", help="Output directory (default: creates folder next to image)")
    parser.add_argument("--smooth", action="store_true", help="Apply smoothing")
    parser.add_argument("--model", help="Model directory", default=str(MODEL_PATH))
    parser.add_argument("--json-only", action="store_true", help="Only output JSON, no OBJ")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    # Create output directory with timestamp
    tile_name = image_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: test_output/model_outputs/<timestamp>_<tile_name>/
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{tile_name}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(args.model)
    
    print(f"=" * 60)
    print(f"WoW Minimap to Mesh V3")
    print(f"=" * 60)
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_path}")
    print()
    
    print(f"Loading model...")
    model, stats, device = load_model(model_path)
    
    print(f"Predicting heights and normals...")
    heights, normals = predict_heights(model, image_path, stats, device)
    
    print(f"Height range: [{heights.min():.2f}, {heights.max():.2f}]")
    if normals is not None:
        print(f"Normals shape: {normals.shape}")
    
    # Copy input image to output
    import shutil
    minimap_copy = output_dir / f"{tile_name}_minimap.png"
    shutil.copy(image_path, minimap_copy)
    print(f"Copied input minimap: {minimap_copy}")
    
    # Generate VLM dataset JSON
    print(f"\nGenerating VLM dataset JSON...")
    dataset = create_vlm_dataset_json(heights, normals, tile_name, output_dir)
    
    # Generate heightmap image
    print(f"Generating heightmap image...")
    heightmap_path = output_dir / f"{tile_name}_heightmap.png"
    save_heightmap_image(heights, heightmap_path)
    
    # Generate normal map image
    if normals is not None:
        print(f"Generating normal map image...")
        normalmap_path = output_dir / f"{tile_name}_normalmap.png"
        save_normalmap_image(normals, normalmap_path)
    
    # Generate OBJ mesh
    if not args.json_only:
        print(f"\nGenerating OBJ mesh...")
        obj_path = output_dir / f"{tile_name}.obj"
        heights_to_mesh(heights, obj_path, smooth=args.smooth)
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"Output files in {output_dir}:")
    print(f"  - {tile_name}.json          (VLM dataset for minimap regeneration)")
    print(f"  - {tile_name}_minimap.png   (input minimap copy)")
    print(f"  - {tile_name}_heightmap.png (16-bit heightmap)")
    if normals is not None:
        print(f"  - {tile_name}_normalmap.png (RGB normal map)")
    if not args.json_only:
        print(f"  - {tile_name}.obj           (3D mesh)")
    print(f"=" * 60)
    print(f"\nTo regenerate minimap from this data, use MinimapBakeService with the JSON.")


if __name__ == "__main__":
    main()
