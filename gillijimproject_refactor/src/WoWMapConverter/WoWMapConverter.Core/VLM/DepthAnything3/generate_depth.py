#!/usr/bin/env python3
"""
DepthAnything3 Depth Map Generator for VLM Dataset Tool

Generates depth maps from minimap PNG images using the DA3Mono-Large model.

Usage:
    python generate_depth.py --input <dir> --output <dir> [--model <name>]
    
Example:
    python generate_depth.py --input ./vlm_output/images --output ./vlm_output/depths
"""

import argparse
import os
import glob
import torch
import numpy as np
from pathlib import Path

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: DepthAnything3 not installed.")
    print("Run: pip install -e . from the Depth-Anything-3 directory")
    print("Or run the setup script: ./setup_da3.sh (Linux) or .\\setup_da3.ps1 (Windows)")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: PIL not installed. Run: pip install pillow")
    exit(1)


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_depth(depth: np.ndarray, path: str):
    """Save depth map as 16-bit grayscale PNG."""
    # Normalize to 0-65535 range for 16-bit precision
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)
    
    depth_16bit = (depth_normalized * 65535).astype(np.uint16)
    img = Image.fromarray(depth_16bit, mode='I;16')
    img.save(path)


def generate_depth_maps(input_dir: str, output_dir: str, model_name: str = "depth-anything/DA3MONO-LARGE"):
    """Generate depth maps for all PNG images in input directory."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PNG images
    patterns = ['*.png', '*.PNG', '*.jpg', '*.jpeg']
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return 0
    
    print(f"Found {len(image_paths)} images")
    print(f"Loading model: {model_name}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    
    # Process images in batches
    batch_size = 4  # Adjust based on GPU memory
    processed = 0
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        try:
            # Run inference
            prediction = model.inference(batch_paths)
            
            # Save depths
            for j, img_path in enumerate(batch_paths):
                depth = prediction.depth[j]  # [H, W] float32
                
                # Generate output path
                filename = Path(img_path).stem + "_depth.png"
                output_path = os.path.join(output_dir, filename)
                
                save_depth(depth, output_path)
                processed += 1
                
            print(f"Processed {processed}/{len(image_paths)} images")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    print(f"Complete: {processed} depth maps saved to {output_dir}")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps from minimap images using DepthAnything3"
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory with PNG images")
    parser.add_argument("--output", "-o", required=True, help="Output directory for depth maps")
    parser.add_argument("--model", "-m", default="depth-anything/DA3MONO-LARGE",
                        help="Model name (default: DA3MONO-LARGE)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input directory not found: {args.input}")
        exit(1)
    
    count = generate_depth_maps(args.input, args.output, args.model)
    exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
