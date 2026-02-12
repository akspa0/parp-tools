"""
Convert VLM terrain dataset to Unsloth/Qwen3-VL format.

Input: VLM JSON files with terrain data + minimap images
Output: JSONL file compatible with Unsloth vision fine-tuning

Format:
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image"},
      {"type": "text", "text": "Analyze this World of Warcraft terrain tile..."}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "<terrain JSON>"}
    ]}
  ],
  "images": ["path/to/image.png"]
}
"""

import json
import os
import argparse
from pathlib import Path
from typing import Optional
import base64

def create_instruction(tile_name: str) -> str:
    """Create the instruction prompt for the VLM."""
    return f"""Analyze this World of Warcraft terrain minimap tile ({tile_name}). 
Extract and describe the terrain data including:
- Height values for each chunk (145 values per chunk, 256 chunks total)
- Chunk positions (X, Y, Z coordinates)
- Terrain holes (bitmask per chunk)
- Texture information
- Liquid data (water, lava, etc.)
- Object placements (M2 models and WMO world objects)

Output the terrain analysis as structured JSON."""


def clean_texture_name(path: str) -> str:
    """Match C# logic: Extract filename and change extension to .png."""
    # C# VlmDatasetExporter uses: Path.GetFileName(texture) -> Path.ChangeExtension(..., ".png")
    # This means 'Textures/Grass/Dirt.blp' -> 'Dirt.png'
    stem = Path(path).stem
    return f"tilesets/{stem}.png"

def create_response(terrain_data: dict) -> str:
    """Create the assistant response with COMPLETE terrain data."""
    
    # Process textures to point to local PNGs
    raw_textures = terrain_data.get("textures") or []
    processed_textures = [clean_texture_name(t) for t in raw_textures]
    
    # Process chunk_layers to update texture_path references
    chunk_layers = terrain_data.get("chunk_layers") or []
    for layer_obj in chunk_layers:
        if isinstance(layer_obj, dict) and "layers" in layer_obj:
            for layer in layer_obj.get("layers", []):
                if isinstance(layer, dict) and layer.get("texture_path"):
                    layer["texture_path"] = clean_texture_name(layer["texture_path"])

    # Build COMPLETE terrain response - includes ALL exported data
    response = {
        "tile": terrain_data.get("adt_tile", ""),
        
        # Height data (256 chunks × 145 heights each)
        "heights": terrain_data.get("heights") or [],
        "chunk_positions": terrain_data.get("chunk_positions") or [],
        "holes": terrain_data.get("holes") or [],
        
        # Height statistics
        "height_min": terrain_data.get("height_min", 0),
        "height_max": terrain_data.get("height_max", 0),
        
        # Texture data
        "textures": processed_textures,
        
        # Per-chunk layer data (MCLY with normals, alpha, flags)
        "chunk_layers": chunk_layers,
        
        # Shadow data
        "shadow_maps": terrain_data.get("shadow_maps") or [],
        "shadow_bits": terrain_data.get("shadow_bits") or [],
        
        # Alpha masks
        "alpha_masks": terrain_data.get("alpha_masks") or [],
        
        # Liquid data
        "liquids": terrain_data.get("liquids") or [],
        "liquid_mask": terrain_data.get("liquid_mask"),
        "liquid_height": terrain_data.get("liquid_height"),
        "liquid_min": terrain_data.get("liquid_min", 0),
        "liquid_max": terrain_data.get("liquid_max", 0),
        
        # Object placements (M2/WMO)
        "objects": terrain_data.get("objects") or [],
        
        # WDL low-res heightmap
        "wdl_heights": terrain_data.get("wdl_heights"),
    }
    
    return json.dumps(response, separators=(',', ':'))


def convert_sample(json_path: Path, images_dir: Path, output_mode: str = "path") -> Optional[dict]:
    """
    Convert a single VLM sample to Unsloth format.
    
    Args:
        json_path: Path to the VLM JSON file
        images_dir: Directory containing minimap images
        output_mode: "path" for image paths, "base64" for embedded images
    
    Returns:
        Unsloth-formatted sample or None if no image found
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    terrain = data.get("terrain_data", {})
    tile_name = terrain.get("adt_tile", json_path.stem)
    
    # Find corresponding image
    image_rel_path = data.get("image_path", "")
    if image_rel_path:
        image_path = images_dir.parent / image_rel_path
    else:
        # Try to find by tile name
        image_path = images_dir / f"{tile_name}.png"
    
    if not image_path.exists():
        print(f"Warning: Image not found for {tile_name}: {image_path}")
        return None
    
    # Create the Unsloth format
    instruction = create_instruction(tile_name)
    response = create_response(terrain)
    
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ],
        "images": [str(image_path.absolute())]
    }
    
    return sample


def convert_dataset(input_dir: str, output_file: str, limit: Optional[int] = None):
    """
    Convert entire VLM dataset to Unsloth JSONL format.
    
    Args:
        input_dir: Directory containing VLM export (dataset/, images/ folders)
        output_file: Output JSONL file path
        limit: Optional limit on number of samples
    """
    input_path = Path(input_dir)
    dataset_dir = input_path / "dataset"
    images_dir = input_path / "images"
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    if not images_dir.exists():
        print(f"Warning: Images directory not found: {images_dir}")
    
    json_files = sorted(dataset_dir.glob("*.json"))
    if limit:
        json_files = json_files[:limit]
    
    print(f"Converting {len(json_files)} samples...")
    
    converted = 0
    skipped = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for json_path in json_files:
            sample = convert_sample(json_path, images_dir)
            if sample:
                out.write(json.dumps(sample) + '\n')
                converted += 1
            else:
                skipped += 1
            
            if converted % 100 == 0 and converted > 0:
                print(f"  Converted {converted} samples...")
    
    print(f"\nConversion complete:")
    print(f"  Converted: {converted}")
    print(f"  Skipped (no image): {skipped}")
    print(f"  Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert VLM dataset to Unsloth format")
    parser.add_argument("input_dir", help="VLM export directory (containing dataset/ and images/)")
    parser.add_argument("-o", "--output", default="unsloth_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("-l", "--limit", type=int, help="Limit number of samples")
    
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output, args.limit)


if __name__ == "__main__":
    main()
