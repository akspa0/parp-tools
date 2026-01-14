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


def create_response(terrain_data: dict) -> str:
    """Create the assistant response with terrain data."""
    # Create a simplified but complete terrain summary
    summary = {
        "tile": terrain_data.get("adt_tile", ""),
        "height_range": {
            "min": terrain_data.get("height_min", 0),
            "max": terrain_data.get("height_max", 0)
        },
        "chunk_count": len(terrain_data.get("heights", [])) if terrain_data.get("heights") else 0,
        "texture_count": len(terrain_data.get("textures", [])),
        "textures": terrain_data.get("textures", []),
        "object_count": len(terrain_data.get("objects", [])),
        "liquid_count": len(terrain_data.get("liquids", [])) if terrain_data.get("liquids") else 0,
        "heights": terrain_data.get("heights", []),
        "chunk_positions": terrain_data.get("chunk_positions", []),
        "holes": terrain_data.get("holes", []),
        "objects": terrain_data.get("objects", []),
        "liquids": terrain_data.get("liquids", []),
        "wdl_heights": terrain_data.get("wdl_heights", None)
    }
    return json.dumps(summary, separators=(',', ':'))


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
