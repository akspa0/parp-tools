"""
WoW Terrain VLM Training Guide for Unsloth / Qwen3-VL
======================================================

This notebook demonstrates how to use the COMPLETE VLM terrain dataset
for fine-tuning a Vision Language Model (Qwen3-VL) with Unsloth.

Dataset Structure:
- image: Minimap image path (256x256 PNG)
- terrain_data: Full terrain reconstruction data including:
  * heights: Per-chunk height values (145 floats x 256 chunks)
  * chunk_positions: World coordinates for each chunk (256 x 3)
  * holes: Terrain hole bitmasks per chunk (256 ints)
  * textures: List of texture paths used in tile
  * chunk_layers: Per-chunk texture layer data with alpha masks & normals
  * liquids: Water/lava data per chunk
  * objects: M2 and WMO placements with positions/rotations
  * shadows: Shadow map data (Base64 or PNG paths)
  * wdl_heights: Low-res world heightmap (17x17 + 16x16)

Training Approach:
We train the VLM to understand the relationship between:
  INPUT: Minimap image + terrain description request
  OUTPUT: Structured terrain data (heights, textures, objects, etc.)

This enables the model to learn spatial correlations between
visual minimap appearances and underlying terrain geometry.
"""

# ============================================================================
# SECTION 1: Environment Setup (Run in Colab or local env with GPU)
# ============================================================================

# Uncomment for Colab:
# !pip install unsloth
# !pip install --upgrade transformers datasets accelerate peft trl

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# ============================================================================
# SECTION 2: Dataset Schema Documentation
# ============================================================================

DATASET_SCHEMA = """
VlmTrainingSample (root):
├── image: str                    # Path to minimap PNG (256x256)
├── depth: str | null             # Depth map path (optional)
└── terrain_data: VlmTerrainData
    ├── adt_tile: str             # Tile name e.g. "Azeroth_30_30"
    │
    │ # Height Data (256 chunks × 145 heights each = 37,120 total values)
    ├── heights: VlmChunkHeights[]
    │   ├── idx: int              # Chunk index 0-255
    │   └── h: float[]            # 145 height values
    │
    ├── chunk_positions: float[]  # 256 × 3 (x,y,z) = 768 floats
    ├── holes: int[]              # 256 hole bitmasks
    │
    │ # Texture Data
    ├── textures: str[]           # List of BLP texture paths
    ├── chunk_layers: VlmChunkLayers[]
    │   ├── idx: int              # Chunk index
    │   ├── layers: VlmTextureLayer[]
    │   │   ├── tex_id: int       # Texture index
    │   │   ├── texture_path: str # Full texture path
    │   │   ├── flags: int        # MCLY flags
    │   │   ├── alpha_off: int    # Alpha offset
    │   │   ├── effect_id: int    # Ground effect ID
    │   │   ├── alpha_bits: str   # Base64 alpha mask
    │   │   └── alpha_path: str   # PNG alpha mask path
    │   ├── shadow_path: str      # Per-chunk shadow PNG
    │   ├── normals: int[]        # MCNR normals (448 bytes)
    │   ├── mccv_colors: int[]    # Vertex colors (580 bytes)
    │   ├── area_id: int          # Area ID
    │   └── flags: int            # Chunk flags
    │
    │ # Liquid Data
    ├── liquids: VlmLiquidData[]
    │   ├── idx: int              # Chunk index
    │   ├── type: int             # Liquid type (water=0, ocean=1, lava=2...)
    │   ├── min_height: float
    │   ├── max_height: float
    │   ├── mask_path: str        # Liquid mask PNG
    │   └── heights: float[]      # 9×9 = 81 liquid height values
    │
    ├── liquid_mask: str          # Stitched liquid mask path
    ├── liquid_height: str        # Stitched liquid height path
    ├── liquid_min: float
    ├── liquid_max: float
    │
    │ # Object Placements
    ├── objects: VlmObjectPlacement[]
    │   ├── name: str             # Model name
    │   ├── name_id: int
    │   ├── unique_id: int
    │   ├── x, y, z: float        # World position
    │   ├── rot_x, rot_y, rot_z: float  # Rotation
    │   ├── scale: float
    │   └── category: str         # "wmo" or "m2"
    │
    │ # Shadow Data
    ├── shadow_maps: str[]        # Per-chunk shadow PNG paths
    ├── shadow_bits: VlmChunkShadowBits[]
    │   ├── idx: int
    │   └── bits: str             # Base64 (64 bytes = 512 bits)
    │
    │ # Alpha Masks
    ├── alpha_masks: str[]        # Per-layer alpha PNG paths
    │
    │ # WDL Low-Res Heightmap
    ├── wdl_heights: VlmWdlData
    │   ├── outer_17: int[]       # 17×17 = 289 values
    │   └── inner_16: int[]       # 16×16 = 256 values
    │
    │ # Statistics
    ├── height_min: float
    └── height_max: float
"""

print(DATASET_SCHEMA)

# ============================================================================
# SECTION 3: Load and Explore Dataset
# ============================================================================

def load_vlm_sample(json_path: str) -> dict:
    """Load a single VLM training sample."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize_sample(sample: dict) -> dict:
    """Create a human-readable summary of a sample."""
    terrain = sample.get("terrain_data", {})
    
    # Count data elements
    heights = terrain.get("heights") or []
    chunk_layers = terrain.get("chunk_layers") or []
    liquids = terrain.get("liquids") or []
    objects = terrain.get("objects") or []
    textures = terrain.get("textures") or []
    
    return {
        "tile": terrain.get("adt_tile", "Unknown"),
        "image": sample.get("image", ""),
        "num_chunks_with_heights": len(heights),
        "num_chunk_layers": len(chunk_layers),
        "num_textures": len(textures),
        "num_liquids": len(liquids),
        "num_objects": len(objects),
        "height_range": (terrain.get("height_min", 0), terrain.get("height_max", 0)),
        "has_wdl": terrain.get("wdl_heights") is not None,
        "has_liquid_maps": terrain.get("liquid_mask") is not None,
    }

# Example usage:
# sample = load_vlm_sample("path/to/Azeroth_30_30.json")
# print(summarize_sample(sample))

# ============================================================================
# SECTION 4: Create Training Prompts (Full Data)
# ============================================================================

def create_instruction(tile_name: str, task_type: str = "full") -> str:
    """
    Create instruction prompts for different training tasks.
    
    Task types:
    - "full": Extract complete terrain data
    - "heights": Extract only heightmap data
    - "objects": Extract object placements
    - "textures": Extract texture/material data
    - "liquids": Extract water/lava data
    """
    
    base_instruction = f"Analyze this World of Warcraft terrain minimap tile ({tile_name})."
    
    if task_type == "full":
        return f"""{base_instruction}
Extract ALL terrain data including:
1. Heights: Per-chunk elevation values (145 per chunk, 256 chunks)
2. Chunk Positions: World X,Y,Z coordinates for each chunk
3. Terrain Holes: Bitmasks indicating terrain holes
4. Textures: List of ground textures used
5. Texture Layers: Per-chunk texture blending with alpha masks
6. Liquids: Water/lava areas with heights and types
7. Objects: M2 models and WMO buildings with positions
8. Shadows: Shadow map data
9. Statistics: Height min/max

Output as structured JSON."""

    elif task_type == "heights":
        return f"""{base_instruction}
Extract the terrain height data:
- Per-chunk heights (145 values per chunk)
- Chunk world positions (X, Y, Z)
- Height range (min, max)

Output as JSON with 'heights' and 'chunk_positions' arrays."""

    elif task_type == "objects":
        return f"""{base_instruction}
Extract all object placements:
- M2 models (doodads, trees, rocks)
- WMO buildings (structures, caves)
Include position (x,y,z), rotation, scale, and model name.

Output as JSON array of objects."""

    elif task_type == "textures":
        return f"""{base_instruction}
Extract texture and material information:
- List of ground textures used
- Per-chunk texture layers with blending
- Alpha mask references

Output as JSON with 'textures' and 'chunk_layers' data."""

    elif task_type == "liquids":
        return f"""{base_instruction}
Extract liquid (water/lava) data:
- Liquid types and locations
- Height values for liquid surfaces
- Coverage masks

Output as JSON with 'liquids' array."""
    
    return base_instruction


def create_response(terrain_data: dict, task_type: str = "full") -> str:
    """
    Create the expected model response with terrain data.
    
    For training, we provide the COMPLETE ground truth data.
    """
    
    if task_type == "full":
        # Include everything
        response = {
            "tile": terrain_data.get("adt_tile", ""),
            "heights": terrain_data.get("heights", []),
            "chunk_positions": terrain_data.get("chunk_positions", []),
            "holes": terrain_data.get("holes", []),
            "textures": terrain_data.get("textures", []),
            "chunk_layers": terrain_data.get("chunk_layers", []),
            "liquids": terrain_data.get("liquids", []),
            "objects": terrain_data.get("objects", []),
            "shadow_bits": terrain_data.get("shadow_bits", []),
            "wdl_heights": terrain_data.get("wdl_heights"),
            "height_min": terrain_data.get("height_min", 0),
            "height_max": terrain_data.get("height_max", 0),
            "liquid_min": terrain_data.get("liquid_min", 0),
            "liquid_max": terrain_data.get("liquid_max", 0),
        }
    
    elif task_type == "heights":
        response = {
            "heights": terrain_data.get("heights", []),
            "chunk_positions": terrain_data.get("chunk_positions", []),
            "height_min": terrain_data.get("height_min", 0),
            "height_max": terrain_data.get("height_max", 0),
        }
    
    elif task_type == "objects":
        response = {
            "objects": terrain_data.get("objects", []),
        }
    
    elif task_type == "textures":
        response = {
            "textures": terrain_data.get("textures", []),
            "chunk_layers": terrain_data.get("chunk_layers", []),
        }
    
    elif task_type == "liquids":
        response = {
            "liquids": terrain_data.get("liquids", []),
            "liquid_min": terrain_data.get("liquid_min", 0),
            "liquid_max": terrain_data.get("liquid_max", 0),
        }
    
    else:
        response = terrain_data
    
    # Compact JSON (no extra whitespace)
    return json.dumps(response, separators=(',', ':'))


# ============================================================================
# SECTION 5: Convert to Unsloth/Qwen3-VL Format
# ============================================================================

def convert_to_unsloth_format(
    json_path: str,
    images_dir: Path,
    task_type: str = "full"
) -> Optional[dict]:
    """
    Convert a VLM sample to Unsloth conversation format.
    
    Format:
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "<instruction>"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "<terrain JSON>"}
            ]}
        ],
        "images": ["<absolute path to image>"]
    }
    """
    sample = load_vlm_sample(json_path)
    terrain = sample.get("terrain_data", {})
    tile_name = terrain.get("adt_tile", Path(json_path).stem)
    
    # Find image
    image_rel = sample.get("image", "")
    if image_rel:
        image_path = images_dir.parent / image_rel
    else:
        image_path = images_dir / f"{tile_name}.png"
    
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return None
    
    instruction = create_instruction(tile_name, task_type)
    response = create_response(terrain, task_type)
    
    return {
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


def create_training_dataset(
    dataset_dir: str,
    output_file: str,
    task_type: str = "full",
    limit: Optional[int] = None
):
    """
    Convert entire dataset to JSONL for Unsloth training.
    
    Args:
        dataset_dir: Directory with 'dataset/' and 'images/' folders
        output_file: Output JSONL path
        task_type: "full", "heights", "objects", "textures", or "liquids"
        limit: Optional limit on samples
    """
    dataset_path = Path(dataset_dir)
    json_dir = dataset_path / "dataset"
    images_dir = dataset_path / "images"
    
    json_files = sorted(json_dir.glob("*.json"))
    if limit:
        json_files = json_files[:limit]
    
    print(f"Converting {len(json_files)} samples (task: {task_type})...")
    
    converted = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        for json_path in json_files:
            sample = convert_to_unsloth_format(str(json_path), images_dir, task_type)
            if sample:
                out.write(json.dumps(sample) + '\n')
                converted += 1
    
    print(f"Created {output_file} with {converted} samples.")
    return converted


# ============================================================================
# SECTION 6: Unsloth Training Code
# ============================================================================

UNSLOTH_TRAINING_CODE = '''
# ============================================================================
# Unsloth Training Script for WoW Terrain VLM
# Run this in Google Colab or a machine with GPU
# ============================================================================

# 1. Install dependencies
!pip install unsloth
!pip install --upgrade transformers datasets accelerate peft trl

# 2. Load the model
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct",  # or Qwen3-VL when available
    load_in_4bit=True,
    max_seq_length=8192,  # Terrain data is large
)

# 3. Add LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. Load dataset
from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": "train.jsonl"}, split="train")
print(f"Loaded {len(dataset)} training samples")

# 5. Setup trainer
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        max_seq_length=8192,
        per_device_train_batch_size=1,  # Terrain data is large
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Adjust based on dataset size
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="wow_terrain_vlm",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# 6. Train
trainer_stats = trainer.train()
print(f"Training completed in {trainer_stats.metrics['train_runtime']:.1f}s")

# 7. Save model
model.save_pretrained("wow_terrain_vlm_lora")
tokenizer.save_pretrained("wow_terrain_vlm_lora")

# 8. Optional: Save to GGUF for local inference
# model.save_pretrained_gguf("wow_terrain_vlm", tokenizer, quantization_method="q4_k_m")

# 9. Test inference
FastVisionModel.for_inference(model)

from PIL import Image
test_image = Image.open("test_minimap.png")

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Analyze this WoW terrain minimap. Extract the terrain heights and object placements."}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(test_image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=4096, temperature=0.7)
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
'''

print("\n" + "="*60)
print("UNSLOTH TRAINING CODE")
print("="*60)
print(UNSLOTH_TRAINING_CODE)


# ============================================================================
# SECTION 7: Command-Line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert WoW Terrain VLM dataset for Unsloth training"
    )
    parser.add_argument("input_dir", help="VLM export directory")
    parser.add_argument("-o", "--output", default="train.jsonl", 
                        help="Output JSONL file")
    parser.add_argument("-t", "--task", default="full",
                        choices=["full", "heights", "objects", "textures", "liquids"],
                        help="Training task type")
    parser.add_argument("-l", "--limit", type=int, help="Limit samples")
    parser.add_argument("--schema", action="store_true", 
                        help="Print dataset schema and exit")
    
    args = parser.parse_args()
    
    if args.schema:
        print(DATASET_SCHEMA)
    else:
        create_training_dataset(
            args.input_dir,
            args.output,
            task_type=args.task,
            limit=args.limit
        )
