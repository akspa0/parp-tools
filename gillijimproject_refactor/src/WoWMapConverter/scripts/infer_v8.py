#!/usr/bin/env python3
"""
V8 Inference Assembler
Stitches Neural Network predictions with Library lookups to generate valid ADT data.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from train_v8 import MultiChannelUNetV8, HEIGHT_GLOBAL_MIN, HEIGHT_GLOBAL_MAX, INPUT_SIZE, OUTPUT_SIZE
from texture_library import TextureLibrary
from object_library import ObjectLibrary # Assuming this exists or we mock it
# from v8_utils import ... 

class V8Assembler:
    def __init__(self, model_path, texture_db_path, object_db_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing V8 Assembler on {self.device}...")
        
        # Load Model
        self.model = MultiChannelUNetV8().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print("WARNING: No model loaded. Using random weights (Debug Mode).")

        # Load Libraries
        self.tex_lib = TextureLibrary()
        if texture_db_path:
            self.tex_lib.load(texture_db_path)
            print(f"Loaded Texture Library from {texture_db_path}")
            
        self.obj_lib = None
        if object_db_path:
            # self.obj_lib = ObjectLibrary()
            # self.obj_lib.load(object_db_path)
            print("Object Library loading not fully implemented yet.")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def prepare_input(self, minimap_path):
        # Similar to Dataset __getitem__ but for single inference file
        img = Image.open(minimap_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        mm_t = self.normalize(self.to_tensor(img))
        
        # Channels: 3(RGB) + 3(Normal Mock) + 3(MCCV Mock) + 1(Shadow Mock) + 1(WDL Mock) + 4(Mask Mocks)
        # Total 15. For inference from JUST minimap, we mock the rest or use a separate "Estimator" model.
        # V8 Requirement: Requires specific inputs or we use black for missing?
        # Let's assume we mock missing channels with zeros for now or use V7-style heuristics.
        
        zeros_3 = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)
        zeros_1 = torch.zeros(1, INPUT_SIZE, INPUT_SIZE)
        
        # TODO: Better estimation for Normal/MCCV from Minimap?
        # For now: Just Minimap + Zeros
        input_tensor = torch.cat([mm_t, zeros_3, zeros_3, zeros_1, zeros_1, zeros_1, zeros_1, zeros_1, zeros_1], dim=0)
        return input_tensor.unsqueeze(0).to(self.device)

    def assemble_tile(self, minimap_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tile_name = Path(minimap_path).stem
        
        # 1. Inference
        inp = self.prepare_input(minimap_path)
        with torch.no_grad():
            preds = self.model(inp)
            
        # 2. Terrain Processing
        hm_pred = preds["terrain"][0, 0].cpu().numpy() # [H, W]
        # Denormalize
        hm_world = hm_pred * (HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN) + HEIGHT_GLOBAL_MIN
        
        # Save Heightmap
        hm_img = Image.fromarray(hm_world.astype(np.float32), mode='F')
        hm_img.save(output_dir / f"{tile_name}_height.tif") 
        # TIF for float precision, or scale to PNG? 
        # VlmDatasetExporter usually handles normalized. Let's stick to raw.
        
        # 3. Texture Assembly
        tex_emb = preds["tex_emb"][0].cpu().numpy() # [16]
        
        texture_result = "Unknown"
        if self.tex_lib:
            # Search Library
            # FAISS expects list of vectors
            results = self.tex_lib.search(tex_emb, k=1)
            if results:
                texture_result = results[0]["path"] # e.g. "Tileset/..."
        
        # 4. Generate Output JSON (VLM Format)
        # This JSON is what "vlm-decode" C# tool expects
        vlm_json = {
            "tile_name": tile_name,
            "terrain_data": {
                "height_values": hm_world.flatten().tolist(), # Flat array
                # "normalmap": ...
            },
            "texture_layers": [
                {
                    "texture_path": texture_result,
                    "layer_index": 0,
                    # "alpha_map": ... (Use predicted alpha from preds["alpha"])
                }
            ]
        }
        
        with open(output_dir / f"{tile_name}.json", 'w') as f:
            json.dump(vlm_json, f, indent=2)
            
        print(f"Assembled {tile_name}: Texture='{texture_result}'")

def run_batch_assembly(args):
    assembler = V8Assembler(args.model, args.texture_db, args.object_db)
    
    input_dir = Path(args.input)
    output_dir = Path(args.out)
    
    files = list(input_dir.glob("*.png"))
    print(f"Found {len(files)} files to assemble.")
    
    for f in files:
        assembler.assemble_tile(f, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V8 Assembler")
    parser.add_argument("--model", help="Path to best.pt")
    parser.add_argument("--texture-db", help="Path to texture library index")
    parser.add_argument("--object-db", help="Path to object library index")
    parser.add_argument("--input", required=True, help="Input directory of minimaps")
    parser.add_argument("--out", required=True, help="Output directory")
    
    args = parser.parse_args()
    run_batch_assembly(args)
