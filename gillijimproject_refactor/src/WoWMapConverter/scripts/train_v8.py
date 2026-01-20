#!/usr/bin/env python3
"""
WoW Height Regressor V8 - Multi-Head Reconstruction
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from v8_utils import generate_synthetic_minimap

# Configuration
INPUT_SIZE = 160 # Multiple of 32 for UNet
OUTPUT_SIZE = 145 # Native ADT resolution

HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0

class MultiChannelUNetV8(nn.Module):
    def __init__(self, in_channels=16):  # 3mm + 3nm + 4mccv + 1sh + 1wdl + 4masks = 16
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 1024)
        
        # Bottleneck
        self.bottleneck = self._conv_block(1024, 2048)
        
        # Global Heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.head_bounds = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 4)
        )
        
        self.head_tex_emb = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 16)
        )
        
        self.head_obj_emb = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 128)
        )
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec5 = self._conv_block(2048, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Spatial Heads: Terrain(2) + Alpha(4) = 6
        self.out_conv = nn.Conv2d(64, 6, 1)
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        b = self.bottleneck(self.pool(e5))
        
        # Global Heads
        g = self.global_pool(b).view(b.size(0), -1)
        bounds = self.head_bounds(g)
        tex_emb = self.head_tex_emb(g)
        obj_emb = self.head_obj_emb(g)
        
        # Decoder
        d5 = self.up5(b)
        if d5.shape != e5.shape: d5 = F.interpolate(d5, size=e5.shape[2:])
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        if d4.shape != e4.shape: d4 = F.interpolate(d4, size=e4.shape[2:])
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape != e3.shape: d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape: d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape: d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Spatial Output
        out_spatial = torch.sigmoid(self.out_conv(d1)) # [B, 6, H, W]
        
        # Interpolate to 145x145
        if out_spatial.shape[-1] != OUTPUT_SIZE:
            out_spatial = F.interpolate(out_spatial, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear', align_corners=False)
            
        return {
            "terrain": out_spatial[:, 0:2], # Global, Local
            "alpha": out_spatial[:, 2:6],   # 4 Layers
            "bounds": bounds,
            "tex_emb": tex_emb,
            "obj_emb": obj_emb
        }

from v8_utils import generate_synthetic_minimap, load_adt_bin

class WoWTileDatasetV8(Dataset):
    def __init__(self, dataset_roots, texture_library=None, augment=True, limit=None):
        self.samples = []
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.texture_library = texture_library
        
        print("Loading V8 Datasets...")
        count = 0
        for root in dataset_roots:
            if not root.exists(): continue
            
            # Recursively find all 'dataset' directories
            dataset_dirs = list(root.glob("**/dataset"))
            if not dataset_dirs:
                # Fallback: check if root itself contains dataset
                dataset_dir = root / "dataset"
                if dataset_dir.exists():
                    dataset_dirs = [dataset_dir]
            
            for dataset_dir in dataset_dirs:
                for entry in os.scandir(dataset_dir):
                    if entry.is_file() and entry.name.endswith(".json"):
                        try:
                            # Check for binary file
                            bin_path = dataset_dir / f"{entry.name.replace('.json', '.bin')}"
                            has_bin = bin_path.exists()
                            
                            with open(entry.path, 'r') as f: data = json.load(f)
                            
                            self.samples.append({
                                "json": Path(entry.path),
                                "root": root,
                                "data": data,
                                "bin_path": bin_path if has_bin else None
                            })
                            count += 1
                        except: continue
                        
                    if limit and count >= limit: break
            if limit and count >= limit: break
            
        print(f"Loaded {len(self.samples)} samples (Limit: {limit}).")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = sample["data"]
        terrain = data.get("terrain_data", {})
        root = sample["root"]
        
        # 1. Load Data (Binary or JSON/Image fallback)
        if sample["bin_path"]:
            # Load from Binary
            bin_data = load_adt_bin(sample["bin_path"])
            
            if bin_data is None:
                # Fall back to legacy loading if binary is corrupt
                sample["bin_path"] = None
                return self.__getitem__(idx)  # Re-enter with legacy path
            
            # Extract maps (keeping original resolution for processing)
            h_raw = bin_data['heights'] # 145x145
            n_raw = bin_data['normals'] # 145x145x3
            c_raw = bin_data['mccv']    # 145x145x4
            s_raw = bin_data['shadows'] # 1024x1024
            a_raw = bin_data['alphas']  # 4x1024x1024
            
            # Inputs: Resize to INPUT_SIZE (160)
            # Use torch interpolation for resizing
            def resize_tensor(t, size, mode='bilinear'):
                # t: [C, H, W] or [1, H, W]
                if len(t.shape) == 2: t = t.unsqueeze(0)
                if len(t.shape) == 3: t = t.unsqueeze(0) # Batch dim
                if mode == 'nearest':
                    resized = F.interpolate(t, size=(size, size), mode=mode)
                else:
                    resized = F.interpolate(t, size=(size, size), mode=mode, align_corners=False)
                return resized.squeeze(0)

            # Convert to Tensors
            t_norm = torch.from_numpy(n_raw).permute(2,0,1).float() / 127.0 # [-1, 1] approx
            t_mccv = torch.from_numpy(c_raw).permute(2,0,1).float() / 255.0
            t_shadow = torch.from_numpy(s_raw).float().unsqueeze(0) / 1.0 # [1, 1024, 1024]
            t_height = torch.from_numpy(h_raw).float().unsqueeze(0) # [1, 145, 145]
            t_alpha = torch.from_numpy(a_raw).float() / 255.0 # [4, 1024, 1024]

            nm_t = resize_tensor(t_norm, INPUT_SIZE)
            mccv_t = resize_tensor(t_mccv, INPUT_SIZE)
            sh_t = resize_tensor(t_shadow, INPUT_SIZE, mode='nearest') # Shadow is mask
            
            # Prepare Targets (145x145)
            # Heights are already 145.
            # Alphas need downsampling to 145.
            alpha_target = resize_tensor(t_alpha, 145, mode='bilinear')
            
            # Minimap (Still explicit image)
            mm_path = root / ("images/" + sample["json"].stem + ".png")
            if mm_path.exists():
                mm_img = Image.open(mm_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
                mm_t = self.normalize(self.to_tensor(mm_img))
            else:
                mm_t = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)

        else:
            # Legacy Image Loading ... (Simplified)
            # ... reuse existing logic or implement minimal fallback
            def load_img(subpath, mode="RGB", size=INPUT_SIZE):
                path = root / subpath
                if not path.exists(): return Image.new(mode, (size, size))
                try: return Image.open(path).convert(mode).resize((size, size), Image.BILINEAR)
                except: return Image.new(mode, (size, size))

            minimap = load_img("images/" + sample["json"].stem + ".png")
            normal = load_img(terrain.get("normalmap", ""))
            mccv = load_img(terrain.get("mccv_map", ""))
            
            shadow_path = terrain.get("shadow_maps", [None])[0]
            shadow = load_img(shadow_path, "L") if shadow_path else Image.new("L", (INPUT_SIZE, INPUT_SIZE))

            mm_t = self.normalize(self.to_tensor(minimap))
            nm_t = self.normalize(self.to_tensor(normal))
            mccv_t = self.to_tensor(mccv) 
            sh_t = self.to_tensor(shadow)
            
            # Targets (Legacy)
            t_height = torch.tensor(data.get("terrain_data", {}).get("height_values", np.zeros((145,145)))).float().unsqueeze(0)
            
            # Legacy Alpha Target (Zeros for now or load if needed)
            alpha_target = torch.zeros(4, 145, 145)

        # WDL (Placeholder)
        wdl = torch.zeros(1, INPUT_SIZE, INPUT_SIZE) 
        masks = [torch.zeros(1, INPUT_SIZE, INPUT_SIZE) for _ in range(5)]
        
        input_tensor = torch.cat([mm_t, nm_t, mccv_t, sh_t, wdl] + masks[:4], dim=0)

        # Finalize Targets
        # Normalize Heights
        heights_norm = (t_height - HEIGHT_GLOBAL_MIN) / (HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN)
        
        # Embeddings
        layers = data.get("texture_layers", [])
        tex_embeddings = []
        if self.texture_library:
            for l in layers[:4]:
                tex_embeddings.append(self.texture_library.get_embedding(l.get("texture_path", "")))
        
        primary_tex_emb = tex_embeddings[0] if tex_embeddings else np.zeros(16, dtype=np.float32)

        return {
            "input": input_tensor.float(), 
            "targets": {
                "height": heights_norm,
                "alpha": alpha_target,
                "tex_emb": torch.tensor(primary_tex_emb).float(),
                "obj_emb": torch.zeros(128).float()
            }
        }

    def get_targets(self, sample):
        pass # Deprecated by inline logic

def train_v8(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training V8 on {device}")
    
    # Init Models & Libraries
    model = MultiChannelUNetV8().to(device)
    
    # Load Libraries
    from texture_library import TextureLibrary
    tex_lib = TextureLibrary(Path(args.dataset) / "tilesets") if Path(args.dataset).joinpath("tilesets").exists() else None
    if args.texture_db and Path(args.texture_db).exists():
        tex_lib = TextureLibrary(Path("."))
        tex_lib.load(Path(args.texture_db))
    
    # Dataset
    dataset = WoWTileDatasetV8([Path(args.dataset)], texture_library=tex_lib, limit=args.limit)
    if len(dataset) == 0:
        print("ERROR: No samples found in dataset!")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 for Windows
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Losses
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()  # Use logits version to avoid NaN
    criterion_cos = nn.CosineEmbeddingLoss()
    
    best_loss = float('inf')
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in loop:
            # Inputs
            inputs = batch["input"].to(device)
            targets = batch["targets"]
            
            # Move targets to device and sanitize
            t_height = targets["height"].to(device)
            t_height = torch.nan_to_num(t_height, nan=0.0, posinf=HEIGHT_GLOBAL_MAX, neginf=HEIGHT_GLOBAL_MIN)
            # Normalize heights to [0, 1] range
            t_height = (t_height - HEIGHT_GLOBAL_MIN) / (HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN)
            t_height = t_height.clamp(0, 1)
            
            t_alpha = targets["alpha"].to(device).clamp(0, 1)
            t_tex_emb = targets["tex_emb"].to(device)
            t_tex_emb = torch.nan_to_num(t_tex_emb, nan=0.0)
            t_obj_emb = targets["obj_emb"].to(device)
            t_obj_emb = torch.nan_to_num(t_obj_emb, nan=0.0)
            
            # Forward
            outputs = model(inputs)
            
            # --- Loss Calculation ---
            # 1. Terrain Loss (MSE on height prediction)
            # outputs["terrain"] is [B, 2, 145, 145], use first channel for global height
            pred_height = outputs["terrain"][:, 0:1]  # [B, 1, 145, 145]
            loss_height = criterion_mse(pred_height, t_height)
            
            # 2. Alpha Loss (BCE on alpha masks)
            pred_alpha = outputs["alpha"]  # [B, 4, 145, 145] (raw logits)
            t_alpha = t_alpha.clamp(0, 1)  # Ensure valid range
            loss_alpha = criterion_bce(pred_alpha, t_alpha)
            
            # 3. Texture Embedding Loss (Cosine Similarity)
            pred_tex = outputs["tex_emb"]  # [B, 16]
            # CosineEmbeddingLoss expects target labels (+1 = similar)
            ones = torch.ones(pred_tex.size(0)).to(device)
            loss_tex = criterion_cos(pred_tex, t_tex_emb, ones)
            
            # 4. Object Embedding Loss (Cosine Similarity)
            pred_obj = outputs["obj_emb"]  # [B, 128]
            loss_obj = criterion_cos(pred_obj, t_obj_emb, ones)
            
            # Combined Loss (Weighted)
            loss = loss_height * 1.0 + loss_alpha * 0.5 + loss_tex * 0.1 + loss_obj * 0.1
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), h=loss_height.item(), a=loss_alpha.item())
        
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved new best model to {args.save_path}")
            
    print(f"Training complete. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--texture-db", help="Path to texture library index")
    parser.add_argument("--limit", type=int, help="Limit samples")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", default="vlm_v8.pt")
    
    args = parser.parse_args()
    
    if args.dataset:
        train_v8(args)
    else:
        # Test Init
        model = MultiChannelUNetV8()
        print("V8 Model Initialized (Test Mode)")
