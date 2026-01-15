"""
WoW Height Regressor V3 - Full ADT Resolution
==============================================
Outputs FULL ADT heightmap (256 chunks × 145 vertices = 37,120 values)
from a 256×256 minimap tile image (native WoW minimap resolution).

Architecture:
- Uses a CNN encoder-decoder (U-Net style) for spatial coherence
- Processes 256×256 input (native minimap), outputs 256×145 height grid
- Preserves spatial relationships across the entire tile

Key differences from V1/V2:
- Full tile input (256×256 native minimap) not chunk crops (64×64)
- Full tile output (37,120 heights) not single chunk (145)
- U-Net architecture instead of ViT for better spatial output
- Proper 2D smoothness loss on the heightmap grid

Usage:
    python train_height_regressor_v3.py
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Configuration
DATASET_ROOTS = [
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalidar_v2"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
]
OUTPUT_DIR = Path(r"j:\vlm_output\wow_height_regressor_v3")
NUM_CHUNKS = 256  # 16×16 chunks per ADT
NUM_HEIGHTS_PER_CHUNK = 145  # WoW vertex count per chunk
TOTAL_HEIGHTS = NUM_CHUNKS * NUM_HEIGHTS_PER_CHUNK  # 37,120

# For output, we'll reshape to a 2D grid
# WoW heightmap is effectively 17×17 outer + 16×16 inner per chunk
# For simplicity, output as 256×145 and reshape during inference
OUTPUT_H = 256  # chunks
OUTPUT_W = 145  # heights per chunk

# Training hyperparameters
BATCH_SIZE = 2  # Small batch due to large images
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
SMOOTHNESS_WEIGHT = 0.05
GRADIENT_WEIGHT = 0.1  # Weight for gradient matching loss


class ConvBlock(nn.Module):
    """Double convolution block with batch norm and ReLU"""
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


class HeightmapUNet(nn.Module):
    """
    U-Net style encoder-decoder for full ADT heightmap prediction.
    
    Input: 4096×4096×3 RGB minimap
    Output: 256×145 heights (reshaped from spatial features)
    
    The network learns to extract terrain features and predict heights
    while maintaining spatial coherence across the entire tile.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder (downsampling path)
        # 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.enc1 = ConvBlock(3, 32)      # 4096 -> 4096
        self.enc2 = ConvBlock(32, 64)     # 2048 -> 2048
        self.enc3 = ConvBlock(64, 128)    # 1024 -> 1024
        self.enc4 = ConvBlock(128, 256)   # 512 -> 512
        self.enc5 = ConvBlock(256, 512)   # 256 -> 256
        self.enc6 = ConvBlock(512, 512)   # 128 -> 128
        self.enc7 = ConvBlock(512, 512)   # 64 -> 64
        self.enc8 = ConvBlock(512, 512)   # 32 -> 32
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck at 16×16
        self.bottleneck = ConvBlock(512, 1024)  # 16 -> 16
        
        # Decoder - output 16×16 feature map, then project to heights
        self.dec1 = ConvBlock(1024 + 512, 512)  # 32
        self.dec2 = ConvBlock(512 + 512, 512)   # 64
        self.dec3 = ConvBlock(512 + 512, 256)   # 128
        self.dec4 = ConvBlock(256 + 512, 256)   # 256
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final projection: 256×256 feature map -> 256×145 heights
        # Each spatial position (16×16 grid) corresponds to a chunk
        self.height_proj = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 145, 1),  # Output 145 channels = heights per chunk
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # 4096
        e2 = self.enc2(self.pool(e1))  # 2048
        e3 = self.enc3(self.pool(e2))  # 1024
        e4 = self.enc4(self.pool(e3))  # 512
        e5 = self.enc5(self.pool(e4))  # 256
        e6 = self.enc6(self.pool(e5))  # 128
        e7 = self.enc7(self.pool(e6))  # 64
        e8 = self.enc8(self.pool(e7))  # 32
        
        # Bottleneck
        b = self.bottleneck(self.pool(e8))  # 16
        
        # Decoder with skip connections
        d1 = self.dec1(torch.cat([self.up(b), e8], dim=1))   # 32
        d2 = self.dec2(torch.cat([self.up(d1), e7], dim=1))  # 64
        d3 = self.dec3(torch.cat([self.up(d2), e6], dim=1))  # 128
        d4 = self.dec4(torch.cat([self.up(d3), e5], dim=1))  # 256
        
        # Project to heights: [B, 256, 256, 256] -> [B, 145, 16, 16]
        # We need 16×16 spatial output for 256 chunks
        d4_pooled = F.adaptive_avg_pool2d(d4, (16, 16))  # [B, 256, 16, 16]
        heights = self.height_proj(d4_pooled)  # [B, 145, 16, 16]
        
        # Reshape to [B, 256, 145] where 256 = 16×16 chunks
        B = heights.shape[0]
        heights = heights.permute(0, 2, 3, 1)  # [B, 16, 16, 145]
        heights = heights.reshape(B, 256, 145)  # [B, 256, 145]
        
        return heights


class HeightmapUNetLite(nn.Module):
    """
    U-Net for 256×256 native WoW minimap input.
    Outputs 256 chunks × 145 heights = 37,120 values.
    
    Input: 256×256×3 RGB minimap (native WoW resolution)
    Output: [B, 256, 145] heights
    """
    def __init__(self):
        super().__init__()
        
        # Encoder for 256×256 input
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = ConvBlock(3, 64)      # 256
        self.enc2 = ConvBlock(64, 128)    # 128
        self.enc3 = ConvBlock(128, 256)   # 64
        self.enc4 = ConvBlock(256, 512)   # 32
        self.enc5 = ConvBlock(512, 512)   # 16
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck at 8×8
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.dec1 = ConvBlock(1024 + 512, 512)  # 16
        self.dec2 = ConvBlock(512 + 512, 512)   # 32
        self.dec3 = ConvBlock(512 + 256, 256)   # 64
        self.dec4 = ConvBlock(256 + 128, 128)   # 128
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Height projection: 16×16 spatial -> 145 heights per position
        # Each of the 16×16 = 256 positions corresponds to one chunk
        self.height_proj = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 145, 1),  # 145 heights per chunk
        )
        
    def forward(self, x):
        # Ensure input is 256×256
        if x.shape[2] != 256 or x.shape[3] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        
        # Encoder
        e1 = self.enc1(x)              # 256
        e2 = self.enc2(self.pool(e1))  # 128
        e3 = self.enc3(self.pool(e2))  # 64
        e4 = self.enc4(self.pool(e3))  # 32
        e5 = self.enc5(self.pool(e4))  # 16
        
        # Bottleneck
        b = self.bottleneck(self.pool(e5))  # 8
        
        # Decoder - go back up to 16×16 (one position per chunk)
        d1 = self.dec1(torch.cat([self.up(b), e5], dim=1))   # 16
        
        # Project 16×16 feature map to heights
        # Each spatial position = one chunk, 145 channels = heights
        heights = self.height_proj(d1)  # [B, 145, 16, 16]
        
        # Reshape to [B, 256, 145]
        B = heights.shape[0]
        heights = heights.permute(0, 2, 3, 1)  # [B, 16, 16, 145]
        heights = heights.reshape(B, 256, 145)  # [B, 256, 145]
        
        return heights


def compute_smoothness_loss_2d(heights):
    """
    2D smoothness loss that respects WoW's chunk grid.
    Heights shape: [B, 256, 145] where 256 = 16×16 chunks
    """
    B = heights.shape[0]
    
    # Reshape to [B, 16, 16, 145] for spatial operations
    h = heights.reshape(B, 16, 16, 145)
    
    # Smoothness within each chunk (adjacent vertices)
    intra_chunk = torch.mean((h[:, :, :, 1:] - h[:, :, :, :-1]) ** 2)
    
    # Smoothness between adjacent chunks (edge continuity)
    # Compare last vertices of chunk[i] with first vertices of chunk[i+1]
    inter_chunk_h = torch.mean((h[:, :, 1:, :] - h[:, :, :-1, :]) ** 2)
    inter_chunk_v = torch.mean((h[:, 1:, :, :] - h[:, :-1, :, :]) ** 2)
    
    return intra_chunk + 0.5 * (inter_chunk_h + inter_chunk_v)


def compute_gradient_loss(pred, target):
    """
    Loss that encourages matching terrain gradients (slopes).
    """
    # Gradient in height dimension
    pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
    target_grad = target[:, :, 1:] - target[:, :, :-1]
    
    return F.mse_loss(pred_grad, target_grad)


class WoWTileDataset(Dataset):
    """
    Full-tile dataset for V3 training.
    Each sample is a complete ADT tile with all 256 chunks.
    """
    def __init__(self, dataset_roots, augment=True):
        self.augment = augment
        self.samples = []
        self.all_heights = []
        
        for root in dataset_roots:
            if not root.exists():
                print(f"Warning: Dataset root not found: {root}")
                continue
            
            dataset_dir = root / "dataset"
            images_dir = root / "images"
            
            if not dataset_dir.exists():
                continue
            
            print(f"Loading from {root.name}...")
            
            for json_path in dataset_dir.glob("*.json"):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Find stitched minimap image
                    stitched_dir = root / "stitched"
                    tile_name = json_path.stem
                    
                    # Try multiple image locations
                    img_candidates = [
                        stitched_dir / f"{tile_name}_minimap.png",
                        images_dir / f"{tile_name}.png",
                        root / data.get("image", ""),
                    ]
                    
                    img_path = None
                    for candidate in img_candidates:
                        if candidate.exists():
                            img_path = candidate
                            break
                    
                    if img_path is None:
                        continue
                    
                    # Extract all chunk heights
                    td = data.get("terrain_data", {})
                    heights_list = td.get("heights", [])
                    
                    if len(heights_list) < 256:
                        continue  # Need all chunks
                    
                    # Build height array [256, 145]
                    tile_heights = np.zeros((256, 145), dtype=np.float32)
                    valid_chunks = 0
                    
                    for chunk in heights_list:
                        idx = chunk.get("idx", -1)
                        h_vals = chunk.get("h", chunk.get("heights", []))
                        
                        if 0 <= idx < 256 and len(h_vals) == 145:
                            tile_heights[idx] = h_vals
                            valid_chunks += 1
                    
                    if valid_chunks < 200:  # Need most chunks
                        continue
                    
                    self.samples.append({
                        "img_path": img_path,
                        "heights": tile_heights,
                        "tile_name": tile_name,
                    })
                    self.all_heights.append(tile_heights.flatten())
                    
                except Exception as e:
                    pass
        
        print(f"Loaded {len(self.samples)} complete tiles")
        
        # Global normalization stats
        if self.all_heights:
            all_h = np.concatenate(self.all_heights)
            self.global_min = float(np.min(all_h))
            self.global_max = float(np.max(all_h))
            self.global_mean = float(np.mean(all_h))
            self.global_std = float(np.std(all_h)) + 1e-6
            
            print(f"Height range: [{self.global_min:.2f}, {self.global_max:.2f}]")
            print(f"Height stats: mean={self.global_mean:.2f}, std={self.global_std:.2f}")
        else:
            self.global_min = 0.0
            self.global_max = 1.0
            self.global_mean = 0.0
            self.global_std = 1.0
    
    def __len__(self):
        return len(self.samples)
    
    def normalize_heights(self, heights):
        """Normalize to [-1, 1] range"""
        h_norm = 2.0 * (heights - self.global_min) / (self.global_max - self.global_min + 1e-6) - 1.0
        return h_norm.astype(np.float32)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load tile minimap image (native 256×256 or resize if different)
        img = Image.open(sample["img_path"]).convert("RGB")
        
        # Ensure 256×256 (native WoW minimap resolution)
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BILINEAR)
        
        # Convert to tensor
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, 256, 256]
        
        # Normalize heights
        heights = self.normalize_heights(sample["heights"])
        heights_tensor = torch.from_numpy(heights)  # [256, 145]
        
        # Data augmentation
        if self.augment and random.random() > 0.5:
            # Horizontal flip
            img_tensor = torch.flip(img_tensor, dims=[2])
            # Flip chunk order horizontally (reverse columns in 16×16 grid)
            h_2d = heights_tensor.reshape(16, 16, 145)
            h_2d = torch.flip(h_2d, dims=[1])
            heights_tensor = h_2d.reshape(256, 145)
        
        return {
            "pixel_values": img_tensor,
            "labels": heights_tensor,
        }


def train():
    print("=" * 60)
    print("WoW Height Regressor V3 - Full ADT Resolution")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = WoWTileDataset(DATASET_ROOTS, augment=True)
    
    if len(dataset) == 0:
        print("ERROR: No complete tiles loaded!")
        print("V3 requires stitched minimap images and full tile height data.")
        return
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    # Create model
    print("Creating HeightmapUNetLite model...")
    model = HeightmapUNetLite()
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Loss
    mse_loss = nn.MSELoss()
    
    # Training loop
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values)
            
            # MSE loss
            loss_mse = mse_loss(outputs, labels)
            
            # Smoothness loss
            loss_smooth = compute_smoothness_loss_2d(outputs)
            
            # Gradient loss
            loss_grad = compute_gradient_loss(outputs, labels)
            
            # Combined loss
            loss = loss_mse + SMOOTHNESS_WEIGHT * loss_smooth + GRADIENT_WEIGHT * loss_grad
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{loss_mse.item():.4f}",
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(pixel_values)
                loss = mse_loss(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= max(len(val_loader), 1)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, OUTPUT_DIR / "best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
    
    # Save final model and stats
    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    
    # Save normalization stats
    stats = {
        "global_min": dataset.global_min,
        "global_max": dataset.global_max,
        "global_mean": dataset.global_mean,
        "global_std": dataset.global_std,
        "output_shape": [256, 145],
        "input_size": 1024,
    }
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
