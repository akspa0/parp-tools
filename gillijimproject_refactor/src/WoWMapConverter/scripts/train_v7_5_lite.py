#!/usr/bin/env python3
"""
WoW Height Regressor V7.5 LITE - Optimized CPU Training
- Loads cached tensors (.pt) for instant I/O
- Residual Learning (Predicts Height - WDL)
- 4-Level UNet (Reduced Complexity)
- 256x256 Resolution
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

# Config
OUTPUT_DIR = Path(r"./vlm_output/v7_5_lite")
CACHE_FILE_TRAIN = OUTPUT_DIR / "train_cache.pt"
CACHE_FILE_VAL = OUTPUT_DIR / "val_cache.pt"

BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
EARLY_STOP_PATIENCE = 25

class CachedDataset(Dataset):
    def __init__(self, cache_path, augment=False):
        print(f"Loading cache: {cache_path}...")
        self.data = torch.load(cache_path) # List of (input_fp16, target_fp16)
        self.augment = augment
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        inp, tgt = self.data[idx]
        # Dequantize fp16 -> fp32 on the fly
        inp = inp.float()
        tgt = tgt.float()
        
        # Augmentation (Horizontal Flip)
        if self.augment and torch.rand(1).item() > 0.5:
            inp = torch.flip(inp, [2])
            tgt = torch.flip(tgt, [2])
            
        return inp, tgt

class UNetLite(nn.Module):
    """
    4-Level UNet for 256x256. 
    Channels: 32 -> 64 -> 128 -> 256
    """
    def __init__(self, in_channels=10, out_channels=1):
        super().__init__()
        
        # Encoders
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoders
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_channels, 1)
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # No sigmoid, we predict residual which can be negative/positive
        return self.out_conv(d1)

def train(epochs=None):
    print("="*60 + "\nWoW V7.5 LITE - Residual Training (256x256)\n" + "="*60)
    device = torch.device("cpu") # Focus on CPU optimization as requested
    if torch.cuda.is_available(): device = torch.device("cuda")
    print(f"Device: {device}")
    
    if not CACHE_FILE_TRAIN.exists():
        print("Cache not found. Run cache_v7_5_data.py first.")
        return

    train_ds = CachedDataset(CACHE_FILE_TRAIN, augment=True)
    val_ds = CachedDataset(CACHE_FILE_VAL, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Update num_workers=0 for cached tensors
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UNetLite(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs or NUM_EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            # Slice inputs/outputs for PURE MINIMAP training
            # Input: Channel 0-2 (Minimap Only). IDrop Normals/WDL/Masks.
            img = inputs[:, 0:3] 
            # Target: Channel 0 (Global Height). IDrop Residuals.
            target = targets[:, 0:1]

            optimizer.zero_grad()
            # Mixed precision for speed
            with torch.amp.autocast('cuda'):
                pred_height = model(img)
                loss = criterion(pred_height, target)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Progress bar update
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                # Same strict slicing for validation
                img = x[:, 0:3]
                target = y[:, 0:1] # Global Height
                
                with torch.amp.autocast('cuda'):
                    pred_height = model(img)
                    val_loss += criterion(pred_height, target).item()
                    
        avg_t = total_loss / len(train_loader)
        avg_v = val_loss / len(val_loader)
        print(f"Train: {avg_t:.4f} | Val: {avg_v:.4f}")
        
        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)
        
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
            torch.save(model.state_dict(), OUTPUT_DIR / "best_lite.pt")
            print("✓ Saved Best")
            
        scheduler.step(avg_v)

        # --- Preview Generation (Epoch 0 only for speed, or every 5) ---
        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.eval()
            # Just grab the last batch values
            
            # Visualize 1st sample
            # Col 1: Minimap (Input)
            # Col 2: Predicted Height (0-1)
            # Col 3: GT Height (0-1)
            
            # Height is already 0-1, just clamp
            def vis_h(x): return torch.clamp(x, 0, 1)
            
            row = torch.cat([
                img[0], # Minimap is already 0-1 RGB
                vis_h(pred_height[0]).repeat(3, 1, 1), # Grayscale to RGB
                vis_h(target[0]).repeat(3, 1, 1)       # Grayscale to RGB
            ], dim=2)
            
            (OUTPUT_DIR / "previews").mkdir(parents=True, exist_ok=True)
            transforms.ToPILImage()(row).save(OUTPUT_DIR / "previews" / "preview_lite.png")

if __name__ == "__main__":
    train()
