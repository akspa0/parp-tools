#!/usr/bin/env python3
"""
WoW Height Regressor V5 - Multi-Channel Input

Uses minimap (RGB) + normalmap (RGB) as 6-channel input to predict
129x129 heightmap output. This approach leverages:
- Minimap: texture/biome context
- Normalmap: surface orientation (directly correlates with height gradients)

Output: 129x129 heightmap (smooth, continuous grid)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Dataset paths
DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_shadowfang_v1"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
]
OUTPUT_DIR = Path(r"J:\vlm_output\wow_height_regressor_v5_multichannel")

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
EARLY_STOP_PATIENCE = 50

# Image sizes
INPUT_SIZE = 256  # Resize inputs to this
OUTPUT_SIZE = 129  # Native heightmap grid size


class MultiChannelUNet(nn.Module):
    """
    U-Net style encoder-decoder for 6-channel input -> 129x129 heightmap.
    
    Input: [B, 6, 256, 256] (minimap RGB + normalmap RGB)
    Output: [B, 1, 129, 129] (heightmap)
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder (6 channels -> features)
        self.enc1 = self._conv_block(6, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, 1, 1)
        
        # Pool for encoder
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
        e1 = self.enc1(x)  # 256 -> 256
        e2 = self.enc2(self.pool(e1))  # 128
        e3 = self.enc3(self.pool(e2))  # 64
        e4 = self.enc4(self.pool(e3))  # 32
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 16
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))  # 32
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # 64
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # 128
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # 256
        
        # Final output
        out = self.final(d1)  # [B, 1, 256, 256]
        
        # Resize to 129x129
        out = F.interpolate(out, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear', align_corners=True)
        
        return out


class WoWTileDatasetV5(Dataset):
    """Dataset loading minimap + normalmap -> heightmap pairs."""
    
    def __init__(self, dataset_roots, input_size=256, augment=True):
        self.input_size = input_size
        self.augment = augment
        self.samples = []
        
        print("Loading dataset...")
        
        for root in dataset_roots:
            if not root.exists():
                continue
            
            images_dir = root / "images"
            if not images_dir.exists():
                continue
            
            # Find all tiles that have minimap, normalmap, and heightmap_v2
            minimap_files = list(images_dir.glob("*[!_]*.png"))
            minimap_files = [f for f in minimap_files if "_" not in f.stem or 
                           (f.stem.count("_") == 2 and not any(x in f.stem for x in ["heightmap", "normalmap", "preview"]))]
            
            for minimap_path in minimap_files:
                tile_name = minimap_path.stem
                normalmap_path = images_dir / f"{tile_name}_normalmap.png"
                heightmap_path = images_dir / f"{tile_name}_heightmap_v2_preview.png"
                
                if normalmap_path.exists() and heightmap_path.exists():
                    self.samples.append({
                        "minimap": minimap_path,
                        "normalmap": normalmap_path,
                        "heightmap": heightmap_path,
                        "tile_name": tile_name,
                    })
        
        print(f"Loaded {len(self.samples)} tiles with all required images")
        
        # Image transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        minimap = Image.open(sample["minimap"]).convert("RGB")
        normalmap = Image.open(sample["normalmap"]).convert("RGB")
        heightmap = Image.open(sample["heightmap"]).convert("L")
        
        # Resize inputs to INPUT_SIZE
        minimap = minimap.resize((self.input_size, self.input_size), Image.BILINEAR)
        normalmap = normalmap.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Heightmap stays at 129x129 (or resize if needed)
        heightmap = heightmap.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        
        # Convert to tensors
        minimap_t = self.to_tensor(minimap)
        normalmap_t = self.to_tensor(normalmap)
        heightmap_t = self.to_tensor(heightmap)
        
        # Normalize RGB inputs
        minimap_t = self.normalize(minimap_t)
        normalmap_t = self.normalize(normalmap_t)
        
        # Concatenate minimap + normalmap -> 6 channels
        input_tensor = torch.cat([minimap_t, normalmap_t], dim=0)
        
        # Augmentation: random horizontal flip
        if self.augment and torch.rand(1).item() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            heightmap_t = torch.flip(heightmap_t, dims=[2])
        
        return {
            "input": input_tensor,  # [6, 256, 256]
            "target": heightmap_t,  # [1, 129, 129]
            "tile_name": sample["tile_name"],
        }


def train(resume_from=None, epochs=None):
    print("=" * 70)
    print("WoW Height Regressor V5 - Multi-Channel (Minimap + Normalmap)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = WoWTileDatasetV5(DATASET_ROOTS, input_size=INPUT_SIZE, augment=True)
    
    if len(dataset) == 0:
        print("ERROR: No tiles loaded!")
        return
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = MultiChannelUNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs or NUM_EPOCHS)
    
    # Loss function
    criterion = nn.L1Loss()
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    max_epochs = epochs or NUM_EPOCHS
    patience_counter = 0
    
    for epoch in range(start_epoch, max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for batch in pbar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, OUTPUT_DIR / "best_model.pt")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pt")
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train WoW Height Regressor V5")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    args = parser.parse_args()
    
    train(resume_from=args.resume, epochs=args.epochs)


if __name__ == "__main__":
    main()
