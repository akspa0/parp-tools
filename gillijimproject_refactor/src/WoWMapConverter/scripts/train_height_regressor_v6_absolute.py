#!/usr/bin/env python3
"""
WoW Height Regressor V6 - Absolute Height Awareness

Key improvements over V5:
- Loads height_min/height_max from JSON for absolute height context
- Adds WDL low-res heightmap as 7th input channel (upscaled 17x17 -> 256x256)
- Model predicts BOTH normalized heightmap AND height bounds (min/max)
- Loss combines heightmap accuracy + height range prediction
- Enables reconstruction of TRUE world heights, not just relative gradients

Input: minimap (3ch) + normalmap (3ch) + WDL hint (1ch) = 7 channels
Output: 256x256 heightmap + height_min + height_max scalars
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
OUTPUT_DIR = Path(r"J:\vlm_output\wow_height_regressor_v6_absolute")

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
EARLY_STOP_PATIENCE = 50

# Image sizes
INPUT_SIZE = 256
OUTPUT_SIZE = 256

# Height normalization - WoW world coordinate range (approximate)
# Most terrain is between -500 and 2000 world units
HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
HEIGHT_GLOBAL_RANGE = HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN


class MultiChannelUNetV6(nn.Module):
    """
    7-channel U-Net that predicts:
    - 256x256 normalized heightmap
    - height_min scalar
    - height_max scalar
    """
    
    def __init__(self, in_channels=7, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Global pooling branch for height bounds prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.height_bounds_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),  # min, max
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        
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
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Height bounds prediction from global features
        global_features = self.global_pool(b)
        global_features = global_features.view(global_features.size(0), -1)
        height_bounds = self.height_bounds_fc(global_features)  # [batch, 2]
        
        # Decoder path
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
        
        # Output heightmap (sigmoid for 0-1 range)
        heightmap = torch.sigmoid(self.out_conv(d1))
        
        # Ensure output is exactly OUTPUT_SIZE
        if heightmap.shape[-1] != OUTPUT_SIZE:
            heightmap = F.interpolate(heightmap, size=(OUTPUT_SIZE, OUTPUT_SIZE), 
                                     mode='bilinear', align_corners=False)
        
        return heightmap, height_bounds


class WoWTileDatasetV6(Dataset):
    """Dataset loading minimap + normalmap + WDL -> heightmap + height bounds."""
    
    def __init__(self, dataset_roots, input_size=256, augment=True):
        self.input_size = input_size
        self.augment = augment
        self.samples = []
        
        print("Loading dataset with height bounds...")
        
        for root in dataset_roots:
            if not root.exists():
                continue
            
            images_dir = root / "images"
            dataset_dir = root / "dataset"
            
            if not images_dir.exists() or not dataset_dir.exists():
                continue
            
            # Find JSON files with height data
            json_files = list(dataset_dir.glob("*.json"))
            
            for json_path in json_files:
                tile_name = json_path.stem
                minimap_path = images_dir / f"{tile_name}.png"
                normalmap_path = images_dir / f"{tile_name}_normalmap.png"
                heightmap_path = images_dir / f"{tile_name}_heightmap_v2_preview.png"
                
                # Check images exist
                if not (minimap_path.exists() and normalmap_path.exists() and heightmap_path.exists()):
                    continue
                
                # Load JSON to check height bounds exist
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    terrain = data.get("terrain_data", {})
                    if terrain.get("height_min") is None:
                        continue
                except Exception:
                    continue
                
                self.samples.append({
                    "json": json_path,
                    "minimap": minimap_path,
                    "normalmap": normalmap_path,
                    "heightmap": heightmap_path,
                    "tile_name": tile_name,
                })
        
        print(f"Loaded {len(self.samples)} tiles with JSON + images")
        
        # Image transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.samples)
    
    def _render_wdl_to_image(self, wdl_data):
        """Convert WDL 17x17 outer grid to 256x256 image."""
        if wdl_data is None or "outer_17" not in wdl_data:
            # Return flat gray if no WDL data
            return np.full((256, 256), 0.5, dtype=np.float32)
        
        outer = np.array(wdl_data["outer_17"], dtype=np.float32)
        if len(outer) != 289:  # 17x17
            return np.full((256, 256), 0.5, dtype=np.float32)
        
        # Reshape to 17x17
        wdl_grid = outer.reshape(17, 17)
        
        # Normalize WDL heights to 0-1 (WDL uses shorts, typically -32768 to 32767)
        wdl_min, wdl_max = wdl_grid.min(), wdl_grid.max()
        if wdl_max - wdl_min > 1e-6:
            wdl_norm = (wdl_grid - wdl_min) / (wdl_max - wdl_min)
        else:
            wdl_norm = np.full_like(wdl_grid, 0.5)
        
        # Upscale to 256x256 using bilinear interpolation
        wdl_img = Image.fromarray((wdl_norm * 255).astype(np.uint8), mode='L')
        wdl_img = wdl_img.resize((256, 256), Image.BILINEAR)
        
        return np.array(wdl_img, dtype=np.float32) / 255.0
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load JSON for height bounds and WDL
        with open(sample["json"], 'r') as f:
            data = json.load(f)
        
        terrain = data.get("terrain_data", {})
        height_min = terrain.get("height_min", 0.0)
        height_max = terrain.get("height_max", 100.0)
        wdl_data = terrain.get("wdl_heights", None)
        
        # Normalize height bounds to global range
        height_min_norm = (height_min - HEIGHT_GLOBAL_MIN) / HEIGHT_GLOBAL_RANGE
        height_max_norm = (height_max - HEIGHT_GLOBAL_MIN) / HEIGHT_GLOBAL_RANGE
        
        # Clamp to 0-1
        height_min_norm = np.clip(height_min_norm, 0.0, 1.0)
        height_max_norm = np.clip(height_max_norm, 0.0, 1.0)
        
        # Load images
        minimap = Image.open(sample["minimap"]).convert("RGB")
        normalmap = Image.open(sample["normalmap"]).convert("RGB")
        heightmap = Image.open(sample["heightmap"]).convert("L")
        
        # Render WDL to image
        wdl_img = self._render_wdl_to_image(wdl_data)
        
        # Resize inputs to INPUT_SIZE
        minimap = minimap.resize((self.input_size, self.input_size), Image.BILINEAR)
        normalmap = normalmap.resize((self.input_size, self.input_size), Image.BILINEAR)
        heightmap = heightmap.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        
        # Convert to tensors
        minimap_t = self.to_tensor(minimap)
        normalmap_t = self.to_tensor(normalmap)
        heightmap_t = self.to_tensor(heightmap)
        wdl_t = torch.tensor(wdl_img, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]
        
        # Normalize RGB inputs
        minimap_t = self.normalize(minimap_t)
        normalmap_t = self.normalize(normalmap_t)
        
        # Concatenate minimap + normalmap + WDL -> 7 channels
        input_tensor = torch.cat([minimap_t, normalmap_t, wdl_t], dim=0)
        
        # Height bounds tensor
        height_bounds = torch.tensor([height_min_norm, height_max_norm], dtype=torch.float32)
        
        # Augmentation: random horizontal flip
        if self.augment and torch.rand(1).item() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            heightmap_t = torch.flip(heightmap_t, dims=[2])
        
        return {
            "input": input_tensor,          # [7, 256, 256]
            "target": heightmap_t,          # [1, 256, 256]
            "height_bounds": height_bounds, # [2] - normalized min/max
            "tile_name": sample["tile_name"],
        }


def combined_loss(pred_heightmap, pred_bounds, target_heightmap, target_bounds):
    """
    Combined loss:
    - L1 loss on heightmap
    - MSE loss on height bounds
    - Gradient loss for sharp features
    """
    # Heightmap L1 loss
    heightmap_loss = F.l1_loss(pred_heightmap, target_heightmap)
    
    # Height bounds MSE loss (weighted higher - these are critical)
    bounds_loss = F.mse_loss(pred_bounds, target_bounds) * 10.0
    
    # Gradient loss for edge preservation
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy
    
    pred_dx, pred_dy = gradient(pred_heightmap)
    target_dx, target_dy = gradient(target_heightmap)
    gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    
    total = heightmap_loss + bounds_loss + gradient_loss * 0.5
    
    return total, {
        "heightmap": heightmap_loss.item(),
        "bounds": bounds_loss.item() / 10.0,  # Report unweighted
        "gradient": gradient_loss.item(),
    }


def train(resume_from=None, epochs=None):
    print("=" * 70)
    print("WoW Height Regressor V6 - Absolute Height Awareness")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    full_dataset = WoWTileDatasetV6(DATASET_ROOTS, input_size=INPUT_SIZE, augment=True)
    
    if len(full_dataset) == 0:
        print("ERROR: No samples found!")
        return
    
    # Split 90/10 train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = MultiChannelUNetV6(in_channels=7, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    
    num_epochs = epochs if epochs else NUM_EPOCHS
    patience_counter = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            target_bounds = batch["height_bounds"].to(device)
            
            optimizer.zero_grad()
            pred_heightmap, pred_bounds = model(inputs)
            
            loss, loss_parts = combined_loss(pred_heightmap, pred_bounds, targets, target_bounds)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", hm=f"{loss_parts['heightmap']:.4f}", 
                           bounds=f"{loss_parts['bounds']:.4f}")
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_bounds_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                target_bounds = batch["height_bounds"].to(device)
                
                pred_heightmap, pred_bounds = model(inputs)
                loss, loss_parts = combined_loss(pred_heightmap, pred_bounds, targets, target_bounds)
                
                val_losses.append(loss.item())
                val_bounds_errors.append(loss_parts['bounds'])
        
        avg_val_loss = np.mean(val_losses)
        avg_bounds_error = np.mean(val_bounds_errors)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"bounds_err={avg_bounds_error:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, OUTPUT_DIR / "best_model.pt")
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pt")
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WoW Height Regressor V6")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    args = parser.parse_args()
    
    train(resume_from=args.resume, epochs=args.epochs)
