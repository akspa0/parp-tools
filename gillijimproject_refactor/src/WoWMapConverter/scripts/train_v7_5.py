#!/usr/bin/env python3
"""
WoW Height Regressor V7.5 - High-Resolution with Object Footprints (0.5.3 Data ONLY)

Based on V7.1. Constrained to Alpha 0.5.3 data.

Key Features:
- Resolution: 512x512
- 11-Channel Input
- Strict 0.5.3 Dataset
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

# Configuration
DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalimdor_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalidar_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Shadowfang_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_PVPZone01_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_PVPZone02_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_RazorfenKraulInstance_v30"),
]
OUTPUT_DIR = Path(r"./vlm_output/v7_5")

BATCH_SIZE = 4 
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
EARLY_STOP_PATIENCE = 25

INPUT_SIZE = 512
OUTPUT_SIZE = 512

# Global Height Range 
HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
HEIGHT_GLOBAL_RANGE = HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN

ALPHA_LAYERS = 4
VAL_MAP_FRACTION = 0.1
HOLDOUT_MAPS = []

LOSS_WEIGHTS = {
    "heightmap_global": 0.15,
    "heightmap_local": 0.35,
    "alpha_masks": 0.10,
    "bounds": 0.05,
    "ssim": 0.05,
    "gradient": 0.05,
    "edge": 0.25,
}

class MultiChannelUNetV7(nn.Module):
    """
    5-level U-Net for 512x512 inputs.
    """
    def __init__(self, in_channels=11, out_channels=2 + ALPHA_LAYERS):
        super().__init__()
        
        # Encoder (512 -> 256 -> 128 -> 64 -> 32 -> 16)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        self.enc5 = self._conv_block(512, 1024)
        
        # Bottleneck (16x16)
        self.bottleneck = self._conv_block(1024, 2048)
        
        # Global pooling for bounds
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.height_bounds_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4), # tile_min, tile_max, global_min, global_max
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        b = self.bottleneck(self.pool(e5))
        
        # Bounds
        g = self.global_pool(b).view(b.size(0), -1)
        bounds = self.height_bounds_fc(g)
        
        # Decode
        d5 = self.up5(b)
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
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
        
        out = torch.sigmoid(self.out_conv(d1))
        
        # Ensure exact size
        if out.shape[-1] != OUTPUT_SIZE:
             out = F.interpolate(out, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear', align_corners=False)
             
        return out, bounds

def load_heightmap_16bit(path: Path, target_size: int = 512) -> torch.Tensor:
    img = Image.open(path)
    if img.mode == 'I;16':
        arr = np.array(img, dtype=np.float32) / 65535.0
    elif img.mode == 'I':
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        img = img.convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        
    if arr.shape[0] != target_size:
        from scipy.ndimage import zoom
        scale = target_size / arr.shape[0]
        arr = zoom(arr, scale, order=1)
        
    return torch.from_numpy(arr).unsqueeze(0)

class WoWTileDatasetV7(Dataset):
    def __init__(self, dataset_roots, input_size=512, augment=True):
        self.input_size = input_size
        self.augment = augment
        self.samples = []
        self._map_to_indices = {}
        
        # Setup transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Blur minimap to remove texture noise (V7 Feature)
        # Reduced to Sigma=0.5 to preserve structure while suppressing dithering
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        
        print("Loading V7.5 dataset (053)...")
        
        for root in dataset_roots:
            if not root.exists(): continue
            dataset_dir = root / "dataset"
            
            for json_path in dataset_dir.glob("*.json"):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    terrain = data.get("terrain_data", {})
                    
                    # Must have heightmaps and NORMALMAP
                    hm_g = terrain.get("heightmap_global") or terrain.get("heightmap")
                    hm_l = terrain.get("heightmap_local") or terrain.get("heightmap")
                    nm = terrain.get("normalmap") # V7 Property
                    
                    if not (hm_g and hm_l and nm):
                        continue
                        
                    hm_g_path = root / hm_g
                    hm_l_path = root / hm_l
                    nm_path = root / nm
                    mm_path = root / "images" / f"{json_path.stem}.png" # Legacy minimap path
                    
                    if not (hm_g_path.exists() and hm_l_path.exists() and nm_path.exists() and mm_path.exists()):
                        continue
                        
                    self.samples.append({
                        "json": json_path,
                        "minimap": mm_path,
                        "normalmap": nm_path,
                        "heightmap_global": hm_g_path,
                        "heightmap_local": hm_l_path,
                        "tile_name": json_path.stem,
                        "alpha_masks": terrain.get("alpha_masks", []),
                        "liquid_mask": terrain.get("liquid_mask") # V7 Water Integration
                    })
                except:
                    continue
        
        # Map indices grouping
        for idx, sample in enumerate(self.samples):
            parts = sample["tile_name"].split("_")
            map_name = "_".join(parts[:-2]) if len(parts) >= 3 else sample["tile_name"]
            if map_name not in self._map_to_indices: self._map_to_indices[map_name] = []
            self._map_to_indices[map_name].append(idx)

        print(f"Loaded {len(self.samples)} valid samples (V7.5 strict mode)")

    def get_map_indices(self): return self._map_to_indices
    def __len__(self): return len(self.samples)
    
    def _render_wdl(self, wdl_data):
        if not wdl_data: return torch.full((1, self.input_size, self.input_size), 0.5)
        outer = np.array(wdl_data.get("outer_17", []), dtype=np.float32)
        if len(outer) != 289: return torch.full((1, self.input_size, self.input_size), 0.5)
        
        # 17x17 -> 512x512
        grid = outer.reshape(17, 17)
        vmin, vmax = grid.min(), grid.max()
        if vmax - vmin > 1e-6:
            grid = (grid - vmin) / (vmax - vmin)
        else:
            grid[:] = 0.5
            
        img = Image.fromarray((grid * 255).astype(np.uint8), 'L')
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        return self.to_tensor(img)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with open(sample["json"], 'r') as f: data = json.load(f)
        terrain = data.get("terrain_data", {})
        
        # Inputs
        minimap = Image.open(sample["minimap"]).convert("RGB")
        normalmap = Image.open(sample["normalmap"]).convert("RGB")
        
        # Resize inputs
        minimap = minimap.resize((self.input_size, self.input_size), Image.BILINEAR)
        normalmap = normalmap.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Data Alignment Fix: Rotate inputs -90 (270) to match heightmap orientation
        minimap = minimap.rotate(-90)
        normalmap = normalmap.rotate(-90)

        
        # Augmentation
        if self.augment:
            minimap = self.color_jitter(minimap)
            # Blur minimap (V7 Requirement) - Only minimap, not normalmap
            minimap = self.blur(minimap)
            
        minimap_t = self.normalize(self.to_tensor(minimap))
        normalmap_t = self.normalize(self.to_tensor(normalmap))
        
        # WDL
        wdl_t = self._render_wdl(terrain.get("wdl_heights"))
        
        # Bounds Hint
        h_min = terrain.get("height_min", 0.0)
        h_max = terrain.get("height_max", 100.0)
        
        g_min = terrain.get("height_global_min", HEIGHT_GLOBAL_MIN)
        g_max = terrain.get("height_global_max", HEIGHT_GLOBAL_MAX)
        g_range = g_max - g_min
        
        h_min_n = np.clip((h_min - g_min) / g_range, 0, 1)
        h_max_n = np.clip((h_max - g_min) / g_range, 0, 1)
        g_min_n = np.clip((g_min - g_min) / g_range, 0, 1)
        g_max_n = np.clip((g_max - g_min) / g_range, 0, 1)
        
        # Channel 7: H_Min Mask (constant per tile)
        h_min_mask = torch.full((1, self.input_size, self.input_size), h_min_n, dtype=torch.float32)
        
        # Channel 8: H_Max Mask (constant per tile)
        h_max_mask = torch.full((1, self.input_size, self.input_size), h_max_n, dtype=torch.float32)
        
        # Channel 9: Water Mask
        liquid_mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        if sample.get("liquid_mask"):
            l_path = sample["json"].parent.parent / sample["liquid_mask"]
            if l_path.exists():
                l_img = Image.open(l_path).convert("L").resize((self.input_size, self.input_size), Image.NEAREST)
                l_t = self.to_tensor(l_img)
                liquid_mask = (l_t > 0.1).float()
        
        # Channel 10: Object Footprint Mask (render squares from objects list)
        object_mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        objects = terrain.get("objects")
        if objects:
            obj_img = np.zeros((self.input_size, self.input_size), dtype=np.float32)
            tile_size = 533.33333  # World units per tile
            for obj in objects:
                # obj = {pos_x, pos_y, pos_z, scale, ...}
                px = obj.get("pos_x", 0)
                py = obj.get("pos_y", 0)
                scale = obj.get("scale", 1.0)
                
                # Use actual bounding box if available, otherwise fallback to scale
                bounds_min = obj.get("bounds_min")
                bounds_max = obj.get("bounds_max")
                
                if bounds_min and bounds_max and len(bounds_min) >= 2 and len(bounds_max) >= 2:
                    # Calculate footprint from actual bounding box (in world units)
                    half_width = abs(bounds_max[0] - bounds_min[0]) * 0.5 * scale
                    half_depth = abs(bounds_max[1] - bounds_min[1]) * 0.5 * scale
                    # Convert to pixels (1 tile = 533.33 units = 512 pixels)
                    pixels_per_unit = self.input_size / tile_size
                    radius_x = max(1, int(half_width * pixels_per_unit))
                    radius_y = max(1, int(half_depth * pixels_per_unit))
                else:
                    # Fallback: use scale * 5 pixels
                    radius_x = radius_y = max(1, int(5 * scale))
                
                # Normalize to tile coordinates
                if abs(px) < 2 and abs(py) < 2:  # Already normalized
                    nx = int((px + 1) * 0.5 * self.input_size)
                    ny = int((py + 1) * 0.5 * self.input_size)
                else:
                    # Assume coords are within tile bounds
                    nx = int((px / tile_size) * self.input_size) % self.input_size
                    ny = int((py / tile_size) * self.input_size) % self.input_size
                
                # Draw rectangular footprint
                x1, y1 = max(0, nx - radius_x), max(0, ny - radius_y)
                x2, y2 = min(self.input_size, nx + radius_x), min(self.input_size, ny + radius_y)
                obj_img[y1:y2, x1:x2] = 1.0
                
            object_mask = torch.from_numpy(obj_img).unsqueeze(0)
                
        input_tensor = torch.cat([minimap_t, normalmap_t, wdl_t, h_min_mask, h_max_mask, liquid_mask, object_mask], dim=0)
        
        # Target
        hm_g_t = load_heightmap_16bit(sample["heightmap_global"], OUTPUT_SIZE)
        hm_l_t = load_heightmap_16bit(sample["heightmap_local"], OUTPUT_SIZE)
        
        # Alpha (simple load)
        alpha_targets = []
        alpha_paths = sample.get("alpha_masks") or []
        for i in range(ALPHA_LAYERS):
            if i < len(alpha_paths):
                ap = sample["json"].parent.parent / alpha_paths[i]
                if ap.exists():
                    aimg = Image.open(ap).convert("L").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
                    alpha_targets.append(self.to_tensor(aimg))
                    continue
            alpha_targets.append(torch.zeros(1, OUTPUT_SIZE, OUTPUT_SIZE))
        alpha_t = torch.cat(alpha_targets, dim=0)

        bounds_t = torch.tensor([h_min_n, h_max_n, g_min_n, g_max_n], dtype=torch.float32)
        
        if self.augment and torch.rand(1).item() > 0.5:
             input_tensor = torch.flip(input_tensor, [2])
             hm_g_t = torch.flip(hm_g_t, [2])
             hm_l_t = torch.flip(hm_l_t, [2])
             alpha_t = torch.flip(alpha_t, [2])
             
        return {
            "input": input_tensor,
            "target": torch.cat([hm_g_t, hm_l_t, alpha_t], dim=0),
            "height_bounds": bounds_t
        }

# Losses
def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    def gaussian_window(size: int, sigma: float = 1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()
    g = gaussian_window(window_size).to(pred.device)
    window = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0).expand(pred.shape[1], 1, window_size, window_size)
    mu_pred = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu_target = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
    mu_pred_sq, mu_target_sq = mu_pred.pow(2), mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target
    sigma_pred_sq = F.conv2d(pred*pred, window, padding=window_size//2, groups=pred.shape[1]) - mu_pred_sq
    sigma_target_sq = F.conv2d(target*target, window, padding=window_size//2, groups=target.shape[1]) - mu_target_sq
    sigma_pred_target = F.conv2d(pred*target, window, padding=window_size//2, groups=pred.shape[1]) - mu_pred_target
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return 1 - ssim_map.mean()

def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)
    def compute(x):
        edges = torch.zeros_like(x)
        for c in range(x.shape[1]):
            ch = x[:, c:c+1]
            edges[:, c:c+1] = F.conv2d(ch, sobel_x, padding=1).abs() + F.conv2d(ch, sobel_y, padding=1).abs()
        return edges
    return F.l1_loss(compute(pred[:, :2]), compute(target[:, :2]))

def combined_loss(pred_heightmap, pred_bounds, target_heightmap, target_bounds):
    g_loss = F.l1_loss(pred_heightmap[:, 0:1], target_heightmap[:, 0:1])
    l_loss = F.l1_loss(pred_heightmap[:, 1:2], target_heightmap[:, 1:2])
    a_loss = F.l1_loss(pred_heightmap[:, 2:], target_heightmap[:, 2:]) if pred_heightmap.shape[1] > 2 else torch.tensor(0.0).to(pred_heightmap.device)
    b_loss = F.mse_loss(pred_bounds, target_bounds)
    
    # Gradient loss
    def get_grad(x): 
        return x[:,:,:,1:]-x[:,:,:,:-1], x[:,:,1:,:]-x[:,:,:-1,:]
    p_dx, p_dy = get_grad(pred_heightmap[:, :2])
    t_dx, t_dy = get_grad(target_heightmap[:, :2])
    grad_loss = F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)
    
    s_loss = ssim_loss(pred_heightmap[:, :2], target_heightmap[:, :2])
    e_loss = edge_loss(pred_heightmap, target_heightmap)
    
    total = (LOSS_WEIGHTS["heightmap_global"] * g_loss + LOSS_WEIGHTS["heightmap_local"] * l_loss + 
             LOSS_WEIGHTS["alpha_masks"] * a_loss + LOSS_WEIGHTS["bounds"] * b_loss +
             LOSS_WEIGHTS["gradient"] * grad_loss + LOSS_WEIGHTS["ssim"] * s_loss +
             LOSS_WEIGHTS["edge"] * e_loss)
             
    return total, {"heightmap_global": g_loss.item(), "heightmap_local": l_loss.item(), 
                   "alpha": a_loss.item(), "bounds": b_loss.item(), 
                   "gradient": grad_loss.item(), "ssim": s_loss.item(), "edge": e_loss.item()}

def save_training_preview(model, batch, epoch, output_dir, device):
    """Save a grid of inference previews: Minimap | Normal | Pred Height | GT Height"""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        bounds = batch["height_bounds"].to(device)
        
        # Run inference
        pred_heightmap, _ = model(inputs)
        
        # Take first 4 samples
        n_samples = min(4, inputs.shape[0])
        
        # Prepare grid
        # Columns: Minimap (RGB), Normal (RGB), Pred (Gray), GT (Gray)
        rows = []
        for i in range(n_samples):
            # Input Channels: 0-2=Minimap, 3-5=Normal, 6=WDL, 7=Bounds, 8=Water
            minimap = inputs[i, 0:3].cpu()
            normal = inputs[i, 3:6].cpu()
            water = inputs[i, 8:9].cpu()
            
            # Predict & Target (Channel 0=Global, 1=Local)
            # Visualize Global Height (Ch 0)
            pred = pred_heightmap[i, 0].cpu().unsqueeze(0)
            gt = targets[i, 0].cpu().unsqueeze(0)
            
            # Denormalize for visualization (simple min-max stretch)
            def normalize_viz(x):
                x = x - x.min()
                return x / (x.max() + 1e-6)
                
            pred_viz = normalize_viz(pred).repeat(3, 1, 1)
            gt_viz = normalize_viz(gt).repeat(3, 1, 1)
            
            # Un-normalize inputs (approximate)
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            minimap = minimap * std + mean
            normal = normal * std + mean
            
            # Water Viz (Blue tint)
            water_viz = water.repeat(3, 1, 1) * torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
            
            row = torch.cat([minimap, normal, water_viz, pred_viz, gt_viz], dim=2)
            rows.append(row)
            
        grid = torch.cat(rows, dim=1)
        grid = torch.clamp(grid, 0, 1)
        
        # Save
        save_path = output_dir / f"val_epoch_{epoch:04d}.png"
        transforms.ToPILImage()(grid).save(save_path)
        print(f"  Saved preview: {save_path.name}")

def train(resume_from=None, epochs=None):
    print("="*60 + "\nWoW V7.5 Training (512x512) - 0.5.3 Data Only\n" + "="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    full_dataset = WoWTileDatasetV7(DATASET_ROOTS, input_size=INPUT_SIZE)
    if len(full_dataset) == 0:
        print("No samples found. Please regenerate dataset with V7 exporter.")
        return

    # Split
    indices = list(range(len(full_dataset)))
    split = int(len(indices) * (1 - VAL_MAP_FRACTION))
    train_ds = torch.utils.data.Subset(full_dataset, indices[:split])
    val_ds = torch.utils.data.Subset(full_dataset, indices[split:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MultiChannelUNetV7().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    
    # Mixed Precision Training - DISABLED due to NaN instability with SSIM/edge loss
    scaler = None  # Disabled
    use_amp = False
    print(f"Mixed Precision Training: DISABLED (stability mode)")
    
    # Gradient clipping to prevent exploding gradients
    MAX_GRAD_NORM = 1.0
    
    # Training history for analysis
    training_history = {"epochs": [], "train_loss": [], "val_loss": [], "components": []}
    
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['val_loss']
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs or NUM_EPOCHS):
        model.train()
        t_losses = []
        epoch_parts = {}  # Track per-component losses
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            bounds = batch["height_bounds"].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    out, out_bounds = model(inputs)
                    loss, parts = combined_loss(out, out_bounds, targets, bounds)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                out, out_bounds = model(inputs)
                loss, parts = combined_loss(out, out_bounds, targets, bounds)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            
            t_losses.append(loss.item())
            for k, v in parts.items():
                epoch_parts[k] = epoch_parts.get(k, 0) + v
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Average component losses
        for k in epoch_parts:
            epoch_parts[k] /= len(train_loader)
            
        # Validation
        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                bounds = batch["height_bounds"].to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out, out_bounds = model(inputs)
                        loss, _ = combined_loss(out, out_bounds, targets, bounds)
                else:
                    out, out_bounds = model(inputs)
                    loss, _ = combined_loss(out, out_bounds, targets, bounds)
                v_losses.append(loss.item())
                
        avg_t_loss = np.mean(t_losses)
        avg_v_loss = np.mean(v_losses)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log all losses
        print(f"\nEpoch {epoch+1}/{epochs or NUM_EPOCHS}")
        print(f"  Train: {avg_t_loss:.4f} | Val: {avg_v_loss:.4f} | Best: {best_loss:.4f}")
        print(f"  HM_G: {epoch_parts.get('heightmap_global',0):.4f} | HM_L: {epoch_parts.get('heightmap_local',0):.4f} | Edge: {epoch_parts.get('edge',0):.4f}")
        print(f"  LR: {current_lr:.2e} | Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
        
        # Save to history
        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(avg_t_loss)
        training_history["val_loss"].append(avg_v_loss)
        training_history["components"].append(epoch_parts)
        
        # Save history after each epoch
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(training_history, f, indent=2)
        
        scheduler.step(avg_v_loss)
        
        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            patience_counter = 0
            ckpt_path = OUTPUT_DIR / "best.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': best_loss}, ckpt_path)
            print("  ✓ Saved Best Model")
            
            # Save visual preview
            try:
                preview_batch = next(iter(val_loader))
                save_training_preview(model, preview_batch, epoch, OUTPUT_DIR / "previews", device)
            except Exception as e:
                print(f"  Failed to save preview: {e}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⛔ Early stopping: No improvement for {EARLY_STOP_PATIENCE} epochs")
                break
    
    print(f"\n✓ Training complete. Best Val Loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    train(args.resume, args.epochs)
