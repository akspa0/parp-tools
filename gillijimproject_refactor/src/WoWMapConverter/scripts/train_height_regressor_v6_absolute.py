#!/usr/bin/env python3
"""
WoW Height Regressor V6 - Absolute Height Awareness

Key improvements over V5:
- Loads height_min/height_max from JSON for absolute height context
- Adds WDL low-res heightmap as 7th input channel (upscaled 17x17 -> 256x256)
- Model predicts BOTH normalized heightmap AND height bounds (min/max)
- Loss combines heightmap accuracy + height range prediction
- Enables reconstruction of TRUE world heights, not just relative gradients
- V6.1: Added 8th channel (height bounds mask) + SSIM/edge losses

Input: minimap (3ch) + normalmap (3ch) + WDL hint (1ch) + bounds hint (1ch) = 8 channels
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


def load_heightmap_16bit(path: Path, target_size: int = 256) -> torch.Tensor:
    """
    Load a 16-bit grayscale heightmap PNG and normalize to [0, 1] range.
    
    The heightmaps are stored as I;16 (16-bit integer) PNGs with values 0-65535.
    Using .convert('L') corrupts them by truncating to 8-bit.
    """
    img = Image.open(path)
    
    # Handle different modes
    if img.mode == 'I;16':
        # 16-bit integer mode - convert via numpy
        arr = np.array(img, dtype=np.float32) / 65535.0
    elif img.mode == 'I':
        # 32-bit integer mode
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    elif img.mode == 'L':
        # Already 8-bit (legacy format)
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        # Fallback: convert to grayscale
        img = img.convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
    
    # Resize if needed
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        from scipy.ndimage import zoom
        scale = target_size / arr.shape[0]
        arr = zoom(arr, scale, order=1)
    
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

# Dataset paths - All v20 datasets (1,790+ complete tiles)
DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v20"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalimdor_v20"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalidar_v20"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v20"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_PVPZone02_v20"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Shadowfang_v20"),
    # PVPZone01_v20 and RazorfenKraulInstance_v20 excluded - no minimaps in original data
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

ALPHA_LAYERS = 4
ALPHA_WEIGHT = 0.10  # V6 spec: 0.10
REQUIRE_WDL = False  # Fallback to grayscale if missing
REQUIRE_ALPHA = False
VAL_MAP_FRACTION = 0.1
HOLDOUT_MAPS = []  # e.g., ["Azeroth", "Kalimdor"]

# V6 Loss Weights (from v6_resnet34.yaml config)
LOSS_WEIGHTS = {
    "heightmap_global": 0.15,
    "heightmap_local": 0.35,
    "alpha_masks": 0.10,
    "bounds": 0.05,
    "ssim": 0.05,
    "gradient": 0.05,
    "edge": 0.25,
}

class MultiChannelUNetV6(nn.Module):
    """
    8-channel U-Net that predicts:
    - 256x256 normalized heightmaps (global + local)
    - alpha mask layers (optional auxiliary target)
    - height bounds (tile min/max + global min/max)
    
    V6.1: Added 8th channel (height bounds mask broadcast to 256x256)
    """
    
    def __init__(self, in_channels=8, out_channels=2 + ALPHA_LAYERS):
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
            nn.Linear(256, 4),  # tile min/max, global min/max
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
        height_bounds = self.height_bounds_fc(global_features)  # [batch, 4]
        
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
        
        # Output heightmaps + alpha (sigmoid for 0-1 range)
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
        self._map_to_indices = {}
        self.stats = {
            "total_json": 0,
            "missing_height_bounds": 0,
            "missing_wdl": 0,
            "missing_alpha": 0,
            "missing_heightmaps": 0,
            "missing_inputs": 0,
        }
        
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
                self.stats["total_json"] += 1
                tile_name = json_path.stem
                minimap_path = images_dir / f"{tile_name}.png"
                normalmap_path = images_dir / f"{tile_name}_normalmap.png"

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    terrain = data.get("terrain_data", {})
                    if terrain.get("height_min") is None:
                        self.stats["missing_height_bounds"] += 1
                        continue

                    wdl_data = terrain.get("wdl_heights")
                    if REQUIRE_WDL:
                        outer = wdl_data.get("outer_17") if isinstance(wdl_data, dict) else None
                        if not outer or len(outer) != 289:
                            self.stats["missing_wdl"] += 1
                            continue

                    alpha_masks = terrain.get("alpha_masks") or []
                    if REQUIRE_ALPHA and len(alpha_masks) == 0:
                        self.stats["missing_alpha"] += 1
                        continue

                    heightmap_global_rel = terrain.get("heightmap_global") or terrain.get("heightmap")
                    heightmap_local_rel = terrain.get("heightmap_local") or terrain.get("heightmap")

                    heightmap_global_path = (root / heightmap_global_rel) if heightmap_global_rel else None
                    heightmap_local_path = (root / heightmap_local_rel) if heightmap_local_rel else None
                except Exception:
                    continue

                if heightmap_global_path is None or heightmap_local_path is None:
                    self.stats["missing_heightmaps"] += 1
                    continue

                # Check images exist - normalmap is optional
                if not (minimap_path.exists() and heightmap_global_path.exists() and heightmap_local_path.exists()):
                    self.stats["missing_inputs"] += 1
                    continue
                
                # Normalmap is optional - fallback to gray if missing
                normalmap_actual = normalmap_path if normalmap_path.exists() else None

                self.samples.append({
                    "json": json_path,
                    "minimap": minimap_path,
                    "normalmap": normalmap_actual,  # None = use fallback
                    "heightmap_global": heightmap_global_path,
                    "heightmap_local": heightmap_local_path,
                    "alpha_masks": alpha_masks,
                    "tile_name": tile_name,
                })
        
        for idx, sample in enumerate(self.samples):
            map_name = self._extract_map_name(sample["tile_name"])
            if map_name not in self._map_to_indices:
                self._map_to_indices[map_name] = []
            self._map_to_indices[map_name].append(idx)

        print(f"Loaded {len(self.samples)} tiles with JSON + images")
        self._print_sanity_report()
        
        # Image transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        )
    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _extract_map_name(tile_name: str) -> str:
        parts = tile_name.split("_")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            return "_".join(parts[:-2])
        return tile_name

    def get_map_indices(self):
        return self._map_to_indices

    def _print_sanity_report(self):
        print("Dataset sanity report:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        print(f"  maps: {len(self._map_to_indices)}")
    
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

        global_min = terrain.get("height_global_min", HEIGHT_GLOBAL_MIN)
        global_max = terrain.get("height_global_max", HEIGHT_GLOBAL_MAX)
        global_range = global_max - global_min if global_max != global_min else HEIGHT_GLOBAL_RANGE

        # Normalize height bounds to global range
        height_min_norm = (height_min - global_min) / global_range
        height_max_norm = (height_max - global_min) / global_range
        global_min_norm = (global_min - global_min) / global_range
        global_max_norm = (global_max - global_min) / global_range
        
        # Clamp to 0-1
        height_min_norm = np.clip(height_min_norm, 0.0, 1.0)
        height_max_norm = np.clip(height_max_norm, 0.0, 1.0)
        global_min_norm = np.clip(global_min_norm, 0.0, 1.0)
        global_max_norm = np.clip(global_max_norm, 0.0, 1.0)
        
        # Load images
        minimap = Image.open(sample["minimap"]).convert("RGB")
        
        # Normalmap - use gray fallback if not available (v20 datasets)
        if sample["normalmap"] is not None:
            normalmap = Image.open(sample["normalmap"]).convert("RGB")
        else:
            # Gray fallback (0.5, 0.5, 0.5) = neutral normal
            normalmap = Image.new("RGB", (self.input_size, self.input_size), color=(128, 128, 128))
        
        heightmap_global_t = load_heightmap_16bit(sample["heightmap_global"], OUTPUT_SIZE)
        heightmap_local_t = load_heightmap_16bit(sample["heightmap_local"], OUTPUT_SIZE)
        
        # Render WDL to image (require if configured)
        if REQUIRE_WDL:
            outer = wdl_data.get("outer_17") if isinstance(wdl_data, dict) else None
            if not outer or len(outer) != 289:
                raise ValueError("Missing WDL outer_17 data")

        wdl_img = self._render_wdl_to_image(wdl_data)
        
        # Resize inputs to INPUT_SIZE
        minimap = minimap.resize((self.input_size, self.input_size), Image.BILINEAR)
        normalmap = normalmap.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Augment minimap color before tensor conversion
        if self.augment:
            minimap = self.color_jitter(minimap)

        # Convert to tensors (heightmaps already tensors from load_heightmap_16bit)
        minimap_t = self.to_tensor(minimap)
        normalmap_t = self.to_tensor(normalmap)
        wdl_t = torch.tensor(wdl_img, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]
        
        # Normalize RGB inputs
        minimap_t = self.normalize(minimap_t)
        normalmap_t = self.normalize(normalmap_t)
        
        # V6.1: Create 8th channel - height bounds hint (broadcast tile min to full image)
        # This gives the model a hint about the absolute height range
        bounds_hint = torch.full((1, self.input_size, self.input_size), height_min_norm, dtype=torch.float32)
        
        # Concatenate minimap + normalmap + WDL + bounds hint -> 8 channels
        input_tensor = torch.cat([minimap_t, normalmap_t, wdl_t, bounds_hint], dim=0)
        
        # Alpha mask targets (up to ALPHA_LAYERS)
        alpha_targets = []
        alpha_paths = sample.get("alpha_masks") or []
        for i in range(ALPHA_LAYERS):
            if i < len(alpha_paths):
                alpha_path = sample["json"].parent.parent / alpha_paths[i]
                if alpha_path.exists():
                    alpha_img = Image.open(alpha_path).convert("L")
                    alpha_img = alpha_img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
                    alpha_targets.append(self.to_tensor(alpha_img))
                    continue
            alpha_targets.append(torch.zeros(1, OUTPUT_SIZE, OUTPUT_SIZE))

        alpha_tensor = torch.cat(alpha_targets, dim=0) if alpha_targets else torch.zeros(ALPHA_LAYERS, OUTPUT_SIZE, OUTPUT_SIZE)

        # Height bounds tensor: [tile_min, tile_max, global_min, global_max]
        height_bounds = torch.tensor([height_min_norm, height_max_norm, global_min_norm, global_max_norm], dtype=torch.float32)
        
        # Augmentation: random horizontal flip
        if self.augment and torch.rand(1).item() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            heightmap_global_t = torch.flip(heightmap_global_t, dims=[2])
            heightmap_local_t = torch.flip(heightmap_local_t, dims=[2])
            alpha_tensor = torch.flip(alpha_tensor, dims=[2])
        
        return {
            "input": input_tensor,          # [8, 256, 256] - V6.1: Added bounds hint channel
            "target": torch.cat([heightmap_global_t, heightmap_local_t, alpha_tensor], dim=0),  # [2+ALPHA, 256, 256]
            "height_bounds": height_bounds, # [4] - normalized min/max
            "tile_name": sample["tile_name"],
        }


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute SSIM loss (1 - SSIM).
    V6.1: Added for perceptual quality on heightmaps.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g
    
    g = gaussian_window(window_size).to(pred.device)
    window = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.shape[1], 1, window_size, window_size)
    
    mu_pred = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu_target = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size//2, groups=pred.shape[1]) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size//2, groups=target.shape[1]) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.shape[1]) - mu_pred_target
    
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return 1 - ssim_map.mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Edge preservation loss using Sobel filters.
    V6.1: Added for crisp terrain boundaries.
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=pred.device)
    sobel_y = sobel_x.T
    
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    def compute_edges(x: torch.Tensor) -> torch.Tensor:
        # Process each channel independently and sum
        edges = torch.zeros_like(x)
        for c in range(x.shape[1]):
            channel = x[:, c:c+1, :, :]
            ex = F.conv2d(channel, sobel_x, padding=1).abs()
            ey = F.conv2d(channel, sobel_y, padding=1).abs()
            edges[:, c:c+1] = ex + ey
        return edges
    
    pred_edges = compute_edges(pred[:, :2])  # Only heightmaps, not alpha
    target_edges = compute_edges(target[:, :2])
    
    return F.l1_loss(pred_edges, target_edges)


def combined_loss(pred_heightmap, pred_bounds, target_heightmap, target_bounds):
    """
    Combined loss (V6.1 - matches v6_resnet34.yaml config):
    - L1 loss on heightmaps (global + local)
    - L1 loss on alpha masks (aux)
    - MSE loss on height bounds
    - Gradient loss for smooth gradients
    - SSIM loss for perceptual quality
    - Edge loss for crisp boundaries (V6.1)
    """
    device = pred_heightmap.device
    
    # Heightmap L1 loss (global + local)
    if pred_heightmap.shape[1] >= 2:
        global_loss = F.l1_loss(pred_heightmap[:, 0:1], target_heightmap[:, 0:1])
        local_loss = F.l1_loss(pred_heightmap[:, 1:2], target_heightmap[:, 1:2])
    else:
        global_loss = F.l1_loss(pred_heightmap, target_heightmap)
        local_loss = global_loss

    alpha_loss = torch.tensor(0.0, device=device)
    if pred_heightmap.shape[1] > 2:
        pred_alpha = pred_heightmap[:, 2:2 + ALPHA_LAYERS]
        tgt_alpha = target_heightmap[:, 2:2 + ALPHA_LAYERS]
        alpha_loss = F.l1_loss(pred_alpha, tgt_alpha)
    
    # Height bounds MSE loss
    bounds_loss = F.mse_loss(pred_bounds, target_bounds)
    
    # Gradient loss for smooth transitions
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy
    
    pred_dx, pred_dy = gradient(pred_heightmap[:, :2])
    target_dx, target_dy = gradient(target_heightmap[:, :2])
    gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    
    # SSIM loss on heightmaps (V6.1)
    ssim = ssim_loss(pred_heightmap[:, :2], target_heightmap[:, :2])
    
    # Edge loss on heightmaps (V6.1)
    edge = edge_loss(pred_heightmap, target_heightmap)
    
    # Combine with V6 spec weights
    total = (
        LOSS_WEIGHTS["heightmap_global"] * global_loss +
        LOSS_WEIGHTS["heightmap_local"] * local_loss +
        LOSS_WEIGHTS["alpha_masks"] * alpha_loss +
        LOSS_WEIGHTS["bounds"] * bounds_loss +
        LOSS_WEIGHTS["gradient"] * gradient_loss +
        LOSS_WEIGHTS["ssim"] * ssim +
        LOSS_WEIGHTS["edge"] * edge
    )
    
    return total, {
        "heightmap_global": global_loss.item(),
        "heightmap_local": local_loss.item(),
        "alpha": alpha_loss.item(),
        "bounds": bounds_loss.item(),
        "gradient": gradient_loss.item(),
        "ssim": ssim.item(),
        "edge": edge.item(),
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
    
    map_indices = full_dataset.get_map_indices()
    all_maps = sorted(map_indices.keys())
    if HOLDOUT_MAPS:
        val_maps = [m for m in all_maps if m in HOLDOUT_MAPS]
    else:
        rng = np.random.default_rng(42)
        val_count = max(1, int(len(all_maps) * VAL_MAP_FRACTION))
        val_maps = rng.choice(all_maps, size=val_count, replace=False).tolist()

    train_indices = []
    val_indices = []
    for map_name, indices in map_indices.items():
        if map_name in val_maps:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Val maps ({len(val_maps)}): {', '.join(val_maps)}")
    
    # Create model - 8 input channels: RGB(3) + normalmap(3) + WDL(1) + bounds_hint(1)
    model = MultiChannelUNetV6(in_channels=8, out_channels=2 + ALPHA_LAYERS).to(device)
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
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                hg=f"{loss_parts['heightmap_global']:.4f}",
                hl=f"{loss_parts['heightmap_local']:.4f}",
                bounds=f"{loss_parts['bounds']:.4f}",
                alpha=f"{loss_parts['alpha']:.4f}"
            )
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_loss_parts = {k: [] for k in ["heightmap_global", "heightmap_local", "alpha", "bounds", "ssim", "edge", "gradient"]}
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                target_bounds = batch["height_bounds"].to(device)
                
                pred_heightmap, pred_bounds = model(inputs)
                loss, loss_parts = combined_loss(pred_heightmap, pred_bounds, targets, target_bounds)
                
                val_losses.append(loss.item())
                for k, v in loss_parts.items():
                    val_loss_parts[k].append(v)
        
        avg_val_loss = np.mean(val_losses)
        avg_val_parts = {k: np.mean(v) for k, v in val_loss_parts.items()}
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Detailed epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Losses: hg={avg_val_parts['heightmap_global']:.4f} hl={avg_val_parts['heightmap_local']:.4f} "
              f"alpha={avg_val_parts['alpha']:.4f} bounds={avg_val_parts['bounds']:.4f} "
              f"ssim={avg_val_parts['ssim']:.4f} edge={avg_val_parts['edge']:.4f} grad={avg_val_parts['gradient']:.4f}")
        
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
