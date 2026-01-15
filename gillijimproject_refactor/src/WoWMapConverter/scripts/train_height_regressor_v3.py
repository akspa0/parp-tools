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
import sys
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from datetime import datetime

# Configuration
DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_shadowfang_v1"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
]
OUTPUT_DIR = Path(r"j:\vlm_output\wow_height_regressor_v3")

# Resume training from checkpoint
RESUME_CHECKPOINT = None  # Set to path to resume training
NUM_CHUNKS = 256  # 16×16 chunks per ADT
NUM_HEIGHTS_PER_CHUNK = 145  # WoW vertex count per chunk
TOTAL_HEIGHTS = NUM_CHUNKS * NUM_HEIGHTS_PER_CHUNK  # 37,120

# For output, we'll reshape to a 2D grid
# WoW heightmap is effectively 17×17 outer + 16×16 inner per chunk
# For simplicity, output as 256×145 and reshape during inference
OUTPUT_H = 256  # chunks
OUTPUT_W = 145  # heights per chunk

# Training hyperparameters - optimized for 16GB VRAM
BATCH_SIZE = 8  # Increased for better GPU utilization (will auto-reduce if OOM)
LEARNING_RATE = 3e-4  # Higher LR works well with larger batches
NUM_EPOCHS = 500  # Max epochs - early stopping will terminate sooner
SMOOTHNESS_WEIGHT = 0.1  # Smoother terrain
GRADIENT_WEIGHT = 0.5  # CRITICAL: Weight for gradient matching - learns terrain SHAPE
NORMAL_WEIGHT = 0.2  # Weight for normal prediction loss
BOUNDARY_WEIGHT = 0.3  # Weight for chunk boundary continuity
CONSISTENCY_WEIGHT = 0.3  # Height-normal consistency - prevents height collapse

# Early stopping configuration
EARLY_STOP_PATIENCE = 50  # Stop if no improvement for N epochs (50 is standard for image tasks)
EARLY_STOP_MIN_DELTA = 1e-5  # Minimum improvement to count as progress

# Mixed precision training (uses less VRAM, faster on modern GPUs)
USE_AMP = True  # Automatic Mixed Precision


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
    U-Net for 256×256 native WoW minimap input with optional auxiliary channels.
    Outputs 256 chunks × 145 heights = 37,120 values.
    Optionally also outputs 256 × 145 × 3 normals.
    
    Input channels:
    - RGB minimap (3 channels) - required
    - Shadow map (1 channel) - optional, correlates with terrain slope
    - Alpha masks (N channels) - optional, texture transitions at edges/slopes
    
    Output: heights [B, 256, 145], normals [B, 256, 145, 3] (optional)
    """
    def __init__(self, predict_normals=True, in_channels=3):
        super().__init__()
        self.predict_normals = predict_normals
        self.in_channels = in_channels
        
        # Encoder for 256×256 input
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = ConvBlock(in_channels, 64)  # 256 - accepts variable input channels
        self.enc2 = ConvBlock(64, 128)    # 128
        self.enc3 = ConvBlock(128, 256)   # 64
        self.enc4 = ConvBlock(256, 512)   # 32
        self.enc5 = ConvBlock(512, 512)   # 16
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck at 8×8
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.dec1 = ConvBlock(1024 + 512, 512)  # 16
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Chunk position embedding: [256, 3] -> [256, 64] -> inject into decoder
        # This provides absolute height context from world coordinates
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        # Height projection: 16×16 spatial -> 145 heights per position
        # Each of the 16×16 = 256 positions corresponds to one chunk
        # Input: 512 (decoder) + 64 (position embedding) = 576 channels
        self.height_proj = nn.Sequential(
            nn.Conv2d(512 + 64, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 145, 1),  # 145 heights per chunk
        )
        
        # Normal projection: 16×16 spatial -> 145*3 = 435 normal components per chunk
        if predict_normals:
            self.normal_proj = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 145 * 3, 1),  # 145 normals × 3 components
                nn.Tanh(),  # Normals are in [-1, 1] range
            )
        
    def forward(self, x, chunk_positions=None):
        """
        Args:
            x: Input tensor [B, C, 256, 256] where C = 5 (RGB + Shadow + Alpha)
            chunk_positions: Optional [B, 256, 3] world coordinates per chunk
        """
        B = x.shape[0]
        
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
        d1 = self.dec1(torch.cat([self.up(b), e5], dim=1))   # [B, 512, 16, 16]
        
        # Embed chunk positions and concatenate with decoder features
        if chunk_positions is not None:
            # chunk_positions: [B, 256, 3] -> embed -> [B, 256, 64]
            pos_flat = chunk_positions.reshape(B * 256, 3)
            pos_embed = self.pos_embed(pos_flat)  # [B*256, 64]
            pos_embed = pos_embed.reshape(B, 256, 64)  # [B, 256, 64]
            pos_embed = pos_embed.reshape(B, 16, 16, 64)  # [B, 16, 16, 64]
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [B, 64, 16, 16]
        else:
            pos_embed = torch.zeros(B, 64, 16, 16, device=x.device)
        
        # Concatenate position embedding with decoder features
        d1_with_pos = torch.cat([d1, pos_embed], dim=1)  # [B, 576, 16, 16]
        
        # Project 16×16 feature map to heights
        # Each spatial position = one chunk, 145 channels = heights
        heights = self.height_proj(d1_with_pos)  # [B, 145, 16, 16]
        
        # Reshape to [B, 256, 145]
        heights = heights.permute(0, 2, 3, 1)  # [B, 16, 16, 145]
        heights = heights.reshape(B, 256, 145)  # [B, 256, 145]
        
        if self.predict_normals:
            # Project to normals (uses d1 without positions - normals are local)
            normals = self.normal_proj(d1)  # [B, 435, 16, 16]
            normals = normals.permute(0, 2, 3, 1)  # [B, 16, 16, 435]
            normals = normals.reshape(B, 256, 145, 3)  # [B, 256, 145, 3]
            return heights, normals
        
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
    inter_chunk_h = torch.mean((h[:, :, 1:, :] - h[:, :, :-1, :]) ** 2)
    inter_chunk_v = torch.mean((h[:, 1:, :, :] - h[:, :-1, :, :]) ** 2)
    
    return intra_chunk + 0.5 * (inter_chunk_h + inter_chunk_v)


def compute_boundary_continuity_loss(heights):
    """
    CRITICAL: Enforce that adjacent chunks share the same edge heights.
    
    WoW's 145 vertex layout per chunk (9+8 interleaved pattern):
    - Row 0: vertices 0-8 (9 outer vertices) - TOP EDGE
    - Row 1: vertices 9-16 (8 inner vertices)
    - Row 2: vertices 17-25 (9 outer vertices)
    - ...
    - Row 16: vertices 136-144 (9 outer vertices) - BOTTOM EDGE
    
    For horizontal adjacency (chunk to the right):
    - Right edge of chunk A = vertices at positions 8, 16, 25, 33, 42, 50, 59, 67, 76, 84, 93, 101, 110, 118, 127, 135, 144
    - Left edge of chunk B = vertices at positions 0, 9, 17, 26, 34, 43, 51, 60, 68, 77, 85, 94, 102, 111, 119, 128, 136
    
    For vertical adjacency (chunk below):
    - Bottom edge of chunk A = vertices 136-144 (last 9 outer vertices)
    - Top edge of chunk B = vertices 0-8 (first 9 outer vertices)
    """
    B = heights.shape[0]
    h = heights.reshape(B, 16, 16, 145)
    
    # Vertical boundary: bottom edge of row i must match top edge of row i+1
    # Bottom edge = vertices 136-144 (indices in the 145-array)
    # Top edge = vertices 0-8
    bottom_edge = h[:, :-1, :, 136:145]  # [B, 15, 16, 9]
    top_edge = h[:, 1:, :, 0:9]          # [B, 15, 16, 9]
    vertical_loss = F.mse_loss(bottom_edge, top_edge)
    
    # Horizontal boundary: right edge of col j must match left edge of col j+1
    # Right edge vertices (every 17th starting from 8, then every 17th for outer, 16th for inner)
    # Simplified: extract column 8 of each row (rightmost outer vertex per row)
    # For the 9+8 pattern, outer rows are at indices: 0, 17, 34, 51, 68, 85, 102, 119, 136
    # The rightmost vertex of each outer row is at: 8, 25, 42, 59, 76, 93, 110, 127, 144
    right_edge_indices = torch.tensor([8, 25, 42, 59, 76, 93, 110, 127, 144], device=heights.device)
    left_edge_indices = torch.tensor([0, 17, 34, 51, 68, 85, 102, 119, 136], device=heights.device)
    
    right_edge = h[:, :, :-1, :][:, :, :, right_edge_indices]  # [B, 16, 15, 9]
    left_edge = h[:, :, 1:, :][:, :, :, left_edge_indices]     # [B, 16, 15, 9]
    horizontal_loss = F.mse_loss(right_edge, left_edge)
    
    return vertical_loss + horizontal_loss


def compute_gradient_loss(pred, target):
    """
    Loss that encourages matching terrain gradients (slopes) in 2D.
    This is CRITICAL for learning terrain shape with per-tile normalization.
    """
    B = pred.shape[0]
    
    # Reshape to [B, 16, 16, 145] for spatial operations
    pred_2d = pred.reshape(B, 16, 16, 145)
    target_2d = target.reshape(B, 16, 16, 145)
    
    # Gradient within each chunk (vertex-to-vertex)
    pred_grad_v = pred_2d[:, :, :, 1:] - pred_2d[:, :, :, :-1]
    target_grad_v = target_2d[:, :, :, 1:] - target_2d[:, :, :, :-1]
    loss_v = F.mse_loss(pred_grad_v, target_grad_v)
    
    # Gradient between chunks (chunk-to-chunk in X direction)
    pred_grad_x = pred_2d[:, :, 1:, :] - pred_2d[:, :, :-1, :]
    target_grad_x = target_2d[:, :, 1:, :] - target_2d[:, :, :-1, :]
    loss_x = F.mse_loss(pred_grad_x, target_grad_x)
    
    # Gradient between chunks (chunk-to-chunk in Y direction)
    pred_grad_y = pred_2d[:, 1:, :, :] - pred_2d[:, :-1, :, :]
    target_grad_y = target_2d[:, 1:, :, :] - target_2d[:, :-1, :, :]
    loss_y = F.mse_loss(pred_grad_y, target_grad_y)
    
    return loss_v + loss_x + loss_y


def compute_normal_height_consistency_loss(pred_heights, pred_normals):
    """
    Enforces that predicted height gradients are consistent with predicted normals.
    
    Normal vectors encode surface orientation:
    - N.x = -dh/dx (slope in X)
    - N.y = -dh/dy (slope in Y)
    - N.z = 1 / sqrt(1 + (dh/dx)^2 + (dh/dy)^2)
    
    This loss ensures the model's height and normal predictions are self-consistent,
    which helps prevent the height output from collapsing to mean while normals
    capture terrain features.
    """
    B = pred_heights.shape[0]
    
    # Reshape heights to [B, 16, 16, 145]
    h = pred_heights.reshape(B, 16, 16, 145)
    
    # Compute height gradients (finite differences)
    # Within-chunk gradients (vertex to vertex)
    dh_v = h[:, :, :, 1:] - h[:, :, :, :-1]  # [B, 16, 16, 144]
    
    # Between-chunk gradients
    dh_x = h[:, :, 1:, :] - h[:, :, :-1, :]  # [B, 16, 15, 145]
    dh_y = h[:, 1:, :, :] - h[:, :-1, :, :]  # [B, 15, 16, 145]
    
    # Reshape normals to [B, 16, 16, 145, 3]
    n = pred_normals.reshape(B, 16, 16, 145, 3)
    
    # Extract normal components (X and Y encode slopes)
    # Normal X component should correlate with -dh/dx
    # Normal Y component should correlate with -dh/dy
    
    # For within-chunk: compare dh_v with average of adjacent normal X components
    n_x_avg = (n[:, :, :, 1:, 0] + n[:, :, :, :-1, 0]) / 2  # [B, 16, 16, 144]
    
    # For between-chunk X: compare dh_x with normal X
    n_x_chunk = (n[:, :, 1:, :, 0] + n[:, :, :-1, :, 0]) / 2  # [B, 16, 15, 145]
    
    # For between-chunk Y: compare dh_y with normal Y
    n_y_chunk = (n[:, 1:, :, :, 1] + n[:, :-1, :, :, 1]) / 2  # [B, 15, 16, 145]
    
    # The relationship is: gradient ≈ -normal_component (for unit-scale normals)
    # We use correlation loss: gradients should be negatively correlated with normals
    # Simplified: MSE between normalized gradient and -normal component
    
    # Normalize gradients to similar scale as normals [-1, 1]
    dh_v_norm = torch.tanh(dh_v * 5)  # Scale and bound
    dh_x_norm = torch.tanh(dh_x * 5)
    dh_y_norm = torch.tanh(dh_y * 5)
    
    # Loss: height gradients should match negative of normal components
    loss_v = F.mse_loss(dh_v_norm, -n_x_avg)
    loss_x = F.mse_loss(dh_x_norm, -n_x_chunk)
    loss_y = F.mse_loss(dh_y_norm, -n_y_chunk)
    
    return (loss_v + loss_x + loss_y) / 3


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
                        stitched_dir / f"{tile_name}.png",
                        images_dir / f"{tile_name}.png",
                    ]
                    # Add image path from JSON if present
                    img_from_json = data.get("image", "")
                    if img_from_json:
                        img_candidates.append(root / img_from_json)
                    
                    img_path = None
                    for candidate in img_candidates:
                        if candidate.exists() and candidate.is_file():
                            img_path = candidate
                            break
                    
                    if img_path is None:
                        continue
                    
                    # Find shadow map (stitched 1024×1024 or per-chunk)
                    shadow_path = None
                    shadow_candidates = [
                        stitched_dir / f"{tile_name}_shadow.png",
                        stitched_dir / f"{tile_name}_shadow.webp",
                    ]
                    for candidate in shadow_candidates:
                        if candidate.exists() and candidate.is_file():
                            shadow_path = candidate
                            break
                    
                    # Find alpha mask (first layer - usually base texture boundary)
                    alpha_path = None
                    alpha_candidates = [
                        stitched_dir / f"{tile_name}_alpha_l0.png",
                        stitched_dir / f"{tile_name}_alpha_l1.png",  # Layer 1 often more informative
                    ]
                    for candidate in alpha_candidates:
                        if candidate.exists() and candidate.is_file():
                            alpha_path = candidate
                            break
                    
                    # Extract all chunk heights
                    td = data.get("terrain_data", {})
                    heights_list = td.get("heights", [])
                    chunk_layers = td.get("chunk_layers", [])
                    
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
                    
                    # Build normals array [256, 145, 3] from chunk_layers
                    # Normals are stored as signed bytes (sbyte) in range [-127, 127]
                    tile_normals = np.zeros((256, 145, 3), dtype=np.float32)
                    has_normals = False
                    
                    for layer in chunk_layers:
                        idx = layer.get("idx", -1)
                        normals_raw = layer.get("normals", None)
                        
                        if normals_raw is not None and 0 <= idx < 256:
                            # Normals are 145*3 = 435 signed bytes
                            if len(normals_raw) >= 435:
                                for v in range(145):
                                    for c in range(3):
                                        # Convert signed byte to normalized float [-1, 1]
                                        tile_normals[idx, v, c] = normals_raw[v * 3 + c] / 127.0
                                has_normals = True
                    
                    # Extract chunk positions [256, 3] - absolute world coordinates
                    # Format: flattened array of 256*3 floats (x,y,z per chunk)
                    chunk_positions = td.get("chunk_positions", None)
                    if chunk_positions is not None and len(chunk_positions) >= 256 * 3:
                        chunk_pos_array = np.array(chunk_positions, dtype=np.float32).reshape(256, 3)
                    else:
                        chunk_pos_array = None
                    
                    # Extract holes bitmask [256] - one bitmask per chunk
                    # Holes indicate missing terrain (should mask loss)
                    holes_data = td.get("holes", None)
                    if holes_data is not None and len(holes_data) >= 256:
                        holes_array = np.array(holes_data[:256], dtype=np.int32)
                    else:
                        holes_array = np.zeros(256, dtype=np.int32)
                    
                    self.samples.append({
                        "img_path": img_path,
                        "shadow_path": shadow_path,
                        "alpha_path": alpha_path,
                        "heights": tile_heights,
                        "normals": tile_normals if has_normals else None,
                        "chunk_positions": chunk_pos_array,
                        "holes": holes_array,
                        "tile_name": tile_name,
                    })
                    self.all_heights.append(tile_heights.flatten())
                    
                except Exception as e:
                    pass
        
        # Count auxiliary data availability
        n_with_shadow = sum(1 for s in self.samples if s.get("shadow_path") is not None)
        n_with_alpha = sum(1 for s in self.samples if s.get("alpha_path") is not None)
        n_with_normals = sum(1 for s in self.samples if s.get("normals") is not None)
        n_with_positions = sum(1 for s in self.samples if s.get("chunk_positions") is not None)
        n_with_holes = sum(1 for s in self.samples if np.any(s.get("holes", np.zeros(1)) != 0))
        
        print(f"Loaded {len(self.samples)} complete tiles")
        print(f"  With shadows:   {n_with_shadow} ({100*n_with_shadow/max(len(self.samples),1):.1f}%)")
        print(f"  With alphas:    {n_with_alpha} ({100*n_with_alpha/max(len(self.samples),1):.1f}%)")
        print(f"  With normals:   {n_with_normals} ({100*n_with_normals/max(len(self.samples),1):.1f}%)")
        print(f"  With positions: {n_with_positions} ({100*n_with_positions/max(len(self.samples),1):.1f}%)")
        print(f"  With holes:     {n_with_holes} ({100*n_with_holes/max(len(self.samples),1):.1f}%)")
        
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
        """Normalize to [-1, 1] range using GLOBAL stats"""
        h_norm = 2.0 * (heights - self.global_min) / (self.global_max - self.global_min + 1e-6) - 1.0
        return h_norm.astype(np.float32)
    
    def normalize_heights_per_tile(self, heights):
        """
        Normalize to [0, 1] range using PER-TILE min/max.
        This teaches the model terrain SHAPE rather than absolute elevation.
        Returns: (normalized_heights, tile_min, tile_max)
        """
        tile_min = float(np.min(heights))
        tile_max = float(np.max(heights))
        h_range = tile_max - tile_min + 1e-6
        h_norm = (heights - tile_min) / h_range
        return h_norm.astype(np.float32), tile_min, tile_max
    
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
        
        # Load shadow map as auxiliary channel (correlates with terrain slope)
        shadow_tensor = None
        if sample.get("shadow_path") is not None:
            try:
                shadow_img = Image.open(sample["shadow_path"]).convert("L")
                if shadow_img.size != (256, 256):
                    shadow_img = shadow_img.resize((256, 256), Image.BILINEAR)
                shadow_np = np.array(shadow_img, dtype=np.float32) / 255.0
                shadow_tensor = torch.from_numpy(shadow_np).unsqueeze(0)  # [1, 256, 256]
            except:
                pass
        
        # Load alpha mask as auxiliary channel (texture transitions at edges/slopes)
        alpha_tensor = None
        if sample.get("alpha_path") is not None:
            try:
                alpha_img = Image.open(sample["alpha_path"]).convert("L")
                if alpha_img.size != (256, 256):
                    alpha_img = alpha_img.resize((256, 256), Image.BILINEAR)
                alpha_np = np.array(alpha_img, dtype=np.float32) / 255.0
                alpha_tensor = torch.from_numpy(alpha_np).unsqueeze(0)  # [1, 256, 256]
            except:
                pass
        
        # Concatenate auxiliary channels: RGB (3) + Shadow (1) + Alpha (1) = 5 channels
        # If shadow/alpha not available, use zeros as placeholder
        if shadow_tensor is None:
            shadow_tensor = torch.zeros(1, 256, 256)
        if alpha_tensor is None:
            alpha_tensor = torch.zeros(1, 256, 256)
        
        # Concatenate all channels: [5, 256, 256]
        input_tensor = torch.cat([img_tensor, shadow_tensor, alpha_tensor], dim=0)
        
        # Normalize heights using PER-TILE normalization
        # This teaches terrain SHAPE rather than absolute elevation
        raw_heights = sample["heights"]
        tile_min = float(np.min(raw_heights))
        tile_max = float(np.max(raw_heights))
        tile_range = tile_max - tile_min + 1e-6
        
        # Normalize to [0, 1] based on this tile's range
        heights = ((raw_heights - tile_min) / tile_range).astype(np.float32)
        heights_tensor = torch.from_numpy(heights)  # [256, 145]
        
        # Get normals if available (already normalized to [-1, 1])
        normals = sample.get("normals", None)
        if normals is not None:
            normals_tensor = torch.from_numpy(normals.astype(np.float32))  # [256, 145, 3]
        else:
            normals_tensor = torch.zeros(256, 145, 3)  # Placeholder
        
        # Get chunk positions [256, 3] - provides absolute height context
        chunk_positions = sample.get("chunk_positions", None)
        if chunk_positions is not None:
            # Normalize positions to reasonable range (WoW coords can be large)
            pos_tensor = torch.from_numpy(chunk_positions.astype(np.float32))
            # Normalize Z (height) component to [-1, 1] using global stats
            pos_tensor[:, 2] = 2.0 * (pos_tensor[:, 2] - self.global_min) / (self.global_max - self.global_min + 1e-6) - 1.0
        else:
            pos_tensor = torch.zeros(256, 3)
        
        # Get holes bitmask [256] - used to mask loss for missing terrain
        holes = sample.get("holes", np.zeros(256, dtype=np.int32))
        # Convert bitmask to per-vertex mask: 0 = has hole, 1 = valid terrain
        # Each chunk has 16 bits for 4x4 sub-holes, if any bit is set, chunk has holes
        holes_mask = torch.from_numpy((holes == 0).astype(np.float32))  # [256] - 1.0 for valid, 0.0 for holes
        
        # Data augmentation
        do_flip = self.augment and random.random() > 0.5
        if do_flip:
            # Horizontal flip
            input_tensor = torch.flip(input_tensor, dims=[2])
            if shadow_tensor is not None:
                shadow_tensor = torch.flip(shadow_tensor, dims=[2])
            if alpha_tensor is not None:
                alpha_tensor = torch.flip(alpha_tensor, dims=[2])
            # Flip chunk order horizontally (reverse columns in 16×16 grid)
            h_2d = heights_tensor.reshape(16, 16, 145)
            h_2d = torch.flip(h_2d, dims=[1])
            heights_tensor = h_2d.reshape(256, 145)
            # Flip normals too
            n_2d = normals_tensor.reshape(16, 16, 145, 3)
            n_2d = torch.flip(n_2d, dims=[1])
            # Also flip X component of normals for horizontal flip
            n_2d[:, :, :, 0] = -n_2d[:, :, :, 0]
            normals_tensor = n_2d.reshape(256, 145, 3)
            # Flip chunk positions (reverse columns)
            p_2d = pos_tensor.reshape(16, 16, 3)
            p_2d = torch.flip(p_2d, dims=[1])
            pos_tensor = p_2d.reshape(256, 3)
            # Flip holes mask
            hm_2d = holes_mask.reshape(16, 16)
            hm_2d = torch.flip(hm_2d, dims=[1])
            holes_mask = hm_2d.reshape(256)
        
        return {
            "pixel_values": input_tensor,
            "labels": heights_tensor,
            "normals": normals_tensor,
            "chunk_positions": pos_tensor,
            "holes_mask": holes_mask,
            "tile_min": tile_min,  # For reconstructing absolute heights
            "tile_max": tile_max,
            "has_shadow": shadow_tensor is not None,
            "has_alpha": alpha_tensor is not None,
        }


def train(resume_from=None, epochs=None):
    print("=" * 60)
    print("WoW Height Regressor V3 - Full ADT Resolution")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # GPU info and VRAM-based batch size optimization
    actual_batch_size = BATCH_SIZE
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
        
        # Auto-adjust batch size based on VRAM
        if vram_gb >= 24:
            actual_batch_size = 16
        elif vram_gb >= 16:
            actual_batch_size = 8
        elif vram_gb >= 12:
            actual_batch_size = 6
        elif vram_gb >= 8:
            actual_batch_size = 4
        else:
            actual_batch_size = 2
        
        print(f"Auto-selected batch size: {actual_batch_size} (based on {vram_gb:.1f}GB VRAM)")
        
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("Enabled: TF32, cuDNN benchmark")
    
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
    
    # Optimize num_workers based on CPU cores
    num_workers = min(8, os.cpu_count() or 4)
    
    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=actual_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    print(f"DataLoader: {num_workers} workers, batch_size={actual_batch_size}")
    
    # Create model with 5 input channels: RGB (3) + Shadow (1) + Alpha (1)
    print("Creating HeightmapUNetLite model with 5 input channels...")
    model = HeightmapUNetLite(predict_normals=True, in_channels=5)
    model = model.to(device)
    
    # Note: torch.compile can cause issues with some GPU configurations
    # Disabled by default - enable with --compile flag if desired
    # try:
    #     if hasattr(torch, 'compile') and device.type == "cuda":
    #         model = torch.compile(model, mode="reduce-overhead")
    #         print("Model compiled with torch.compile (reduce-overhead mode)")
    # except Exception as e:
    #     print(f"torch.compile not available: {e}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Use ReduceLROnPlateau for adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    use_amp = USE_AMP and device.type == "cuda"
    print(f"Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")
    
    # Training loop
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0
    epochs_without_improvement = 0
    num_epochs = epochs if epochs else NUM_EPOCHS
    
    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    
    print(f"\nTraining for {num_epochs - start_epoch} epochs (epochs {start_epoch+1} to {num_epochs})")
    print(f"Loss weights: smooth={SMOOTHNESS_WEIGHT}, grad={GRADIENT_WEIGHT}, boundary={BOUNDARY_WEIGHT}, normal={NORMAL_WEIGHT}, consistency={CONSISTENCY_WEIGHT}")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    print()
    
    epoch = start_epoch  # Initialize for exception handlers
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                target_normals = batch["normals"].to(device, non_blocking=True)
                holes_mask = batch["holes_mask"].to(device, non_blocking=True)
                chunk_positions = batch["chunk_positions"].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_heights, pred_normals = model(pixel_values, chunk_positions)
                    
                    # Apply holes mask to loss - don't penalize predictions for hole regions
                    mask_expanded = holes_mask.unsqueeze(-1).expand_as(pred_heights)
                    
                    # Masked MSE loss for heights
                    height_diff = (pred_heights - labels) ** 2
                    masked_height_diff = height_diff * mask_expanded
                    loss_mse = masked_height_diff.sum() / (mask_expanded.sum() + 1e-6)
                    
                    # Smoothness loss
                    loss_smooth = compute_smoothness_loss_2d(pred_heights)
                    
                    # Gradient loss - CRITICAL for terrain shape
                    loss_grad = compute_gradient_loss(pred_heights, labels)
                    
                    # Boundary continuity loss
                    loss_boundary = compute_boundary_continuity_loss(pred_heights)
                    
                    # Normal prediction loss (masked)
                    normal_mask = holes_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_normals)
                    normal_diff = (pred_normals - target_normals) ** 2
                    masked_normal_diff = normal_diff * normal_mask
                    loss_normal = masked_normal_diff.sum() / (normal_mask.sum() + 1e-6)
                    
                    # Height-normal consistency loss - ensures height gradients match normal directions
                    # This prevents height collapse while normals capture terrain features
                    loss_consistency = compute_normal_height_consistency_loss(pred_heights, pred_normals)
                    
                    # Combined loss
                    loss = (loss_mse + 
                            SMOOTHNESS_WEIGHT * loss_smooth + 
                            GRADIENT_WEIGHT * loss_grad + 
                            BOUNDARY_WEIGHT * loss_boundary +
                            NORMAL_WEIGHT * loss_normal +
                            CONSISTENCY_WEIGHT * loss_consistency)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{loss_mse.item():.4f}",
                    "cons": f"{loss_consistency.item():.4f}",
                })
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    for batch in val_loader:
                        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                        labels = batch["labels"].to(device, non_blocking=True)
                        target_normals = batch["normals"].to(device, non_blocking=True)
                        holes_mask = batch["holes_mask"].to(device, non_blocking=True)
                        chunk_positions = batch["chunk_positions"].to(device, non_blocking=True)
                        
                        pred_heights, pred_normals = model(pixel_values, chunk_positions)
                        
                        # Masked height loss
                        mask_expanded = holes_mask.unsqueeze(-1).expand_as(pred_heights)
                        height_diff = (pred_heights - labels) ** 2
                        loss_h = (height_diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
                        
                        # Gradient loss for validation too
                        loss_g = compute_gradient_loss(pred_heights, labels)
                        
                        loss_b = compute_boundary_continuity_loss(pred_heights)
                        
                        # Masked normal loss
                        normal_mask = holes_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_normals)
                        normal_diff = (pred_normals - target_normals) ** 2
                        loss_n = (normal_diff * normal_mask).sum() / (normal_mask.sum() + 1e-6)
                        
                        # Height-normal consistency loss
                        loss_c = compute_normal_height_consistency_loss(pred_heights, pred_normals)
                        
                        loss = loss_h + GRADIENT_WEIGHT * loss_g + BOUNDARY_WEIGHT * loss_b + NORMAL_WEIGHT * loss_n + CONSISTENCY_WEIGHT * loss_c
                        val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= max(len(val_loader), 1)
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.2e}")
            
            # Early stopping check
            if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                }, OUTPUT_DIR / "best_model.pt")
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"\n*** Early stopping triggered after {epochs_without_improvement} epochs without improvement ***")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            
            # Periodic checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                }, checkpoint_path)
                print(f"  -> Saved checkpoint: {checkpoint_path.name}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving emergency checkpoint...")
        emergency_path = OUTPUT_DIR / "emergency_checkpoint.pt"
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
            }, emergency_path)
            print(f"Saved: {emergency_path}")
            print("You can resume with: --resume emergency_checkpoint.pt")
        except Exception as save_err:
            print(f"Could not save emergency checkpoint: {save_err}")
        return
    
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        print("Saving emergency checkpoint...")
        emergency_path = OUTPUT_DIR / "emergency_checkpoint.pt"
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
            }, emergency_path)
            print(f"Saved: {emergency_path}")
        except Exception as save_err:
            print(f"Could not save emergency checkpoint: {save_err}")
        raise
    
    # Save final model and stats
    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    
    # Save normalization stats
    stats = {
        "global_min": dataset.global_min,
        "global_max": dataset.global_max,
        "global_mean": dataset.global_mean,
        "global_std": dataset.global_std,
        "output_shape": [256, 145],
        "output_normals_shape": [256, 145, 3],
        "input_size": 256,
        "in_channels": 5,  # RGB (3) + Shadow (1) + Alpha (1)
        "uses_chunk_positions": True,
        "predict_normals": True,
        "normalization_mode": "per_tile",  # Model outputs [0,1] relative heights
    }
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    global BATCH_SIZE, LEARNING_RATE, OUTPUT_DIR, USE_AMP, EARLY_STOP_PATIENCE
    
    parser = argparse.ArgumentParser(
        description="Train WoW Height Regressor V3 - Auto-optimized for your GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_height_regressor_v3.py                    # Auto-optimized training
  python train_height_regressor_v3.py --resume best_model.pt  # Resume training
  python train_height_regressor_v3.py --no-amp           # Disable mixed precision
  
Training will automatically:
  - Detect GPU VRAM and set optimal batch size
  - Use mixed precision (AMP) for faster training
  - Stop early when validation loss plateaus
  - Save best model and periodic checkpoints
        """
    )
    parser.add_argument("--resume", "-r", help="Resume from checkpoint file")
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS,
                        help=f"Max epochs (default: {NUM_EPOCHS}, early stopping may end sooner)")
    parser.add_argument("--batch-size", "-b", type=int, 
                        help="Batch size (default: auto-detected based on VRAM)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--output", "-o", help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--no-amp", action="store_true", 
                        help="Disable mixed precision training")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE,
                        help=f"Early stopping patience (default: {EARLY_STOP_PATIENCE})")
    
    args = parser.parse_args()
    
    # Override globals if specified
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.lr:
        LEARNING_RATE = args.lr
    if args.output:
        OUTPUT_DIR = Path(args.output)
    if args.no_amp:
        USE_AMP = False
    if args.patience:
        EARLY_STOP_PATIENCE = args.patience
    
    train(resume_from=args.resume, epochs=args.epochs)


if __name__ == "__main__":
    main()
