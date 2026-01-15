#!/usr/bin/env python3
"""
WoW Height Regressor V4 - Comprehensive Training with Modern Deep Learning

Key improvements over V3:
- Pretrained encoder backbone (EfficientNet/ResNet) for transfer learning
- L1 loss instead of MSE (reduces mean-regression)
- SSIM loss for structural similarity
- Perceptual loss using VGG features
- Much stronger gradient/edge loss (terrain shape is PRIMARY objective)
- Cosine annealing with warm restarts
- Multi-scale feature fusion
- Attention mechanisms
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
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

# Try to import timm for better pretrained models
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Note: Install 'timm' for better pretrained backbones: pip install timm")

# Dataset paths
DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_shadowfang_v1"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
]
OUTPUT_DIR = Path(r"J:\vlm_output\wow_height_regressor_v4")

# WoW ADT structure
OUTPUT_H = 256  # chunks per tile (16x16)
OUTPUT_W = 145  # heights per chunk

# Training hyperparameters
BATCH_SIZE = 4  # Smaller due to larger model
LEARNING_RATE = 1e-4  # Lower for pretrained backbone
NUM_EPOCHS = 500
EARLY_STOP_PATIENCE = 50

# Loss weights - TERRAIN SHAPE IS PRIMARY
L1_WEIGHT = 1.0           # Base height loss
SSIM_WEIGHT = 1.0         # Structural similarity
PERCEPTUAL_WEIGHT = 0.5   # VGG feature matching
GRADIENT_WEIGHT = 5.0     # CRITICAL: Very high - terrain shape is PRIMARY
EDGE_WEIGHT = 2.0         # Edge-aware loss
NORMAL_WEIGHT = 0.5       # Normal prediction
CONSISTENCY_WEIGHT = 1.0  # Height-normal consistency

USE_AMP = True


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
        
    def _create_window(self, window_size, channel):
        _1D_window = torch.Tensor(
            [np.exp(-(x - window_size//2)**2 / float(2 * 1.5**2)) 
             for x in range(window_size)]
        )
        _1D_window = _1D_window / _1D_window.sum()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        # Reshape to image format if needed [B, H, W] -> [B, 1, H, W]
        if img1.dim() == 3:
            img1 = img1.unsqueeze(1)
            img2 = img2.unsqueeze(1)
        
        channel = img1.size(1)
        window = self.window.to(img1.device).type_as(img1)
        if channel != self.channel:
            window = self._create_window(self.window_size, channel).to(img1.device).type_as(img1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for feature matching"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        # Convert single channel heightmap to 3-channel for VGG
        # pred/target: [B, 256, 145] -> [B, 3, 16, 16*145] or similar
        B = pred.shape[0]
        
        # Reshape to 2D image: [B, 16, 16, 145] -> [B, 1, 16*9, 16*16] approximately
        # Simplified: just use the raw 256x145 as a pseudo-image
        pred_img = pred.reshape(B, 1, 256, 145)
        target_img = target.reshape(B, 1, 256, 145)
        
        # Resize to VGG-friendly size and expand to 3 channels
        pred_img = F.interpolate(pred_img, size=(224, 224), mode='bilinear', align_corners=False)
        target_img = F.interpolate(target_img, size=(224, 224), mode='bilinear', align_corners=False)
        
        pred_img = pred_img.expand(-1, 3, -1, -1)
        target_img = target_img.expand(-1, 3, -1, -1)
        
        # Normalize
        pred_img = (pred_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std
        
        # Extract features
        pred_feat = self.features(pred_img)
        target_feat = self.features(target_img)
        
        return F.l1_loss(pred_feat, target_feat)


class SobelEdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters"""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        B = pred.shape[0]
        
        # Reshape to 2D
        pred_2d = pred.reshape(B, 1, 256, 145)
        target_2d = target.reshape(B, 1, 256, 145)
        
        # Apply Sobel filters
        pred_edge_x = F.conv2d(pred_2d, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_2d, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target_2d, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_2d, self.sobel_y, padding=1)
        
        # Edge magnitude
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        return F.l1_loss(pred_edge, target_edge)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with skip connection and attention"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attention = CBAM(out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


class TerrainRegressorV4(nn.Module):
    """
    Advanced terrain height regressor with:
    - Pretrained encoder backbone
    - Feature Pyramid Network style decoder
    - Attention mechanisms
    - Multi-scale output
    """
    def __init__(self, in_channels=5, predict_normals=True, backbone='efficientnet_b0'):
        super().__init__()
        self.predict_normals = predict_normals
        
        # Input projection (5 channels -> 3 for pretrained backbone)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),  # Project to 3 channels for backbone
        )
        
        # Pretrained encoder
        if HAS_TIMM:
            self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
            encoder_channels = self.encoder.feature_info.channels()
        else:
            # Fallback to torchvision ResNet
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.encoder = nn.ModuleList([
                nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            ])
            encoder_channels = [64, 64, 128, 256, 512]
        
        self.has_timm = HAS_TIMM
        
        # Feature Pyramid Network style decoder
        self.decoder4 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], 256)
        self.decoder3 = DecoderBlock(256, encoder_channels[-3], 128)
        self.decoder2 = DecoderBlock(128, encoder_channels[-4], 64)
        self.decoder1 = DecoderBlock(64, encoder_channels[-5], 32)
        
        # Final upsampling to match input size
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Output heads
        # Height head - outputs 256 chunks x 145 heights = 37120 values
        self.height_head = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((256, 145)),  # Force output to ADT structure
            nn.Conv2d(32, 1, 1),
        )
        
        # Normal head
        if predict_normals:
            self.normal_head = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((256, 145)),
                nn.Conv2d(32, 3, 1),
                nn.Tanh(),  # Normals in [-1, 1]
            )
    
    def forward(self, x, chunk_positions=None):
        B = x.shape[0]
        
        # Project input channels
        x = self.input_proj(x)
        
        # Encode
        if self.has_timm:
            features = self.encoder(x)
        else:
            features = []
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                features.append(x)
        
        # Decode with skip connections
        d4 = self.decoder4(features[-1], features[-2])
        d3 = self.decoder3(d4, features[-3])
        d2 = self.decoder2(d3, features[-4])
        d1 = self.decoder1(d2, features[-5])
        
        # Final upsampling
        out = self.final_up(d1)
        
        # Height output
        heights = self.height_head(out)
        heights = heights.squeeze(1)  # [B, 256, 145]
        heights = torch.sigmoid(heights)  # Output in [0, 1] for per-tile normalization
        
        # Normal output
        if self.predict_normals:
            normals = self.normal_head(out)
            normals = normals.permute(0, 2, 3, 1)  # [B, 256, 145, 3]
            return heights, normals
        
        return heights, None


class WoWTileDataset(Dataset):
    """Full-tile dataset for V4 training"""
    def __init__(self, dataset_roots, augment=True):
        self.augment = augment
        self.samples = []
        self.all_heights = []
        
        for root in dataset_roots:
            if not root.exists():
                continue
            
            images_dir = root / "images"
            labels_dir = root / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            print(f"Loading from {root.name}...")
            
            for img_path in images_dir.glob("*.png"):
                tile_name = img_path.stem
                json_path = labels_dir / f"{tile_name}.json"
                
                if not json_path.exists():
                    continue
                
                try:
                    with open(json_path, 'r') as f:
                        label_data = json.load(f)
                    
                    if "chunks" not in label_data or len(label_data["chunks"]) != 256:
                        continue
                    
                    # Check all chunks have 145 heights
                    all_heights_valid = all(
                        len(c.get("heights", [])) == 145 for c in label_data["chunks"]
                    )
                    if not all_heights_valid:
                        continue
                    
                    # Extract all heights for normalization stats
                    tile_heights = []
                    for chunk in label_data["chunks"]:
                        tile_heights.extend(chunk["heights"])
                    
                    self.all_heights.extend(tile_heights)
                    
                    # Check for optional data
                    has_shadow = (root / "shadows" / f"{tile_name}_shadow.png").exists()
                    has_alpha = (root / "alphas" / f"{tile_name}_alpha.png").exists()
                    has_normals = all("normals" in c for c in label_data["chunks"])
                    has_positions = all("chunk_position" in c for c in label_data["chunks"])
                    has_holes = all("is_hole" in c for c in label_data["chunks"])
                    
                    self.samples.append({
                        "image_path": img_path,
                        "json_path": json_path,
                        "root": root,
                        "tile_name": tile_name,
                        "has_shadow": has_shadow,
                        "has_alpha": has_alpha,
                        "has_normals": has_normals,
                        "has_positions": has_positions,
                        "has_holes": has_holes,
                    })
                except Exception as e:
                    continue
        
        # Compute global stats
        if self.all_heights:
            self.global_min = min(self.all_heights)
            self.global_max = max(self.all_heights)
            self.global_mean = sum(self.all_heights) / len(self.all_heights)
            self.global_std = (sum((h - self.global_mean)**2 for h in self.all_heights) / len(self.all_heights)) ** 0.5
        else:
            self.global_min = 0
            self.global_max = 1
            self.global_mean = 0.5
            self.global_std = 1
        
        # Print stats
        n_shadows = sum(1 for s in self.samples if s["has_shadow"])
        n_alphas = sum(1 for s in self.samples if s["has_alpha"])
        n_normals = sum(1 for s in self.samples if s["has_normals"])
        n_positions = sum(1 for s in self.samples if s["has_positions"])
        n_holes = sum(1 for s in self.samples if s["has_holes"])
        
        print(f"Loaded {len(self.samples)} complete tiles")
        print(f"  With shadows:   {n_shadows} ({100*n_shadows/max(len(self.samples),1):.1f}%)")
        print(f"  With alphas:    {n_alphas} ({100*n_alphas/max(len(self.samples),1):.1f}%)")
        print(f"  With normals:   {n_normals} ({100*n_normals/max(len(self.samples),1):.1f}%)")
        print(f"  With positions: {n_positions} ({100*n_positions/max(len(self.samples),1):.1f}%)")
        print(f"  With holes:     {n_holes} ({100*n_holes/max(len(self.samples),1):.1f}%)")
        print(f"Height range: [{self.global_min:.2f}, {self.global_max:.2f}]")
        print(f"Height stats: mean={self.global_mean:.2f}, std={self.global_std:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def _normalize_per_tile(self, heights):
        """Normalize heights to [0, 1] within each tile"""
        tile_min = min(heights)
        tile_max = max(heights)
        tile_range = tile_max - tile_min
        if tile_range < 1e-6:
            return [0.5] * len(heights), tile_min, tile_max
        return [(h - tile_min) / tile_range for h in heights], tile_min, tile_max
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load main image
        img = Image.open(sample["image_path"]).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        
        # Load shadow map
        if sample["has_shadow"]:
            shadow_path = sample["root"] / "shadows" / f"{sample['tile_name']}_shadow.png"
            shadow = Image.open(shadow_path).convert("L")
            shadow_tensor = transforms.ToTensor()(shadow)
        else:
            shadow_tensor = torch.zeros(1, img_tensor.shape[1], img_tensor.shape[2])
        
        # Load alpha map
        if sample["has_alpha"]:
            alpha_path = sample["root"] / "alphas" / f"{sample['tile_name']}_alpha.png"
            alpha = Image.open(alpha_path).convert("L")
            alpha_tensor = transforms.ToTensor()(alpha)
        else:
            alpha_tensor = torch.ones(1, img_tensor.shape[1], img_tensor.shape[2])
        
        # Resize to 256x256
        target_size = (256, 256)
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        shadow_tensor = F.interpolate(shadow_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        alpha_tensor = F.interpolate(alpha_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        # Stack channels: RGB (3) + Shadow (1) + Alpha (1) = 5
        pixel_values = torch.cat([img_tensor, shadow_tensor, alpha_tensor], dim=0)
        
        # Load labels
        with open(sample["json_path"], 'r') as f:
            label_data = json.load(f)
        
        # Extract heights
        all_heights = []
        for chunk in label_data["chunks"]:
            all_heights.extend(chunk["heights"])
        
        # Per-tile normalization
        normalized_heights, tile_min, tile_max = self._normalize_per_tile(all_heights)
        
        # Reshape to [256, 145]
        heights_tensor = torch.tensor(normalized_heights, dtype=torch.float32).reshape(256, 145)
        
        # Extract normals
        if sample["has_normals"]:
            all_normals = []
            for chunk in label_data["chunks"]:
                chunk_normals = chunk.get("normals", [[0, 0, 1]] * 145)
                all_normals.extend(chunk_normals)
            normals_tensor = torch.tensor(all_normals, dtype=torch.float32).reshape(256, 145, 3)
        else:
            normals_tensor = torch.zeros(256, 145, 3)
            normals_tensor[:, :, 2] = 1.0
        
        # Extract chunk positions
        if sample["has_positions"]:
            positions = [chunk.get("chunk_position", [0, 0, 0]) for chunk in label_data["chunks"]]
            positions_tensor = torch.tensor(positions, dtype=torch.float32)
        else:
            positions_tensor = torch.zeros(256, 3)
        
        # Extract holes mask
        if sample["has_holes"]:
            holes_mask = torch.tensor(
                [0.0 if chunk.get("is_hole", False) else 1.0 for chunk in label_data["chunks"]],
                dtype=torch.float32
            )
        else:
            holes_mask = torch.ones(256, dtype=torch.float32)
        
        # Augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            # Horizontal flip
            pixel_values = torch.flip(pixel_values, dims=[2])
            heights_tensor = torch.flip(heights_tensor.reshape(16, 16, 145), dims=[1]).reshape(256, 145)
            normals_tensor = torch.flip(normals_tensor.reshape(16, 16, 145, 3), dims=[1]).reshape(256, 145, 3)
            normals_tensor[:, :, 0] = -normals_tensor[:, :, 0]
        
        return {
            "pixel_values": pixel_values,
            "labels": heights_tensor,
            "normals": normals_tensor,
            "holes_mask": holes_mask,
            "chunk_positions": positions_tensor,
            "tile_min": tile_min,
            "tile_max": tile_max,
        }


def compute_gradient_loss(pred, target):
    """Strong 2D gradient matching loss - PRIMARY for terrain shape"""
    B = pred.shape[0]
    
    # Reshape to 2D grid
    pred_2d = pred.reshape(B, 1, 256, 145)
    target_2d = target.reshape(B, 1, 256, 145)
    
    # Multi-scale gradients for better shape capture
    losses = []
    
    for scale in [1, 2, 4]:
        if scale > 1:
            p = F.avg_pool2d(pred_2d, scale)
            t = F.avg_pool2d(target_2d, scale)
        else:
            p, t = pred_2d, target_2d
        
        # X gradient
        grad_x_pred = p[:, :, :, 1:] - p[:, :, :, :-1]
        grad_x_target = t[:, :, :, 1:] - t[:, :, :, :-1]
        
        # Y gradient
        grad_y_pred = p[:, :, 1:, :] - p[:, :, :-1, :]
        grad_y_target = t[:, :, 1:, :] - t[:, :, :-1, :]
        
        # L1 loss on gradients (more robust than MSE)
        losses.append(F.l1_loss(grad_x_pred, grad_x_target))
        losses.append(F.l1_loss(grad_y_pred, grad_y_target))
    
    return sum(losses) / len(losses)


def compute_normal_height_consistency(pred_heights, pred_normals):
    """Enforce height gradients match normal directions"""
    B = pred_heights.shape[0]
    
    h = pred_heights.reshape(B, 1, 256, 145)
    n = pred_normals.reshape(B, 256, 145, 3)
    
    # Height gradients
    dh_x = h[:, :, :, 1:] - h[:, :, :, :-1]
    dh_y = h[:, :, 1:, :] - h[:, :, :-1, :]
    
    # Normal X and Y components (averaged at gradient locations)
    n_x = (n[:, :, 1:, 0] + n[:, :, :-1, 0]) / 2
    n_y = (n[:, 1:, :, 1] + n[:, :-1, :, 1]) / 2
    
    # Gradients should correlate negatively with normals
    dh_x_norm = torch.tanh(dh_x.squeeze(1) * 10)
    dh_y_norm = torch.tanh(dh_y.squeeze(1) * 10)
    
    loss_x = F.l1_loss(dh_x_norm, -n_x)
    loss_y = F.l1_loss(dh_y_norm, -n_y)
    
    return (loss_x + loss_y) / 2


def train(resume_from=None, epochs=None):
    print("=" * 70)
    print("WoW Height Regressor V4 - Comprehensive Training")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # GPU info
    actual_batch_size = BATCH_SIZE
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
        
        # Auto-adjust batch size
        if vram_gb >= 24:
            actual_batch_size = 8
        elif vram_gb >= 16:
            actual_batch_size = 4
        elif vram_gb >= 12:
            actual_batch_size = 3
        else:
            actual_batch_size = 2
        
        print(f"Auto-selected batch size: {actual_batch_size}")
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("Enabled: TF32, cuDNN benchmark")
    
    # Create dataset
    dataset = WoWTileDataset(DATASET_ROOTS, augment=True)
    
    if len(dataset) == 0:
        print("ERROR: No tiles loaded!")
        return
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    num_workers = min(4, os.cpu_count() or 2)  # Fewer workers for larger model
    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=actual_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    print(f"DataLoader: {num_workers} workers, batch_size={actual_batch_size}")
    
    # Create model
    backbone = 'efficientnet_b0' if HAS_TIMM else 'resnet34'
    print(f"Creating TerrainRegressorV4 with {backbone} backbone...")
    model = TerrainRegressorV4(in_channels=5, predict_normals=True, backbone=backbone)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss functions
    ssim_loss = SSIMLoss().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    edge_loss = SobelEdgeLoss().to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == "cuda")
    use_amp = USE_AMP and device.type == "cuda"
    print(f"Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")
    
    # Training state
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0
    epochs_without_improvement = 0
    num_epochs = epochs if epochs else NUM_EPOCHS
    
    # Resume
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    
    print(f"\nTraining for up to {num_epochs - start_epoch} epochs")
    print(f"Loss weights: L1={L1_WEIGHT}, SSIM={SSIM_WEIGHT}, Perceptual={PERCEPTUAL_WEIGHT}")
    print(f"              Gradient={GRADIENT_WEIGHT}, Edge={EDGE_WEIGHT}, Normal={NORMAL_WEIGHT}")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    print()
    
    epoch = start_epoch
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
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_heights, pred_normals = model(pixel_values)
                    
                    # Mask for holes
                    mask = holes_mask.unsqueeze(-1).expand_as(pred_heights)
                    
                    # L1 loss (masked)
                    l1 = (torch.abs(pred_heights - labels) * mask).sum() / (mask.sum() + 1e-6)
                    
                    # SSIM loss
                    loss_ssim = ssim_loss(pred_heights * mask, labels * mask)
                    
                    # Perceptual loss
                    loss_perc = perceptual_loss(pred_heights, labels)
                    
                    # Gradient loss - PRIMARY FOR TERRAIN SHAPE
                    loss_grad = compute_gradient_loss(pred_heights, labels)
                    
                    # Edge loss
                    loss_edge = edge_loss(pred_heights, labels)
                    
                    # Normal loss (masked)
                    normal_mask = holes_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_normals)
                    loss_normal = (torch.abs(pred_normals - target_normals) * normal_mask).sum() / (normal_mask.sum() + 1e-6)
                    
                    # Height-normal consistency
                    loss_cons = compute_normal_height_consistency(pred_heights, pred_normals)
                    
                    # Total loss - GRADIENT IS PRIMARY
                    loss = (L1_WEIGHT * l1 +
                            SSIM_WEIGHT * loss_ssim +
                            PERCEPTUAL_WEIGHT * loss_perc +
                            GRADIENT_WEIGHT * loss_grad +
                            EDGE_WEIGHT * loss_edge +
                            NORMAL_WEIGHT * loss_normal +
                            CONSISTENCY_WEIGHT * loss_cons)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "grad": f"{loss_grad.item():.4f}",
                    "l1": f"{l1.item():.4f}",
                })
            
            scheduler.step()
            
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
                        
                        pred_heights, pred_normals = model(pixel_values)
                        
                        mask = holes_mask.unsqueeze(-1).expand_as(pred_heights)
                        l1 = (torch.abs(pred_heights - labels) * mask).sum() / (mask.sum() + 1e-6)
                        loss_grad = compute_gradient_loss(pred_heights, labels)
                        loss_edge = edge_loss(pred_heights, labels)
                        
                        loss = L1_WEIGHT * l1 + GRADIENT_WEIGHT * loss_grad + EDGE_WEIGHT * loss_edge
                        val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= max(len(val_loader), 1)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.2e}")
            
            # Early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epochs_without_improvement': 0,
                }, OUTPUT_DIR / "best_model.pt")
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"\n*** Early stopping after {epochs_without_improvement} epochs without improvement ***")
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                }, OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pt")
                print(f"  -> Saved checkpoint: checkpoint_epoch{epoch+1}.pt")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving emergency checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'epochs_without_improvement': epochs_without_improvement,
        }, OUTPUT_DIR / "emergency_checkpoint.pt")
        print(f"Saved: {OUTPUT_DIR / 'emergency_checkpoint.pt'}")
        return
    
    except Exception as e:
        print(f"\n\nError: {e}")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
            }, OUTPUT_DIR / "emergency_checkpoint.pt")
            print(f"Saved emergency checkpoint")
        except:
            pass
        raise
    
    # Save final
    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    
    stats = {
        "global_min": dataset.global_min,
        "global_max": dataset.global_max,
        "global_mean": dataset.global_mean,
        "global_std": dataset.global_std,
        "output_shape": [256, 145],
        "input_size": 256,
        "in_channels": 5,
        "predict_normals": True,
        "normalization_mode": "per_tile",
        "backbone": backbone,
        "version": "v4",
    }
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="WoW Height Regressor V4 - Comprehensive Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This V4 training includes:
  - Pretrained EfficientNet/ResNet backbone
  - L1 + SSIM + Perceptual losses
  - Strong gradient and edge losses (terrain shape is PRIMARY)
  - Attention mechanisms (CBAM)
  - Cosine annealing with warm restarts
  - Height-normal consistency loss

Examples:
  python train_height_regressor_v4.py
  python train_height_regressor_v4.py --resume best_model.pt
        """
    )
    parser.add_argument("--resume", "-r", help="Resume from checkpoint")
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS, help="Max epochs")
    
    args = parser.parse_args()
    train(resume_from=args.resume, epochs=args.epochs)


if __name__ == "__main__":
    main()
