#!/usr/bin/env python3
"""
WoW Height Regressor V4 - DPT (Dense Prediction Transformer) Based

Uses Hugging Face transformers DPT model pretrained on depth estimation,
which is EXACTLY the same task as predicting terrain heights from minimaps.

Key advantages:
- DPT is state-of-the-art for monocular depth estimation
- Pretrained on large depth datasets (NYU, KITTI, etc.)
- Vision Transformer backbone with global receptive field
- Dense prediction head designed for pixel-wise regression
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

# Hugging Face transformers
from transformers import DPTForDepthEstimation, DPTImageProcessor, DPTConfig
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Dataset paths
DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_shadowfang_v1"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
]
OUTPUT_DIR = Path(r"J:\vlm_output\wow_height_regressor_v4_dpt")

# Training hyperparameters
BATCH_SIZE = 2  # DPT is memory-intensive
LEARNING_RATE = 1e-5  # Much lower LR to prevent gradient explosion
NUM_EPOCHS = 500
EARLY_STOP_PATIENCE = 50

# Loss weights - TERRAIN SHAPE IS PRIMARY
L1_WEIGHT = 1.0
GRADIENT_WEIGHT = 2.0  # Reduced to prevent NaN
SSIM_WEIGHT = 1.0
EDGE_WEIGHT = 2.0
SCALE_INV_WEIGHT = 1.0  # Scale-invariant loss (common in depth estimation)

USE_AMP = True

# DPT model to use - pretrained on depth estimation
DPT_MODEL_NAME = "Intel/dpt-large"  # or "Intel/dpt-hybrid-midas" for faster


class ScaleInvariantLoss(nn.Module):
    """
    Scale-invariant loss commonly used in depth estimation.
    Allows the model to learn relative depths without being penalized for global scale.
    """
    def __init__(self, lambda_weight=0.5):
        super().__init__()
        self.lambda_weight = lambda_weight
    
    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred * mask
            target = target * mask
            n_valid = mask.sum() + 1e-6
        else:
            n_valid = pred.numel()
        
        diff = pred - target
        
        # Scale-invariant term
        si_term = (diff ** 2).sum() / n_valid
        
        # Variance reduction term
        var_term = (diff.sum() ** 2) / (n_valid ** 2)
        
        return si_term - self.lambda_weight * var_term


class GradientMatchingLoss(nn.Module):
    """Multi-scale gradient matching loss for terrain shape"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        total_loss = 0.0
        
        for scale in [1, 2, 4]:
            if scale > 1:
                p = F.avg_pool2d(pred.unsqueeze(1), scale).squeeze(1)
                t = F.avg_pool2d(target.unsqueeze(1), scale).squeeze(1)
            else:
                p, t = pred, target
            
            # Gradients in both directions
            grad_x_pred = p[:, :, 1:] - p[:, :, :-1]
            grad_x_target = t[:, :, 1:] - t[:, :, :-1]
            grad_y_pred = p[:, 1:, :] - p[:, :-1, :]
            grad_y_target = t[:, 1:, :] - t[:, :-1, :]
            
            total_loss += F.l1_loss(grad_x_pred, grad_x_target)
            total_loss += F.l1_loss(grad_y_pred, grad_y_target)
        
        return total_loss / 6  # Average over 3 scales * 2 directions


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                              for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        self.register_buffer('window', _2d.unsqueeze(0).unsqueeze(0))
    
    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        
        window = self.window.to(pred.device).type_as(pred)
        
        mu1 = F.conv2d(pred, window, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, padding=self.window_size//2)
        
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred**2, window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target**2, window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2) - mu1_mu2
        
        C1, C2 = 0.01**2, 0.03**2
        
        ssim = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim.mean()


class SobelEdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters"""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        
        pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_x = F.conv2d(target, self.sobel_x, padding=1)
        target_y = F.conv2d(target, self.sobel_y, padding=1)
        
        pred_edge = torch.sqrt(pred_x**2 + pred_y**2 + 1e-6)
        target_edge = torch.sqrt(target_x**2 + target_y**2 + 1e-6)
        
        return F.l1_loss(pred_edge, target_edge)


class TerrainDPT(nn.Module):
    """
    DPT-based terrain height regressor.
    
    Uses pretrained DPT model and adapts it for WoW terrain prediction.
    DPT was designed for monocular depth estimation, which is essentially
    the same task as predicting terrain heights from top-down minimaps.
    """
    def __init__(self, model_name=DPT_MODEL_NAME, predict_normals=True):
        super().__init__()
        self.predict_normals = predict_normals
        
        # Load pretrained DPT
        print(f"Loading pretrained DPT: {model_name}")
        self.dpt = DPTForDepthEstimation.from_pretrained(model_name)
        
        # Get the image processor for proper preprocessing
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        
        # Freeze early layers, fine-tune later layers
        # This is a common transfer learning strategy
        for name, param in self.dpt.named_parameters():
            if 'neck' in name or 'head' in name:
                param.requires_grad = True
            else:
                # Freeze backbone initially, can unfreeze later
                param.requires_grad = True  # Fine-tune everything
        
        # Additional head for WoW-specific output shape adaptation
        # DPT outputs at input resolution, we need 256x145 for WoW ADT
        self.output_adapter = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        # Normal prediction head
        if predict_normals:
            self.normal_head = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 1),
                nn.Tanh(),
            )
    
    def forward(self, pixel_values):
        """
        Forward pass.
        
        Args:
            pixel_values: [B, 3, H, W] RGB image tensor (normalized)
        
        Returns:
            heights: [B, 256, 145] terrain heights
            normals: [B, 256, 145, 3] surface normals (if predict_normals=True)
        """
        B = pixel_values.shape[0]
        
        # DPT forward pass
        outputs = self.dpt(pixel_values)
        predicted_depth = outputs.predicted_depth  # [B, H, W]
        
        # Add channel dimension for conv layers
        depth_features = predicted_depth.unsqueeze(1)  # [B, 1, H, W]
        
        # Adapt to WoW output shape
        depth_adapted = self.output_adapter(depth_features)
        
        # Resize to WoW ADT shape [256, 145]
        heights = F.interpolate(depth_adapted, size=(256, 145), mode='bilinear', align_corners=False)
        heights = heights.squeeze(1)  # [B, 256, 145]
        
        # Normalize to [0, 1] for per-tile normalization
        heights = torch.sigmoid(heights)
        
        # Predict normals
        if self.predict_normals:
            normal_features = F.interpolate(depth_features, size=(256, 145), mode='bilinear', align_corners=False)
            normals = self.normal_head(normal_features)
            normals = normals.permute(0, 2, 3, 1)  # [B, 256, 145, 3]
            return heights, normals
        
        return heights, None


class WoWTileDataset(Dataset):
    """Full-tile dataset for DPT training"""
    def __init__(self, dataset_roots, image_size=384, augment=True):
        self.augment = augment
        self.image_size = image_size
        self.samples = []
        self.all_heights = []
        
        # Standard ImageNet normalization for DPT
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        for root in dataset_roots:
            if not root.exists():
                continue
            
            dataset_dir = root / "dataset"
            images_dir = root / "images"
            stitched_dir = root / "stitched"
            
            if not dataset_dir.exists():
                continue
            
            print(f"Loading from {root.name}...")
            
            for json_path in dataset_dir.glob("*.json"):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Parse terrain_data structure (V3 format)
                    td = data.get("terrain_data", {})
                    heights_list = td.get("heights", [])
                    chunk_layers = td.get("chunk_layers", [])
                    
                    if len(heights_list) < 256:
                        continue
                    
                    # Build height array [256, 145]
                    tile_heights = np.zeros((256, 145), dtype=np.float32)
                    valid_chunks = 0
                    
                    for chunk in heights_list:
                        idx = chunk.get("idx", -1)
                        h_vals = chunk.get("h", chunk.get("heights", []))
                        
                        if 0 <= idx < 256 and len(h_vals) == 145:
                            tile_heights[idx] = h_vals
                            valid_chunks += 1
                    
                    if valid_chunks < 200:
                        continue
                    
                    # Find image
                    tile_name = json_path.stem
                    img_candidates = [
                        stitched_dir / f"{tile_name}_minimap.png",
                        stitched_dir / f"{tile_name}.png",
                        images_dir / f"{tile_name}.png",
                    ]
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
                    
                    # Build normals array [256, 145, 3]
                    tile_normals = np.zeros((256, 145, 3), dtype=np.float32)
                    has_normals = False
                    
                    for layer in chunk_layers:
                        idx = layer.get("idx", -1)
                        normals_raw = layer.get("normals", None)
                        
                        if normals_raw is not None and 0 <= idx < 256:
                            if len(normals_raw) >= 435:
                                for v in range(145):
                                    for c in range(3):
                                        tile_normals[idx, v, c] = normals_raw[v * 3 + c] / 127.0
                                has_normals = True
                    
                    # Extract holes
                    holes_data = td.get("holes", None)
                    if holes_data is not None and len(holes_data) >= 256:
                        holes_array = np.array(holes_data[:256], dtype=np.int32)
                    else:
                        holes_array = np.zeros(256, dtype=np.int32)
                    
                    self.all_heights.append(tile_heights.flatten())
                    
                    self.samples.append({
                        "image_path": img_path,
                        "heights": tile_heights,
                        "normals": tile_normals if has_normals else None,
                        "holes": holes_array,
                        "tile_name": tile_name,
                    })
                except Exception:
                    continue
        
        # Global stats from all height arrays
        if self.all_heights:
            all_h = np.concatenate(self.all_heights)
            self.global_min = float(all_h.min())
            self.global_max = float(all_h.max())
            self.global_mean = float(all_h.mean())
            self.global_std = float(all_h.std())
        else:
            self.global_min, self.global_max = 0, 1
            self.global_mean, self.global_std = 0.5, 1
        
        print(f"Loaded {len(self.samples)} complete tiles")
        print(f"Height range: [{self.global_min:.2f}, {self.global_max:.2f}]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        img = Image.open(sample["image_path"]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img)
        img_tensor = self.normalize(img_tensor)
        
        # Get pre-computed heights [256, 145]
        tile_heights = sample["heights"]
        
        # Per-tile normalization to [0, 1]
        tile_min = tile_heights.min()
        tile_max = tile_heights.max()
        tile_range = tile_max - tile_min
        if tile_range < 1e-6:
            normalized = np.full_like(tile_heights, 0.5)
        else:
            normalized = (tile_heights - tile_min) / tile_range
        
        heights_tensor = torch.tensor(normalized, dtype=torch.float32)
        
        # Get normals [256, 145, 3]
        tile_normals = sample["normals"]
        if tile_normals is not None:
            normals_tensor = torch.tensor(tile_normals, dtype=torch.float32)
        else:
            normals_tensor = torch.zeros(256, 145, 3)
            normals_tensor[:, :, 2] = 1.0
        
        # Holes mask - convert bitmask to float mask
        holes = sample["holes"]
        holes_mask = torch.tensor([0.0 if h != 0 else 1.0 for h in holes], dtype=torch.float32)
        
        # Augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[2])
            heights_tensor = torch.flip(heights_tensor.reshape(16, 16, 145), dims=[1]).reshape(256, 145)
            normals_tensor = torch.flip(normals_tensor.reshape(16, 16, 145, 3), dims=[1]).reshape(256, 145, 3)
            normals_tensor[:, :, 0] = -normals_tensor[:, :, 0]
        
        return {
            "pixel_values": img_tensor,
            "labels": heights_tensor,
            "normals": normals_tensor,
            "holes_mask": holes_mask,
            "tile_min": float(tile_min),
            "tile_max": float(tile_max),
        }


def train(resume_from=None, epochs=None):
    print("=" * 70)
    print("WoW Height Regressor V4 - DPT (Dense Prediction Transformer)")
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
        
        # DPT is memory intensive
        if vram_gb >= 24:
            actual_batch_size = 4
        elif vram_gb >= 16:
            actual_batch_size = 2
        else:
            actual_batch_size = 1
        
        print(f"Auto-selected batch size: {actual_batch_size}")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Create dataset
    dataset = WoWTileDataset(DATASET_ROOTS, image_size=384, augment=True)
    
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
    
    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=actual_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    print(f"DataLoader: {num_workers} workers, batch_size={actual_batch_size}")
    
    # Create model
    print(f"\nLoading DPT model: {DPT_MODEL_NAME}")
    model = TerrainDPT(model_name=DPT_MODEL_NAME, predict_normals=True)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Loss functions
    scale_inv_loss = ScaleInvariantLoss().to(device)
    gradient_loss = GradientMatchingLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    edge_loss = SobelEdgeLoss().to(device)
    
    # Optimizer - lower LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
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
        print(f"Resumed from epoch {start_epoch}")
    
    print(f"\nTraining for up to {num_epochs - start_epoch} epochs")
    print(f"Loss weights: L1={L1_WEIGHT}, Gradient={GRADIENT_WEIGHT}, SSIM={SSIM_WEIGHT}, Edge={EDGE_WEIGHT}")
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
                    
                    # Mask
                    mask = holes_mask.unsqueeze(-1).expand_as(pred_heights)
                    
                    # L1 loss
                    l1 = (torch.abs(pred_heights - labels) * mask).sum() / (mask.sum() + 1e-6)
                    
                    # Scale-invariant loss (with NaN protection)
                    try:
                        loss_si = scale_inv_loss(pred_heights, labels, mask)
                        if torch.isnan(loss_si):
                            loss_si = torch.tensor(0.0, device=device)
                    except:
                        loss_si = torch.tensor(0.0, device=device)
                    
                    # Gradient loss - PRIMARY FOR TERRAIN SHAPE
                    try:
                        loss_grad = gradient_loss(pred_heights, labels)
                        if torch.isnan(loss_grad):
                            loss_grad = torch.tensor(0.0, device=device)
                    except:
                        loss_grad = torch.tensor(0.0, device=device)
                    
                    # SSIM loss (can be unstable)
                    try:
                        loss_ssim = ssim_loss(pred_heights, labels)
                        if torch.isnan(loss_ssim):
                            loss_ssim = torch.tensor(0.0, device=device)
                    except:
                        loss_ssim = torch.tensor(0.0, device=device)
                    
                    # Edge loss
                    try:
                        loss_edge = edge_loss(pred_heights, labels)
                        if torch.isnan(loss_edge):
                            loss_edge = torch.tensor(0.0, device=device)
                    except:
                        loss_edge = torch.tensor(0.0, device=device)
                    
                    # Normal loss
                    if pred_normals is not None:
                        normal_mask = holes_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_normals)
                        loss_normal = (torch.abs(pred_normals - target_normals) * normal_mask).sum() / (normal_mask.sum() + 1e-6)
                    else:
                        loss_normal = torch.tensor(0.0, device=device)
                    
                    # Total loss - focus on L1 + gradient for stability
                    loss = (L1_WEIGHT * l1 +
                            SCALE_INV_WEIGHT * loss_si +
                            GRADIENT_WEIGHT * loss_grad +
                            SSIM_WEIGHT * loss_ssim +
                            EDGE_WEIGHT * loss_edge +
                            0.5 * loss_normal)
                    
                    # Clamp loss to prevent explosion
                    loss = torch.clamp(loss, 0, 100)
                
                # Skip NaN losses
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Tighter clipping
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
            val_height_var = 0.0
            val_count = 0
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    for batch in val_loader:
                        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                        labels = batch["labels"].to(device, non_blocking=True)
                        holes_mask = batch["holes_mask"].to(device, non_blocking=True)
                        
                        pred_heights, _ = model(pixel_values)
                        
                        # Skip if predictions contain NaN
                        if torch.isnan(pred_heights).any():
                            continue
                        
                        mask = holes_mask.unsqueeze(-1).expand_as(pred_heights)
                        l1 = (torch.abs(pred_heights - labels) * mask).sum() / (mask.sum() + 1e-6)
                        
                        try:
                            loss_grad = gradient_loss(pred_heights, labels)
                            if torch.isnan(loss_grad):
                                loss_grad = torch.tensor(0.0, device=device)
                        except:
                            loss_grad = torch.tensor(0.0, device=device)
                        
                        loss = L1_WEIGHT * l1 + GRADIENT_WEIGHT * loss_grad
                        
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            val_loss += loss.item()
                            val_height_var += pred_heights.var().item()
                            val_count += 1
            
            train_loss /= len(train_loader)
            val_loss /= max(val_count, 1)
            val_height_var /= max(val_count, 1)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, HeightVar={val_height_var:.4f}, LR={current_lr:.2e}")
            
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
                    print(f"\n*** Early stopping after {epochs_without_improvement} epochs ***")
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
        return
    
    except Exception as e:
        print(f"\n\nError: {e}")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, OUTPUT_DIR / "emergency_checkpoint.pt")
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
        "input_size": 384,
        "in_channels": 3,
        "predict_normals": True,
        "normalization_mode": "per_tile",
        "dpt_model": DPT_MODEL_NAME,
        "version": "v4_dpt",
    }
    with open(OUTPUT_DIR / "normalization_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="WoW Height Regressor V4 - DPT Based",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This V4-DPT training uses:
  - Intel DPT-Large: Pretrained dense prediction transformer
  - Designed specifically for monocular depth estimation
  - Vision Transformer backbone with global receptive field
  - Scale-invariant loss + strong gradient matching
  - Per-tile normalization for terrain shape learning

Example:
  python train_height_regressor_v4_dpt.py
  python train_height_regressor_v4_dpt.py --resume best_model.pt
        """
    )
    parser.add_argument("--resume", "-r", help="Resume from checkpoint")
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS, help="Max epochs")
    
    args = parser.parse_args()
    train(resume_from=args.resume, epochs=args.epochs)


if __name__ == "__main__":
    main()
