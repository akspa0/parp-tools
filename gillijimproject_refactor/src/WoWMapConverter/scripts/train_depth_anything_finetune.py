#!/usr/bin/env python3
"""
Fine-tune Depth Anything V3 on WoW Terrain Data

Uses the pretrained Depth Anything V3 model as a foundation and fine-tunes
it on WoW minimap → heightmap pairs. This allows the model to learn
WoW-specific texture-to-height correlations.

Requirements:
    - DA3 installed from VLM/DepthAnything3/Depth-Anything-3
    - pip install albumentations

Usage:
    python train_depth_anything_finetune.py
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from tqdm import tqdm

# Add DA3 to path
DA3_PATH = Path(__file__).parent.parent / "WoWMapConverter.Core" / "VLM" / "DepthAnything3" / "Depth-Anything-3" / "src"
sys.path.insert(0, str(DA3_PATH))

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print(f"Error: Could not import DepthAnything3 from {DA3_PATH}")
    print("Make sure DA3 is installed: pip install -e . from the Depth-Anything-3 directory")
    sys.exit(1)

# Dataset paths
DATASET_ROOTS = [
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_shadowfang_v1"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
    Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
]

OUTPUT_DIR = Path(r"J:\vlm_output\depth_anything_wow_finetune")

# Model config
MODEL_NAME = "da3-large"  # DA3 model name
INPUT_SIZE = 518  # Depth Anything expects 518x518
OUTPUT_SIZE = 129  # Our heightmap size


class SiLogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss for depth estimation."""
    
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, pred, target, valid_mask=None):
        if valid_mask is None:
            valid_mask = target > 0
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        log_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
        silog = torch.sqrt(
            torch.mean(log_diff ** 2) - self.lambd * (torch.mean(log_diff) ** 2)
        )
        return silog


class WoWDepthDataset(Dataset):
    """Dataset for WoW minimap → heightmap training."""
    
    def __init__(
        self,
        dataset_roots: List[Path],
        mode: str = "train",
        input_size: int = 518,
        output_size: int = 129,
        augment: bool = True,
        variance_threshold: float = 10.0,
    ):
        self.mode = mode
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment and mode == "train"
        self.variance_threshold = variance_threshold
        
        # Collect samples
        self.samples = []
        for root in dataset_roots:
            self._scan_dataset(root)
        
        # Split train/val
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * 0.9)
        
        if mode == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]
        
        print(f"[{mode}] Loaded {len(self.samples)} samples")
        
        # Augmentations
        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        ])
        
        # Normalization for Depth Anything
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _scan_dataset(self, root: Path):
        """Scan dataset for valid samples with heightmaps."""
        dataset_dir = root / "dataset"
        images_dir = root / "images"
        
        if not dataset_dir.exists() or not images_dir.exists():
            return
        
        for json_path in dataset_dir.glob("*.json"):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check for image
                img_name = json_path.stem + ".png"
                img_path = images_dir / img_name
                if not img_path.exists():
                    continue
                
                # Check for heightmap (v2 preferred)
                heightmap_path = images_dir / f"{json_path.stem}_heightmap_v2.png"
                if not heightmap_path.exists():
                    heightmap_path = images_dir / f"{json_path.stem}_heightmap.png"
                if not heightmap_path.exists():
                    continue
                
                # Load heights from JSON to check variance
                td = data.get("terrain_data", {})
                heights_list = td.get("heights", [])
                if not heights_list:
                    continue
                
                all_heights = []
                for chunk in heights_list:
                    h = chunk.get("h", chunk.get("heights", []))
                    all_heights.extend(h)
                
                if len(all_heights) < 100:
                    continue
                
                variance = np.var(all_heights)
                if variance < self.variance_threshold:
                    continue
                
                self.samples.append({
                    "image": img_path,
                    "heightmap": heightmap_path,
                    "json": json_path,
                    "variance": variance,
                })
                
            except Exception as e:
                pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image"]).convert("RGB")
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image = np.array(image)
        
        # Load heightmap
        heightmap = Image.open(sample["heightmap"])
        if heightmap.mode == 'I;16':
            heightmap = np.array(heightmap, dtype=np.float32) / 65535.0
        else:
            heightmap = np.array(heightmap.convert('L'), dtype=np.float32) / 255.0
        
        # Resize heightmap to output size
        heightmap = Image.fromarray((heightmap * 255).astype(np.uint8))
        heightmap = heightmap.resize((self.output_size, self.output_size), Image.BILINEAR)
        heightmap = np.array(heightmap, dtype=np.float32) / 255.0
        
        # Augmentations
        if self.augment:
            # Need to resize heightmap to match image for augmentation
            hm_large = Image.fromarray((heightmap * 255).astype(np.uint8))
            hm_large = hm_large.resize((self.input_size, self.input_size), Image.BILINEAR)
            hm_large = np.array(hm_large, dtype=np.float32) / 255.0
            
            augmented = self.augs(image=image, mask=hm_large)
            image = augmented["image"]
            hm_large = augmented["mask"]
            
            # Resize back to output size
            hm_img = Image.fromarray((hm_large * 255).astype(np.uint8))
            hm_img = hm_img.resize((self.output_size, self.output_size), Image.BILINEAR)
            heightmap = np.array(hm_img, dtype=np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        
        heightmap = torch.from_numpy(heightmap).float()
        
        return {
            "pixel_values": image,
            "labels": heightmap,
        }


class DepthAnythingWoW(nn.Module):
    """Depth Anything V3 fine-tuned for WoW terrain."""
    
    def __init__(self, model_name: str = MODEL_NAME, output_size: int = OUTPUT_SIZE):
        super().__init__()
        
        # Load pretrained Depth Anything V3
        self.da3 = DepthAnything3(model_name=model_name)
        self.backbone = self.da3.model  # Get the underlying model for training
        
        self.output_size = output_size
        self.input_processor = self.da3.input_processor
    
    def forward(self, pixel_values):
        """
        Forward pass for training.
        pixel_values: [B, 3, H, W] normalized tensor
        """
        # DA3 model expects features from encoder
        # Get depth prediction from backbone
        B, C, H, W = pixel_values.shape
        
        # Run through backbone
        outputs = self.backbone(pixel_values)
        
        # Extract depth from outputs (DA3 returns a dict)
        if isinstance(outputs, dict):
            depth = outputs.get('depth', outputs.get('disparity', None))
            if depth is None:
                # Try to get first tensor output
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        depth = v
                        break
        else:
            depth = outputs
        
        # Ensure 3D: [B, H, W]
        if depth.dim() == 4:
            depth = depth.squeeze(1)
        
        # Resize to our output size
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)
        
        # Normalize to 0-1 range (per-sample)
        depth_min = depth.view(depth.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        depth_max = depth.view(depth.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        depth_range = depth_max - depth_min + 1e-8
        depth = (depth - depth_min) / depth_range
        
        return depth


def compute_metrics(pred, target):
    """Compute depth estimation metrics."""
    valid_mask = target > 0.01
    
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    if pred_valid.numel() == 0:
        return {}
    
    # AbsRel
    abs_rel = torch.mean(torch.abs(pred_valid - target_valid) / (target_valid + 1e-8))
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))
    
    # MAE
    mae = torch.mean(torch.abs(pred_valid - target_valid))
    
    # Delta1 (threshold accuracy)
    thresh = torch.max(pred_valid / (target_valid + 1e-8), target_valid / (pred_valid + 1e-8))
    delta1 = (thresh < 1.25).float().mean()
    
    return {
        "abs_rel": abs_rel.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "delta1": delta1.item(),
    }


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(pixel_values)
                loss = criterion(pred, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(pixel_values)
            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = {"abs_rel": [], "rmse": [], "mae": [], "delta1": []}
    
    for batch in tqdm(dataloader, desc="Validating"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        pred = model(pixel_values)
        loss = criterion(pred, labels)
        total_loss += loss.item()
        
        # Compute metrics
        metrics = compute_metrics(pred, labels)
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
    avg_metrics["loss"] = total_loss / len(dataloader)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Depth Anything V2 on WoW terrain")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = WoWDepthDataset(DATASET_ROOTS, mode="train", augment=True)
    val_dataset = WoWDepthDataset(DATASET_ROOTS, mode="val", augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Create model
    print(f"\nLoading Depth Anything V2 model: {MODEL_NAME}")
    model = DepthAnythingWoW(MODEL_NAME).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer with different LR for encoder/decoder
    encoder_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr},
        {"params": decoder_params, "lr": args.lr * 10},
    ], weight_decay=0.01)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    
    # Loss
    criterion = SiLogLoss(lambd=0.5)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # Training loop
    best_val_loss = float("inf")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AbsRel: {val_metrics.get('abs_rel', 0):.4f}")
        print(f"Val RMSE: {val_metrics.get('rmse', 0):.4f}")
        print(f"Val Delta1: {val_metrics.get('delta1', 0):.4f}")
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "metrics": val_metrics,
            }, output_dir / "best_model.pt")
            print(f"  *** New best model saved! ***")
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
        }, output_dir / "checkpoint.pt")
    
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
