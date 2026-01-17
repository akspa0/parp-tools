#!/usr/bin/env python3
"""
V6 Heightmap Regression Training Script
Trains U-Net with ResNet-34 encoder for dual heightmap + alpha mask prediction
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.unet_v6 import UNetV6
from datasets.v6_dataset import V6HeightmapDataset
from training.trainer_v6 import V6Trainer
from utils.logging import setup_logging
from utils.config import load_config, validate_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train V6 heightmap regression model")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--resume", help="Checkpoint path to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("train_v6", debug=args.debug)
    
    # Load configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = UNetV6(
        encoder_name=config["model"]["encoder"],
        input_channels=config["model"]["input_channels"],
        output_channels=config["model"]["output_channels"],
        pretrained=config["model"]["pretrained"]
    )
    model.to(device)
    
    # Create datasets
    train_dataset = V6HeightmapDataset(
        root=args.dataset,
        split="train",
        config=config["data"],
        augmentation=True
    )
    
    val_dataset = V6HeightmapDataset(
        root=args.dataset,
        split="val",
        config=config["data"],
        augmentation=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"]
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = V6Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
