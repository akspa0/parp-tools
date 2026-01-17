#!/usr/bin/env python3
"""
V6 Heightmap Regression Inference Script
Runs trained model on minimap tiles to generate heightmaps
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json

from models.unet_v6 import UNetV6
from datasets.v6_dataset import V6HeightmapDataset
from inference.inference_engine import V6InferenceEngine
from utils.logging import setup_logging
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run V6 heightmap inference")
    parser.add_argument("--model", required=True, help="Trained model checkpoint path")
    parser.add_argument("--input", required=True, help="Input directory with minimaps")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Configuration file (overrides model config)")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("infer_v6")
    
    # Load model and config
    checkpoint = torch.load(args.model, map_location="cpu")
    model_config = checkpoint.get("config", {})
    
    # Override with provided config
    if args.config:
        external_config = load_config(args.config)
        model_config.update(external_config)
    
    # Override batch size if specified
    if args.batch_size:
        model_config["inference"]["batch_size"] = args.batch_size
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = UNetV6(
        encoder_name=model_config["model"]["encoder"],
        input_channels=model_config["model"]["input_channels"],
        output_channels=model_config["model"]["output_channels"],
        pretrained=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {args.model}")
    
    # Create inference engine
    engine = V6InferenceEngine(
        model=model,
        config=model_config["inference"],
        device=device,
        logger=logger
    )
    
    # Run inference
    results = engine.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        test_time_augmentation=args.tta or model_config["inference"]["test_time_augmentation"]
    )
    
    # Generate summary
    summary = {
        "model_path": args.model,
        "input_dir": args.input,
        "output_dir": args.output,
        "tiles_processed": len(results),
        "config": model_config
    }
    
    summary_path = Path(args.output) / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Inference completed! Processed {len(results)} tiles")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
