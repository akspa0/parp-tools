#!/usr/bin/env python3
"""CLI for Spec 120 OBB Minimap Detector Trainer (T007).

Dry-run-first: prints model parameter count, loss configuration, learning rate schedule, and training plan.
Requires --confirm-run to launch actual GPU training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.spec120.obb_contract import STAGE_OBB_DETECTOR
from harvester.spec120.obb_detector_model import MinimapOBBDetector
from harvester.spec120.obb_detector_train import generate_dry_run_report, train_obb_detector_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Spec 120 Minimap OBB Object Detector."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../output/spec120/obb_dataset"),
        help="Path to converted OBB training dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../output/spec120/runs/obb_detector_v1"),
        help="Directory to save checkpoints and run records.",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=16,
        help="Base width parameter for model channels (default: 16).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Maximum learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Explicit user confirmation flag required to launch GPU training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = MinimapOBBDetector(in_channels=3, num_classes=4, base=args.base)
    report = generate_dry_run_report(model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    print(f"=== Spec 120 OBB Detector Trainer ({STAGE_OBB_DETECTOR}) ===")
    print(f"Architecture: {report['arch']}")
    print(f"Base Width:   {report['base']}")
    print(f"Parameters:   {report['num_params']:,}")
    print(f"Epochs:       {report['epochs']}")
    print(f"Batch Size:   {report['batch_size']}")
    print(f"Max LR:       {report['max_lr']}")
    print(f"Device:       {report['device']}")
    print(f"Dataset Dir:  {args.dataset_dir.resolve()}")
    print(f"Output Dir:   {args.output_dir.resolve()}")
    print(f"Confirm Run:  {args.confirm_run}")

    if not args.confirm_run:
        print("\n[DRY-RUN COMPLETE] Training plan generated successfully. No training was launched.")
        print("Pass --confirm-run to execute GPU training.")
    else:
        print("\n[USER CONFIRMED] Launching PyTorch OBB Detector training loop...")
        ckpt_path = train_obb_detector_loop(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            base=args.base,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print(f"[RUN SUCCESS] Training finished. Model saved to {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
