"""Authorize bounded M0 continuation only from measured undertraining evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.spec102.m0 import PRECISE_MASK_KEY


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize corrected Spec 102 M0 continuation")
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    history = training.get("history", [])
    ious = [float(item["validation"]["iou"]) for item in history]
    train_losses = [float(item["train_loss"]) for item in history]
    late_best = len(ious) == 3 and int(max(range(len(ious)), key=ious.__getitem__)) >= 1
    train_loss_decreasing = len(train_losses) == 3 and all(
        later < earlier for earlier, later in zip(train_losses, train_losses[1:])
    )
    gain = ious[-1] - ious[0] if len(ious) == 3 else 0.0
    conditions = {
        "canonical_precise_target": training.get("target") == PRECISE_MASK_KEY,
        "three_epoch_decision_run": training.get("epochs") == 3,
        "validation_best_in_final_two_epochs": late_best,
        "validation_iou_gain_at_least_0_02": gain >= 0.02,
        "train_loss_decreasing": train_loss_decreasing,
        "calibrated_validation_above_uncalibrated": (
            float(calibration.get("selected_validation_iou", 0.0)) > float(training.get("best_validation_iou", 1.0))
        ),
        "peak_vram_below_7gb": float(training.get("peak_vram_gb", 99.0)) < 7.0,
    }
    report = {
        "schema": "spec102-m0-extension-authorization-v1",
        "extension_authorized": all(conditions.values()),
        "maximum_total_epochs": 12,
        "conditions": conditions,
        "validation_ious": ious,
        "validation_iou_gain": gain,
        "selected_threshold": calibration.get("selected_threshold"),
        "selected_validation_iou": calibration.get("selected_validation_iou"),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["extension_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
