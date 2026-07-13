"""Train Spec 102 M0: RGB minimap -> one object-visibility mask."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from harvester.spec102.m0 import (
    PRECISE_MASK_KEY,
    M0ObjectMask,
    precise_object_target_256,
    segmentation_loss,
)


class MaskDataset(Dataset):
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, rows: list[int]) -> None:
        self.rgb = rgb
        self.mask = mask
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        rgb = np.ascontiguousarray(self.rgb[row].transpose(2, 0, 1), dtype=np.float32) / 255.0
        mask = np.ascontiguousarray(self.mask[row] > 0.5, dtype=np.float32)[None]
        return torch.from_numpy(rgb), torch.from_numpy(mask)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    intersection = union = predicted = target = loss_sum = samples = 0.0
    with torch.inference_mode():
        for rgb, mask in loader:
            rgb = rgb.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(rgb)
                loss, _ = segmentation_loss(logits, mask)
            binary = torch.sigmoid(logits) >= 0.5
            truth = mask >= 0.5
            intersection += float((binary & truth).sum().item())
            union += float((binary | truth).sum().item())
            predicted += float(binary.sum().item())
            target += float(truth.sum().item())
            loss_sum += float(loss.item()) * rgb.shape[0]
            samples += rgb.shape[0]
    return {
        "loss": loss_sum / max(samples, 1.0),
        "iou": intersection / max(union, 1.0),
        "dice": (2.0 * intersection) / max(predicted + target, 1.0),
    }


def write_validation_grid(
    path: Path,
    model: nn.Module,
    dataset: MaskDataset,
    device: torch.device,
    count: int = 4,
) -> None:
    rows = min(count, len(dataset))
    canvas = Image.new("RGB", (256 * 3, 256 * rows))
    model.eval()
    for index in range(rows):
        rgb, target = dataset[index]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probability = torch.sigmoid(model(rgb[None].to(device)))[0, 0].float().cpu().numpy()
        source = np.clip(rgb.permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
        predicted = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
        truth = (target[0].numpy() * 255.0).astype(np.uint8)
        canvas.paste(Image.fromarray(source, "RGB"), (0, index * 256))
        canvas.paste(Image.fromarray(predicted, "L").convert("RGB"), (256, index * 256))
        canvas.paste(Image.fromarray(truth, "L").convert("RGB"), (512, index * 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Spec 102 M0 object-mask model")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--prior-gate-report", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.epochs < 1 or args.epochs > 12:
        raise ValueError("Spec 102 M0 permits 1-12 epochs; extension requires a passed prior gate")
    if args.epochs > 3:
        if args.prior_gate_report is None or not args.prior_gate_report.is_file():
            raise RuntimeError("M0 runs beyond 3 epochs require --prior-gate-report")
        prior_gate = json.loads(args.prior_gate_report.read_text(encoding="utf-8"))
        if not (prior_gate.get("gate_passed") or prior_gate.get("extension_authorized")):
            raise RuntimeError("M0 extended run blocked: prior bounded run did not pass or authorize undertraining continuation")
    if not torch.cuda.is_available():
        raise RuntimeError("M0 is CUDA-only; CPU fallback is prohibited")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spec102-curated-split-v2":
        raise RuntimeError("M0 refuses an uncurated split manifest")
    rows_by_split = {
        split: [
            int(row["row"]) for row in manifest["rows"]
            if row["split"] == split and row.get("eligible_m0") is True
        ]
        for split in ("train", "validation_map", "test_era")
    }
    if any(not rows_by_split[split] for split in rows_by_split):
        raise RuntimeError(f"M0 curated split is empty: { {key: len(value) for key, value in rows_by_split.items()} }")
    group = zarr.open_group(str(args.store), mode="r")
    target_name = PRECISE_MASK_KEY
    if target_name not in group:
        raise RuntimeError(
            f"M0 canonical target '{target_name}' is missing; refusing coarse-mask or visibility-mask fallbacks"
        )
    rgb = np.asarray(group["minimap_rgb"][:])
    precise_masks = np.asarray(group[target_name][:], dtype=np.float32)
    if precise_masks.shape[1:] != (257, 257):
        raise RuntimeError(f"M0 canonical precise target has invalid shape {precise_masks.shape}")
    masks = np.stack([precise_object_target_256(mask) for mask in precise_masks], axis=0)
    if rgb.shape[0] != len(manifest["rows"]) or masks.shape[0] != len(manifest["rows"]):
        raise RuntimeError("Store row count does not match the frozen split")
    eligible_rows = np.asarray(
        [row for split_rows in rows_by_split.values() for row in split_rows], dtype=np.int64
    )
    positive_prevalence = float((masks[eligible_rows] > 0.5).mean())
    if positive_prevalence <= 0.0:
        raise RuntimeError(f"M0 target '{target_name}' is empty; refusing a meaningless GPU run")

    datasets = {split: MaskDataset(rgb, masks, rows) for split, rows in rows_by_split.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=True),
        "validation_map": DataLoader(datasets["validation_map"], batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True),
        "test_era": DataLoader(datasets["test_era"], batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True),
    }

    model = M0ObjectMask().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 3_000_000 <= parameter_count <= 12_000_000:
        raise RuntimeError(f"M0 parameter count {parameter_count} violates the 3-12M contract")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "schema": "spec102-m0-input-v1",
        "forward_inputs": [{"name": "minimap_rgb", "shape": ["batch", 3, 256, 256], "deployment_source": "raw minimap RGB"}],
        "target_only": [{
            "name": target_name,
            "shape": ["batch", 257, 257],
            "projection": "four_corner_max_to_256",
        }],
        "output": {"name": "object_visibility_mask_logits", "shape": ["batch", 1, 256, 256]},
        "prohibited_inputs": [
            "height_257", "normal_xyz_257", "wdl_height_33", target_name,
            "object_mask_256", "object_visibility_256",
        ],
    }
    (args.output_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

    history: list[dict] = []
    best_iou = -1.0
    start_epoch = 0
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if resume.get("schema") != "spec102-m0-checkpoint-v1":
            raise RuntimeError("--resume-checkpoint is not a Spec 102 M0 checkpoint")
        model.load_state_dict(resume["model"], strict=True)
        start_epoch = int(resume["epoch"])
        best_iou = float(resume["validation"]["iou"])
        torch.save(resume, args.output_dir / "checkpoint_best.pt")
    if start_epoch >= args.epochs:
        raise RuntimeError(f"resume checkpoint epoch {start_epoch} already reaches requested epoch {args.epochs}")
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = samples = 0.0
        for batch_rgb, batch_mask in loaders["train"]:
            batch_rgb = batch_rgb.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch_rgb)
                loss, _ = segmentation_loss(
                    logits,
                    batch_mask,
                    positive_weight=args.positive_weight,
                    dice_weight=args.dice_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite M0 loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * batch_rgb.shape[0]
            samples += batch_rgb.shape[0]
        validation = evaluate(model, loaders["validation_map"], device)
        record = {"epoch": epoch, "train_loss": train_loss / max(samples, 1), "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        write_validation_grid(args.output_dir / "validation" / f"epoch_{epoch:02d}.png", model, datasets["validation_map"], device)
        if validation["iou"] > best_iou:
            best_iou = validation["iou"]
            torch.save(
                {
                    "schema": "spec102-m0-checkpoint-v1",
                    "model": model.state_dict(),
                    "config": {
                        "base_channels": 40,
                        "parameter_count": parameter_count,
                        "positive_weight": args.positive_weight,
                        "dice_weight": args.dice_weight,
                        "target": target_name,
                        "single_output": True,
                    },
                    "input_manifest": input_manifest,
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output_dir / "checkpoint_best.pt",
            )

    checkpoint = torch.load(args.output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test_era"], device)
    gate_passed = best_iou >= 0.25 and test_metrics["iou"] >= 0.10
    report = {
        "schema": "spec102-m0-training-report-v1",
        "parameter_count": parameter_count,
        "target": target_name,
        "target_positive_prevalence": positive_prevalence,
        "epochs": args.epochs,
        "resumed_from_epoch": start_epoch,
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
        "history": history,
        "best_validation_iou": best_iou,
        "test_era": test_metrics,
        "zero_mask_baseline_iou": 0.0,
        "gate_requirements": {"validation_iou_min": 0.25, "test_era_iou_min": 0.10},
        "gate_passed": gate_passed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
