"""Train Spec 102 W1: cleaned RGB + frozen H0 -> one 545-sample residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from harvester.spec102.w1 import RELIEF_SCALE, W1WdlResidual, masked_residual_l1
from harvester.v25.h0_offset import H0OffsetModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def block_mean_rgb(rgb: np.ndarray, size: int = 128) -> np.ndarray:
    factor = 256 // size
    small = rgb.reshape(len(rgb), size, factor, size, factor, 3).mean(axis=(2, 4))
    return np.moveaxis(small.astype(np.float32) / 255.0, -1, 1)


def liquid_vertex_valid(liquid: np.ndarray) -> np.ndarray:
    cells = liquid > 127
    covered = np.zeros((len(liquid), 257, 257), dtype=bool)
    covered[:, :-1, :-1] |= cells
    covered[:, 1:, :-1] |= cells
    covered[:, :-1, 1:] |= cells
    covered[:, 1:, 1:] |= cells
    return ~covered


def freeze_h0(raw_rgb: np.ndarray, checkpoint: dict, device: torch.device, batch_size: int) -> np.ndarray:
    model = H0OffsetModel().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    config = checkpoint["config"]
    outputs: list[np.ndarray] = []
    for start in range(0, len(raw_rgb), batch_size):
        rgb = torch.from_numpy(block_mean_rgb(raw_rgb[start : start + batch_size], 64)).to(device)
        with torch.inference_mode():
            baseline = float(config["rgb_flat_slope"]) * rgb.float().mean(dim=(1, 2, 3)) + float(config["rgb_flat_intercept"])
            outputs.append((baseline + model(rgb).float()).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def prepare_split(
    rows: list[int],
    raw_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    heights: np.ndarray,
    liquids: np.ndarray,
    h0_all: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = np.asarray(rows, dtype=np.int64)
    clean = block_mean_rgb(cleaned_rgb[selected], 128)
    height = heights[selected]
    valid257 = liquid_vertex_valid(liquids[selected]) & np.isfinite(height)
    target = np.concatenate(
        [height[:, ::16, ::16].reshape(len(rows), -1), height[:, 8::16, 8::16].reshape(len(rows), -1)],
        axis=1,
    ).astype(np.float32)
    valid = np.concatenate(
        [valid257[:, ::16, ::16].reshape(len(rows), -1), valid257[:, 8::16, 8::16].reshape(len(rows), -1)],
        axis=1,
    )
    h0 = h0_all[selected]
    residual = (target - h0[:, None]) / RELIEF_SCALE
    return (
        torch.from_numpy(np.ascontiguousarray(clean)),
        torch.from_numpy(np.ascontiguousarray(h0)),
        torch.from_numpy(np.ascontiguousarray(residual)),
        torch.from_numpy(np.ascontiguousarray(valid)),
    )


def evaluate(model: W1WdlResidual, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    error = baseline_error = valid_count = 0.0
    with torch.inference_mode():
        for rgb, h0, residual, valid in loader:
            rgb, h0, residual, valid = rgb.to(device), h0.to(device), residual.to(device), valid.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(rgb, h0)
            target_world = h0[:, None] + (residual * RELIEF_SCALE)
            prediction_world = h0[:, None] + (prediction.float() * RELIEF_SCALE)
            weights = valid.float()
            error += float((torch.abs(prediction_world - target_world) * weights).sum().item())
            baseline_error += float((torch.abs(h0[:, None] - target_world) * weights).sum().item())
            valid_count += float(weights.sum().item())
    return {"l1": error / max(valid_count, 1.0), "h0_plane_l1": baseline_error / max(valid_count, 1.0)}


def write_grid(path: Path, model: W1WdlResidual, data: tuple[torch.Tensor, ...], device: torch.device) -> None:
    rgb, h0, residual, _ = data
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(rgb[:4].to(device), h0[:4].to(device)).float().cpu()
    predicted = h0[:4, None] + (prediction * RELIEF_SCALE)
    target = h0[:4, None] + (residual[:4] * RELIEF_SCALE)
    canvas = Image.new("L", (17 * 16, 17 * 8))
    for row in range(min(4, len(rgb))):
        lo = float(min(predicted[row, :289].min(), target[row, :289].min()))
        hi = float(max(predicted[row, :289].max(), target[row, :289].max()))
        scale = 255.0 / max(hi - lo, 1e-6)
        for column, values in enumerate((predicted[row, :289], target[row, :289])):
            image = np.clip((values.reshape(17, 17).numpy() - lo) * scale, 0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(image, "L").resize((17 * 8, 17 * 8), Image.Resampling.NEAREST), (column * 17 * 8, row * 17 * 8))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Spec 102 W1 numeric WDL residual")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--m0-store", required=True, type=Path)
    parser.add_argument("--m0-checkpoint", required=True, type=Path)
    parser.add_argument("--m0-gate-report", required=True, type=Path)
    parser.add_argument("--h0-checkpoint", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=102)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 3:
        raise ValueError("W1 decision runs are capped at 3 epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("W1 is CUDA-only; CPU fallback is prohibited")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    m0_checkpoint = torch.load(args.m0_checkpoint, map_location="cpu", weights_only=False)
    m0_gate = json.loads(args.m0_gate_report.read_text(encoding="utf-8"))
    if m0_checkpoint.get("schema") != "spec102-m0-checkpoint-v1" or not m0_gate.get("gate_passed"):
        raise RuntimeError("W1 blocked: frozen M0 gate did not pass")
    h0_checkpoint = torch.load(args.h0_checkpoint, map_location="cpu", weights_only=False)
    if h0_checkpoint.get("config", {}).get("stage") != "H0":
        raise RuntimeError("W1 requires the frozen single-output H0 checkpoint")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spec102-curated-split-v2":
        raise RuntimeError("W1 refuses an uncurated split manifest")
    rows_by_split = {
        split: [
            int(row["row"]) for row in manifest["rows"]
            if row["split"] == split and row.get("eligible_w1") is True
        ]
        for split in ("train", "validation_map", "test_era")
    }
    if any(not rows_by_split[split] for split in rows_by_split):
        raise RuntimeError(f"W1 curated split is empty: { {key: len(value) for key, value in rows_by_split.items()} }")
    source = zarr.open_group(str(args.store), mode="r")
    m0 = zarr.open_group(str(args.m0_store), mode="r")
    if m0.attrs.get("checkpoint_sha256") != sha256_file(args.m0_checkpoint):
        raise RuntimeError("M0 materialized outputs do not match the supplied frozen checkpoint")
    raw_rgb = np.asarray(source["minimap_rgb"][:], dtype=np.uint8)
    cleaned_rgb = np.asarray(m0["clean_minimap_256"][:], dtype=np.uint8)
    heights = np.asarray(source["height_257"][:], dtype=np.float32)
    liquids = np.asarray(source["liquid_mask_256"][:])
    h0_all = freeze_h0(raw_rgb, h0_checkpoint, device, args.batch_size * 4)
    data = {
        split: prepare_split(rows, raw_rgb, cleaned_rgb, heights, liquids, h0_all)
        for split, rows in rows_by_split.items()
    }
    loaders = {
        split: DataLoader(TensorDataset(*values), batch_size=args.batch_size, shuffle=(split == "train"), num_workers=0, pin_memory=True)
        for split, values in data.items()
    }
    model = W1WdlResidual().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if not 3_000_000 <= parameters <= 12_000_000:
        raise RuntimeError(f"W1 parameter count {parameters} violates the 3-12M contract")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "schema": "spec102-w1-input-v1",
        "forward_inputs": ["clean_minimap_256 from frozen M0", "frozen H0 datum"],
        "target_only": ["wdl_outer_17", "wdl_inner_16", "liquid validity mask"],
        "prohibited_inputs": ["wdl_height_33", "height_257", "native normals", "target tile mean"],
        "output": {"name": "wdl_lattice_residual", "shape": ["batch", 545]},
    }
    (args.output_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

    best_l1 = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = samples = 0.0
        for rgb, h0, residual, valid in loaders["train"]:
            rgb, h0, residual, valid = rgb.to(device), h0.to(device), residual.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(rgb, h0)
                loss = masked_residual_l1(prediction, residual, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite W1 loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * rgb.shape[0]
            samples += rgb.shape[0]
        validation = evaluate(model, loaders["validation_map"], device)
        record = {"epoch": epoch, "train_scaled_l1": loss_sum / max(samples, 1), "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        write_grid(args.output_dir / "validation" / f"epoch_{epoch:02d}.png", model, data["validation_map"], device)
        if validation["l1"] < best_l1:
            best_l1 = validation["l1"]
            torch.save(
                {
                    "schema": "spec102-w1-checkpoint-v1",
                    "model": model.state_dict(),
                    "config": {"base_channels": 50, "parameters": parameters, "relief_scale": RELIEF_SCALE, "single_output": True},
                    "input_manifest": input_manifest,
                    "m0_checkpoint_sha256": sha256_file(args.m0_checkpoint),
                    "h0_checkpoint_sha256": sha256_file(args.h0_checkpoint),
                    "epoch": epoch,
                    "validation": validation,
                },
                args.output_dir / "checkpoint_best.pt",
            )

    checkpoint = torch.load(args.output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, loaders["test_era"], device)
    baseline = history[0]["validation"]["h0_plane_l1"]
    report = {
        "schema": "spec102-w1-training-report-v1",
        "parameters": parameters,
        "epochs": args.epochs,
        "history": history,
        "best_validation_l1": best_l1,
        "validation_h0_plane_l1": baseline,
        "required_20_percent_improvement_l1": baseline * 0.8,
        "test_era": test_metrics,
        "gate_passed": best_l1 <= baseline * 0.8,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
