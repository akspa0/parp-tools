"""Train Spec 102 H0 only: RGB minimap -> one tile-offset residual."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harvester.v24.train_common import RunLogger, configure_perf, peak_vram_gb, set_determinism
from harvester.v25.h0_offset import H0OffsetModel, OFFSET_SCALE, parameter_count


def validate_run_contract(epochs: int, device: str) -> None:
    if epochs < 1 or epochs > 3:
        raise ValueError("H0 decision runs are hard-capped at 3 epochs")
    if device != "cuda":
        raise ValueError("H0 is CUDA-only; CPU fallback is prohibited")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")


def validate_model_inputs(model: torch.nn.Module) -> dict:
    inputs = list(inspect.signature(model.forward).parameters)
    if inputs != ["minimap_rgb"]:
        raise RuntimeError(f"H0 input leakage: expected ['minimap_rgb'], got {inputs}")
    return {
        "stage": "H0",
        "deployment_inputs": ["minimap_rgb"],
        "output_signal": "tile_offset_residual",
        "target_only": ["height_257", "liquid_mask_256"],
        "prohibited_inputs": ["wdl_height_33", "height_257", "normal_xyz_257", "alpha_256"],
    }


def liquid_vertex_mask(liquid: np.ndarray) -> np.ndarray:
    cells = np.asarray(liquid) > 127
    covered = np.zeros((cells.shape[0], 257, 257), dtype=bool)
    covered[:, :-1, :-1] |= cells
    covered[:, 1:, :-1] |= cells
    covered[:, :-1, 1:] |= cells
    covered[:, 1:, 1:] |= cells
    return ~covered


def preload_samples(
    store: Path, split_by_row: dict[int, str], rgb_slope: float, rgb_intercept: float,
    batch: int = 64,
):
    group = zarr.open_group(str(store), mode="r")
    rgb_array, height_array, liquid_array = (
        group["minimap_rgb"], group["height_257"], group["liquid_mask_256"]
    )
    samples = {name: {"rgb": [], "target": []} for name in set(split_by_row.values())}
    count = len(split_by_row)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        rgb = np.asarray(rgb_array[start:stop])
        heights = np.asarray(height_array[start:stop], dtype=np.float32)
        valid = liquid_vertex_mask(np.asarray(liquid_array[start:stop])) & np.isfinite(heights)
        rgb64 = rgb.reshape(stop - start, 64, 4, 64, 4, 3).mean(axis=(2, 4))
        for offset, row in enumerate(range(start, stop)):
            split = split_by_row[row]
            rgb_input = np.moveaxis(rgb64[offset], -1, 0).astype(np.float32) / 255.0
            rgb_mean = float(rgb_input.mean())
            rgb_flat = rgb_slope * rgb_mean + rgb_intercept
            samples[split]["rgb"].append(rgb_input)
            samples[split]["target"].append(float(heights[offset][valid[offset]].mean() - rgb_flat))
    return {
        name: (
            torch.from_numpy(np.stack(values["rgb"])),
            torch.tensor(values["target"], dtype=torch.float32),
        )
        for name, values in samples.items()
    }


def make_loader(tensors, batch_size: int, shuffle: bool):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=0,
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for rgb, target in loader:
            rgb = rgb.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(rgb)
                loss = torch.nn.functional.mse_loss(
                    prediction.float() / OFFSET_SCALE, target / OFFSET_SCALE
                )
            if training:
                loss.backward()
                optimizer.step()
            total += float(torch.abs(prediction.float() - target).sum().item())
            count += int(target.numel())
    return total / max(count, 1)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Spec 102 H0 offset residual")
    parser.add_argument("--v25-store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=102)
    args = parser.parse_args()
    validate_run_contract(args.epochs, args.device)

    set_determinism(args.seed, strict=False)
    configure_perf(fast=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    global_mean = float(baseline["fit"]["train_global_mean"])
    rgb_slope = float(baseline["fit"]["rgb_flat_slope"])
    rgb_intercept = float(baseline["fit"]["rgb_flat_intercept"])
    split_by_row = {int(record["row"]): str(record["split"]) for record in manifest["rows"]}

    started = time.time()
    tensors = preload_samples(args.v25_store, split_by_row, rgb_slope, rgb_intercept)
    train_loader = make_loader(tensors["train"], args.batch_size, True)
    val_loader = make_loader(tensors["validation_map"], args.batch_size, False)
    era_loader = make_loader(tensors["test_era"], args.batch_size, False)

    model = H0OffsetModel().to(device)
    input_manifest = validate_model_inputs(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger = RunLogger(args.output_dir)
    logger.write_json("input_manifest.json", input_manifest)
    config = {
        "stage": "H0", "output_signal": "tile_offset_residual", "inputs": ["minimap_rgb"],
        "target_only": ["height_257", "liquid_mask_256"],
        "residual_baseline": "rgb_flat", "global_mean": global_mean,
        "rgb_flat_slope": rgb_slope, "rgb_flat_intercept": rgb_intercept,
        "offset_scale": OFFSET_SCALE,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "device": "cuda", "parameters": parameter_count(model), "git_revision": git_revision(),
        "split_manifest": str(args.split_manifest.resolve()), "baseline_report": str(args.baseline_report.resolve()),
        "preload_seconds": round(time.time() - started, 3),
    }
    logger.write_json("config.json", config)

    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_mae = run_epoch(model, train_loader, device, optimizer)
        val_mae = run_epoch(model, val_loader, device)
        logger.log_epoch(
            epoch, train_offset_mae=train_mae, val_offset_mae=val_mae,
            epoch_seconds=round(time.time() - epoch_start, 3),
            peak_vram_gb=round(peak_vram_gb() or 0.0, 3),
        )
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_offset_mae": val_mae}
        torch.save(checkpoint, args.output_dir / "checkpoint_last.pt")
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            torch.save(checkpoint, args.output_dir / "checkpoint_best.pt")

    era_mae = run_epoch(model, era_loader, device)
    deployable = baseline["splits"]["validation_map"]["tile_mean_mae"]
    baseline_name, baseline_mae = min(deployable.items(), key=lambda item: item[1])
    threshold = 0.8 * float(baseline_mae)
    report = {
        "stage": "H0", "best_epoch": best_epoch, "best_val_offset_mae": best_val,
        "era_test_offset_mae": era_mae, "baseline_name": baseline_name,
        "baseline_offset_mae": baseline_mae, "required_mae": threshold,
        "gate_pass": best_val <= threshold, "peak_vram_gb": peak_vram_gb(),
    }
    logger.write_json("report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
