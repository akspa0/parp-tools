from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "ml-training" / "v10_stage1"
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 80
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 1337
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREVIEW_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bounded v10 Stage 1 minimap-to-height_17 model from NPZ shards.")
    parser.add_argument("input", help="NPZ shard, directory of NPZ shards, or JSON manifest containing shard paths.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--preview-count", type=int, default=DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional hard cap after discovery.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_npz_paths(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".npz":
        return [input_path]

    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.npz") if path.is_file())

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        collected: list[Path] = []
        collect_json_npz_paths(payload, input_path.parent, collected)
        return sorted({path.resolve() for path in collected if path.exists()})

    raise FileNotFoundError(f"Could not resolve NPZ input from {input_path}")


def collect_json_npz_paths(value: Any, base_dir: Path, collected: list[Path]) -> None:
    if isinstance(value, str) and value.lower().endswith(".npz"):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        collected.append(candidate)
        return

    if isinstance(value, dict):
        for nested in value.values():
            collect_json_npz_paths(nested, base_dir, collected)
        return

    if isinstance(value, list):
        for nested in value:
            collect_json_npz_paths(nested, base_dir, collected)


def load_metadata(npz_file: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata.json" not in npz_file.files:
        return {}
    raw = npz_file["metadata.json"]
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.ndim == 1:
            raw = b"".join(raw.tolist())
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError("Unsupported metadata payload in NPZ shard")


@dataclass(slots=True)
class Stage1Sample:
    path: Path
    tile_name: str
    source_tag: str
    minimap_rgb: np.ndarray
    height_17: np.ndarray
    min_height: float
    max_height: float


def discover_samples(npz_paths: Iterable[Path], max_samples: int) -> list[Stage1Sample]:
    samples: list[Stage1Sample] = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as shard:
            if "minimap_rgb_256" not in shard.files or "height_17" not in shard.files:
                continue

            metadata = load_metadata(shard)
            tile_name = str(metadata.get("tile_name") or path.stem)
            source_tag = str(metadata.get("minimap_source_tag") or "unknown")
            minimap_rgb = np.asarray(shard["minimap_rgb_256"], dtype=np.uint8)
            height_17 = np.asarray(shard["height_17"], dtype=np.float32)
            if minimap_rgb.shape != (256, 256, 3) or height_17.shape != (17, 17):
                continue

            samples.append(
                Stage1Sample(
                    path=path,
                    tile_name=tile_name,
                    source_tag=source_tag,
                    minimap_rgb=minimap_rgb,
                    height_17=height_17,
                    min_height=float(np.min(height_17)),
                    max_height=float(np.max(height_17)),
                )
            )

        if max_samples > 0 and len(samples) >= max_samples:
            return samples[:max_samples]

    return samples


class Stage1Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[Stage1Sample], tag_to_index: dict[str, int], height_mean: float, height_std: float):
        self.samples = samples
        self.tag_to_index = tag_to_index
        self.height_mean = float(height_mean)
        self.height_std = float(height_std)
        self.tag_count = len(tag_to_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        minimap = torch.from_numpy(sample.minimap_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        tag_planes = torch.zeros((self.tag_count, 256, 256), dtype=torch.float32)
        tag_planes[self.tag_to_index[sample.source_tag]].fill_(1.0)
        inputs = torch.cat([minimap, tag_planes], dim=0)

        height = torch.from_numpy(((sample.height_17 - self.height_mean) / self.height_std).astype(np.float32)).unsqueeze(0)
        range_target = torch.tensor(
            [
                (sample.min_height - self.height_mean) / self.height_std,
                (sample.max_height - self.height_mean) / self.height_std,
            ],
            dtype=torch.float32,
        )

        return {
            "inputs": inputs,
            "height": height,
            "range": range_target,
            "minimap": minimap,
            "tile_name": sample.tile_name,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(inputs) + self.skip(inputs))


class Stage1MinimapModel(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ConvBlock(32, 32),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 96, stride=2),
            ConvBlock(96, 128, stride=2),
            ConvBlock(128, 160, stride=2),
        )
        self.height_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((17, 17)),
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 1, kernel_size=1),
        )
        self.range_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(self.stem(inputs))
        return self.height_head(features), self.range_head(features)


def split_samples(samples: list[Stage1Sample], val_fraction: float, seed: int) -> tuple[list[Stage1Sample], list[Stage1Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError("Need at least two valid NPZ shards to create train and validation splits.")

    val_count = max(1, min(len(shuffled) - 1, int(math.ceil(len(shuffled) * val_fraction))))
    return shuffled[val_count:], shuffled[:val_count]


def make_loader(dataset: Stage1Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def maybe_channels_last(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def run_epoch(
    model: Stage1MinimapModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    height_mean: float,
    height_std: float,
    channels_last: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_batches = 0
    autocast_enabled = device.type == "cuda"

    for batch in loader:
        inputs = maybe_channels_last(batch["inputs"].to(device, non_blocking=True), channels_last)
        target_height = batch["height"].to(device, non_blocking=True)
        target_range = batch["range"].to(device, non_blocking=True)

        context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled) if autocast_enabled else torch.no_grad() if False else torch.enable_grad()
        with context:
            predicted_height, predicted_range = model(inputs)
            height_loss = F.l1_loss(predicted_height, target_height)
            gradient_loss = F.l1_loss(predicted_height[:, :, 1:, :] - predicted_height[:, :, :-1, :], target_height[:, :, 1:, :] - target_height[:, :, :-1, :])
            gradient_loss = gradient_loss + F.l1_loss(predicted_height[:, :, :, 1:] - predicted_height[:, :, :, :-1], target_height[:, :, :, 1:] - target_height[:, :, :, :-1])
            range_loss = F.l1_loss(predicted_range, target_range)
            loss = height_loss + 0.15 * gradient_loss + 0.05 * range_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        predicted_height_m = predicted_height.detach() * height_std + height_mean
        target_height_m = target_height.detach() * height_std + height_mean
        diff = predicted_height_m - target_height_m

        total_loss += float(loss.detach().cpu())
        total_mae += float(diff.abs().mean().cpu())
        total_rmse += float(torch.sqrt(torch.mean(diff.square())).cpu())
        total_batches += 1

    return {
        "loss": total_loss / max(1, total_batches),
        "mae_m": total_mae / max(1, total_batches),
        "rmse_m": total_rmse / max(1, total_batches),
    }


def save_preview(
    model: Stage1MinimapModel,
    dataset: Stage1Dataset,
    device: torch.device,
    output_path: Path,
    height_mean: float,
    height_std: float,
    channels_last: bool,
) -> None:
    if len(dataset) == 0:
        return

    sample = dataset[0]
    with torch.no_grad():
        inputs = maybe_channels_last(sample["inputs"].unsqueeze(0).to(device), channels_last)
        predicted_height, _ = model(inputs)

    minimap = (sample["minimap"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    target_height = (sample["height"].squeeze(0).numpy() * height_std) + height_mean
    predicted_height = (predicted_height.squeeze(0).squeeze(0).cpu().numpy() * height_std) + height_mean
    difference = predicted_height - target_height

    target_image = height_to_rgb(target_height)
    predicted_image = height_to_rgb(predicted_height)
    difference_image = difference_to_rgb(difference)
    target_image = resize_preview_image(target_image, minimap.shape[1], minimap.shape[0])
    predicted_image = resize_preview_image(predicted_image, minimap.shape[1], minimap.shape[0])
    difference_image = resize_preview_image(difference_image, minimap.shape[1], minimap.shape[0])

    composite = np.concatenate([minimap, target_image, predicted_image, difference_image], axis=1)
    Image.fromarray(composite).save(output_path)


def height_to_rgb(height: np.ndarray) -> np.ndarray:
    min_value = float(np.min(height))
    max_value = float(np.max(height))
    scale = max(1e-5, max_value - min_value)
    normalized = ((height - min_value) / scale * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([normalized, normalized, normalized], axis=-1)


def difference_to_rgb(difference: np.ndarray) -> np.ndarray:
    max_abs = max(1e-5, float(np.max(np.abs(difference))))
    normalized = (difference / max_abs).clip(-1.0, 1.0)
    red = np.where(normalized > 0.0, normalized, 0.0)
    blue = np.where(normalized < 0.0, -normalized, 0.0)
    green = 1.0 - np.abs(normalized)
    return np.stack([
        (red * 255.0).astype(np.uint8),
        (green * 255.0).astype(np.uint8),
        (blue * 255.0).astype(np.uint8),
    ], axis=-1)


def resize_preview_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return np.asarray(Image.fromarray(image).resize((width, height), resample=Image.Resampling.NEAREST))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    previews_dir = output_dir / "previews"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = find_npz_paths(input_path)
    samples = discover_samples(npz_paths, args.max_samples)
    if len(samples) < 2:
        raise RuntimeError("Need at least two v10 NPZ shards containing minimap_rgb_256 and height_17.")

    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    height_values = np.concatenate([sample.height_17.reshape(-1) for sample in train_samples], axis=0)
    height_mean = float(np.mean(height_values))
    height_std = float(np.std(height_values))
    if height_std < 1e-5:
        height_std = 1.0

    tag_names = sorted({sample.source_tag for sample in samples})
    tag_to_index = {tag: index for index, tag in enumerate(tag_names)}

    train_dataset = Stage1Dataset(train_samples, tag_to_index, height_mean, height_std)
    val_dataset = Stage1Dataset(val_samples, tag_to_index, height_mean, height_std)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = Stage1MinimapModel(input_channels=3 + len(tag_to_index)).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    use_compile = bool(args.use_compile and hasattr(torch, "compile") and device.type == "cuda")
    if args.use_compile and not use_compile:
        print("torch.compile disabled for this run because the selected device is not CUDA.")

    if use_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, height_mean, height_std, args.channels_last)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, None, device, height_mean, height_std, args.channels_last)

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} | val loss {val_metrics['loss']:.4f} | "
            f"val mae {val_metrics['mae_m']:.2f}m | val rmse {val_metrics['rmse_m']:.2f}m"
        )

        last_checkpoint = checkpoints_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "height_mean": height_mean,
                "height_std": height_std,
                "source_tags": tag_names,
                "history": history,
            },
            last_checkpoint,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "height_mean": height_mean,
                    "height_std": height_std,
                    "source_tags": tag_names,
                    "history": history,
                },
                checkpoints_dir / "best.pt",
            )
            save_preview(
                model,
                val_dataset,
                device,
                previews_dir / f"epoch_{epoch:03d}_best.png",
                height_mean,
                height_std,
                args.channels_last,
            )

    summary = {
        "input": str(input_path),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "source_tags": tag_names,
        "height_mean": height_mean,
        "height_std": height_std,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()