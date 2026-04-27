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
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "ml-training" / "v10_stage2"
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 60
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 1337
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREVIEW_COUNT = 4
DEFAULT_SIGNAL_DROOUT = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bounded v10 Stage 2 terrain synth model from NPZ shards.")
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
    parser.add_argument("--signal-dropout", type=float, default=DEFAULT_SIGNAL_DROOUT,
                        help="Probability of zeroing out an optional signal channel during training.")
    parser.add_argument("--stage1-checkpoint", help="Optional Stage 1 checkpoint to use for coarse prior at inference time.")
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


def load_optional_array(npz_file: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in npz_file.files:
        for alias in SIGNAL_ALIASES.get(key, ()):
            if alias in npz_file.files:
                return np.asarray(npz_file[alias])
        return None
    return np.asarray(npz_file[key])


@dataclass(slots=True)
class SignalSpec:
    key: str
    channels: int
    target_size: int
    dtype: np.dtype


# Ordered list of optional signals that augment the mandatory minimap + coarse prior
OPTIONAL_SIGNALS: list[SignalSpec] = [
    SignalSpec("mcal_alpha_pack_256", 4, 256, np.float32),
    SignalSpec("mccv_rgb", 3, 257, np.float32),
    SignalSpec("mcnr_normal_xyz", 3, 257, np.float32),
    SignalSpec("mh2o_surface_height", 1, 257, np.float32),
    SignalSpec("mh2o_depth", 1, 257, np.float32),
    SignalSpec("object_mask_257", 1, 257, np.float32),
    SignalSpec("object_precise_mask_257", 1, 257, np.float32),
    SignalSpec("pm4_path_mask", 1, 257, np.float32),
    SignalSpec("pm4_building_footprint_mask", 1, 257, np.float32),
    SignalSpec("hole_mask_16", 1, 16, np.uint8),
    SignalSpec("wl_liquid_mask", 1, 257, np.float32),
    SignalSpec("wl_liquid_height", 1, 257, np.float32),
    SignalSpec("mclq_surface_height", 1, 129, np.float32),
    SignalSpec("mtxf_animated_mask", 1, 16, np.int32),
]


SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {
    # Legacy v9 cache names. These keep the v10 trainer usable before every
    # archive-backed client has been regenerated through native v10 extraction.
    "hole_mask_16": ("hole_mask_16x16",),
    "object_precise_mask_257": ("object_mask_precise_257",),
    "pm4_path_mask": ("pm4_mask_257",),
    "wl_liquid_mask": ("liquid_mask_257",),
    "wl_liquid_height": ("liquid_height_257",),
}


@dataclass(slots=True)
class Stage2Sample:
    path: Path
    tile_name: str
    minimap_rgb: np.ndarray
    height_257: np.ndarray
    height_65: np.ndarray
    height_17: np.ndarray
    signals: dict[str, np.ndarray]
    available_signal_keys: set[str]
    min_height: float
    max_height: float


def discover_samples(npz_paths: Iterable[Path], max_samples: int) -> list[Stage2Sample]:
    samples: list[Stage2Sample] = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as shard:
            if "minimap_rgb_256" not in shard.files or "height_257" not in shard.files or "height_17" not in shard.files:
                continue

            minimap_rgb = np.asarray(shard["minimap_rgb_256"], dtype=np.uint8)
            height_257 = np.asarray(shard["height_257"], dtype=np.float32)
            height_17 = np.asarray(shard["height_17"], dtype=np.float32)
            height_65 = np.asarray(shard["height_65"], dtype=np.float32) if "height_65" in shard.files else None

            if minimap_rgb.shape != (256, 256, 3) or height_257.shape != (257, 257) or height_17.shape != (17, 17):
                continue

            metadata = load_metadata(shard)
            tile_name = str(metadata.get("tile_name") or path.stem)

            signals: dict[str, np.ndarray] = {}
            available: set[str] = set()
            for spec in OPTIONAL_SIGNALS:
                arr = load_optional_array(shard, spec.key)
                if arr is None:
                    continue
                # Normalize shape expectations
                if spec.key == "hole_mask_16" and arr.ndim == 2:
                    arr = arr.astype(np.float32)
                elif spec.key == "mtxf_animated_mask" and arr.ndim == 2:
                    arr = arr.astype(np.float32)
                elif spec.channels == 1 and arr.ndim == 2:
                    arr = arr[np.newaxis, ...].astype(np.float32)
                elif spec.channels > 1 and arr.ndim == 3 and arr.shape[2] == spec.channels:
                    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
                signals[spec.key] = arr
                available.add(spec.key)

            if height_65 is None:
                height_65 = downsample_heightmap(height_257, 65)

            samples.append(
                Stage2Sample(
                    path=path,
                    tile_name=tile_name,
                    minimap_rgb=minimap_rgb,
                    height_257=height_257,
                    height_65=height_65,
                    height_17=height_17,
                    signals=signals,
                    available_signal_keys=available,
                    min_height=float(np.min(height_257)),
                    max_height=float(np.max(height_257)),
                )
            )

        if max_samples > 0 and len(samples) >= max_samples:
            return samples[:max_samples]

    return samples


def downsample_heightmap(source: np.ndarray, target_size: int) -> np.ndarray:
    """Bilinear downsample a 2D heightmap."""
    source_size = source.shape[0]
    result = np.empty((target_size, target_size), dtype=np.float32)
    scale = (source_size - 1) / (target_size - 1)
    for y in range(target_size):
        for x in range(target_size):
            sx = x * scale
            sy = y * scale
            ix = min(int(sx), source_size - 2)
            iy = min(int(sy), source_size - 2)
            fx = sx - ix
            fy = sy - iy
            v00 = source[iy, ix]
            v10 = source[iy, ix + 1]
            v01 = source[iy + 1, ix]
            v11 = source[iy + 1, ix + 1]
            top = v00 + (v10 - v00) * fx
            bottom = v01 + (v11 - v01) * fx
            result[y, x] = top + (bottom - top) * fy
    return result


class Stage2Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[Stage2Sample],
        height_mean: float,
        height_std: float,
        signal_dropout: float,
    ):
        self.samples = samples
        self.height_mean = float(height_mean)
        self.height_std = float(height_std)
        self.signal_dropout = signal_dropout

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_dropout(self, tensor: torch.Tensor, key: str, available: set[str]) -> torch.Tensor:
        if key not in available:
            return tensor
        if self.signal_dropout > 0 and random.random() < self.signal_dropout:
            return torch.zeros_like(tensor)
        return tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        minimap = torch.from_numpy(sample.minimap_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        # Build optional signal planes at 256x256
        signal_planes: list[torch.Tensor] = []

        # MCAL alpha pack (4 channels, 256)
        if "mcal_alpha_pack_256" in sample.signals:
            alpha = torch.from_numpy(sample.signals["mcal_alpha_pack_256"].astype(np.float32))
            if alpha.shape != (4, 256, 256):
                alpha = torch.zeros((4, 256, 256), dtype=torch.float32)
            signal_planes.append(self._maybe_dropout(alpha, "mcal_alpha_pack_256", sample.available_signal_keys))
        else:
            signal_planes.append(torch.zeros((4, 256, 256), dtype=torch.float32))

        # 257-res signals → interpolate to 256
        for key, channels in [
            ("mccv_rgb", 3),
            ("mcnr_normal_xyz", 3),
            ("mh2o_surface_height", 1),
            ("mh2o_depth", 1),
            ("object_mask_257", 1),
            ("object_precise_mask_257", 1),
            ("pm4_path_mask", 1),
            ("pm4_building_footprint_mask", 1),
            ("wl_liquid_mask", 1),
            ("wl_liquid_height", 1),
            ("mclq_surface_height", 1),
        ]:
            if key in sample.signals:
                arr = sample.signals[key]
                t = torch.from_numpy(arr.astype(np.float32))
                if t.ndim == 2:
                    t = t.unsqueeze(0)
                # Interpolate 257 → 256 or 129 → 256
                if t.shape[-1] != 256 or t.shape[-2] != 256:
                    t = F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)
                signal_planes.append(self._maybe_dropout(t, key, sample.available_signal_keys))
            else:
                signal_planes.append(torch.zeros((channels, 256, 256), dtype=torch.float32))

        # Hole mask 16 → upsample to 256
        if "hole_mask_16" in sample.signals:
            hole = torch.from_numpy(sample.signals["hole_mask_16"].astype(np.float32))
            if hole.ndim == 2:
                hole = hole.unsqueeze(0)
            if hole.shape[-1] != 256:
                hole = F.interpolate(hole.unsqueeze(0), size=(256, 256), mode="nearest").squeeze(0)
            signal_planes.append(self._maybe_dropout(hole, "hole_mask_16", sample.available_signal_keys))
        else:
            signal_planes.append(torch.zeros((1, 256, 256), dtype=torch.float32))

        # MTXF animated mask 16 → upsample to 256
        if "mtxf_animated_mask" in sample.signals:
            mtxf = torch.from_numpy(sample.signals["mtxf_animated_mask"].astype(np.float32))
            if mtxf.ndim == 2:
                mtxf = mtxf.unsqueeze(0)
            if mtxf.shape[-1] != 256:
                mtxf = F.interpolate(mtxf.unsqueeze(0), size=(256, 256), mode="nearest").squeeze(0)
            signal_planes.append(self._maybe_dropout(mtxf, "mtxf_animated_mask", sample.available_signal_keys))
        else:
            signal_planes.append(torch.zeros((1, 256, 256), dtype=torch.float32))

        # Coarse prior: height_17 upsampled to 256
        coarse_prior = torch.from_numpy(((sample.height_17 - self.height_mean) / self.height_std).astype(np.float32))
        coarse_prior = F.interpolate(coarse_prior.unsqueeze(0).unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        signal_planes.append(coarse_prior.unsqueeze(0))

        inputs = torch.cat([minimap] + signal_planes, dim=0)

        # Targets
        height_257 = torch.from_numpy(((sample.height_257 - self.height_mean) / self.height_std).astype(np.float32)).unsqueeze(0)
        height_65 = torch.from_numpy(((sample.height_65 - self.height_mean) / self.height_std).astype(np.float32)).unsqueeze(0)
        height_17 = torch.from_numpy(((sample.height_17 - self.height_mean) / self.height_std).astype(np.float32)).unsqueeze(0)

        return {
            "inputs": inputs,
            "height_257": height_257,
            "height_65": height_65,
            "height_17": height_17,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + self.skip(x))


class Stage2TerrainSynthModel(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.enc1 = ConvBlock(32, 32)             # 256
        self.enc2 = ConvBlock(32, 64, stride=2)   # 128
        self.enc3 = ConvBlock(64, 96, stride=2)   # 64
        self.enc4 = ConvBlock(96, 128, stride=2)  # 32
        self.enc5 = ConvBlock(128, 160, stride=2) # 16

        # Coarse head: 17x17
        self.coarse_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((17, 17)),
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # Mid head: 65x65
        self.mid_up = nn.Upsample(size=(65, 65), mode="bilinear", align_corners=False)
        self.mid_head = nn.Sequential(
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 1, kernel_size=1),
        )

        # Fine head: 257x257 via progressive upsampling
        self.fine_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(160, 128),
        )
        self.fine_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 96),
        )
        self.fine_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(96, 64),
        )
        self.fine_up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
        )
        self.fine_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.enc1(x0)  # 256
        x2 = self.enc2(x1)  # 128
        x3 = self.enc3(x2)  # 64
        x4 = self.enc4(x3)  # 32
        x5 = self.enc5(x4)  # 16

        coarse = self.coarse_head(x5)
        mid = self.mid_head(self.mid_up(x5))

        f = self.fine_up1(x5)   # 32
        f = self.fine_up2(f)    # 64
        f = self.fine_up3(f)    # 128
        f = self.fine_up4(f)    # 256
        fine = self.fine_head(f)
        fine = F.interpolate(fine, size=(257, 257), mode="bilinear", align_corners=False)

        return coarse, mid, fine


def split_samples(samples: list[Stage2Sample], val_fraction: float, seed: int) -> tuple[list[Stage2Sample], list[Stage2Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError("Need at least two valid NPZ shards to create train and validation splits.")

    val_count = max(1, min(len(shuffled) - 1, int(math.ceil(len(shuffled) * val_fraction))))
    return shuffled[val_count:], shuffled[:val_count]


def make_loader(dataset: Stage2Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
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
    model: Stage2TerrainSynthModel,
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
    total_full_l1 = 0.0
    total_mid_l1 = 0.0
    total_coarse_l1 = 0.0
    total_gradient = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_batches = 0
    autocast_enabled = device.type == "cuda"

    for batch in loader:
        inputs = maybe_channels_last(batch["inputs"].to(device, non_blocking=True), channels_last)
        target_257 = batch["height_257"].to(device, non_blocking=True)
        target_65 = batch["height_65"].to(device, non_blocking=True)
        target_17 = batch["height_17"].to(device, non_blocking=True)

        context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled) if autocast_enabled else torch.enable_grad() if training else torch.no_grad()
        with context:
            pred_17, pred_65, pred_257 = model(inputs)

            full_l1 = F.l1_loss(pred_257, target_257)
            mid_l1 = F.l1_loss(pred_65, target_65)
            coarse_l1 = F.l1_loss(pred_17, target_17)

            # Gradient loss on 257
            grad_pred_x = pred_257[:, :, :, 1:] - pred_257[:, :, :, :-1]
            grad_pred_y = pred_257[:, :, 1:, :] - pred_257[:, :, :-1, :]
            grad_target_x = target_257[:, :, :, 1:] - target_257[:, :, :, :-1]
            grad_target_y = target_257[:, :, 1:, :] - target_257[:, :, :-1, :]
            gradient = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

            # Residual losses
            pred_65_up = F.interpolate(pred_17, size=(65, 65), mode="bilinear", align_corners=False)
            target_65_up = F.interpolate(target_17, size=(65, 65), mode="bilinear", align_corners=False)
            mid_residual = F.l1_loss(pred_65 - pred_65_up, target_65 - target_65_up)

            pred_257_up = F.interpolate(pred_65, size=(257, 257), mode="bilinear", align_corners=False)
            target_257_up = F.interpolate(target_65, size=(257, 257), mode="bilinear", align_corners=False)
            detail_res = F.l1_loss(pred_257 - pred_257_up, target_257 - target_257_up)

            loss = (
                full_l1
                + 0.5 * mid_l1
                + 0.25 * coarse_l1
                + 0.3 * gradient
                + 0.3 * mid_residual
                + 0.3 * detail_res
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        pred_height_m = pred_257.detach() * height_std + height_mean
        target_height_m = target_257.detach() * height_std + height_mean
        diff = pred_height_m - target_height_m

        total_loss += float(loss.detach().cpu())
        total_full_l1 += float(full_l1.detach().cpu())
        total_mid_l1 += float(mid_l1.detach().cpu())
        total_coarse_l1 += float(coarse_l1.detach().cpu())
        total_gradient += float(gradient.detach().cpu())
        total_mae += float(diff.abs().mean().cpu())
        total_rmse += float(torch.sqrt(torch.mean(diff.square())).cpu())
        total_batches += 1

    return {
        "loss": total_loss / max(1, total_batches),
        "full_l1": total_full_l1 / max(1, total_batches),
        "mid_l1": total_mid_l1 / max(1, total_batches),
        "coarse_l1": total_coarse_l1 / max(1, total_batches),
        "gradient": total_gradient / max(1, total_batches),
        "mae_m": total_mae / max(1, total_batches),
        "rmse_m": total_rmse / max(1, total_batches),
    }


def save_preview(
    model: Stage2TerrainSynthModel,
    dataset: Stage2Dataset,
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
        pred_17, pred_65, pred_257 = model(inputs)

    minimap = (sample["minimap"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    target_257 = (sample["height_257"].squeeze(0).numpy() * height_std) + height_mean
    pred_257 = (pred_257.squeeze(0).squeeze(0).cpu().numpy() * height_std) + height_mean
    diff = pred_257 - target_257

    target_img = height_to_rgb(target_257)
    pred_img = height_to_rgb(pred_257)
    diff_img = difference_to_rgb(diff)

    # Resize all to minimap size for composite
    h, w = minimap.shape[:2]
    target_img = resize_preview_image(target_img, w, h)
    pred_img = resize_preview_image(pred_img, w, h)
    diff_img = resize_preview_image(diff_img, w, h)

    composite = np.concatenate([minimap, target_img, pred_img, diff_img], axis=1)
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
        raise RuntimeError("Need at least two v10 NPZ shards containing minimap_rgb_256, height_257, and height_17.")

    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    height_values = np.concatenate([sample.height_257.reshape(-1) for sample in train_samples], axis=0)
    height_mean = float(np.mean(height_values))
    height_std = float(np.std(height_values))
    if height_std < 1e-5:
        height_std = 1.0

    train_dataset = Stage2Dataset(train_samples, height_mean, height_std, args.signal_dropout)
    val_dataset = Stage2Dataset(val_samples, height_mean, height_std, signal_dropout=0.0)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Compute input channels from a sample
    sample_channels = train_dataset[0]["inputs"].shape[0]

    device = torch.device(args.device)
    model = Stage2TerrainSynthModel(input_channels=sample_channels).to(device)
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
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val mae {val_metrics['mae_m']:.2f}m | "
            f"val rmse {val_metrics['rmse_m']:.2f}m | "
            f"val full_l1 {val_metrics['full_l1']:.4f} | "
            f"val gradient {val_metrics['gradient']:.4f}"
        )

        last_checkpoint = checkpoints_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "height_mean": height_mean,
                "height_std": height_std,
                "input_channels": sample_channels,
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
                    "input_channels": sample_channels,
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
        "input_channels": sample_channels,
        "height_mean": height_mean,
        "height_std": height_std,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
