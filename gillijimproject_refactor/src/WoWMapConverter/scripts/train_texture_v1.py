#!/usr/bin/env python3
"""
WoW Texture Decomposer V1 - separate minimap-to-texture-layer model.

This model is intentionally separate from V7 terrain reconstruction.

Inputs:
- minimap RGB

Targets:
- three overlay alpha masks at full tile resolution
- chunk-slot texture classes for up to four terrain layers on the 16x16 chunk grid

The current target is the relatively limited terrain-texture palette available in
the 4.0.0.11927 development-family data, while still allowing mixed real-client
corpora when the dataset roots are provided explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from train_v7 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLUR_SIGMA,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_SPATIAL_GROUP_SIZE,
    DEFAULT_DATASET_SEARCH_ROOTS,
    DEFAULT_VAL_FRACTION,
    INPUT_SIZE,
    OUTPUT_SIZE,
    collect_explicit_roots,
    discover_profile_roots,
    parse_tile_identity,
    split_grouped_indices,
)


CHUNK_GRID_SIZE = 16
MAX_TEXTURE_LAYERS = 4
OVERLAY_ALPHA_LAYERS = 3
IGNORE_INDEX = -100
DEFAULT_OUTPUT_DIR = Path("./texture_output_v1")
DEFAULT_MAX_PALETTE = 96
DEFAULT_MIN_TEXTURE_FREQUENCY = 1

TEXTURE_PROFILE_PRESETS = {
    "manual": {
        "description": "Use only explicit --dataset-root values.",
        "include_maps": [],
        "discover": [],
    },
    "development-textures": {
        "description": "Prioritize 4.0.0.11927 LostIsles/Azeroth/Kalimdor roots for limited-palette texture decomposition.",
        "include_maps": ["LostIsles", "Azeroth", "Kalimdor", "development"],
        "discover": [
            {
                "label": "cata-textures",
                "map_tokens": ["lostisles", "lost_isles", "azeroth", "kalimdor", "development"],
                "build_tokens": ["400", "4.0.0", "11927", "cata", "cataclysm"],
            },
        ],
    },
}


@dataclass(frozen=True)
class TextureTileSample:
    dataset_root: Path
    dataset_name: str
    json_path: Path
    tile_name: str
    map_name: str
    tile_x: int
    tile_y: int
    minimap_path: Path
    alpha_mask_paths: Tuple[Path, ...]
    texture_paths_by_chunk: Tuple[Tuple[Optional[str], ...], ...]


def normalize_texture_path(texture_path: Optional[str]) -> Optional[str]:
    if not texture_path:
        return None
    return texture_path.replace("\\", "/").lower()


def resolve_texture_dataset_roots(args: argparse.Namespace) -> List[Path]:
    explicit = collect_explicit_roots(args.dataset_root)
    if explicit:
        return explicit

    if args.profile == "manual":
        raise SystemExit("No dataset roots provided. Use --dataset-root or choose a non-manual --profile.")

    discovered = discover_profile_roots(args.profile, args.search_root)
    if discovered:
        return discovered

    profile = TEXTURE_PROFILE_PRESETS[args.profile]
    raise SystemExit(
        "No dataset roots resolved for profile "
        f"'{args.profile}'. Expected roots containing the era/map markers for: {profile['description']}. "
        "Pass explicit --dataset-root paths instead."
    )


def chunk_texture_slots(chunk_layers: Sequence[Dict[str, object]]) -> Tuple[Tuple[Optional[str], ...], ...]:
    slots: List[List[Optional[str]]] = [[None for _ in range(MAX_TEXTURE_LAYERS)] for _ in range(CHUNK_GRID_SIZE * CHUNK_GRID_SIZE)]

    for chunk in chunk_layers:
        chunk_index = int(chunk.get("idx", -1))
        if chunk_index < 0 or chunk_index >= len(slots):
            continue
        layers = chunk.get("layers") or []
        if not isinstance(layers, list):
            continue

        for layer_index in range(min(len(layers), MAX_TEXTURE_LAYERS)):
            layer = layers[layer_index]
            if not isinstance(layer, dict):
                continue
            slots[chunk_index][layer_index] = normalize_texture_path(layer.get("texture_path"))

    return tuple(tuple(entry) for entry in slots)


class TextureTileDataset(Dataset):
    def __init__(
        self,
        dataset_roots: Sequence[Path],
        include_maps: Sequence[str],
        exclude_maps: Sequence[str],
        palette: Optional[Sequence[str]] = None,
        augment: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self.include_maps = {value.lower() for value in include_maps if value}
        self.exclude_maps = {value.lower() for value in exclude_maps if value}
        self.augment = augment
        self.samples: List[TextureTileSample] = []
        self.palette_counter: Counter[str] = Counter()
        self.palette = list(palette or [])
        self.palette_index = {name: idx for idx, name in enumerate(self.palette)}

        self.to_tensor = transforms.ToTensor()
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=DEFAULT_BLUR_SIGMA)
        self.color_jitter = transforms.ColorJitter(0.15, 0.15, 0.15, 0.03)

        print("Loading texture dataset roots...")
        for dataset_root in dataset_roots:
            self.samples.extend(self._collect_root_samples(dataset_root, limit))

        if not self.palette:
            self.palette = []
        self.palette_index = {name: idx for idx, name in enumerate(self.palette)}

        print(f"Loaded {len(self.samples)} usable texture samples")

    def _collect_root_samples(self, dataset_root: Path, limit: Optional[int]) -> List[TextureTileSample]:
        dataset_dir = dataset_root / "dataset"
        if not dataset_dir.exists():
            print(f"Warning: dataset folder missing in {dataset_root}")
            return []

        collected: List[TextureTileSample] = []
        for json_path in sorted(dataset_dir.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Warning: failed to read {json_path}: {exc}")
                continue

            terrain = payload.get("terrain_data", {})
            tile_name = terrain.get("adt_tile") or json_path.stem

            try:
                map_name, tile_x, tile_y = parse_tile_identity(tile_name)
            except ValueError:
                continue

            map_key = map_name.lower()
            if self.include_maps and map_key not in self.include_maps:
                continue
            if map_key in self.exclude_maps:
                continue

            minimap_path = dataset_root / "images" / f"{tile_name}.png"
            chunk_layers = terrain.get("chunk_layers")
            alpha_masks = terrain.get("alpha_masks") or []
            if not minimap_path.exists() or not isinstance(chunk_layers, list) or len(chunk_layers) == 0:
                continue

            texture_slots = chunk_texture_slots(chunk_layers)
            for chunk in texture_slots:
                for texture_path in chunk:
                    if texture_path:
                        self.palette_counter[texture_path] += 1

            alpha_mask_paths = tuple(dataset_root / relative_path for relative_path in alpha_masks[:OVERLAY_ALPHA_LAYERS] if relative_path)
            collected.append(
                TextureTileSample(
                    dataset_root=dataset_root,
                    dataset_name=dataset_root.name,
                    json_path=json_path,
                    tile_name=tile_name,
                    map_name=map_name,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    minimap_path=minimap_path,
                    alpha_mask_paths=alpha_mask_paths,
                    texture_paths_by_chunk=texture_slots,
                )
            )

            if limit is not None and len(collected) >= limit:
                break

        print(f"  {dataset_root.name}: {len(collected)} usable texture samples")
        return collected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        minimap = Image.open(sample.minimap_path).convert("RGB").resize((INPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        minimap = self.blur(minimap)
        if self.augment:
            minimap = self.color_jitter(minimap)
        minimap_tensor = self.to_tensor(minimap)

        alpha_targets: List[torch.Tensor] = []
        for alpha_index in range(OVERLAY_ALPHA_LAYERS):
            if alpha_index < len(sample.alpha_mask_paths) and sample.alpha_mask_paths[alpha_index].exists():
                alpha_image = Image.open(sample.alpha_mask_paths[alpha_index]).convert("L").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
                alpha_targets.append(self.to_tensor(alpha_image))
            else:
                alpha_targets.append(torch.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), dtype=torch.float32))
        alpha_tensor = torch.cat(alpha_targets, dim=0)

        texture_targets = torch.full((MAX_TEXTURE_LAYERS, CHUNK_GRID_SIZE, CHUNK_GRID_SIZE), IGNORE_INDEX, dtype=torch.long)
        for chunk_index, chunk_slots in enumerate(sample.texture_paths_by_chunk):
            chunk_y = chunk_index // CHUNK_GRID_SIZE
            chunk_x = chunk_index % CHUNK_GRID_SIZE
            for layer_index, texture_path in enumerate(chunk_slots):
                if layer_index >= MAX_TEXTURE_LAYERS or not texture_path:
                    continue
                palette_index = self.palette_index.get(texture_path)
                if palette_index is None:
                    continue
                texture_targets[layer_index, chunk_y, chunk_x] = palette_index

        if self.augment and torch.rand(1).item() > 0.5:
            minimap_tensor = torch.flip(minimap_tensor, dims=[2])
            alpha_tensor = torch.flip(alpha_tensor, dims=[2])
            texture_targets = torch.flip(texture_targets, dims=[2])

        return {
            "input": minimap_tensor,
            "alpha_target": alpha_tensor,
            "texture_target": texture_targets,
        }


class TextureDecomposerNet(nn.Module):
    def __init__(self, palette_size: int, in_channels: int = 3) -> None:
        super().__init__()
        self.palette_size = palette_size

        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        self.bottleneck = self._conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)

        self.alpha_head = nn.Conv2d(32, OVERLAY_ALPHA_LAYERS, kernel_size=1)
        self.texture_head = nn.Conv2d(32, MAX_TEXTURE_LAYERS * palette_size, kernel_size=1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        alpha_masks = torch.sigmoid(self.alpha_head(dec1))
        chunk_features = F.adaptive_avg_pool2d(dec1, (CHUNK_GRID_SIZE, CHUNK_GRID_SIZE))
        texture_logits = self.texture_head(chunk_features)
        batch_size = texture_logits.shape[0]
        texture_logits = texture_logits.view(batch_size, MAX_TEXTURE_LAYERS, self.palette_size, CHUNK_GRID_SIZE, CHUNK_GRID_SIZE)
        return alpha_masks, texture_logits


def build_texture_palette(dataset: TextureTileDataset, max_palette: int, min_frequency: int) -> List[str]:
    ranked = [
        name
        for name, count in dataset.palette_counter.most_common()
        if count >= min_frequency
    ]
    return ranked[:max_palette]


def texture_loss_components(
    predicted_alpha: torch.Tensor,
    predicted_texture_logits: torch.Tensor,
    target_alpha: torch.Tensor,
    target_texture: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    alpha_loss = F.l1_loss(predicted_alpha, target_alpha)

    slot_losses: List[torch.Tensor] = []
    slot_metrics: Dict[str, float] = {}
    for slot_index in range(MAX_TEXTURE_LAYERS):
        slot_target = target_texture[:, slot_index]
        slot_logits = predicted_texture_logits[:, slot_index]
        valid_mask = slot_target != IGNORE_INDEX
        if valid_mask.any():
            slot_loss = F.cross_entropy(slot_logits, slot_target, ignore_index=IGNORE_INDEX)
            slot_losses.append(slot_loss)
            slot_metrics[f"texture_slot_{slot_index}"] = float(slot_loss.item())
        else:
            slot_metrics[f"texture_slot_{slot_index}"] = 0.0

    if slot_losses:
        texture_loss = torch.stack(slot_losses).mean()
    else:
        texture_loss = torch.tensor(0.0, device=predicted_alpha.device)

    total_loss = alpha_loss * 0.7 + texture_loss * 0.3
    metrics = {
        "alpha": float(alpha_loss.item()),
        "texture": float(texture_loss.item()),
    }
    metrics.update(slot_metrics)
    return total_loss, metrics


def save_texture_preview(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    epoch: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        inputs = batch["input"].to(device)
        alpha_targets = batch["alpha_target"].to(device)
        alpha_predictions, _ = model(inputs)

    rows: List[torch.Tensor] = []
    for index in range(min(4, inputs.shape[0])):
        minimap = inputs[index].cpu()
        alpha_pred = alpha_predictions[index, 0:3].cpu()
        alpha_gt = alpha_targets[index, 0:3].cpu()

        rows.append(
            torch.cat(
                [
                    minimap,
                    torch.cat([alpha_pred[0:1], alpha_pred[1:2], alpha_pred[2:3]], dim=0),
                    torch.cat([alpha_gt[0:1], alpha_gt[1:2], alpha_gt[2:3]], dim=0),
                ],
                dim=2,
            )
        )

    grid = torch.cat(rows, dim=1)
    transforms.ToPILImage()(torch.clamp(grid, 0.0, 1.0)).save(output_dir / f"val_epoch_{epoch:04d}.png")


def texture_metadata(
    args: argparse.Namespace,
    dataset_roots: Sequence[Path],
    sample_count: int,
    train_count: int,
    val_count: int,
    train_groups: int,
    val_groups: int,
    palette: Sequence[str],
) -> Dict[str, object]:
    return {
        "profile": args.profile,
        "dataset_roots": [str(root) for root in dataset_roots],
        "include_maps": list(args.include_map),
        "exclude_maps": list(args.exclude_map),
        "sample_count": sample_count,
        "train_count": train_count,
        "val_count": val_count,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "spatial_group_size": args.spatial_group_size,
        "palette_size": len(palette),
        "palette": list(palette),
        "overlay_alpha_layers": OVERLAY_ALPHA_LAYERS,
        "max_texture_layers": MAX_TEXTURE_LAYERS,
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_roots = resolve_texture_dataset_roots(args)
    include_maps = list(args.include_map)
    if not include_maps and args.profile != "manual":
        include_maps = list(TEXTURE_PROFILE_PRESETS[args.profile]["include_maps"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery_dataset = TextureTileDataset(
        dataset_roots=dataset_roots,
        include_maps=include_maps,
        exclude_maps=args.exclude_map,
        augment=not args.no_augment,
        limit=args.limit,
    )
    if len(discovery_dataset) == 0:
        raise SystemExit("No texture samples found. Export ML datasets with chunk layer data and alpha masks first.")

    palette = build_texture_palette(discovery_dataset, args.max_palette, args.min_texture_frequency)
    if not palette:
        raise SystemExit("No texture palette entries were discovered from chunk layer metadata.")

    (output_dir / "texture_palette.json").write_text(json.dumps(palette, indent=2), encoding="utf-8")

    dataset = TextureTileDataset(
        dataset_roots=dataset_roots,
        include_maps=include_maps,
        exclude_maps=args.exclude_map,
        palette=palette,
        augment=not args.no_augment,
        limit=args.limit,
    )

    train_indices, val_indices, train_groups, val_groups = split_grouped_indices(
        dataset.samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        block_size=args.spatial_group_size,
    )

    val_base_dataset = TextureTileDataset(
        dataset_roots=dataset_roots,
        include_maps=include_maps,
        exclude_maps=args.exclude_map,
        palette=palette,
        augment=False,
        limit=args.limit,
    )

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_base_dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextureDecomposerNet(palette_size=len(palette)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "components": [],
        "metadata": texture_metadata(args, dataset_roots, len(dataset), len(train_indices), len(val_indices), train_groups, val_groups, palette),
    }

    best_loss = float("inf")
    patience_counter = 0

    print("=" * 72)
    print("WoW Texture Decomposer V1")
    print("=" * 72)
    print(f"Palette size: {len(palette)}")
    print(f"Train/val samples: {len(train_indices)} / {len(val_indices)}")

    for epoch in range(args.epochs):
        model.train()
        train_losses: List[float] = []
        epoch_parts: Dict[str, float] = {}

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            inputs = batch["input"].to(device)
            alpha_targets = batch["alpha_target"].to(device)
            texture_targets = batch["texture_target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            alpha_predictions, texture_logits = model(inputs)
            loss, parts = texture_loss_components(alpha_predictions, texture_logits, alpha_targets, texture_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            for key, value in parts.items():
                epoch_parts[key] = epoch_parts.get(key, 0.0) + value
            progress.set_postfix(loss=f"{loss.item():.4f}")

        for key in epoch_parts:
            epoch_parts[key] /= max(len(train_loader), 1)

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                alpha_targets = batch["alpha_target"].to(device)
                texture_targets = batch["texture_target"].to(device)
                alpha_predictions, texture_logits = model(inputs)
                loss, _ = texture_loss_components(alpha_predictions, texture_logits, alpha_targets, texture_targets)
                val_losses.append(float(loss.item()))

        average_train_loss = float(np.mean(train_losses))
        average_val_loss = float(np.mean(val_losses))

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(average_train_loss)
        history["val_loss"].append(average_val_loss)
        history["components"].append(epoch_parts)
        (output_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": average_val_loss,
            "metadata": texture_metadata(args, dataset_roots, len(dataset), len(train_indices), len(val_indices), train_groups, val_groups, palette),
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {average_train_loss:.4f} | Val Loss: {average_val_loss:.4f} | Best: {best_loss:.4f}")
        print(
            "  Alpha: {alpha:.4f} | Texture: {texture:.4f} | Slot0: {slot0:.4f} | Slot1: {slot1:.4f} | Slot2: {slot2:.4f} | Slot3: {slot3:.4f}".format(
                alpha=epoch_parts.get("alpha", 0.0),
                texture=epoch_parts.get("texture", 0.0),
                slot0=epoch_parts.get("texture_slot_0", 0.0),
                slot1=epoch_parts.get("texture_slot_1", 0.0),
                slot2=epoch_parts.get("texture_slot_2", 0.0),
                slot3=epoch_parts.get("texture_slot_3", 0.0),
            )
        )

        scheduler.step(average_val_loss)

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            patience_counter = 0
            torch.save(checkpoint, output_dir / "best.pt")
            try:
                preview_batch = next(iter(val_loader))
                save_texture_preview(model, preview_batch, epoch + 1, output_dir / "previews", device)
            except Exception as exc:
                print(f"  Failed to save texture preview: {exc}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs")
                break

    print(f"\nTexture training complete. Best validation loss: {best_loss:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the separate minimap-to-texture-layer decomposition model.")
    parser.add_argument("--dataset-root", action="append", default=[], help="Explicit dataset root. Repeat for multiple roots.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=[str(path) for path in DEFAULT_DATASET_SEARCH_ROOTS],
        help="Root folder to scan when using an auto-discovery profile.",
    )
    parser.add_argument("--profile", choices=sorted(TEXTURE_PROFILE_PRESETS.keys()), default="development-textures")
    parser.add_argument("--include-map", action="append", default=[], help="Restrict training to these map names.")
    parser.add_argument("--exclude-map", action="append", default=[], help="Exclude these map names.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--spatial-group-size", type=int, default=DEFAULT_SPATIAL_GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, help="Optional per-root sample cap for quick debugging.")
    parser.add_argument("--no-augment", action="store_true", help="Disable RGB jitter and random flips.")
    parser.add_argument("--max-palette", type=int, default=DEFAULT_MAX_PALETTE)
    parser.add_argument("--min-texture-frequency", type=int, default=DEFAULT_MIN_TEXTURE_FREQUENCY)
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())