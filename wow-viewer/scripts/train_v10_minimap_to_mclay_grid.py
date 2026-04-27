from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "ml-training" / "v10_minimap_to_mclay_grid"
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 1337
DEFAULT_NUM_WORKERS = 4
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a v10 minimap-to-MCLY 16x16 chunk palette classifier from NPZ shards or a v10 MCLY label manifest.")
    parser.add_argument("input", help="NPZ shard, directory of NPZ shards, Stage 1 manifest, or v10 MCLY label manifest.")
    parser.add_argument("--dictionary", help="mclay_dictionary.json or mcly_dictionary.json from mine-v10-mcly. Required unless input is a v10 MCLY label manifest.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional hard cap after discovery.")
    parser.add_argument("--min-retained-chunks", type=int, default=8, help="Minimum retained dictionary chunks needed to label a tile.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def is_mcly_label_manifest(input_path: Path) -> bool:
    if not input_path.is_file() or input_path.suffix.lower() != ".json":
        return False

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return str(payload.get("schema_version") or payload.get("SchemaVersion") or "") == "v10-mcly-label-manifest.v1"


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


def normalize_texture_path(texture_path: str) -> str:
    return texture_path.replace("\\", "/").strip()


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    label_index: int
    combination_hash: str
    combination_key: str
    texture_names: tuple[str, ...]
    frequency: int
    inferred_biome_tag: str


def load_dictionary(path: Path) -> dict[str, DictionaryEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("dictionary")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No dictionary entries found in {path}")

    result: dict[str, DictionaryEntry] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue

        combination_key = str(entry.get("combination_key") or "")
        if not combination_key:
            texture_names = entry.get("texture_names") or []
            if isinstance(texture_names, list):
                combination_key = "|".join(normalize_texture_path(str(value)) for value in texture_names[:4])
        if not combination_key:
            continue

        texture_names_value = entry.get("texture_names") or []
        texture_names = tuple(normalize_texture_path(str(value)) for value in texture_names_value[:4]) if isinstance(texture_names_value, list) else ()
        result[combination_key] = DictionaryEntry(
            label_index=index,
            combination_hash=str(entry.get("combination_hash") or combination_key),
            combination_key=combination_key,
            texture_names=texture_names,
            frequency=int(entry.get("frequency") or 0),
            inferred_biome_tag=str(entry.get("inferred_biome_tag") or "unknown"),
        )

    if not result:
        raise ValueError(f"No usable combination keys found in {path}")
    return result


@dataclass(slots=True)
class MclyGridSample:
    path: Path
    tile_name: str
    minimap_rgb: np.ndarray
    label_grid: np.ndarray
    retained_chunk_count: int
    dominant_label_index: int
    dominant_chunk_count: int


def get_json_value(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    lowered = {key.lower(): value for key, value in payload.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def get_json_string(payload: dict[str, Any], *names: str) -> str:
    value = get_json_value(payload, *names)
    return str(value) if value is not None else ""


def get_json_int(payload: dict[str, Any], *names: str, default: int = 0) -> int:
    value = get_json_value(payload, *names)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_texture_names(texture_ids: np.ndarray, texture_names: list[str], y: int, x: int) -> tuple[str, str, str, str]:
    resolved: list[str] = []
    for layer in range(4):
        texture_id = int(texture_ids[y, x, layer]) if layer < texture_ids.shape[2] else -1
        if 0 <= texture_id < len(texture_names):
            resolved.append(normalize_texture_path(texture_names[texture_id]))
        elif texture_id >= 0:
            resolved.append(f"#{texture_id}")
        else:
            resolved.append("")
    return resolved[0], resolved[1], resolved[2], resolved[3]


def build_label_grid(texture_ids: np.ndarray, texture_names: list[str], dictionary: dict[str, DictionaryEntry]) -> tuple[np.ndarray, int, int, int]:
    label_grid = np.full((16, 16), IGNORE_INDEX, dtype=np.int32)
    counts: Counter[int] = Counter()
    for y in range(16):
        for x in range(16):
            key = "|".join(resolve_texture_names(texture_ids, texture_names, y, x))
            entry = dictionary.get(key)
            if entry is None:
                continue
            label_grid[y, x] = entry.label_index
            counts[entry.label_index] += 1

    if not counts:
        return label_grid, 0, IGNORE_INDEX, 0

    dominant_label_index, dominant_chunk_count = counts.most_common(1)[0]
    return label_grid, sum(counts.values()), dominant_label_index, dominant_chunk_count


def discover_samples(npz_paths: Iterable[Path], dictionary: dict[str, DictionaryEntry], min_retained_chunks: int, max_samples: int) -> list[MclyGridSample]:
    samples: list[MclyGridSample] = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as shard:
            if "minimap_rgb_256" not in shard.files or "mcly_texture_ids" not in shard.files:
                continue

            metadata = load_metadata(shard)
            texture_names = [
                normalize_texture_path(str(value))
                for value in metadata.get("mcly_texture_names", [])
                if isinstance(value, str)
            ]
            texture_ids = np.asarray(shard["mcly_texture_ids"], dtype=np.int32)
            minimap_rgb = np.asarray(shard["minimap_rgb_256"], dtype=np.uint8)
            if texture_ids.shape != (16, 16, 4) or minimap_rgb.shape != (256, 256, 3):
                continue

            label_grid, retained_count, dominant_label_index, dominant_chunk_count = build_label_grid(texture_ids, texture_names, dictionary)
            if retained_count < min_retained_chunks:
                continue

            samples.append(
                MclyGridSample(
                    path=path,
                    tile_name=str(metadata.get("tile_name") or path.stem),
                    minimap_rgb=minimap_rgb,
                    label_grid=label_grid,
                    retained_chunk_count=retained_count,
                    dominant_label_index=dominant_label_index,
                    dominant_chunk_count=dominant_chunk_count,
                )
            )

        if max_samples > 0 and len(samples) >= max_samples:
            return samples[:max_samples]

    return samples


def discover_manifest_samples(manifest_path: Path, max_samples: int) -> tuple[list[MclyGridSample], dict[int, DictionaryEntry], int, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    ignore_index = int(payload.get("ignore_index", IGNORE_INDEX))
    dictionary_path = str(payload.get("dictionary") or "")

    index_to_entry: dict[int, DictionaryEntry] = {}
    for label in payload.get("labels", []):
        if not isinstance(label, dict):
            continue

        label_index = get_json_int(label, "dictionary_label_index", "DictionaryLabelIndex", "label_index")
        texture_names_value = get_json_value(label, "texture_names", "TextureNames") or []
        texture_names = tuple(normalize_texture_path(str(value)) for value in texture_names_value[:4]) if isinstance(texture_names_value, list) else ()
        index_to_entry[label_index] = DictionaryEntry(
            label_index=label_index,
            combination_hash=get_json_string(label, "combination_hash", "CombinationHash"),
            combination_key=get_json_string(label, "combination_key", "CombinationKey"),
            texture_names=texture_names,
            frequency=get_json_int(label, "dictionary_frequency", "DictionaryFrequency"),
            inferred_biome_tag=get_json_string(label, "inferred_biome_tag", "InferredBiomeTag") or "unknown",
        )

    if not index_to_entry:
        raise ValueError(f"No label definitions found in {manifest_path}")

    samples: list[MclyGridSample] = []
    base_dir = manifest_path.parent
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            continue

        shard_path = Path(get_json_string(entry, "shard_path", "ShardPath"))
        if not shard_path.is_absolute():
            shard_path = (base_dir / shard_path).resolve()
        if not shard_path.exists():
            continue

        label_grid_value = get_json_value(entry, "label_grid_16", "LabelGrid16")
        label_grid = np.asarray(label_grid_value, dtype=np.int32)
        if label_grid.shape != (16, 16):
            continue
        if ignore_index != IGNORE_INDEX:
            label_grid[label_grid == ignore_index] = IGNORE_INDEX

        with np.load(shard_path, allow_pickle=False) as shard:
            if "minimap_rgb_256" not in shard.files:
                continue
            minimap_rgb = np.asarray(shard["minimap_rgb_256"], dtype=np.uint8)
            if minimap_rgb.shape != (256, 256, 3):
                continue

        retained_count = int(np.count_nonzero(label_grid != IGNORE_INDEX))
        if retained_count <= 0:
            continue

        samples.append(
            MclyGridSample(
                path=shard_path,
                tile_name=get_json_string(entry, "tile_name", "TileName") or shard_path.stem,
                minimap_rgb=minimap_rgb,
                label_grid=label_grid,
                retained_chunk_count=retained_count,
                dominant_label_index=get_json_int(entry, "dominant_dictionary_label_index", "DominantDictionaryLabelIndex", default=IGNORE_INDEX),
                dominant_chunk_count=get_json_int(entry, "dominant_chunk_count", "DominantChunkCount"),
            )
        )

        if max_samples > 0 and len(samples) >= max_samples:
            break

    return samples, index_to_entry, ignore_index, dictionary_path


class MclyGridDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[MclyGridSample], remapped_labels: dict[int, int]):
        self.samples = samples
        self.remapped_labels = remapped_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        minimap = torch.from_numpy(sample.minimap_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        target = np.full_like(sample.label_grid, IGNORE_INDEX, dtype=np.int64)
        for source_label, model_label in self.remapped_labels.items():
            target[sample.label_grid == source_label] = model_label
        return {
            "inputs": minimap,
            "labels": torch.from_numpy(target),
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


class MinimapToMclyGridClassifier(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
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
        self.head = nn.Sequential(
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(96, class_count, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.stem(inputs)))


def split_samples(samples: list[MclyGridSample], val_fraction: float, seed: int) -> tuple[list[MclyGridSample], list[MclyGridSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError("Need at least two valid NPZ shards to create train and validation splits.")

    val_count = max(1, min(len(shuffled) - 1, int(math.ceil(len(shuffled) * val_fraction))))
    return shuffled[val_count:], shuffled[:val_count]


def make_loader(dataset: MclyGridDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
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
    model: MinimapToMclyGridClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    channels_last: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_chunks = 0
    total_batches = 0
    autocast_enabled = device.type == "cuda"

    for batch in loader:
        inputs = maybe_channels_last(batch["inputs"].to(device, non_blocking=True), channels_last)
        labels = batch["labels"].to(device, non_blocking=True)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        autocast_context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled) if autocast_enabled else nullcontext()
        with grad_context:
            with autocast_context:
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(logits.detach(), dim=1)
        valid = labels != IGNORE_INDEX
        total_loss += float(loss.detach().cpu())
        total_correct += int((predictions[valid] == labels[valid]).sum().cpu())
        total_chunks += int(valid.sum().cpu())
        total_batches += 1

    return {
        "loss": total_loss / max(1, total_batches),
        "chunk_accuracy": total_correct / max(1, total_chunks),
        "chunk_count": float(total_chunks),
    }


def summarize_label_distribution(samples: list[MclyGridSample], index_to_entry: dict[int, DictionaryEntry]) -> list[dict[str, Any]]:
    chunk_counts: Counter[int] = Counter()
    tile_counts: Counter[int] = Counter()
    for sample in samples:
        seen: set[int] = set()
        for raw_label in sample.label_grid.reshape(-1):
            label = int(raw_label)
            if label == IGNORE_INDEX:
                continue
            chunk_counts[label] += 1
            seen.add(label)
        for label in seen:
            tile_counts[label] += 1

    return [
        {
            "label_index": label_index,
            "combination_hash": index_to_entry[label_index].combination_hash,
            "combination_key": index_to_entry[label_index].combination_key,
            "inferred_biome_tag": index_to_entry[label_index].inferred_biome_tag,
            "chunk_count": chunk_count,
            "tile_count": tile_counts[label_index],
        }
        for label_index, chunk_count in sorted(chunk_counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    source_label_manifest = is_mcly_label_manifest(input_path)
    if source_label_manifest:
        samples, index_to_entry, source_ignore_index, dictionary_path_text = discover_manifest_samples(input_path, args.max_samples)
        npz_paths = [sample.path for sample in samples]
        dictionary_path_for_checkpoint = dictionary_path_text
        if source_ignore_index != IGNORE_INDEX:
            print(f"normalized manifest ignore_index {source_ignore_index} to trainer ignore_index {IGNORE_INDEX}")
    else:
        if not args.dictionary:
            raise RuntimeError("--dictionary is required unless input is a v10 MCLY label manifest.")
        dictionary_path = Path(args.dictionary).resolve()
        dictionary = load_dictionary(dictionary_path)
        index_to_entry = {entry.label_index: entry for entry in dictionary.values()}
        npz_paths = find_npz_paths(input_path)
        samples = discover_samples(npz_paths, dictionary, args.min_retained_chunks, args.max_samples)
        dictionary_path_for_checkpoint = str(dictionary_path)

    if len(samples) < 2:
        raise RuntimeError("Need at least two v10 NPZ shards with minimap_rgb_256 and retained mcly_texture_ids chunk labels.")

    active_label_indexes = sorted({int(label) for sample in samples for label in sample.label_grid.reshape(-1) if int(label) != IGNORE_INDEX})
    if len(active_label_indexes) < 2:
        raise RuntimeError("Need at least two retained MCLY labels to train a chunk-grid classifier.")

    remapped_labels = {label_index: remapped for remapped, label_index in enumerate(active_label_indexes)}
    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)

    train_dataset = MclyGridDataset(train_samples, remapped_labels)
    val_dataset = MclyGridDataset(val_samples, remapped_labels)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = MinimapToMclyGridClassifier(class_count=len(active_label_indexes)).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args.channels_last)
        val_metrics = run_epoch(model, val_loader, None, None, device, args.channels_last)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} | "
            f"train chunk acc {train_metrics['chunk_accuracy']:.3f} | val loss {val_metrics['loss']:.4f} | "
            f"val chunk acc {val_metrics['chunk_accuracy']:.3f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "active_dictionary_label_indexes": active_label_indexes,
            "remapped_labels": remapped_labels,
            "dictionary_path": dictionary_path_for_checkpoint,
            "ignore_index": IGNORE_INDEX,
            "history": history,
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, checkpoints_dir / "best.pt")
            torch.save(checkpoint, output_dir / "minimap_to_mclay_grid_classifier.pt")

    label_index_payload = [
        {
            "model_label": remapped_labels[label_index],
            "dictionary_label_index": label_index,
            "combination_hash": index_to_entry[label_index].combination_hash,
            "combination_key": index_to_entry[label_index].combination_key,
            "texture_names": list(index_to_entry[label_index].texture_names),
            "inferred_biome_tag": index_to_entry[label_index].inferred_biome_tag,
            "dictionary_frequency": index_to_entry[label_index].frequency,
        }
        for label_index in active_label_indexes
    ]
    (output_dir / "label_index.json").write_text(json.dumps(label_index_payload, indent=2), encoding="utf-8")

    summary = {
        "input": str(input_path),
        "input_schema": "v10-mcly-label-manifest.v1" if source_label_manifest else "npz-or-stage1-manifest",
        "dictionary": dictionary_path_for_checkpoint,
        "discovered_npz_count": len(npz_paths),
        "labeled_sample_count": len(samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "active_label_count": len(active_label_indexes),
        "retained_chunk_count": int(sum(sample.retained_chunk_count for sample in samples)),
        "min_retained_chunks": args.min_retained_chunks,
        "label_distribution": summarize_label_distribution(samples, index_to_entry),
        "best_val_loss": best_val_loss,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
