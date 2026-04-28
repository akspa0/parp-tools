from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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
DEFAULT_MODEL_VARIANT = "structured_fusion_v2"
DEFAULT_NATIVE_V10_BOOST = 1.0
DEFAULT_RARE_SIGNAL_BOOST = 3.0
DEFAULT_MCAL_LOSS_WEIGHT = 0.5
DEFAULT_MCLY_LOSS_WEIGHT = 0.3
DEFAULT_HOLE_LOSS_WEIGHT = 0.5
PREVIEW_ROW_TITLE_HEIGHT = 18
PREVIEW_PANE_LABEL_HEIGHT = 18
PREVIEW_LABEL_PADDING = 4

INPUT_SIGNAL_LAYOUT: list[tuple[str, int]] = [
    ("minimap_rgb_256", 3),
    ("mcal_alpha_pack_256", 4),
    ("mccv_rgb", 3),
    ("mcnr_normal_xyz", 3),
    ("unified_liquid_mask", 1),
    ("unified_liquid_height", 1),
    ("object_mask_257", 1),
    ("object_precise_mask_257", 1),
    ("pm4_path_mask", 1),
    ("pm4_building_footprint_mask", 1),
    ("pm4_mprl_mask", 1),
    ("hole_mask_16", 1),
    ("mtxf_animated_mask", 1),
    ("coarse_height_17_prior", 1),
]

VALIDATION_SUBSET_FIELDS: dict[str, str] = {
    "is_native_v10": "native_v10",
    "is_legacy_only": "legacy_only",
    "has_pm4_signal": "pm4_present",
    "has_mcal_signal": "mcal_present",
    "has_liquid_signal": "liquid_present",
    "has_object_signal": "object_present",
    "has_mccv_signal": "mccv_present",
    "has_normal_signal": "normal_present",
}

VALIDATION_ABLATION_GROUPS: dict[str, tuple[str, ...]] = {
    "pm4": ("pm4_path_mask", "pm4_building_footprint_mask", "pm4_mprl_mask"),
    "mcal": ("mcal_alpha_pack_256",),
    "objects": ("object_mask_257", "object_precise_mask_257"),
    "liquids": ("unified_liquid_mask", "unified_liquid_height"),
    "mccv": ("mccv_rgb",),
    "normals": ("mcnr_normal_xyz",),
}

VALIDATION_ABLATION_SUBSETS: dict[str, str] = {
    "pm4": "pm4_present",
    "mcal": "mcal_present",
    "objects": "object_present",
    "liquids": "liquid_present",
    "mccv": "mccv_present",
    "normals": "normal_present",
}

VALIDATION_SPLIT_PRIORITY_GROUPS: tuple[str, ...] = (
    "native_v10",
    "pm4_present",
    "mcal_present",
    "mccv_present",
    "normal_present",
)

METRIC_KEYS = ("loss", "full_l1", "mid_l1", "coarse_l1", "gradient", "mae_m", "rmse_m")
MULTI_TASK_METRIC_KEYS = ("loss", "height_loss", "mcal_loss", "mcly_loss", "hole_loss", "full_l1", "mid_l1", "coarse_l1", "gradient", "mae_m", "rmse_m")


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
    parser.add_argument(
        "--native-v10-boost",
        type=float,
        default=DEFAULT_NATIVE_V10_BOOST,
        help="Additional weighted-sampler emphasis for native v10 training rows.",
    )
    parser.add_argument(
        "--rare-signal-boost",
        type=float,
        default=DEFAULT_RARE_SIGNAL_BOOST,
        help="Additional weighted-sampler emphasis for PM4, MCAL, MCCV, and normal-bearing training rows.",
    )
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="",
        help=(
            "Model architecture variant. Defaults to structured_fusion_v2 for new training runs. "
            "Checkpoint-only evaluation reuses the checkpoint variant when available, otherwise falls back to early_fusion_v1."
        ),
    )
    parser.add_argument(
        "--validation-ablation-groups",
        default="pm4,mcal,objects,liquids",
        help="Comma-separated validation ablation groups to zero during post-train analysis. Use 'none' to disable.",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        help="Optional checkpoint path. When provided, skip training and only run validation analysis against the checkpoint.",
    )
    parser.add_argument(
        "--force-validation-tiles",
        default="",
        help="Comma-separated tile names that must be placed in the validation split when present in the selected input corpus.",
    )
    parser.add_argument("--stage1-checkpoint", help="Optional Stage 1 checkpoint to use for coarse prior at inference time.")
    parser.add_argument("--mcal-loss-weight", type=float, default=DEFAULT_MCAL_LOSS_WEIGHT)
    parser.add_argument("--mcly-loss-weight", type=float, default=DEFAULT_MCLY_LOSS_WEIGHT)
    parser.add_argument("--hole-loss-weight", type=float, default=DEFAULT_HOLE_LOSS_WEIGHT)
    parser.add_argument("--resume-from", help="Resume training from a checkpoint file path.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Accumulate gradients over N steps for effective larger batch.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs.")
    parser.add_argument("--save-every", type=int, default=10, help="Save a checkpoint every N epochs.")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Max gradient norm for clipping (0 to disable).")
    parser.add_argument("--no-cosine-scheduler", action="store_true", help="Disable cosine annealing LR scheduler.")
    return parser.parse_args()


def build_input_channel_ranges() -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for key, width in INPUT_SIGNAL_LAYOUT:
        ranges[key] = (start, start + width)
        start += width
    return ranges


INPUT_CHANNEL_RANGES = build_input_channel_ranges()
TOTAL_INPUT_CHANNELS = sum(width for _, width in INPUT_SIGNAL_LAYOUT)

MODEL_VARIANTS = ("early_fusion_v1", "structured_fusion_v2", "multi_task_v3")

MODEL_BRANCH_SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "surface": (
        "minimap_rgb_256",
        "mcal_alpha_pack_256",
        "mccv_rgb",
        "mcnr_normal_xyz",
        "coarse_height_17_prior",
    ),
    "structure": (
        "object_mask_257",
        "object_precise_mask_257",
        "pm4_path_mask",
        "pm4_building_footprint_mask",
        "pm4_mprl_mask",
        "hole_mask_16",
        "mtxf_animated_mask",
    ),
    "liquids": (
        "unified_liquid_mask",
        "unified_liquid_height",
    ),
}


def parse_validation_ablation_groups(raw_value: str) -> list[str]:
    value = raw_value.strip().lower()
    if value in {"", "none", "off"}:
        return []

    groups: list[str] = []
    for part in raw_value.split(","):
        group = part.strip().lower()
        if not group:
            continue
        if group not in VALIDATION_ABLATION_GROUPS:
            raise ValueError(
                f"Unknown validation ablation group '{group}'. Supported groups: {', '.join(sorted(VALIDATION_ABLATION_GROUPS))}"
            )
        if group not in groups:
            groups.append(group)

    return groups


def parse_tile_name_list(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []

    seen: set[str] = set()
    result: list[str] = []
    for part in raw_value.split(","):
        tile_name = part.strip()
        if not tile_name:
            continue
        normalized = tile_name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(tile_name)

    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class ShardReference:
    path: Path
    dataset_key: str
    source_schema: str
    source_manifest: str

    def __hash__(self) -> int:
        return hash((str(self.path), self.dataset_key, self.source_schema, self.source_manifest))


def infer_dataset_key(path: Path) -> str:
    parts = list(path.parts)
    lower_parts = [part.lower() for part in parts]
    if "shards" in lower_parts:
        index = lower_parts.index("shards")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name or path.stem


def infer_source_schema(path: Path) -> str:
    normalized = str(path).replace("\\", "/").lower()
    if "/output/build-validation/v10-stage1-development-corpus/" in normalized:
        return "v10-stage1-manifest.v1"
    if "/output/ml-training/cache/" in normalized and "/cache/shards/" in normalized:
        return "v9-native-tensor-cache.v2"
    return "unknown"


def resolve_shard_reference(path: Path, dataset_key: str = "", source_schema: str = "", source_manifest: str = "") -> ShardReference:
    return ShardReference(
        path=path,
        dataset_key=dataset_key or infer_dataset_key(path),
        source_schema=source_schema or infer_source_schema(path),
        source_manifest=source_manifest,
    )


def find_npz_paths(input_path: Path) -> list[ShardReference]:
    if input_path.is_file() and input_path.suffix.lower() == ".npz":
        return [resolve_shard_reference(input_path)]

    if input_path.is_dir():
        return sorted(
            (resolve_shard_reference(path) for path in input_path.rglob("*.npz") if path.is_file()),
            key=lambda item: str(item.path),
        )

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
            references: list[ShardReference] = []
            for entry in payload["entries"]:
                if not isinstance(entry, dict):
                    continue
                shard_value = entry.get("shard_path") or entry.get("npz_path") or entry.get("ShardPath") or entry.get("NpzPath")
                if not isinstance(shard_value, str) or not shard_value.lower().endswith(".npz"):
                    continue
                candidate = Path(shard_value)
                if not candidate.is_absolute():
                    candidate = (input_path.parent / candidate).resolve()
                references.append(
                    resolve_shard_reference(
                        candidate,
                        dataset_key=str(entry.get("dataset_key") or entry.get("DatasetKey") or ""),
                        source_schema=str(entry.get("source_schema") or entry.get("SourceSchema") or ""),
                        source_manifest=str(entry.get("source_manifest") or entry.get("SourceManifest") or ""),
                    )
                )
            if references:
                return sorted(references, key=lambda item: str(item.path))

        collected: list[Path] = []
        collect_json_npz_paths(payload, input_path.parent, collected)
        return sorted(
            {resolve_shard_reference(path.resolve()) for path in collected if path.exists()},
            key=lambda item: str(item.path),
        )

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


def decode_signal_array(key: str, array: np.ndarray, source_key: str) -> np.ndarray:
    if key == "mcnr_normal_xyz" and source_key == "normal_rgb_256":
        decoded = (array.astype(np.float32) / 127.5) - 1.0
        return np.clip(decoded, -1.0, 1.0)
    if key == "unified_liquid_mask" and source_key == "mh2o_surface_height":
        return (array != 0).astype(np.float32)
    return array


def load_optional_array(npz_file: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, str] | None:
    if key not in npz_file.files:
        for alias in SIGNAL_ALIASES.get(key, ()):
            if alias in npz_file.files:
                array = np.asarray(npz_file[alias])
                return decode_signal_array(key, array, alias), alias
        return None
    array = np.asarray(npz_file[key])
    return decode_signal_array(key, array, key), key


@dataclass(slots=True)
class SignalSpec:
    key: str
    channels: int
    target_size: int
    dtype: np.dtype


OPTIONAL_SIGNALS: list[SignalSpec] = [
    SignalSpec("mcal_alpha_pack_256", 4, 256, np.float32),
    SignalSpec("mccv_rgb", 3, 257, np.float32),
    SignalSpec("mcnr_normal_xyz", 3, 257, np.float32),
    SignalSpec("unified_liquid_mask", 1, 257, np.float32),
    SignalSpec("unified_liquid_height", 1, 257, np.float32),
    SignalSpec("object_mask_257", 1, 257, np.float32),
    SignalSpec("object_precise_mask_257", 1, 257, np.float32),
    SignalSpec("pm4_path_mask", 1, 257, np.float32),
    SignalSpec("pm4_building_footprint_mask", 1, 257, np.float32),
    SignalSpec("pm4_mprl_mask", 1, 257, np.float32),
    SignalSpec("hole_mask_16", 1, 16, np.uint8),
    SignalSpec("mtxf_animated_mask", 1, 16, np.int32),
]


SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {
    "hole_mask_16": ("hole_mask_16x16",),
    "mcnr_normal_xyz": ("normal_rgb_256",),
    "object_precise_mask_257": ("object_mask_precise_257",),
    "pm4_path_mask": ("pm4_mask_257",),
    "unified_liquid_mask": ("liquid_mask_257", "wl_liquid_mask", "mh2o_surface_height"),
    "unified_liquid_height": ("liquid_height_257", "wl_liquid_height", "mh2o_surface_height", "mclq_surface_height"),
}


@dataclass(slots=True)
class Stage2Sample:
    path: Path
    tile_name: str
    dataset_key: str
    source_schema: str
    source_manifest: str
    minimap_rgb: np.ndarray
    height_257: np.ndarray
    height_65: np.ndarray
    height_17: np.ndarray
    signals: dict[str, np.ndarray]
    available_signal_keys: set[str]
    min_height: float
    max_height: float
    mcal_alpha: np.ndarray | None = None
    mcly_texture_ids: np.ndarray | None = None
    hole_mask: np.ndarray | None = None


def is_native_v10_source(sample: Stage2Sample) -> bool:
    if sample.source_schema == "v10-stage1-manifest.v1":
        return True
    normalized_dataset = sample.dataset_key.lower()
    normalized_path = str(sample.path).replace("\\", "/").lower()
    return normalized_dataset.startswith("v10-") or "/v10-stage1-development-corpus/" in normalized_path


def sample_has_pm4_signal(sample: Stage2Sample) -> bool:
    return any(
        key in sample.available_signal_keys
        for key in ("pm4_path_mask", "pm4_building_footprint_mask", "pm4_mprl_mask")
    )


def sample_has_mcal_signal(sample: Stage2Sample) -> bool:
    return "mcal_alpha_pack_256" in sample.available_signal_keys


def sample_has_mccv_signal(sample: Stage2Sample) -> bool:
    return "mccv_rgb" in sample.available_signal_keys


def sample_has_normal_signal(sample: Stage2Sample) -> bool:
    return "mcnr_normal_xyz" in sample.available_signal_keys


def sample_has_liquid_signal(sample: Stage2Sample) -> bool:
    return any(
        key in sample.available_signal_keys
        for key in ("unified_liquid_mask", "unified_liquid_height")
    )


def sample_has_object_signal(sample: Stage2Sample) -> bool:
    return any(key in sample.available_signal_keys for key in ("object_mask_257", "object_precise_mask_257"))


def sample_matches_subset(sample: Stage2Sample, subset_name: str) -> bool:
    if subset_name == "native_v10":
        return is_native_v10_source(sample)
    if subset_name == "legacy_only":
        return not is_native_v10_source(sample)
    if subset_name == "pm4_present":
        return sample_has_pm4_signal(sample)
    if subset_name == "mcal_present":
        return sample_has_mcal_signal(sample)
    if subset_name == "liquid_present":
        return sample_has_liquid_signal(sample)
    if subset_name == "object_present":
        return sample_has_object_signal(sample)
    if subset_name == "mccv_present":
        return sample_has_mccv_signal(sample)
    if subset_name == "normal_present":
        return sample_has_normal_signal(sample)
    raise ValueError(f"Unknown subset name '{subset_name}'.")


def discover_samples(npz_paths: Iterable[ShardReference], max_samples: int) -> list[Stage2Sample]:
    samples: list[Stage2Sample] = []
    for shard_ref in npz_paths:
        with np.load(shard_ref.path, allow_pickle=False) as shard:
            if "minimap_rgb_256" not in shard.files or "height_257" not in shard.files or "height_17" not in shard.files:
                continue

            minimap_rgb = np.asarray(shard["minimap_rgb_256"], dtype=np.uint8)
            height_257 = np.asarray(shard["height_257"], dtype=np.float32)
            height_17 = np.asarray(shard["height_17"], dtype=np.float32)
            height_65 = np.asarray(shard["height_65"], dtype=np.float32) if "height_65" in shard.files else None

            if minimap_rgb.shape != (256, 256, 3) or height_257.shape != (257, 257) or height_17.shape != (17, 17):
                continue

            metadata = load_metadata(shard)
            tile_name = str(metadata.get("tile_name") or shard_ref.path.stem)

            signals: dict[str, np.ndarray] = {}
            available: set[str] = set()
            for spec in OPTIONAL_SIGNALS:
                loaded = load_optional_array(shard, spec.key)
                if loaded is None:
                    continue
                arr, source_key = loaded
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
                if source_key != spec.key:
                    available.add(source_key)

            if height_65 is None:
                height_65 = downsample_heightmap(height_257, 65)

            # ── Multi-task targets (optional) ────────────────────────────
            mcal_alpha = None
            if "mcal_alpha_pack_256" in shard.files:
                mcal_arr = np.asarray(shard["mcal_alpha_pack_256"], dtype=np.float32)
                if mcal_arr.ndim == 3 and mcal_arr.shape[2] == 4:
                    mcal_alpha = np.transpose(mcal_arr, (2, 0, 1))
                elif mcal_arr.ndim == 3 and mcal_arr.shape[0] == 4:
                    mcal_alpha = mcal_arr
                # MCAL is stored at 1024×1024 (64×64 per chunk × 16 chunks).
                # Downsample to 256×256 for the multi-task head via block averaging.
                if mcal_alpha is not None and mcal_alpha.shape[1] != 256:
                    c, h, w = mcal_alpha.shape
                    block = h // 256
                    mcal_alpha = mcal_alpha.reshape(c, 256, block, 256, block).mean(axis=(2, 4))

            mcly_ids = None
            if "mcly_texture_ids" in shard.files:
                mcly_arr = np.asarray(shard["mcly_texture_ids"], dtype=np.int32)
                if mcly_arr.shape == (16, 16, 4):
                    mcly_ids = mcly_arr

            hole_mask = None
            if "hole_mask_16" in shard.files:
                hole_arr = np.asarray(shard["hole_mask_16"], dtype=np.float32)
                if hole_arr.shape == (16, 16):
                    hole_mask = hole_arr

            samples.append(
                Stage2Sample(
                    path=shard_ref.path,
                    tile_name=tile_name,
                    dataset_key=shard_ref.dataset_key,
                    source_schema=shard_ref.source_schema,
                    source_manifest=shard_ref.source_manifest,
                    minimap_rgb=minimap_rgb,
                    height_257=height_257,
                    height_65=height_65,
                    height_17=height_17,
                    signals=signals,
                    available_signal_keys=available,
                    min_height=float(np.min(height_257)),
                    max_height=float(np.max(height_257)),
                    mcal_alpha=mcal_alpha,
                    mcly_texture_ids=mcly_ids,
                    hole_mask=hole_mask,
                )
            )

        if max_samples > 0 and len(samples) >= max_samples:
            return samples[:max_samples]

    return samples


def downsample_heightmap(source: np.ndarray, target_size: int) -> np.ndarray:
    """Bilinear downsample a 2D heightmap using vectorized numpy."""
    source_size = source.shape[0]
    scale = (source_size - 1) / (target_size - 1)
    target_coords = np.arange(target_size, dtype=np.float64) * scale
    ix = np.clip(np.floor(target_coords).astype(np.intp), 0, source_size - 2)
    fx = target_coords - ix

    source_rows = source[np.ix_(ix, ix)]
    source_cols = source[np.ix_(ix + 1, ix + 1)]
    source_row_next = source[np.ix_(ix + 1, ix)]
    source_col_next = source[np.ix_(ix, ix + 1)]

    top = source_rows + (source_col_next - source_rows) * fx[np.newaxis, :]
    bottom = source_row_next + (source_cols - source_row_next) * fx[np.newaxis, :]
    result = (top + (bottom - top) * fx[:, np.newaxis]).astype(np.float32)
    return result


class Stage2Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[Stage2Sample],
        height_mean: float,
        height_std: float,
        signal_dropout: float,
        mcly_label_index: dict[int, int] | None = None,
        mcly_num_classes: int = 0,
    ):
        self.samples = samples
        self.height_mean = float(height_mean)
        self.height_std = float(height_std)
        self.signal_dropout = signal_dropout
        self.mcly_label_index = mcly_label_index or {}
        self.mcly_num_classes = mcly_num_classes

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
            ("unified_liquid_mask", 1),
            ("unified_liquid_height", 1),
            ("object_mask_257", 1),
            ("object_precise_mask_257", 1),
            ("pm4_path_mask", 1),
            ("pm4_building_footprint_mask", 1),
            ("pm4_mprl_mask", 1),
        ]:
            if key in sample.signals:
                arr = sample.signals[key]
                t = torch.from_numpy(arr.astype(np.float32))
                if t.ndim == 2:
                    t = t.unsqueeze(0)
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

        # Height targets
        height_257 = torch.from_numpy(((sample.height_257 - self.height_mean) / self.height_std).astype(np.float32)).unsqueeze(0)
        height_65 = torch.from_numpy(((sample.height_65 - self.height_mean) / self.height_std).astype(np.float32)).unsqueeze(0)
        height_17 = torch.from_numpy(((sample.height_17 - self.height_mean) / self.height_std).astype(np.float32)).unsqueeze(0)

        # Multi-task targets
        mcal_target = torch.zeros((4, 256, 256), dtype=torch.float32)
        has_mcal = False
        if sample.mcal_alpha is not None:
            mcal_target = torch.from_numpy(sample.mcal_alpha.astype(np.float32))
            has_mcal = True

        mcly_target = torch.zeros((16, 16), dtype=torch.long)
        has_mcly = False
        if sample.mcly_texture_ids is not None and self.mcly_label_index:
            label_grid = np.full((16, 16), -100, dtype=np.int64)  # -100 = ignore_index
            for cy in range(16):
                for cx in range(16):
                    tex_id = int(sample.mcly_texture_ids[cy, cx, 0])
                    if tex_id >= 0 and tex_id in self.mcly_label_index:
                        label_grid[cy, cx] = self.mcly_label_index[tex_id]
            mcly_target = torch.from_numpy(label_grid)
            has_mcly = True

        hole_target = torch.zeros((1, 16, 16), dtype=torch.float32)
        has_hole = False
        if sample.hole_mask is not None:
            hole_target = torch.from_numpy(sample.hole_mask.astype(np.float32)).unsqueeze(0)
            has_hole = True

        return {
            "inputs": inputs,
            "height_257": height_257,
            "height_65": height_65,
            "height_17": height_17,
            "mcal_target": mcal_target,
            "has_mcal": torch.tensor(has_mcal, dtype=torch.bool),
            "mcly_target": mcly_target,
            "has_mcly": torch.tensor(has_mcly, dtype=torch.bool),
            "hole_target": hole_target,
            "has_hole": torch.tensor(has_hole, dtype=torch.bool),
            "minimap": minimap,
            "tile_name": sample.tile_name,
            "dataset_key": sample.dataset_key,
            "source_schema": sample.source_schema,
            "is_native_v10": torch.tensor(is_native_v10_source(sample), dtype=torch.bool),
            "is_legacy_only": torch.tensor(not is_native_v10_source(sample), dtype=torch.bool),
            "has_pm4_signal": torch.tensor(sample_has_pm4_signal(sample), dtype=torch.bool),
            "has_mcal_signal": torch.tensor(sample_has_mcal_signal(sample), dtype=torch.bool),
            "has_liquid_signal": torch.tensor(sample_has_liquid_signal(sample), dtype=torch.bool),
            "has_object_signal": torch.tensor(sample_has_object_signal(sample), dtype=torch.bool),
            "has_mccv_signal": torch.tensor(sample_has_mccv_signal(sample), dtype=torch.bool),
            "has_normal_signal": torch.tensor(sample_has_normal_signal(sample), dtype=torch.bool),
        }


@dataclass(slots=True)
class MetricAccumulator:
    totals: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in MULTI_TASK_METRIC_KEYS})
    count: int = 0

    def add(self, metric_tensors: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> None:
        selected: dict[str, torch.Tensor] = metric_tensors
        if mask is not None:
            if mask.ndim > 1:
                mask = mask.reshape(-1)
            selected = {key: value[mask] for key, value in metric_tensors.items()}

        if not selected:
            return

        sample_count = int(next(iter(selected.values())).numel())
        if sample_count == 0:
            return

        for key, value in selected.items():
            self.totals[key] += float(value.detach().sum().cpu())
        self.count += sample_count

    def to_report(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, **{key: None for key in MULTI_TASK_METRIC_KEYS}}
        return {
            "count": self.count,
            **{key: self.totals[key] / self.count for key in MULTI_TASK_METRIC_KEYS},
        }


def compute_metric_tensors(
    pred_17: torch.Tensor,
    pred_65: torch.Tensor,
    pred_257: torch.Tensor,
    target_17: torch.Tensor,
    target_65: torch.Tensor,
    target_257: torch.Tensor,
    height_mean: float,
    height_std: float,
    pred_mcal: torch.Tensor | None = None,
    target_mcal: torch.Tensor | None = None,
    has_mcal: torch.Tensor | None = None,
    pred_mcly: torch.Tensor | None = None,
    target_mcly: torch.Tensor | None = None,
    has_mcly: torch.Tensor | None = None,
    pred_hole: torch.Tensor | None = None,
    target_hole: torch.Tensor | None = None,
    has_hole: torch.Tensor | None = None,
    mcal_weight: float = 0.5,
    mcly_weight: float = 0.3,
    hole_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    full_l1 = torch.abs(pred_257 - target_257).mean(dim=(1, 2, 3))
    mid_l1 = torch.abs(pred_65 - target_65).mean(dim=(1, 2, 3))
    coarse_l1 = torch.abs(pred_17 - target_17).mean(dim=(1, 2, 3))

    grad_pred_x = pred_257[:, :, :, 1:] - pred_257[:, :, :, :-1]
    grad_pred_y = pred_257[:, :, 1:, :] - pred_257[:, :, :-1, :]
    grad_target_x = target_257[:, :, :, 1:] - target_257[:, :, :, :-1]
    grad_target_y = target_257[:, :, 1:, :] - target_257[:, :, :-1, :]
    gradient = torch.abs(grad_pred_x - grad_target_x).mean(dim=(1, 2, 3)) + torch.abs(grad_pred_y - grad_target_y).mean(dim=(1, 2, 3))

    pred_65_up = F.interpolate(pred_17, size=(65, 65), mode="bilinear", align_corners=False)
    target_65_up = F.interpolate(target_17, size=(65, 65), mode="bilinear", align_corners=False)
    mid_residual = torch.abs((pred_65 - pred_65_up) - (target_65 - target_65_up)).mean(dim=(1, 2, 3))

    pred_257_up = F.interpolate(pred_65, size=(257, 257), mode="bilinear", align_corners=False)
    target_257_up = F.interpolate(target_65, size=(257, 257), mode="bilinear", align_corners=False)
    detail_res = torch.abs((pred_257 - pred_257_up) - (target_257 - target_257_up)).mean(dim=(1, 2, 3))

    height_loss = full_l1 + 0.5 * mid_l1 + 0.25 * coarse_l1 + 0.3 * gradient + 0.3 * mid_residual + 0.3 * detail_res

    # Multi-task losses (masked per sample)
    mcal_loss = torch.zeros_like(height_loss)
    if pred_mcal is not None and target_mcal is not None and has_mcal is not None:
        mcal_l1 = torch.abs(pred_mcal - target_mcal).mean(dim=(1, 2, 3))
        mcal_loss = mcal_l1 * has_mcal.float() * mcal_weight

    mcly_loss = torch.zeros_like(height_loss)
    if pred_mcly is not None and target_mcly is not None and has_mcly is not None:
        ce = F.cross_entropy(pred_mcly, target_mcly, ignore_index=-100, reduction="none")
        mcly_loss = ce.mean(dim=(1, 2)) * has_mcly.float() * mcly_weight

    hole_loss = torch.zeros_like(height_loss)
    if pred_hole is not None and target_hole is not None and has_hole is not None:
        bce = F.binary_cross_entropy_with_logits(pred_hole, target_hole, reduction="none")
        hole_loss = bce.mean(dim=(1, 2, 3)) * has_hole.float() * hole_weight

    loss = height_loss + mcal_loss + mcly_loss + hole_loss

    pred_height_m = pred_257 * height_std + height_mean
    target_height_m = target_257 * height_std + height_mean
    diff_m = pred_height_m - target_height_m
    mae_m = diff_m.abs().mean(dim=(1, 2, 3))
    rmse_m = torch.sqrt(diff_m.square().mean(dim=(1, 2, 3)))

    return {
        "loss": loss,
        "height_loss": height_loss,
        "mcal_loss": mcal_loss,
        "mcly_loss": mcly_loss,
        "hole_loss": hole_loss,
        "full_l1": full_l1,
        "mid_l1": mid_l1,
        "coarse_l1": coarse_l1,
        "gradient": gradient,
        "mae_m": mae_m,
        "rmse_m": rmse_m,
    }


def apply_input_ablation(inputs: torch.Tensor, group_name: str) -> torch.Tensor:
    masked = inputs.clone()
    for signal_name in VALIDATION_ABLATION_GROUPS[group_name]:
        start, end = INPUT_CHANNEL_RANGES[signal_name]
        masked[:, start:end, :, :] = 0
    return masked


def delta_report(current: dict[str, float | int | None], baseline: dict[str, float | int | None]) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key in MULTI_TASK_METRIC_KEYS:
        current_value = current.get(key)
        baseline_value = baseline.get(key)
        if current_value is None or baseline_value is None:
            delta[key] = None
        else:
            delta[key] = float(current_value) - float(baseline_value)
    return delta


def evaluate_validation_analysis(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    height_mean: float,
    height_std: float,
    channels_last: bool,
    ablation_groups: list[str],
    is_multi_task: bool = False,
    mcal_weight: float = 0.5,
    mcly_weight: float = 0.3,
    hole_weight: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    autocast_enabled = device.type == "cuda"
    baseline_accumulator = MetricAccumulator()
    subset_accumulators = {label: MetricAccumulator() for label in VALIDATION_SUBSET_FIELDS.values()}
    ablation_accumulators = {
        group: {
            "overall": MetricAccumulator(),
            "applicable_subset": MetricAccumulator(),
        }
        for group in ablation_groups
    }

    for batch in loader:
        inputs = maybe_channels_last(batch["inputs"].to(device, non_blocking=True), channels_last)
        target_257 = batch["height_257"].to(device, non_blocking=True)
        target_65 = batch["height_65"].to(device, non_blocking=True)
        target_17 = batch["height_17"].to(device, non_blocking=True)

        with torch.no_grad():
            context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled) if autocast_enabled else torch.no_grad()
            with context:
                if is_multi_task:
                    pred_17, pred_65, pred_257, pred_mcal, pred_mcly, pred_hole = model(inputs)
                    metric_tensors = compute_metric_tensors(
                        pred_17, pred_65, pred_257, target_17, target_65, target_257,
                        height_mean, height_std,
                        pred_mcal=pred_mcal, target_mcal=batch["mcal_target"].to(device, non_blocking=True),
                        has_mcal=batch["has_mcal"].to(device, non_blocking=True),
                        pred_mcly=pred_mcly, target_mcly=batch["mcly_target"].to(device, non_blocking=True),
                        has_mcly=batch["has_mcly"].to(device, non_blocking=True),
                        pred_hole=pred_hole, target_hole=batch["hole_target"].to(device, non_blocking=True),
                        has_hole=batch["has_hole"].to(device, non_blocking=True),
                        mcal_weight=mcal_weight, mcly_weight=mcly_weight, hole_weight=hole_weight,
                    )
                else:
                    pred_17, pred_65, pred_257 = model(inputs)
                    metric_tensors = compute_metric_tensors(pred_17, pred_65, pred_257, target_17, target_65, target_257, height_mean, height_std)

        baseline_accumulator.add(metric_tensors)
        for field_name, label in VALIDATION_SUBSET_FIELDS.items():
            subset_accumulators[label].add(metric_tensors, batch[field_name].to(torch.bool))

        for group in ablation_groups:
            ablated_inputs = apply_input_ablation(inputs, group)
            with torch.no_grad():
                context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled) if autocast_enabled else torch.no_grad()
                with context:
                    if is_multi_task:
                        pred_17, pred_65, pred_257, pred_mcal, pred_mcly, pred_hole = model(ablated_inputs)
                        ablated_metrics = compute_metric_tensors(
                            pred_17, pred_65, pred_257, target_17, target_65, target_257,
                            height_mean, height_std,
                            pred_mcal=pred_mcal, target_mcal=batch["mcal_target"].to(device, non_blocking=True),
                            has_mcal=batch["has_mcal"].to(device, non_blocking=True),
                            pred_mcly=pred_mcly, target_mcly=batch["mcly_target"].to(device, non_blocking=True),
                            has_mcly=batch["has_mcly"].to(device, non_blocking=True),
                            pred_hole=pred_hole, target_hole=batch["hole_target"].to(device, non_blocking=True),
                            has_hole=batch["has_hole"].to(device, non_blocking=True),
                            mcal_weight=mcal_weight, mcly_weight=mcly_weight, hole_weight=hole_weight,
                        )
                    else:
                        pred_17, pred_65, pred_257 = model(ablated_inputs)
                        ablated_metrics = compute_metric_tensors(pred_17, pred_65, pred_257, target_17, target_65, target_257, height_mean, height_std)

            ablation_accumulators[group]["overall"].add(ablated_metrics)
            applicable_field_name = next(
                field_name for field_name, label in VALIDATION_SUBSET_FIELDS.items() if label == VALIDATION_ABLATION_SUBSETS[group]
            )
            ablation_accumulators[group]["applicable_subset"].add(ablated_metrics, batch[applicable_field_name].to(torch.bool))

    baseline_report = baseline_accumulator.to_report()
    subsets_report = {label: accumulator.to_report() for label, accumulator in subset_accumulators.items()}
    ablation_report: dict[str, Any] = {}
    for group, accumulators in ablation_accumulators.items():
        overall_report = accumulators["overall"].to_report()
        applicable_report = accumulators["applicable_subset"].to_report()
        ablation_report[group] = {
            "zeroed_signals": list(VALIDATION_ABLATION_GROUPS[group]),
            "channel_ranges": [
                {
                    "signal": signal_name,
                    "start": INPUT_CHANNEL_RANGES[signal_name][0],
                    "end_exclusive": INPUT_CHANNEL_RANGES[signal_name][1],
                }
                for signal_name in VALIDATION_ABLATION_GROUPS[group]
            ],
            "overall": overall_report,
            "overall_delta_vs_baseline": delta_report(overall_report, baseline_report),
            "applicable_subset": VALIDATION_ABLATION_SUBSETS[group],
            "applicable_subset_metrics": applicable_report,
            "applicable_subset_delta_vs_baseline": delta_report(applicable_report, subsets_report[VALIDATION_ABLATION_SUBSETS[group]]),
        }

    return {
        "baseline": baseline_report,
        "subsets": subsets_report,
        "ablations": ablation_report,
    }


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def load_model_state(model: nn.Module, state_dict: dict[str, Any]) -> None:
    cleaned_state = {
        key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state)


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


class DecoderBlock(nn.Module):
    """U-Net decoder block: upsample, concat skip, conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EarlyFusionStage2TerrainSynthModel(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.enc1 = ConvBlock(32, 32)
        self.enc2 = ConvBlock(32, 64, stride=2)
        self.enc3 = ConvBlock(64, 96, stride=2)
        self.enc4 = ConvBlock(96, 128, stride=2)
        self.enc5 = ConvBlock(128, 160, stride=2)

        self.coarse_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((17, 17)),
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.mid_up = nn.Upsample(size=(65, 65), mode="bilinear", align_corners=False)
        self.mid_head = nn.Sequential(
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 1, kernel_size=1),
        )

        # U-Net decoder with skip connections from encoder
        self.dec1 = DecoderBlock(160, 128, 128)   # 16->32, skip x4(128@32)
        self.dec2 = DecoderBlock(128, 96, 96)      # 32->64, skip x3(96@64)
        self.dec3 = DecoderBlock(96, 64, 64)       # 64->128, skip x2(64@128)
        self.dec4 = DecoderBlock(64, 32, 32)       # 128->256, skip x1(32@256)
        self.fine_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        coarse = self.coarse_head(x5)
        mid = self.mid_head(self.mid_up(x5))

        d = self.dec1(x5, x4)
        d = self.dec2(d, x3)
        d = self.dec3(d, x2)
        d = self.dec4(d, x1)
        fine = self.fine_head(d)
        fine = F.interpolate(fine, size=(257, 257), mode="bilinear", align_corners=False)

        return coarse, mid, fine


class SignalBranchStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StructuredFusionStage2TerrainSynthModel(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        expected_channels = TOTAL_INPUT_CHANNELS
        if input_channels != expected_channels:
            raise ValueError(f"structured_fusion_v2 expects {expected_channels} input channels, got {input_channels}.")

        self.branch_ranges = {
            branch_name: [INPUT_CHANNEL_RANGES[signal_name] for signal_name in signal_names]
            for branch_name, signal_names in MODEL_BRANCH_SIGNAL_GROUPS.items()
        }
        branch_widths = {
            "surface": 24,
            "structure": 16,
            "liquids": 16,
        }

        self.surface_stem = SignalBranchStem(sum(end - start for start, end in self.branch_ranges["surface"]), branch_widths["surface"])
        self.structure_stem = SignalBranchStem(sum(end - start for start, end in self.branch_ranges["structure"]), branch_widths["structure"])
        self.liquid_stem = SignalBranchStem(sum(end - start for start, end in self.branch_ranges["liquids"]), branch_widths["liquids"])

        fused_channels = sum(branch_widths.values())
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.enc1 = ConvBlock(32, 32)
        self.enc2 = ConvBlock(32, 64, stride=2)
        self.enc3 = ConvBlock(64, 96, stride=2)
        self.enc4 = ConvBlock(96, 128, stride=2)
        self.enc5 = ConvBlock(128, 160, stride=2)

        self.coarse_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((17, 17)),
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.mid_up = nn.Upsample(size=(65, 65), mode="bilinear", align_corners=False)
        self.mid_head = nn.Sequential(
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 1, kernel_size=1),
        )

        self.dec1 = DecoderBlock(160, 128, 128)
        self.dec2 = DecoderBlock(128, 96, 96)
        self.dec3 = DecoderBlock(96, 64, 64)
        self.dec4 = DecoderBlock(64, 32, 32)
        self.fine_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def _slice_branch(self, x: torch.Tensor, branch_name: str) -> torch.Tensor:
        return torch.cat([x[:, start:end, :, :] for start, end in self.branch_ranges[branch_name]], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        surface = self.surface_stem(self._slice_branch(x, "surface"))
        structure = self.structure_stem(self._slice_branch(x, "structure"))
        liquids = self.liquid_stem(self._slice_branch(x, "liquids"))

        x0 = self.fusion(torch.cat([surface, structure, liquids], dim=1))
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        coarse = self.coarse_head(x5)
        mid = self.mid_head(self.mid_up(x5))

        d = self.dec1(x5, x4)
        d = self.dec2(d, x3)
        d = self.dec3(d, x2)
        d = self.dec4(d, x1)
        fine = self.fine_head(d)
        fine = F.interpolate(fine, size=(257, 257), mode="bilinear", align_corners=False)

        return coarse, mid, fine


class MultiTaskStructuredFusionModel(nn.Module):
    """Multi-task variant: predicts height + MCAL alpha + MCLY palette + hole mask."""

    def __init__(self, input_channels: int, mcly_num_classes: int) -> None:
        super().__init__()
        expected_channels = TOTAL_INPUT_CHANNELS
        if input_channels != expected_channels:
            raise ValueError(f"multi_task_v3 expects {expected_channels} input channels, got {input_channels}.")

        self.branch_ranges = {
            branch_name: [INPUT_CHANNEL_RANGES[signal_name] for signal_name in signal_names]
            for branch_name, signal_names in MODEL_BRANCH_SIGNAL_GROUPS.items()
        }
        branch_widths = {
            "surface": 24,
            "structure": 16,
            "liquids": 16,
        }

        self.surface_stem = SignalBranchStem(sum(end - start for start, end in self.branch_ranges["surface"]), branch_widths["surface"])
        self.structure_stem = SignalBranchStem(sum(end - start for start, end in self.branch_ranges["structure"]), branch_widths["structure"])
        self.liquid_stem = SignalBranchStem(sum(end - start for start, end in self.branch_ranges["liquids"]), branch_widths["liquids"])

        fused_channels = sum(branch_widths.values())
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.enc1 = ConvBlock(32, 32)
        self.enc2 = ConvBlock(32, 64, stride=2)
        self.enc3 = ConvBlock(64, 96, stride=2)
        self.enc4 = ConvBlock(96, 128, stride=2)
        self.enc5 = ConvBlock(128, 160, stride=2)

        # ── Height heads (U-Net with skip connections) ──────────────────
        self.coarse_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((17, 17)),
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.mid_up = nn.Upsample(size=(65, 65), mode="bilinear", align_corners=False)
        self.mid_head = nn.Sequential(
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 1, kernel_size=1),
        )
        self.dec1 = DecoderBlock(160, 128, 128)
        self.dec2 = DecoderBlock(128, 96, 96)
        self.dec3 = DecoderBlock(96, 64, 64)
        self.dec4 = DecoderBlock(64, 32, 32)
        self.fine_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # ── MCAL head: 160@16 → upsample → 4@256 with sigmoid ────────────
        self.mcal_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(160, 96),
        )
        self.mcal_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(96, 64),
        )
        self.mcal_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
        )
        self.mcal_up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(32, 16),
        )
        self.mcal_head = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # ── MCLY head: 160@16 → 16×16 × num_classes ──────────────────────
        self.mcly_head = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, mcly_num_classes, kernel_size=1),
        )

        # ── Hole head: 160@16 → 1×16×16 with logits ──────────────────────
        self.hole_head = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _slice_branch(self, x: torch.Tensor, branch_name: str) -> torch.Tensor:
        return torch.cat([x[:, start:end, :, :] for start, end in self.branch_ranges[branch_name]], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        surface = self.surface_stem(self._slice_branch(x, "surface"))
        structure = self.structure_stem(self._slice_branch(x, "structure"))
        liquids = self.liquid_stem(self._slice_branch(x, "liquids"))

        x0 = self.fusion(torch.cat([surface, structure, liquids], dim=1))
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)  # [B, 160, 16, 16]

        # Height (with skip connections)
        coarse = self.coarse_head(x5)
        mid = self.mid_head(self.mid_up(x5))
        d = self.dec1(x5, x4)
        d = self.dec2(d, x3)
        d = self.dec3(d, x2)
        d = self.dec4(d, x1)
        fine = self.fine_head(d)
        fine = F.interpolate(fine, size=(257, 257), mode="bilinear", align_corners=False)

        # MCAL
        m = self.mcal_up1(x5)
        m = self.mcal_up2(m)
        m = self.mcal_up3(m)
        m = self.mcal_up4(m)
        mcal = self.mcal_head(m)  # [B, 4, 256, 256]

        # MCLY
        mcly = self.mcly_head(x5)  # [B, num_classes, 16, 16]

        # Hole
        hole = self.hole_head(x5)  # [B, 1, 16, 16] (logits)

        return coarse, mid, fine, mcal, mcly, hole


def build_model(model_variant: str, input_channels: int, mcly_num_classes: int = 0) -> nn.Module:
    if model_variant == "early_fusion_v1":
        return EarlyFusionStage2TerrainSynthModel(input_channels=input_channels)
    if model_variant == "structured_fusion_v2":
        return StructuredFusionStage2TerrainSynthModel(input_channels=input_channels)
    if model_variant == "multi_task_v3":
        if mcly_num_classes <= 0:
            raise ValueError("multi_task_v3 requires mcly_num_classes > 0.")
        return MultiTaskStructuredFusionModel(input_channels=input_channels, mcly_num_classes=mcly_num_classes)
    raise ValueError(f"Unsupported model variant '{model_variant}'.")


def resolve_model_variant(args: argparse.Namespace, checkpoint_payload: dict[str, Any] | None = None) -> str:
    if args.model_variant:
        return args.model_variant
    if checkpoint_payload is not None:
        return str(checkpoint_payload.get("model_variant") or "early_fusion_v1")
    return DEFAULT_MODEL_VARIANT


def build_mcly_label_index(samples: list[Stage2Sample]) -> tuple[dict[int, int], int]:
    """Build a mapping from MCLY texture ID → contiguous class index."""
    all_ids: set[int] = set()
    for sample in samples:
        if sample.mcly_texture_ids is None:
            continue
        ids = sample.mcly_texture_ids
        for cy in range(ids.shape[0]):
            for cx in range(ids.shape[1]):
                tex_id = int(ids[cy, cx, 0])
                if tex_id >= 0:
                    all_ids.add(tex_id)

    sorted_ids = sorted(all_ids)
    label_index = {tex_id: idx for idx, tex_id in enumerate(sorted_ids)}
    return label_index, len(sorted_ids)


def build_split_report(
    samples: list[Stage2Sample],
    train_samples: list[Stage2Sample],
    val_samples: list[Stage2Sample],
    target_val_count: int,
    priority_quotas: dict[str, int],
    forced_validation_tiles: list[str],
) -> dict[str, Any]:
    def count_matches(group_name: str, source: list[Stage2Sample]) -> int:
        return sum(1 for sample in source if sample_matches_subset(sample, group_name))

    return {
        "strategy": "quota-aware-stratified-signal-holdout.v1",
        "target_val_count": target_val_count,
        "actual_val_count": len(val_samples),
        "forced_validation_tiles": forced_validation_tiles,
        "val_dataset_counts": build_dataset_counts(val_samples),
        "priority_groups": {
            group_name: {
                "desired_val_count": priority_quotas[group_name],
                "all_samples": count_matches(group_name, samples),
                "train_samples": count_matches(group_name, train_samples),
                "val_samples": count_matches(group_name, val_samples),
            }
            for group_name in VALIDATION_SPLIT_PRIORITY_GROUPS
        },
    }


def build_dataset_counts(samples: list[Stage2Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.dataset_key] = counts.get(sample.dataset_key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def sample_priority_score(sample: Stage2Sample) -> tuple[int, int, int, int, int, int, int, str]:
    return (
        1 if is_native_v10_source(sample) else 0,
        1 if sample_has_pm4_signal(sample) else 0,
        1 if sample_has_mcal_signal(sample) else 0,
        1 if sample_has_mccv_signal(sample) else 0,
        1 if sample_has_normal_signal(sample) else 0,
        1 if sample_has_liquid_signal(sample) else 0,
        1 if sample_has_object_signal(sample) else 0,
        sample.tile_name,
    )


def order_validation_samples(samples: list[Stage2Sample], forced_validation_tiles: set[str] | None = None) -> list[Stage2Sample]:
    forced_validation_tiles = forced_validation_tiles or set()
    return sorted(
        samples,
        key=lambda sample: (1 if sample.tile_name.lower() in forced_validation_tiles else 0, *sample_priority_score(sample)),
        reverse=True,
    )


def sample_catalog_entry(sample: Stage2Sample) -> dict[str, Any]:
    return {
        "tile_name": sample.tile_name,
        "dataset_key": sample.dataset_key,
        "source_schema": sample.source_schema,
        "source_manifest": sample.source_manifest,
        "shard_path": str(sample.path),
        "groups": {
            "native_v10": is_native_v10_source(sample),
            "legacy_only": not is_native_v10_source(sample),
            "pm4_present": sample_has_pm4_signal(sample),
            "mcal_present": sample_has_mcal_signal(sample),
            "mccv_present": sample_has_mccv_signal(sample),
            "normal_present": sample_has_normal_signal(sample),
            "liquid_present": sample_has_liquid_signal(sample),
            "object_present": sample_has_object_signal(sample),
        },
    }


def write_validation_catalog(output_dir: Path, train_samples: list[Stage2Sample], val_samples: list[Stage2Sample]) -> Path:
    catalog_path = output_dir / "validation_samples.json"
    payload = {
        "schema_version": "v10-stage2-validation-samples.v1",
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "val_dataset_counts": build_dataset_counts(val_samples),
        "val_samples": [sample_catalog_entry(sample) for sample in val_samples],
    }
    catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return catalog_path


def split_samples(
    samples: list[Stage2Sample],
    val_fraction: float,
    seed: int,
    forced_validation_tiles: list[str],
) -> tuple[list[Stage2Sample], list[Stage2Sample], dict[str, Any]]:
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError("Need at least two valid NPZ shards to create train and validation splits.")

    forced_validation_lookup = {tile_name.lower() for tile_name in forced_validation_tiles}
    available_tile_names = {sample.tile_name.lower() for sample in shuffled}
    missing_forced_tiles = sorted(tile_name for tile_name in forced_validation_lookup if tile_name not in available_tile_names)
    if missing_forced_tiles:
        raise ValueError(
            "Forced validation tiles were not found in the selected input corpus: " + ", ".join(missing_forced_tiles)
        )

    val_count = max(1, min(len(shuffled) - 1, int(math.ceil(len(shuffled) * val_fraction))))
    if forced_validation_lookup:
        val_count = max(val_count, len(forced_validation_lookup))
        if val_count >= len(shuffled):
            raise ValueError("Forced validation tiles leave no training samples. Reduce the forced set or increase the input corpus.")

    indexed_samples = list(enumerate(shuffled))
    selected_indices: set[int] = set()
    priority_quotas: dict[str, int] = {}
    current_counts = {group_name: 0 for group_name in VALIDATION_SPLIT_PRIORITY_GROUPS}

    for idx, sample in indexed_samples:
        if sample.tile_name.lower() not in forced_validation_lookup:
            continue
        selected_indices.add(idx)
        for group_name in VALIDATION_SPLIT_PRIORITY_GROUPS:
            if sample_matches_subset(sample, group_name):
                current_counts[group_name] += 1

    for group_name in VALIDATION_SPLIT_PRIORITY_GROUPS:
        group_total = sum(1 for _, sample in indexed_samples if sample_matches_subset(sample, group_name))
        if group_total <= 0:
            priority_quotas[group_name] = 0
            continue

        quota = max(1, int(math.ceil(group_total * val_fraction)))
        if group_total > 1:
            quota = min(quota, group_total - 1)
        priority_quotas[group_name] = min(quota, val_count - 1 if val_count > 1 else 1)

    unmet_groups = lambda idx: [
        group_name
        for group_name in VALIDATION_SPLIT_PRIORITY_GROUPS
        if current_counts[group_name] < priority_quotas[group_name] and sample_matches_subset(indexed_samples[idx][1], group_name)
    ]

    while len(selected_indices) < val_count:
        best_index: int | None = None
        best_score: tuple[int, int, int] | None = None
        for idx, sample in indexed_samples:
            if idx in selected_indices:
                continue
            covers = unmet_groups(idx)
            if not covers:
                continue

            score = (
                len(covers),
                sum(priority_quotas[group_name] - current_counts[group_name] for group_name in covers),
                -idx,
            )
            if best_score is None or score > best_score:
                best_index = idx
                best_score = score

        if best_index is None:
            break

        selected_indices.add(best_index)
        for group_name in VALIDATION_SPLIT_PRIORITY_GROUPS:
            if sample_matches_subset(indexed_samples[best_index][1], group_name):
                current_counts[group_name] += 1

    for idx, _ in indexed_samples:
        if len(selected_indices) >= val_count:
            break
        if idx not in selected_indices:
            selected_indices.add(idx)

    val_samples = [sample for idx, sample in indexed_samples if idx in selected_indices]
    train_samples = [sample for idx, sample in indexed_samples if idx not in selected_indices]

    val_samples = order_validation_samples(val_samples, forced_validation_lookup)

    if not train_samples or not val_samples:
        raise RuntimeError("Split logic produced an empty train or validation set.")

    split_report = build_split_report(shuffled, train_samples, val_samples, val_count, priority_quotas, forced_validation_tiles)
    return train_samples, val_samples, split_report


def compute_training_sample_weights(
    samples: list[Stage2Sample],
    native_v10_boost: float,
    rare_signal_boost: float,
) -> list[float]:
    weights: list[float] = []
    for sample in samples:
        weight = 1.0
        if native_v10_boost > 0 and is_native_v10_source(sample):
            weight += native_v10_boost
        if rare_signal_boost > 0:
            if sample_has_pm4_signal(sample):
                weight += rare_signal_boost
            if sample_has_mcal_signal(sample):
                weight += rare_signal_boost
            if sample_has_mccv_signal(sample):
                weight += rare_signal_boost
            if sample_has_normal_signal(sample):
                weight += rare_signal_boost
        weights.append(weight)
    return weights


def make_loader(
    dataset: Stage2Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sample_weights: list[float] | None = None,
) -> DataLoader:
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def maybe_channels_last(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    height_mean: float,
    height_std: float,
    channels_last: bool,
    is_multi_task: bool = False,
    mcal_weight: float = 0.5,
    mcly_weight: float = 0.3,
    hole_weight: float = 0.5,
    grad_accum_steps: int = 1,
    grad_clip: float = 0.0,
    scheduler: Any = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    accumulator = MetricAccumulator()
    autocast_enabled = device.type == "cuda"
    step = 0

    for batch in loader:
        inputs = maybe_channels_last(batch["inputs"].to(device, non_blocking=True), channels_last)
        target_257 = batch["height_257"].to(device, non_blocking=True)
        target_65 = batch["height_65"].to(device, non_blocking=True)
        target_17 = batch["height_17"].to(device, non_blocking=True)

        context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled) if autocast_enabled else torch.enable_grad() if training else torch.no_grad()
        with context:
            if is_multi_task:
                pred_17, pred_65, pred_257, pred_mcal, pred_mcly, pred_hole = model(inputs)
                metric_tensors = compute_metric_tensors(
                    pred_17, pred_65, pred_257, target_17, target_65, target_257,
                    height_mean, height_std,
                    pred_mcal=pred_mcal, target_mcal=batch["mcal_target"].to(device, non_blocking=True),
                    has_mcal=batch["has_mcal"].to(device, non_blocking=True),
                    pred_mcly=pred_mcly, target_mcly=batch["mcly_target"].to(device, non_blocking=True),
                    has_mcly=batch["has_mcly"].to(device, non_blocking=True),
                    pred_hole=pred_hole, target_hole=batch["hole_target"].to(device, non_blocking=True),
                    has_hole=batch["has_hole"].to(device, non_blocking=True),
                    mcal_weight=mcal_weight, mcly_weight=mcly_weight, hole_weight=hole_weight,
                )
            else:
                pred_17, pred_65, pred_257 = model(inputs)
                metric_tensors = compute_metric_tensors(pred_17, pred_65, pred_257, target_17, target_65, target_257, height_mean, height_std)
            loss = metric_tensors["loss"].mean() / grad_accum_steps

        if training:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step += 1
            if step % grad_accum_steps == 0:
                if grad_clip > 0:
                    if scaler is not None and autocast_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler is not None and autocast_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        accumulator.add(metric_tensors)

    report = accumulator.to_report()
    return {key: float(report[key]) for key in MULTI_TASK_METRIC_KEYS if report[key] is not None}


def save_preview(
    model: nn.Module,
    dataset: Stage2Dataset,
    device: torch.device,
    output_path: Path,
    height_mean: float,
    height_std: float,
    channels_last: bool,
    preview_count: int,
    is_multi_task: bool = False,
) -> None:
    if len(dataset) == 0:
        return

    model.eval()
    row_count = min(max(preview_count, 1), len(dataset))
    rows: list[np.ndarray] = []
    preview_rows: list[dict[str, Any]] = []
    all_preds: list[np.ndarray] = []

    for index in range(row_count):
        sample = dataset[index]
        with torch.no_grad():
            inputs = maybe_channels_last(sample["inputs"].unsqueeze(0).to(device), channels_last)
            if is_multi_task:
                pred_17, pred_65, pred_257, _pred_mcal, _pred_mcly, _pred_hole = model(inputs)
            else:
                pred_17, pred_65, pred_257 = model(inputs)

        pred_257_np = (pred_257.squeeze(0).squeeze(0).cpu().numpy() * height_std) + height_mean
        all_preds.append(pred_257_np)

        minimap = (sample["minimap"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        target_257 = (sample["height_257"].squeeze(0).numpy() * height_std) + height_mean
        diff = pred_257_np - target_257

        target_img = height_to_rgb(target_257)
        pred_img = height_to_rgb(pred_257_np)
        diff_img = difference_to_rgb(diff)

        h, w = minimap.shape[:2]
        target_img = resize_preview_image(target_img, w, h)
        pred_img = resize_preview_image(pred_img, w, h)
        diff_img = resize_preview_image(diff_img, w, h)

        row_title = f"Row {index + 1} | {sample['tile_name']} | {sample['dataset_key']}"
        labeled_row = build_labeled_preview_row(
            row_title,
            [minimap, target_img, pred_img, diff_img],
            ["Minimap", "Target Height", "Prediction", "Difference"],
        )

        rows.append(labeled_row)
        preview_rows.append(
            {
                "row": index,
                "tile_name": str(sample["tile_name"]),
                "dataset_key": str(sample["dataset_key"]),
                "source_schema": str(sample["source_schema"]),
                "row_title": row_title,
                "panes": ["minimap", "target_height", "predicted_height", "difference"],
                "difference_legend": {
                    "green": "close match",
                    "red": "prediction higher than target",
                    "blue": "prediction lower than target",
                },
            }
        )

    # Diagnostic: verify predictions are not all identical
    if len(all_preds) > 1:
        pred_variance = max(np.var(pred) for pred in all_preds)
        pred_diff_max = max(np.max(np.abs(all_preds[i] - all_preds[j])) for i in range(len(all_preds)) for j in range(i + 1, len(all_preds)))
        print(f"  [preview] pred_variance={pred_variance:.2f} max_inter_pred_diff={pred_diff_max:.2f}m")
        if pred_diff_max < 1.0:
            print("  WARNING: Preview predictions are nearly identical across tiles! Model may be collapsed.")

    composite = rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)
    Image.fromarray(composite).save(output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": "v10-stage2-preview.v1",
                "row_count": row_count,
                "rows": preview_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize_signal_coverage(samples: Iterable[Stage2Sample]) -> dict[str, dict[str, float | int]]:
    sample_list = list(samples)
    total = len(sample_list)
    counts: dict[str, int] = {}
    for sample in sample_list:
        for key in sorted(sample.available_signal_keys):
            counts[key] = counts.get(key, 0) + 1

    summary: dict[str, dict[str, float | int]] = {}
    for key, count in sorted(counts.items()):
        summary[key] = {
            "count": count,
            "fraction": (count / total) if total > 0 else 0.0,
        }

    return summary


def build_labeled_preview_row(row_title: str, pane_images: list[np.ndarray], pane_labels: list[str]) -> np.ndarray:
    if not pane_images:
        raise ValueError("Expected at least one pane image when building preview rows.")

    pane_height = pane_images[0].shape[0]
    pane_widths = [image.shape[1] for image in pane_images]
    row_width = sum(pane_widths)
    row_height = PREVIEW_ROW_TITLE_HEIGHT + PREVIEW_PANE_LABEL_HEIGHT + pane_height

    row_image = Image.new("RGB", (row_width, row_height), color=(24, 24, 24))
    draw = ImageDraw.Draw(row_image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, row_width, PREVIEW_ROW_TITLE_HEIGHT), fill=(32, 32, 32))
    draw.text((PREVIEW_LABEL_PADDING, 2), row_title, fill=(255, 255, 255), font=font)

    x_offset = 0
    for pane_image, pane_label in zip(pane_images, pane_labels, strict=False):
        pane_width = pane_image.shape[1]
        label_box = (x_offset, PREVIEW_ROW_TITLE_HEIGHT, x_offset + pane_width, PREVIEW_ROW_TITLE_HEIGHT + PREVIEW_PANE_LABEL_HEIGHT)
        draw.rectangle(label_box, fill=(48, 48, 48))

        text_bbox = draw.textbbox((0, 0), pane_label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x_offset + max(PREVIEW_LABEL_PADDING, (pane_width - text_width) // 2)
        text_y = PREVIEW_ROW_TITLE_HEIGHT + max(1, (PREVIEW_PANE_LABEL_HEIGHT - text_height) // 2)
        draw.text((text_x, text_y), pane_label, fill=(255, 255, 255), font=font)

        row_image.paste(Image.fromarray(pane_image), (x_offset, PREVIEW_ROW_TITLE_HEIGHT + PREVIEW_PANE_LABEL_HEIGHT))
        x_offset += pane_width

    return np.asarray(row_image)


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
    ablation_groups = parse_validation_ablation_groups(args.validation_ablation_groups)
    forced_validation_tiles = parse_tile_name_list(args.force_validation_tiles)

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

    train_samples, val_samples, split_report = split_samples(samples, args.val_fraction, args.seed, forced_validation_tiles)
    height_values = np.concatenate([sample.height_257.reshape(-1) for sample in train_samples], axis=0)
    height_mean = float(np.mean(height_values))
    height_std = float(np.std(height_values))
    if height_std < 1e-5:
        height_std = 1.0

    model_variant = resolve_model_variant(args)
    is_multi_task = model_variant == "multi_task_v3"

    # Build MCLY label index for multi-task training
    mcly_label_index: dict[int, int] = {}
    mcly_num_classes = 0
    if is_multi_task:
        mcly_label_index, mcly_num_classes = build_mcly_label_index(train_samples)
        print(f"MCLY label index: {mcly_num_classes} classes from training set")

    train_dataset = Stage2Dataset(
        train_samples, height_mean, height_std, args.signal_dropout,
        mcly_label_index=mcly_label_index, mcly_num_classes=mcly_num_classes,
    )
    val_dataset = Stage2Dataset(
        val_samples, height_mean, height_std, signal_dropout=0.0,
        mcly_label_index=mcly_label_index, mcly_num_classes=mcly_num_classes,
    )
    train_sample_weights = compute_training_sample_weights(train_samples, args.native_v10_boost, args.rare_signal_boost)
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        sample_weights=train_sample_weights,
    )
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    total_signal_coverage = summarize_signal_coverage(samples)
    train_signal_coverage = summarize_signal_coverage(train_samples)
    val_signal_coverage = summarize_signal_coverage(val_samples)
    validation_catalog_path = write_validation_catalog(output_dir, train_samples, val_samples)

    sample_channels = train_dataset[0]["inputs"].shape[0]

    device = torch.device(args.device)

    if args.evaluate_checkpoint:
        checkpoint_path = Path(args.evaluate_checkpoint).resolve()
        checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)
        checkpoint_height_mean = float(checkpoint_payload.get("height_mean", height_mean))
        checkpoint_height_std = float(checkpoint_payload.get("height_std", height_std))
        checkpoint_input_channels = int(checkpoint_payload.get("input_channels", sample_channels))
        checkpoint_model_variant = resolve_model_variant(args, checkpoint_payload)
        checkpoint_is_multi_task = checkpoint_model_variant == "multi_task_v3"
        checkpoint_mcly_classes = int(checkpoint_payload.get("mcly_num_classes", mcly_num_classes))

        model = build_model(checkpoint_model_variant, checkpoint_input_channels, checkpoint_mcly_classes).to(device)
        if args.channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        load_model_state(model, checkpoint_payload["model_state"])

        validation_analysis = evaluate_validation_analysis(
            model, val_loader, device,
            checkpoint_height_mean, checkpoint_height_std,
            args.channels_last, ablation_groups,
            is_multi_task=checkpoint_is_multi_task,
            mcal_weight=args.mcal_loss_weight,
            mcly_weight=args.mcly_loss_weight,
            hole_weight=args.hole_loss_weight,
        )
        save_preview(
            model, val_dataset, device,
            previews_dir / "checkpoint_eval.png",
            checkpoint_height_mean, checkpoint_height_std,
            args.channels_last, args.preview_count,
            is_multi_task=checkpoint_is_multi_task,
        )

        summary = {
            "input": str(input_path),
            "sample_count": len(samples),
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "input_channels": checkpoint_input_channels,
            "model_variant": checkpoint_model_variant,
            "height_mean": checkpoint_height_mean,
            "height_std": checkpoint_height_std,
            "mcly_num_classes": checkpoint_mcly_classes,
            "train_sampler": {
                "native_v10_boost": args.native_v10_boost,
                "rare_signal_boost": args.rare_signal_boost,
                "min_weight": min(train_sample_weights),
                "max_weight": max(train_sample_weights),
            },
            "preview_count": min(max(args.preview_count, 1), len(val_dataset)),
            "signal_coverage": {
                "all_samples": total_signal_coverage,
                "train_samples": train_signal_coverage,
                "val_samples": val_signal_coverage,
            },
            "split_report": split_report,
            "validation_catalog": str(validation_catalog_path),
            "evaluated_checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint_payload.get("epoch", 0)),
            "validation_analysis": validation_analysis,
            "history": checkpoint_payload.get("history", []),
        }
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    model = build_model(model_variant, sample_channels, mcly_num_classes).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    use_compile = bool(args.use_compile and hasattr(torch, "compile") and device.type == "cuda")
    if args.use_compile and not use_compile:
        print("torch.compile disabled for this run because the selected device is not CUDA.")

    if use_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Resume support ───────────────────────────────────────────────
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        print(f"Resuming from {resume_path}")
        payload = load_checkpoint_payload(resume_path, device)
        load_model_state(model, payload["model_state"])
        if "optimizer_state" in payload and payload["optimizer_state"]:
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        history = list(payload.get("history", []))
        best_val_loss = float(payload.get("best_val_loss", float("inf")))
        print(f"  Resuming at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, history_len={len(history)}")

    # ── LR scheduler: linear warmup then cosine annealing ────────────
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.warmup_epochs
    if args.no_cosine_scheduler:
        scheduler = None
    else:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Model: {model_variant} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Input channels: {sample_channels}")
    print(f"Batch size: {args.batch_size} | Grad accum: {args.gradient_accumulation_steps} | Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"LR: {args.learning_rate} | Weight decay: {args.weight_decay} | Warmup epochs: {args.warmup_epochs}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    import time

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler, device,
            height_mean, height_std, args.channels_last,
            is_multi_task=is_multi_task,
            mcal_weight=args.mcal_loss_weight,
            mcly_weight=args.mcly_loss_weight,
            hole_weight=args.hole_loss_weight,
            grad_accum_steps=args.gradient_accumulation_steps,
            grad_clip=args.gradient_clip,
            scheduler=scheduler,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, None, None, device,
                height_mean, height_std, args.channels_last,
                is_multi_task=is_multi_task,
                mcal_weight=args.mcal_loss_weight,
                mcly_weight=args.mcly_loss_weight,
                hole_weight=args.hole_loss_weight,
            )

        epoch_time = time.time() - epoch_start
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)

        new_best = val_metrics["loss"] < best_val_loss
        best_marker = " >>> NEW BEST <<<" if new_best else ""

        lr_str = ""
        if scheduler is not None:
            lr_str = f" | lr {scheduler.get_last_lr()[0]:.2e}"

        gpu_mem = ""
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            gpu_mem = f" | gpu {alloc:.1f}GB/{reserved:.1f}GB"

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val mae {val_metrics['mae_m']:.2f}m | "
            f"val rmse {val_metrics['rmse_m']:.2f}m"
            f"{best_marker}{lr_str}{gpu_mem} | {epoch_time:.1f}s"
        )
        if is_multi_task:
            print(
                f"         | "
                f"height {val_metrics['height_loss']:.4f} | "
                f"mcal {val_metrics['mcal_loss']:.4f} | "
                f"mcly {val_metrics['mcly_loss']:.4f} | "
                f"hole {val_metrics['hole_loss']:.4f}"
            )

        incremental_summary = {
            "input": str(input_path),
            "sample_count": len(samples),
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "input_channels": sample_channels,
            "model_variant": model_variant,
            "height_mean": height_mean,
            "height_std": height_std,
            "mcly_num_classes": mcly_num_classes,
            "best_val_loss_so_far": best_val_loss,
            "history": history,
        }
        (output_dir / "metrics.json").write_text(json.dumps(incremental_summary, indent=2), encoding="utf-8")

        last_checkpoint = checkpoints_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "height_mean": height_mean,
                "height_std": height_std,
                "input_channels": sample_channels,
                "model_variant": model_variant,
                "mcly_num_classes": mcly_num_classes,
                "history": history,
                "best_val_loss": best_val_loss,
            },
            last_checkpoint,
        )

        if new_best:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "height_mean": height_mean,
                    "height_std": height_std,
                    "input_channels": sample_channels,
                    "model_variant": model_variant,
                    "mcly_num_classes": mcly_num_classes,
                    "history": history,
                    "best_val_loss": best_val_loss,
                },
                checkpoints_dir / "best.pt",
            )
            save_preview(
                model, val_dataset, device,
                previews_dir / f"epoch_{epoch:03d}_best.png",
                height_mean, height_std,
                args.channels_last, args.preview_count,
                is_multi_task=is_multi_task,
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "height_mean": height_mean,
                    "height_std": height_std,
                    "input_channels": sample_channels,
                    "model_variant": model_variant,
                    "mcly_num_classes": mcly_num_classes,
                    "history": history,
                    "best_val_loss": best_val_loss,
                },
                checkpoints_dir / f"epoch_{epoch:03d}.pt",
            )

    del train_loader
    del val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    analysis_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=0)

    best_checkpoint_path = checkpoints_dir / "best.pt"
    best_checkpoint = load_checkpoint_payload(best_checkpoint_path, device)
    best_model_variant = str(best_checkpoint.get("model_variant") or model_variant)
    best_is_multi_task = best_model_variant == "multi_task_v3"
    best_mcly_classes = int(best_checkpoint.get("mcly_num_classes", mcly_num_classes))
    analysis_model = build_model(best_model_variant, sample_channels, best_mcly_classes).to(device)
    if args.channels_last and device.type == "cuda":
        analysis_model = analysis_model.to(memory_format=torch.channels_last)
    load_model_state(analysis_model, best_checkpoint["model_state"])
    validation_analysis = evaluate_validation_analysis(
        analysis_model, analysis_loader, device,
        height_mean, height_std,
        args.channels_last, ablation_groups,
        is_multi_task=best_is_multi_task,
        mcal_weight=args.mcal_loss_weight,
        mcly_weight=args.mcly_loss_weight,
        hole_weight=args.hole_loss_weight,
    )

    summary = {
        "input": str(input_path),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "input_channels": sample_channels,
        "model_variant": best_model_variant,
        "height_mean": height_mean,
        "height_std": height_std,
        "mcly_num_classes": best_mcly_classes,
        "train_sampler": {
            "native_v10_boost": args.native_v10_boost,
            "rare_signal_boost": args.rare_signal_boost,
            "min_weight": min(train_sample_weights),
            "max_weight": max(train_sample_weights),
        },
        "preview_count": min(max(args.preview_count, 1), len(val_dataset)),
        "signal_coverage": {
            "all_samples": total_signal_coverage,
            "train_samples": train_signal_coverage,
            "val_samples": val_signal_coverage,
        },
        "split_report": split_report,
        "validation_catalog": str(validation_catalog_path),
        "validation_analysis": validation_analysis,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
