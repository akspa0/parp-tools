"""Shared V16.1 training entrypoint used by task-specific wrappers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402
from torch.utils.data._utils.collate import default_collate  # noqa: E402

from harvester.v16_1_dataset import V161Dataset  # noqa: E402
from harvester.v16_1_models import (  # noqa: E402
    V161HeightModel,
    V161HolesModel,
    V161LiquidModel,
    V161NormalHeightCombinedModel,
    V161NormalHeightModel,
    V161NormalModel,
    V161NormalRefiner,
    V161TexcompModel,
    recompose_from_mcly_alpha,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_MODELS_ROOT = _PROJECT_ROOT / "models" / "v16_1"
_PANEL_SIZE = 256
_PANEL_LABEL_HEIGHT = 18
_ROW_LABEL_HEIGHT = 18
_DIFFICULTY_BUCKETS = ("easy", "medium", "hard", "pathological")
_BUCKET_SAMPLING_PROFILES: dict[str, dict[str, float]] = {
    "uniform": {bucket: 1.0 for bucket in _DIFFICULTY_BUCKETS},
    "v16_1_1_normal": {
        "easy": 1.0,
        "medium": 1.75,
        "hard": 3.5,
        "pathological": 1.25,
    },
}
_DEFAULT_AUTOTUNE_BATCH_CANDIDATES = (8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
_NORMAL_VARIANTS = (
    "v16_1_1_base",
    "v16_1_2_refiner",
    "v16_1_3_height",
    "v17_hybrid",
    "v17_1_normals",
)


def _resolve_normal_variant(args: argparse.Namespace, task_name: str) -> tuple[str, bool, bool]:
    if task_name != "normal":
        return "not_normal_task", bool(getattr(args, "height_channel", False)), False

    variant = str(getattr(args, "normal_variant", "v17_1_normals"))
    if variant not in _NORMAL_VARIANTS:
        raise RuntimeError(f"Unknown --normal-variant: {variant}")

    expected_height = variant in {"v16_1_3_height", "v17_hybrid"}
    expected_refiner = variant in {"v16_1_2_refiner", "v17_hybrid"}

    manual_height = getattr(args, "height_channel", None)
    if manual_height is not None and bool(manual_height) != bool(expected_height):
        raise RuntimeError(
            f"--normal-variant {variant} requires height_channel={expected_height}, "
            f"but CLI override set height_channel={bool(manual_height)}"
        )

    manual_refiner_disabled = getattr(args, "refiner_disabled", None)
    if manual_refiner_disabled is not None and bool(manual_refiner_disabled) != (not expected_refiner):
        raise RuntimeError(
            f"--normal-variant {variant} requires refiner_enabled={expected_refiner}, "
            f"but CLI override set refiner_disabled={bool(manual_refiner_disabled)}"
        )

    return variant, expected_height, expected_refiner


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return int(requested)
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def _resolve_persistent_workers(requested: bool | None, num_workers: int) -> bool:
    if num_workers <= 0:
        return False
    if requested is None:
        return True
    return bool(requested)


def _resolve_autotune_batch_candidates(base_batch_size: int, requested: list[int] | None) -> list[int]:
    raw = list(requested) if requested else list(_DEFAULT_AUTOTUNE_BATCH_CANDIDATES)
    raw.append(int(base_batch_size))
    candidates = sorted({int(value) for value in raw if int(value) > 0})
    return [value for value in candidates if value >= int(base_batch_size)]


def _normalize_bucket_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in _DIFFICULTY_BUCKETS:
        return label
    return "unbucketed"


def _bucket_sampling_weights(profile: str | None) -> dict[str, float] | None:
    if profile is None:
        return None
    return _BUCKET_SAMPLING_PROFILES.get(str(profile))


class _DeterministicEpochSampler(Sampler[int]):
    """Deterministic sampler with optional per-epoch subset rotation."""

    def __init__(
        self,
        n: int,
        seed: int,
        order_log_path: Path | None = None,
        bucket_log_path: Path | None = None,
        epoch_size: int | None = None,
        build_labels: list[str] | None = None,
        build_balanced: bool = False,
        bucket_labels: list[str] | None = None,
        bucket_sampling_profile: str | None = None,
        sample_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._n = int(n)
        self._seed = int(seed)
        self._epoch = 0
        self._order_log_path = order_log_path
        self._bucket_log_path = bucket_log_path
        self._epoch_size = None if epoch_size is None or int(epoch_size) <= 0 else min(int(epoch_size), self._n)
        self._build_labels = list(build_labels) if build_labels is not None else None
        self._build_balanced = bool(build_balanced)
        self._bucket_labels = [_normalize_bucket_label(label) for label in bucket_labels] if bucket_labels is not None else None
        self._bucket_sampling_profile = bucket_sampling_profile
        self._bucket_sampling_weights = _bucket_sampling_weights(bucket_sampling_profile)
        self._sample_rows = [dict(row) for row in sample_rows] if sample_rows is not None else None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _sample_subset(self, rng: np.random.RandomState) -> list[int]:
        if self._n <= 0:
            return []
        if self._epoch_size is None or self._epoch_size >= self._n:
            return list(range(self._n))
        return _sample_positions(
            self._n,
            seed=self._seed + self._epoch,
            take=self._epoch_size,
            build_labels=self._build_labels,
            build_balanced=self._build_balanced,
            bucket_labels=self._bucket_labels,
            bucket_sampling_weights=self._bucket_sampling_weights,
        )

    def __iter__(self):
        rng = np.random.RandomState(self._seed + self._epoch)
        selected = self._sample_subset(rng)
        order = list(selected)
        rng.shuffle(order)
        available_bucket_counts = _bucket_counts(self._bucket_labels or [])
        selected_bucket_labels = [self._bucket_labels[pos] for pos in selected] if self._bucket_labels is not None else []
        ordered_bucket_labels = [self._bucket_labels[pos] for pos in order] if self._bucket_labels is not None else []
        selected_rows = [self._sample_rows[pos] for pos in selected] if self._sample_rows is not None else None
        order_rows = [self._sample_rows[pos] for pos in order] if self._sample_rows is not None else None
        if self._order_log_path is not None:
            payload = {
                "epoch": self._epoch,
                "num_samples": self._n,
                "epoch_size": len(order),
                "bucket_sampling_profile": self._bucket_sampling_profile,
                "bucket_sampling_weights": self._bucket_sampling_weights,
                "available_bucket_counts": available_bucket_counts,
                "selected_bucket_counts": _bucket_counts(selected_bucket_labels),
                "ordered_bucket_counts": _bucket_counts(ordered_bucket_labels),
                "selected_positions_sha256": hashlib.sha256(np.asarray(selected, dtype=np.int32).tobytes()).hexdigest(),
                "selected_positions": selected,
                "selected_rows": selected_rows,
                "order_sha256": hashlib.sha256(np.asarray(order, dtype=np.int32).tobytes()).hexdigest(),
                "order": order,
                "order_rows": order_rows,
            }
            with self._order_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        if self._bucket_log_path is not None:
            payload = {
                "epoch": self._epoch,
                "num_samples": self._n,
                "epoch_size": len(order),
                "bucket_sampling_profile": self._bucket_sampling_profile,
                "bucket_sampling_weights": self._bucket_sampling_weights,
                "available_bucket_counts": available_bucket_counts,
                "selected_bucket_counts": _bucket_counts(selected_bucket_labels),
                "selected_build_counts": _count_string_values([row.get("build", "unknown") for row in selected_rows] if selected_rows is not None else []),
            }
            with self._bucket_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        return iter(order)

    def __len__(self) -> int:
        return self._epoch_size if self._epoch_size is not None else self._n


def _sample_positions(
    n: int,
    seed: int,
    take: int,
    build_labels: list[str] | None = None,
    build_balanced: bool = True,
    bucket_labels: list[str] | None = None,
    bucket_sampling_weights: dict[str, float] | None = None,
) -> list[int]:
    if n <= 0 or take <= 0:
        return []
    if take >= n:
        return list(range(n))

    rng = np.random.RandomState(int(seed))
    if not build_balanced or not build_labels or len(build_labels) != n:
        return _sample_weighted_positions(
            list(range(n)),
            take=take,
            rng=rng,
            bucket_labels=bucket_labels,
            bucket_sampling_weights=bucket_sampling_weights,
        )

    by_build: dict[str, list[int]] = {}
    for pos, build in enumerate(build_labels):
        by_build.setdefault(str(build), []).append(pos)

    build_order = sorted(by_build.keys())
    rng.shuffle(build_order)
    out: list[int] = []
    while len(out) < take:
        progressed = False
        for build in build_order:
            items = by_build[build]
            if not items:
                continue
            chosen = _sample_weighted_positions(
                items,
                take=1,
                rng=rng,
                bucket_labels=bucket_labels,
                bucket_sampling_weights=bucket_sampling_weights,
            )
            if not chosen:
                continue
            selected = chosen[0]
            items.remove(selected)
            out.append(selected)
            progressed = True
            if len(out) >= take:
                break
        if not progressed:
            break

    if len(out) < take:
        remaining: list[int] = []
        for build in build_order:
            remaining.extend(by_build[build])
        if remaining:
            extra = _sample_weighted_positions(
                remaining,
                take=take - len(out),
                rng=rng,
                bucket_labels=bucket_labels,
                bucket_sampling_weights=bucket_sampling_weights,
            )
            out.extend(extra)

    rng.shuffle(out)
    return out[:take]


def _count_string_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _bucket_counts(bucket_labels: list[str]) -> dict[str, int]:
    counts = _count_string_values([_normalize_bucket_label(label) for label in bucket_labels if str(label or "").strip()])
    return {bucket: counts.get(bucket, 0) for bucket in _DIFFICULTY_BUCKETS if counts.get(bucket, 0) > 0}


def _sample_weighted_positions(
    positions: list[int],
    *,
    take: int,
    rng: np.random.RandomState,
    bucket_labels: list[str] | None = None,
    bucket_sampling_weights: dict[str, float] | None = None,
) -> list[int]:
    if take <= 0 or not positions:
        return []
    if take >= len(positions):
        return list(positions)

    weights = None
    if bucket_labels is not None and bucket_sampling_weights:
        raw = np.asarray(
            [
                max(float(bucket_sampling_weights.get(_normalize_bucket_label(bucket_labels[pos]), 1.0)), 0.0)
                for pos in positions
            ],
            dtype=np.float64,
        )
        if float(raw.sum()) > 0.0 and not np.allclose(raw, raw[0]):
            weights = raw / raw.sum()
    chosen_idx = rng.choice(len(positions), size=take, replace=False, p=weights)
    return [positions[int(idx)] for idx in chosen_idx.tolist()]


def _pool_row(entry: dict[str, Any], subset_pos: int, split_pos: int) -> dict[str, Any]:
    return {
        "subset_pos": int(subset_pos),
        "split_pos": int(split_pos),
        "build": str(entry.get("_build", "unknown")),
        "map": entry.get("map"),
        "tile_id": int(entry.get("tile_id", -1)),
        "tile_x": int(entry.get("tile_x", -1) if entry.get("tile_x") is not None else -1),
        "tile_y": int(entry.get("tile_y", -1) if entry.get("tile_y") is not None else -1),
        "height_mean": float(entry.get("height_mean", 0.0) or 0.0),
        "height_std": float(entry.get("height_std", 0.0) or 0.0),
        "has_normal_xyz": bool(entry.get("has_normal_xyz", False)),
        "has_alpha_256": bool(entry.get("has_alpha_256", False)),
        "has_liquid_mask": bool(entry.get("has_liquid_mask", False)),
        "has_mcly_texture_ids": bool(entry.get("has_mcly_texture_ids", False)),
        "n_mddf": int(entry.get("n_mddf", 0) or 0),
        "n_modf": int(entry.get("n_modf", 0) or 0),
        "difficulty_bucket": str(entry.get("_curation_difficulty_bucket", "")),
        "difficulty_rank": int(entry.get("_curation_difficulty_rank", -1)),
        "quality_score": float(entry.get("_curation_quality_score", 0.0) or 0.0),
        "usefulness_score": float(entry.get("_curation_usefulness_score", 0.0) or 0.0),
        "difficulty_score": float(entry.get("_curation_difficulty_score", 0.0) or 0.0),
    }


def _apply_dataset_pool(
    ds: V161Dataset,
    split: str,
    max_tiles: int,
    seed: int,
    evidence_dir: Path,
    build_balanced: bool = True,
) -> dict[str, Any]:
    available = len(ds._indices)
    build_labels = [
        str(ds._index_entries[global_idx].get("_build", "unknown"))
        for global_idx in ds._indices
    ]
    bucket_labels = [
        _normalize_bucket_label(ds._index_entries[global_idx].get("_curation_difficulty_bucket", ""))
        for global_idx in ds._indices
    ]
    selected_positions = _sample_positions(
        available,
        seed=seed,
        take=max_tiles if max_tiles > 0 else available,
        build_labels=build_labels,
        build_balanced=build_balanced,
    )
    selected_global_indices = [ds._indices[pos] for pos in selected_positions]
    ds._indices = selected_global_indices

    rows: list[dict[str, Any]] = []
    build_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for subset_pos, split_pos in enumerate(selected_positions):
        entry = ds._index_entries[selected_global_indices[subset_pos]]
        row = _pool_row(entry, subset_pos=subset_pos, split_pos=split_pos)
        rows.append(row)
        build = str(row["build"])
        build_counts[build] = build_counts.get(build, 0) + 1
        bucket = _normalize_bucket_label(row.get("difficulty_bucket", ""))
        if bucket != "unbucketed":
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    selection_jsonl = evidence_dir / f"{split}_pool_selection.jsonl"
    with selection_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "split": split,
        "available_tiles": int(available),
        "selected_tiles": int(len(ds._indices)),
        "requested_max_tiles": int(max_tiles),
        "build_balanced": bool(build_balanced),
        "seed": int(seed),
        "selected_positions_sha256": hashlib.sha256(
            np.asarray(selected_positions, dtype=np.int32).tobytes()
        ).hexdigest(),
        "build_tile_counts": build_counts,
        "available_bucket_counts": _bucket_counts(bucket_labels),
        "selected_bucket_counts": dict(sorted(bucket_counts.items())),
        "selection_jsonl": str(selection_jsonl),
    }
    (evidence_dir / f"{split}_pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _apply_dataset_limit(ds: V161Dataset, max_samples: int) -> None:
    if max_samples > 0:
        ds._indices = ds._indices[: min(int(max_samples), len(ds._indices))]


def _autotune_batch_size(
    *,
    task: "TaskSpec",
    task_name: str,
    train_ds: V161Dataset,
    device: torch.device,
    args: argparse.Namespace,
    evidence_dir: Path,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "autotune_batch_size", False)):
        return None
    if device.type != "cuda":
        print("Autotune: skipped because device is not CUDA.", flush=True)
        return None
    if float(args.target_vram_gb) <= 0.0:
        print("Autotune: skipped because --target-vram-gb <= 0.", flush=True)
        return None
    if len(train_ds) <= 0:
        print("Autotune: skipped because train dataset is empty.", flush=True)
        return None

    base_batch_size = int(args.batch_size)
    candidates = _resolve_autotune_batch_candidates(base_batch_size, args.autotune_batch_candidates)
    if not candidates:
        return None

    safety_factor = float(getattr(args, "autotune_safety_factor", 0.0))
    if safety_factor <= 0.0:
        safety_factor = 0.72 if (hasattr(torch, "compile") and not args.no_compile) else 0.82
    effective_target_vram_gb = float(args.target_vram_gb) * float(safety_factor)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    results: list[dict[str, Any]] = []
    chosen_batch_size = base_batch_size
    reached_limit = False

    print(
        "Autotune: probing batch-size ladder "
        f"{candidates} against target_vram_gb={float(args.target_vram_gb):.2f} "
        f"(effective_target={effective_target_vram_gb:.2f}GB, safety_factor={safety_factor:.2f})",
        flush=True,
    )

    def _cleanup_probe_state() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

    warmup_steps = max(1, int(getattr(args, "autotune_probe_warmup_steps", 2)))
    measure_steps = max(1, int(getattr(args, "autotune_probe_measure_steps", 3)))
    total_steps = warmup_steps + measure_steps

    for candidate in candidates:
        probe_batch_size = min(int(candidate), len(train_ds))
        probe_model = None
        probe_optimizer = None
        probe_scaler = None
        probe_loss = None
        probe_metrics = None
        peak_alloc_gb = None
        peak_reserved_gb = None
        measured_alloc_gb = None
        measured_reserved_gb = None
        status = "ok"
        fits_target = False

        _cleanup_probe_state()
        try:
            probe_model = task.model_factory().to(device)
            if task_name == "normal" and bool(getattr(args, "resolved_height_channel", getattr(args, "height_channel", False))):
                probe_model = V161NormalHeightModel().to(device)
            if bool(hasattr(torch, "compile") and not args.no_compile):
                try:
                    probe_model = torch.compile(probe_model)
                except Exception:
                    pass
            probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            probe_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            probe_model.train()
            measured_allocs: list[float] = []
            measured_reserveds: list[float] = []

            for step in range(total_steps):
                probe_batch = default_collate([train_ds[(step * probe_batch_size + idx) % len(train_ds)] for idx in range(probe_batch_size)])
                probe_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    probe_loss, probe_metrics, _probe_outputs = task.loss_fn(probe_model, probe_batch, device, args)
                probe_scaler.scale(probe_loss).backward()
                probe_scaler.unscale_(probe_optimizer)
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), max_norm=1.0)
                probe_scaler.step(probe_optimizer)
                probe_scaler.update()
                del probe_batch
                del probe_loss
                del probe_metrics
                del _probe_outputs

                torch.cuda.synchronize(device)
                current_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                current_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024.0 ** 3)
                peak_alloc_gb = current_alloc_gb if peak_alloc_gb is None else max(peak_alloc_gb, current_alloc_gb)
                peak_reserved_gb = current_reserved_gb if peak_reserved_gb is None else max(peak_reserved_gb, current_reserved_gb)
                if step + 1 == warmup_steps:
                    # Drop compile/warmup transients before measured-window budgeting.
                    torch.cuda.reset_peak_memory_stats(device)
                elif step >= warmup_steps:
                    measured_allocs.append(current_alloc_gb)
                    measured_reserveds.append(current_reserved_gb)

            if measured_allocs:
                measured_alloc_gb = max(measured_allocs)
            if measured_reserveds:
                measured_reserved_gb = max(measured_reserveds)

            torch.cuda.synchronize(device)
            decision_reserved_gb = measured_reserved_gb if measured_reserved_gb is not None else peak_reserved_gb
            fits_target = bool(decision_reserved_gb is not None and decision_reserved_gb <= effective_target_vram_gb)
            if fits_target:
                chosen_batch_size = int(candidate)
            else:
                status = "above_target"
                reached_limit = True
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                status = "oom"
                reached_limit = True
            else:
                raise
        finally:
            del probe_scaler
            del probe_optimizer
            del probe_model
            _cleanup_probe_state()

        result = {
            "candidate_batch_size": int(candidate),
            "probe_batch_size": int(probe_batch_size),
            "status": status,
            "fits_target": bool(fits_target),
            "peak_alloc_gb": peak_alloc_gb,
            "peak_reserved_gb": peak_reserved_gb,
            "measured_alloc_gb": measured_alloc_gb,
            "measured_reserved_gb": measured_reserved_gb,
            "warmup_steps": int(warmup_steps),
            "measure_steps": int(measure_steps),
        }
        results.append(result)
        reserved_text = "n/a" if peak_reserved_gb is None else f"{peak_reserved_gb:.2f}GB"
        alloc_text = "n/a" if peak_alloc_gb is None else f"{peak_alloc_gb:.2f}GB"
        measured_reserved_text = "n/a" if measured_reserved_gb is None else f"{measured_reserved_gb:.2f}GB"
        print(
            f"Autotune: batch-size {candidate} -> status={status} "
            f"alloc_peak={alloc_text} reserved_peak={reserved_text} measured_reserved={measured_reserved_text}",
            flush=True,
        )
        if reached_limit:
            break

    tuned_epoch_tiles = int(args.train_epoch_tiles)
    original_epoch_tiles = int(args.train_epoch_tiles)
    original_batch_size = int(base_batch_size)
    if bool(getattr(args, "autotune_keep_epoch_steps", True)) and original_epoch_tiles > 0 and original_batch_size > 0:
        original_steps = max(1, math.ceil(original_epoch_tiles / original_batch_size))
        tuned_epoch_tiles = min(len(train_ds), chosen_batch_size * original_steps)

    payload = {
        "enabled": True,
        "target_vram_gb": float(args.target_vram_gb),
        "effective_target_vram_gb": effective_target_vram_gb,
        "autotune_safety_factor": safety_factor,
        "compile_enabled_for_run": bool(hasattr(torch, "compile") and not args.no_compile),
        "original_batch_size": int(base_batch_size),
        "chosen_batch_size": int(chosen_batch_size),
        "original_train_epoch_tiles": int(original_epoch_tiles),
        "tuned_train_epoch_tiles": int(tuned_epoch_tiles),
        "keep_epoch_steps": bool(getattr(args, "autotune_keep_epoch_steps", True)),
        "candidate_results": results,
    }
    (evidence_dir / "batch_autotune.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.batch_size = int(chosen_batch_size)
    if tuned_epoch_tiles > 0:
        args.train_epoch_tiles = int(tuned_epoch_tiles)
    print(
        f"Autotune: selected batch-size={args.batch_size} "
        f"train_epoch_tiles={int(args.train_epoch_tiles)}",
        flush=True,
    )
    return payload


def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_map * mask).sum() / mask.sum().clamp_min(1e-8)


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target).abs(), weight)


def _weighted_l2(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target) ** 2, weight)


def _resize_weight(weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(weight.shape[-2:]) == tuple(size):
        return weight
    return F.interpolate(weight, size=size, mode="bilinear", align_corners=False)


def _normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    return ((normals.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0).astype(np.uint8)


def _draw_text_strip(img: Image.Image, text: str, height: int) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), color=(0, 0, 0))
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (canvas.width, height - 1)], fill=(14, 14, 14))
    draw.text((4, 3), str(text), fill=(240, 240, 240))
    return canvas


def _compose_horizontal_panel(panels: list[tuple[str, torch.Tensor]]) -> Image.Image:
    images: list[Image.Image] = []
    for label, tensor in panels:
        arr = _to_uint8_hwc(tensor)
        img = Image.fromarray(arr)
        if img.size != (_PANEL_SIZE, _PANEL_SIZE):
            img = img.resize((_PANEL_SIZE, _PANEL_SIZE), Image.Resampling.BILINEAR)
        img = _draw_text_strip(img, label, _PANEL_LABEL_HEIGHT)
        images.append(img)
    canvas = Image.new("RGB", (_PANEL_SIZE * len(images), _PANEL_SIZE + _PANEL_LABEL_HEIGHT), color=(0, 0, 0))
    for idx, img in enumerate(images):
        canvas.paste(img, (idx * _PANEL_SIZE, 0))
    return canvas


def _save_horizontal_panel(panels: list[tuple[str, torch.Tensor]], out_path: Path) -> None:
    canvas = _compose_horizontal_panel(panels)
    canvas.save(out_path)


def _save_preview_grid(rows: list[list[tuple[str, torch.Tensor]]], out_path: Path, row_titles: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError("Cannot save preview grid with no rows.")
    row_images = []
    for idx, row in enumerate(rows):
        row_img = _compose_horizontal_panel(row)
        if row_titles is not None:
            row_img = _draw_text_strip(row_img, row_titles[idx], _ROW_LABEL_HEIGHT)
        row_images.append(row_img)
    width = max(img.width for img in row_images)
    height = sum(img.height for img in row_images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for img in row_images:
        canvas.paste(img, (0, y))
        y += img.height
    canvas.save(out_path)


def _meta_value(x: Any, idx: int) -> Any:
    if isinstance(x, torch.Tensor):
        return x[idx].item()
    if isinstance(x, np.ndarray):
        return x[idx]
    if isinstance(x, (list, tuple)):
        return x[idx]
    return x


def _preview_row_title(batch: dict[str, Any], idx: int) -> str:
    build = _meta_value(batch.get("meta_build", "unknown"), idx)
    map_name = _meta_value(batch.get("meta_map", ""), idx)
    tile_id = _meta_value(batch.get("meta_tile_id", -1), idx)
    tile_x = _meta_value(batch.get("meta_tile_x", -1), idx)
    tile_y = _meta_value(batch.get("meta_tile_y", -1), idx)
    return f"{build} | {map_name} | tile={tile_id} | ({tile_x},{tile_y})"


def _coarse_type_to_rgb(type_grid: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.45, 0.9],
            [0.1, 0.7, 0.95],
            [0.9, 0.35, 0.1],
            [0.25, 0.85, 0.2],
        ],
        dtype=torch.float32,
        device=type_grid.device,
    )
    rgb = palette[type_grid.long().clamp(0, 4)]
    return rgb.permute(2, 0, 1)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    model_factory: Callable[[], torch.nn.Module]
    loss_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device, argparse.Namespace], tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]]
    save_preview: Callable[[dict[str, Any], dict[str, torch.Tensor], Path], None]


def _height_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["height_norm"].to(device, non_blocking=True)
    weight = batch["weight_257"].to(device, non_blocking=True)
    pred = model(inp)
    loss = _weighted_l1(pred, target, weight)
    return loss, {"height": float(loss.item())}, {"pred": pred, "target": target, "weight": weight}


def _gradient_magnitude_257(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt((dx * dx) + (dy * dy) + 1e-8)


def _normals_from_height(height_norm_257: torch.Tensor) -> torch.Tensor:
    """Build a unit-normal field from normalized height using central differences."""
    dzdx = height_norm_257[:, :, :, 2:] - height_norm_257[:, :, :, :-2]
    dzdy = height_norm_257[:, :, 2:, :] - height_norm_257[:, :, :-2, :]
    dzdx = F.pad(dzdx * 0.5, (1, 1, 0, 0), mode="replicate")
    dzdy = F.pad(dzdy * 0.5, (0, 0, 1, 1), mode="replicate")
    nx = -dzdx
    ny = -dzdy
    nz = torch.ones_like(nx)
    return F.normalize(torch.cat([nx, ny, nz], dim=1), dim=1, eps=1e-6)


def _hard_region_weight_from_targets(
    height_raw: torch.Tensor,
    target_normals: torch.Tensor,
    alpha_painted_256: torch.Tensor,
    mcly_any_16: torch.Tensor,
    terrain_valid_mask: torch.Tensor,
    base_mask: torch.Tensor,
    detail_boost: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    height_grad = _gradient_magnitude_257(height_raw)
    normal_grad = _gradient_magnitude_257(target_normals)
    normal_grad = normal_grad.mean(dim=1, keepdim=True)
    alpha_painted_257 = _resize_weight(alpha_painted_256, target_normals.shape[-2:])
    alpha_grad = _gradient_magnitude_257(alpha_painted_257)
    mcly_any_257 = _resize_weight(mcly_any_16, target_normals.shape[-2:])
    mcly_grad = _gradient_magnitude_257(mcly_any_257)

    valid_mean_height = _masked_mean(height_grad, base_mask)
    valid_mean_normal = _masked_mean(normal_grad, base_mask)
    valid_mean_alpha = _masked_mean(alpha_grad, base_mask)
    valid_mean_mcly = _masked_mean(mcly_grad, base_mask)
    height_grad_n = (height_grad / valid_mean_height.clamp_min(1e-6)).clamp(0.0, 4.0)
    normal_grad_n = (normal_grad / valid_mean_normal.clamp_min(1e-6)).clamp(0.0, 4.0)
    alpha_grad_n = (alpha_grad / valid_mean_alpha.clamp_min(1e-6)).clamp(0.0, 4.0)
    mcly_grad_n = (mcly_grad / valid_mean_mcly.clamp_min(1e-6)).clamp(0.0, 4.0)

    transition_signal = torch.maximum(alpha_grad_n, mcly_grad_n)
    hard_region_signal = ((0.50 * height_grad_n) + (0.25 * normal_grad_n) + (0.25 * transition_signal)).clamp(0.0, 4.0)
    hard_region_signal = hard_region_signal * terrain_valid_mask
    hard_region_weight = 1.0 + (float(detail_boost) * hard_region_signal)
    return hard_region_weight, {
        "hard_region_signal": hard_region_signal,
        "height_grad_signal": height_grad_n,
        "normal_grad_signal": normal_grad_n,
        "alpha_grad_signal": alpha_grad_n,
        "mcly_grad_signal": mcly_grad_n,
        "transition_signal": transition_signal,
    }


def _normal_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["normals"].to(device, non_blocking=True)
    height_raw = batch["height_raw"].to(device, non_blocking=True)
    height_norm = batch["height_norm"].to(device, non_blocking=True)
    normal_mask = batch["normal_mask"].to(device, non_blocking=True)
    terrain_valid_mask = batch["terrain_valid_mask_257"].to(device, non_blocking=True)
    object_weight = batch["weight_257"].to(device, non_blocking=True)
    mddf_mask = batch["mddf_mask"].to(device, non_blocking=True)
    modf_mask = batch["modf_mask"].to(device, non_blocking=True)
    liquid_mask = batch["liquid_mask"].to(device, non_blocking=True)
    what_plate_flag = batch["what_plate_flag"].to(device, non_blocking=True).view(-1, 1, 1, 1)
    alpha_painted_cov = batch["alpha_painted_cov"].to(device, non_blocking=True)
    mcly_cov = batch["mcly_cov"].to(device, non_blocking=True)
    alpha_painted_256 = batch["alpha_painted_256"].to(device, non_blocking=True)
    mcly_any_16 = batch["mcly_any_16"].to(device, non_blocking=True)
    pred = model(inp)
    pred_n = F.normalize(pred, dim=1, eps=1e-6)
    target_n = F.normalize(target, dim=1, eps=1e-6)
    cosine = 1.0 - (pred_n * target_n).sum(dim=1, keepdim=True)
    liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
    object_presence = torch.maximum(mddf_mask, modf_mask)
    liquid_weight = 1.0 - (0.85 * liquid_mask_257)
    instance_weight = 1.0 - (0.75 * object_presence)
    base_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight
    base_mask = base_mask * (1.0 - what_plate_flag)
    hard_region_weight, hard_region_debug = _hard_region_weight_from_targets(
        height_raw=height_raw,
        target_normals=target_n,
        alpha_painted_256=alpha_painted_256,
        mcly_any_16=mcly_any_16,
        terrain_valid_mask=terrain_valid_mask,
        base_mask=base_mask,
        detail_boost=float(args.normal_detail_boost),
    )
    train_mask = base_mask * hard_region_weight
    vec_l1 = (pred_n - target_n).abs().mean(dim=1, keepdim=True)
    nz_l2 = (pred_n[:, 2:3] - target_n[:, 2:3]) ** 2
    loss_cos = _masked_mean(cosine, train_mask)
    loss_vec = _masked_mean(vec_l1, train_mask)
    loss_nz = _masked_mean(nz_l2, train_mask)
    loss = loss_cos + (0.35 * loss_vec) + (0.15 * loss_nz)

    # Neutralize masked/invalid regions so object-heavy areas do not leak into
    # predicted normals when they are excluded from terrain supervision.
    invalid_mask = (1.0 - train_mask).clamp(0.0, 1.0)
    up = torch.zeros_like(pred_n)
    up[:, 2:3, :, :] = 1.0
    loss_invalid_neutral = _masked_mean(1.0 - (pred_n * up).sum(dim=1, keepdim=True), invalid_mask)
    invalid_neutral_weight = float(getattr(args, "invalid_neutral_weight", 0.0))
    if invalid_neutral_weight > 0.0:
        loss = loss + (invalid_neutral_weight * loss_invalid_neutral)

    height_sup_weight = float(getattr(args, "height_supervision_weight", 0.0))
    if str(getattr(args, "resolved_normal_variant", "")) == "v17_1_normals" and height_sup_weight > 0.0:
        height_teacher = _normals_from_height(height_norm)
        loss_height_sup = _masked_mean(1.0 - (pred_n * height_teacher).sum(dim=1, keepdim=True), train_mask)
        loss = loss + (height_sup_weight * loss_height_sup)
    else:
        loss_height_sup = torch.zeros((), device=device, dtype=loss.dtype)
    return loss, {
        "normal": float(loss.item()),
        "normal_cos": float(loss_cos.item()),
        "normal_vec": float(loss_vec.item()),
        "normal_nz": float(loss_nz.item()),
        "normal_mask_cov": float(base_mask.mean().item()),
        "normal_detail_mean": float(_masked_mean(hard_region_weight, base_mask).item()),
        "normal_hard_region_mean": float(_masked_mean(hard_region_debug["hard_region_signal"], base_mask).item()),
        "normal_transition_mean": float(_masked_mean(hard_region_debug["transition_signal"], base_mask).item()),
        "normal_height_sup": float(loss_height_sup.item()),
        "normal_height_sup_weight": float(height_sup_weight),
        "normal_invalid_neutral": float(loss_invalid_neutral.item()),
        "normal_invalid_neutral_weight": float(invalid_neutral_weight),
        "what_plate_rate": float(what_plate_flag.mean().item()),
        "alpha_painted_cov": float(alpha_painted_cov.mean().item()),
        "mcly_cov": float(mcly_cov.mean().item()),
    }, {
        "pred": pred_n,
        "target": target_n,
        "train_mask": train_mask,
        "invalid_mask": invalid_mask,
        "base_mask": base_mask,
        "detail_weight": hard_region_weight,
        "hard_region_signal": hard_region_debug["hard_region_signal"],
        "transition_signal": hard_region_debug["transition_signal"],
        "terrain_valid_mask": terrain_valid_mask,
        "object_weight": object_weight,
        "liquid_mask": liquid_mask_257,
        "instance_weight": instance_weight,
    }


def _combined_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target_normals = batch["normals"].to(device, non_blocking=True)
    target_height = batch["height_norm"].to(device, non_blocking=True)
    height_raw = batch["height_raw"].to(device, non_blocking=True)
    normal_mask = batch["normal_mask"].to(device, non_blocking=True)
    terrain_valid_mask = batch["terrain_valid_mask_257"].to(device, non_blocking=True)
    object_weight = batch["weight_257"].to(device, non_blocking=True)
    mddf_mask = batch["mddf_mask"].to(device, non_blocking=True)
    modf_mask = batch["modf_mask"].to(device, non_blocking=True)
    liquid_mask = batch["liquid_mask"].to(device, non_blocking=True)
    alpha_painted_256 = batch["alpha_painted_256"].to(device, non_blocking=True)
    mcly_any_16 = batch["mcly_any_16"].to(device, non_blocking=True)
    what_plate_flag = batch["what_plate_flag"].to(device, non_blocking=True).view(-1, 1, 1, 1)

    pred_normals, pred_height = model(inp)
    pred_n = F.normalize(pred_normals, dim=1, eps=1e-6)
    target_n = F.normalize(target_normals, dim=1, eps=1e-6)

    # ── Normal loss (same as _normal_loss) ──
    cosine = 1.0 - (pred_n * target_n).sum(dim=1, keepdim=True)
    liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
    object_presence = torch.maximum(mddf_mask, modf_mask)
    liquid_weight = 1.0 - (0.85 * liquid_mask_257)
    instance_weight = 1.0 - (0.75 * object_presence)
    base_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight
    base_mask = base_mask * (1.0 - what_plate_flag)

    hard_region_weight, _hard_debug = _hard_region_weight_from_targets(
        height_raw=height_raw,
        target_normals=target_n,
        alpha_painted_256=alpha_painted_256,
        mcly_any_16=mcly_any_16,
        terrain_valid_mask=terrain_valid_mask,
        base_mask=base_mask,
        detail_boost=float(args.normal_detail_boost),
    )
    train_mask = base_mask * hard_region_weight
    vec_l1 = (pred_n - target_n).abs().mean(dim=1, keepdim=True)
    nz_l2 = (pred_n[:, 2:3] - target_n[:, 2:3]) ** 2
    loss_cos = _masked_mean(cosine, train_mask)
    loss_vec = _masked_mean(vec_l1, train_mask)
    loss_nz = _masked_mean(nz_l2, train_mask)
    normal_loss = loss_cos + (0.35 * loss_vec) + (0.15 * loss_nz)

    # ── Height loss (weighted L1) ──
    height_loss = _weighted_l1(pred_height, target_height, object_weight)

    # ── Combined ──
    w_normal = float(getattr(args, "normal_weight", 1.0))
    w_height = float(getattr(args, "height_weight", 1.0))
    loss = (w_normal * normal_loss) + (w_height * height_loss)

    return loss, {
        "normal": float(normal_loss.item()),
        "normal_cos": float(loss_cos.item()),
        "normal_vec": float(loss_vec.item()),
        "normal_nz": float(loss_nz.item()),
        "height": float(height_loss.item()),
        "combined": float(loss.item()),
    }, {
        "pred": pred_n,
        "target": target_n,
        "pred_height": pred_height,
        "target_height": target_height,
        "train_mask": train_mask,
        "base_mask": base_mask,
    }


def _refiner_refine_and_compare(
    main_model: torch.nn.Module,
    refiner: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, float, float, bool]:
    """Run refiner on validation batch and compare refined vs raw loss.

    Returns: (refined_normals, refined_loss, raw_loss, improved)
    """
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["normals"].to(device, non_blocking=True)
    height_raw = batch["height_raw"].to(device, non_blocking=True)
    normal_mask = batch["normal_mask"].to(device, non_blocking=True)
    terrain_valid_mask = batch["terrain_valid_mask_257"].to(device, non_blocking=True)
    object_weight = batch["weight_257"].to(device, non_blocking=True)
    mddf_mask = batch["mddf_mask"].to(device, non_blocking=True)
    modf_mask = batch["modf_mask"].to(device, non_blocking=True)
    liquid_mask = batch["liquid_mask"].to(device, non_blocking=True)
    alpha_painted_256 = batch["alpha_painted_256"].to(device, non_blocking=True)
    mcly_any_16 = batch["mcly_any_16"].to(device, non_blocking=True)

    with torch.no_grad():
        pred = main_model(inp)
        pred_n = F.normalize(pred, dim=1, eps=1e-6)
        target_n = F.normalize(target, dim=1, eps=1e-6)

        liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
        object_presence = torch.maximum(mddf_mask, modf_mask)
        liquid_weight = 1.0 - (0.85 * liquid_mask_257)
        instance_weight = 1.0 - (0.75 * object_presence)
        base_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight

        hard_region_weight, _hard_debug = _hard_region_weight_from_targets(
            height_raw=height_raw,
            target_normals=target_n,
            alpha_painted_256=alpha_painted_256,
            mcly_any_16=mcly_any_16,
            terrain_valid_mask=terrain_valid_mask,
            base_mask=base_mask,
            detail_boost=float(args.normal_detail_boost),
        )
        train_mask = base_mask * hard_region_weight

        raw_cos = _masked_mean(1.0 - (pred_n * target_n).sum(dim=1, keepdim=True), train_mask)
        raw_vec = _masked_mean((pred_n - target_n).abs().mean(dim=1, keepdim=True), train_mask)
        raw_nz = _masked_mean((pred_n[:, 2:3] - target_n[:, 2:3]) ** 2, train_mask)
        raw_loss = float((raw_cos + 0.35 * raw_vec + 0.15 * raw_nz).item())

        h_mean = batch["height_mean"].to(device).view(-1, 1, 1, 1) if "height_mean" in batch else None
        h_std = batch["height_std"].to(device).view(-1, 1, 1, 1) if "height_std" in batch else None
        norm_height = height_raw
        if h_mean is not None and h_std is not None:
            norm_height = (height_raw - h_mean) / (h_std + 1e-8)

        if str(getattr(args, "resolved_normal_variant", "")) == "v17_1_normals":
            refiner_input = torch.cat([torch.zeros_like(pred), norm_height], dim=1)
        else:
            refiner_input = torch.cat([pred, norm_height], dim=1)
        refined = refiner(refiner_input)
        refined_n = F.normalize(refined, dim=1, eps=1e-6)

        ref_cos = _masked_mean(1.0 - (refined_n * target_n).sum(dim=1, keepdim=True), train_mask)
        ref_vec = _masked_mean((refined_n - target_n).abs().mean(dim=1, keepdim=True), train_mask)
        ref_nz = _masked_mean((refined_n[:, 2:3] - target_n[:, 2:3]) ** 2, train_mask)
        refined_loss = float((ref_cos + 0.35 * ref_vec + 0.15 * ref_nz).item())

    improved = bool(refined_loss < raw_loss)
    return refined, refined_loss, raw_loss, improved


def _holes_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["holes"].to(device, non_blocking=True)
    weight = batch["weight_16"].to(device, non_blocking=True)
    pred = model(inp)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    loss = _masked_mean(bce, weight)
    return loss, {"holes": float(loss.item())}, {"pred": pred, "target": target}


def _liquid_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target_mask = batch["liquid_mask"].to(device, non_blocking=True)
    target_type = batch["liquid_type_16"].to(device, non_blocking=True)
    type_valid = batch["liquid_type_valid_16"].to(device, non_blocking=True)
    weight_256 = batch["weight_256"].to(device, non_blocking=True)
    weight_16 = batch["weight_16"].to(device, non_blocking=True)
    pred_mask, pred_type = model(inp)
    mask_loss = _weighted_l1(pred_mask, target_mask, weight_256)
    type_ce = F.cross_entropy(pred_type, target_type, reduction="none").unsqueeze(1)
    type_loss = _masked_mean(type_ce, type_valid * weight_16)
    loss = mask_loss + (0.5 * type_loss)
    type_pred = pred_type.argmax(dim=1)
    return loss, {"liquid_mask": float(mask_loss.item()), "liquid_type": float(type_loss.item())}, {"pred_mask": pred_mask, "target_mask": target_mask, "pred_type": type_pred, "target_type": target_type}


def _texcomp_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    alpha_target = batch["alpha"].to(device, non_blocking=True)
    mcly_ids = batch["mcly_ids"].to(device, non_blocking=True)
    mcly_mask = batch["mcly_mask"].to(device, non_blocking=True)
    weight_256 = batch["weight_256"].to(device, non_blocking=True)
    weight_16 = batch["weight_16"].to(device, non_blocking=True)
    pred_alpha, pred_mask, pred_ids = model(inp)
    alpha_loss = _weighted_l1(pred_alpha, alpha_target, weight_256)
    mask_bce = F.binary_cross_entropy(pred_mask, mcly_mask.permute(0, 3, 1, 2), reduction="none")
    mask_loss = _masked_mean(mask_bce, weight_16.repeat(1, 4, 1, 1))
    pred_ids_r = pred_ids.permute(0, 1, 3, 4, 2)
    id_ce = F.cross_entropy(pred_ids_r.reshape(-1, 16), mcly_ids.reshape(-1), reduction="none")
    id_ce = id_ce.view_as(mcly_ids).permute(0, 3, 1, 2)
    id_loss = _masked_mean(id_ce, mcly_mask.permute(0, 3, 1, 2) * weight_16.repeat(1, 4, 1, 1))
    recomposed = recompose_from_mcly_alpha(pred_alpha, pred_ids, pred_mask)
    recompose_loss = _weighted_l1(recomposed, inp, weight_256)
    loss = alpha_loss + (0.35 * mask_loss) + (0.25 * id_loss) + (0.5 * recompose_loss)
    pred_ids_cls = pred_ids.argmax(dim=2)
    return loss, {
        "alpha": float(alpha_loss.item()),
        "mcly_mask": float(mask_loss.item()),
        "mcly_id": float(id_loss.item()),
        "recompose": float(recompose_loss.item()),
    }, {
        "pred_alpha": pred_alpha,
        "target_alpha": alpha_target,
        "pred_mask": pred_mask,
        "target_mask": mcly_mask.permute(0, 3, 1, 2),
        "pred_ids": pred_ids_cls,
        "target_ids": mcly_ids,
        "recomposed": recomposed,
    }


def _preview_height(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("height_gt", batch["height_norm"][idx]),
                ("height_pred", outputs["pred"][idx]),
                ("weight", batch["weight_257"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_normal(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    has_refined = "refined_normals" in outputs
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        panels: list[tuple[str, torch.Tensor]] = [
            ("input", batch["input"][idx]),
            ("normal_gt", _normals_to_rgb(outputs["target"][idx])),
            ("normal_pred", _normals_to_rgb(outputs["pred"][idx])),
        ]
        if has_refined:
            panels.append(("refiner_teacher_pred", _normals_to_rgb(outputs["refined_normals"][idx])))
        panels.extend([
            ("terrain_valid", outputs["terrain_valid_mask"][idx]),
            ("base_mask", outputs["base_mask"][idx]),
            ("hard_region", outputs["hard_region_signal"][idx] / outputs["hard_region_signal"][idx].max().clamp_min(1e-6)),
            ("transition", outputs["transition_signal"][idx] / outputs["transition_signal"][idx].max().clamp_min(1e-6)),
            ("detail_weight", outputs["detail_weight"][idx] / outputs["detail_weight"][idx].max().clamp_min(1e-6)),
            ("train_mask", outputs["train_mask"][idx] / outputs["train_mask"][idx].max().clamp_min(1e-6)),
            ("liquid_mask", outputs["liquid_mask"][idx]),
            ("object_weight", outputs["object_weight"][idx]),
        ])
        rows.append(panels)
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_holes(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("holes_gt", outputs["target"][idx]),
                ("holes_pred", outputs["pred"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_liquid(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        pred_type_rgb = _coarse_type_to_rgb(outputs["pred_type"][idx])
        target_type_rgb = _coarse_type_to_rgb(outputs["target_type"][idx])
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("liq_gt", outputs["target_mask"][idx]),
                ("liq_pred", outputs["pred_mask"][idx]),
                ("type_gt", target_type_rgb),
                ("type_pred", pred_type_rgb),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_texcomp(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        gt_alpha_painted = batch["alpha"][idx, 1:].max(dim=0).values.unsqueeze(0)
        pred_alpha_painted = outputs["pred_alpha"][idx, 1:].max(dim=0).values.unsqueeze(0)
        gt_mask = outputs["target_mask"][idx].max(dim=0).values.unsqueeze(0)
        pred_mask = outputs["pred_mask"][idx].max(dim=0).values.unsqueeze(0)
        rows.append(
            [
                ("input", batch["input"][idx]),
                ("alpha_gt", gt_alpha_painted),
                ("alpha_pred", pred_alpha_painted),
                ("mask_gt", gt_mask),
                ("mask_pred", pred_mask),
                ("recomposed", outputs["recomposed"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


def _preview_combined(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    n = min(int(batch["input"].shape[0]), 8)
    rows = []
    row_titles = []
    for idx in range(n):
        row_titles.append(_preview_row_title(batch, idx))
        input_rgb = batch["input"][idx, :3]
        rows.append(
            [
                ("input_rgb", input_rgb),
                ("normal_gt", _normals_to_rgb(outputs["target"][idx])),
                ("normal_pred", _normals_to_rgb(outputs["pred"][idx])),
                ("train_mask", outputs["train_mask"][idx]),
            ]
        )
    _save_preview_grid(rows, out_path, row_titles=row_titles)


TASKS: dict[str, TaskSpec] = {
    "height": TaskSpec("height", V161HeightModel, _height_loss, _preview_height),
    "normal": TaskSpec("normal", V161NormalModel, _normal_loss, _preview_normal),
    "holes": TaskSpec("holes", V161HolesModel, _holes_loss, _preview_holes),
    "liquid": TaskSpec("liquid", V161LiquidModel, _liquid_loss, _preview_liquid),
    "texcomp": TaskSpec("texcomp", V161TexcompModel, _texcomp_loss, _preview_texcomp),
    "combined": TaskSpec("combined", V161NormalHeightCombinedModel, _combined_loss, _preview_combined),
}


def _parse_args(task_name: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train V16.1 {task_name} model")
    p.add_argument("--dataset-dir", type=Path, default=_DATASET_ROOT)
    p.add_argument(
        "--curation-manifest",
        type=Path,
        default=None,
        help="Optional curation manifest directory/file produced by build_v16_curation_manifest.py",
    )
    p.add_argument("--builds", nargs="+", default=["3_3_5_12340"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader worker count. Use -1 to auto-resolve a CUDA-friendly default.",
    )
    p.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader worker processes alive between epochs (defaults to true when workers > 0)",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched per worker (effective when --num-workers > 0)",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--train-max-tiles",
        type=int,
        default=0,
        help="If >0, curate this many tiles into the persistent train pool before epoch rotation.",
    )
    p.add_argument(
        "--train-epoch-tiles",
        type=int,
        default=0,
        help="If >0, sample this many train-pool tiles per epoch.",
    )
    p.add_argument(
        "--train-epoch-build-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance per-epoch train sampling across builds when possible.",
    )
    p.add_argument(
        "--bucket-sampling-profile",
        choices=sorted(_BUCKET_SAMPLING_PROFILES.keys()),
        default=("v16_1_1_normal" if task_name == "normal" else "uniform"),
        help="Per-epoch difficulty-bucket sampling profile used when curated manifests carry bucket metadata.",
    )
    p.add_argument(
        "--val-max-tiles",
        type=int,
        default=0,
        help="If >0, curate this many tiles into the fixed validation pool.",
    )
    p.add_argument(
        "--val-epoch-tiles",
        type=int,
        default=0,
        help="If >0 and --rotate-val-tiles is set, sample this many validation-pool tiles per epoch.",
    )
    p.add_argument(
        "--rotate-val-tiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rotate validation tile subset each epoch (uses deterministic seeded sampler).",
    )
    p.add_argument(
        "--curation-seed",
        type=int,
        default=None,
        help="Seed for train/val pool selection (defaults to --seed).",
    )
    p.add_argument(
        "--curation-min-terrain-validity",
        type=float,
        default=0.0,
        help="Drop manifest tiles below this terrain-validity score.",
    )
    p.add_argument(
        "--curation-min-minimap-usefulness",
        type=float,
        default=0.0,
        help="Drop manifest tiles below this minimap-target-usefulness score.",
    )
    p.add_argument(
        "--curation-reject-what-plate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject manifest tiles flagged as what-plate/noise tiles.",
    )
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument(
        "--val-preview-interval",
        type=int,
        default=1,
        help="If >0, write a validation preview only when a new best checkpoint is found. 0 disables preview writes.",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--normal-detail-boost",
        type=float,
        default=1.0,
        help="Extra weight placed on high-deformation normal targets relative to broad flat terrain.",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (useful for CPU-only or limited toolchains)",
    )
    p.add_argument(
        "--target-vram-gb",
        type=float,
        default=0.0,
        help="Soft VRAM target used by startup batch autotune and per-epoch guidance logs (0 disables both).",
    )
    p.add_argument(
        "--autotune-batch-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Probe a batch-size ladder against --target-vram-gb before the run starts and pick the largest safe candidate.",
    )
    p.add_argument(
        "--autotune-batch-candidates",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit batch-size ladder for startup autotune.",
    )
    p.add_argument(
        "--autotune-keep-epoch-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When autotune changes batch-size, rescale train-epoch-tiles to preserve the original steps-per-epoch budget.",
    )
    p.add_argument(
        "--autotune-safety-factor",
        type=float,
        default=0.0,
        help="Override autotune VRAM safety factor (0 uses variant defaults).",
    )
    p.add_argument(
        "--autotune-probe-warmup-steps",
        type=int,
        default=2,
        help="Autotune probe warmup steps per candidate before measured peak capture.",
    )
    p.add_argument(
        "--autotune-probe-measure-steps",
        type=int,
        default=3,
        help="Autotune probe measured steps per candidate used for batch-size decision.",
    )
    p.add_argument(
        "--refiner-disabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable the height-derived normal refiner and distillation (normal task only).",
    )
    p.add_argument(
        "--refiner-distill-weight",
        type=float,
        default=0.25,
        help="Weight of the distillation term when refiner is active (normal task only).",
    )
    p.add_argument(
        "--refiner-probe-plateau-epochs",
        type=int,
        default=3,
        help="Probe refiner only after this many non-best epochs.",
    )
    p.add_argument(
        "--refiner-probe-min-improvement",
        type=float,
        default=0.0,
        help="Minimum (raw-refined) improvement required to activate/retain refiner.",
    )
    p.add_argument(
        "--preview-refiner-teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include refiner teacher panel in validation preview images.",
    )
    p.add_argument(
        "--height-supervision-weight",
        type=float,
        default=0.0,
        help="Additional cosine supervision weight against normals derived from height (normal task only).",
    )
    p.add_argument(
        "--invalid-neutral-weight",
        type=float,
        default=0.20,
        help="Weight for neutralizing masked/invalid normal regions toward up-vector (normal task only).",
    )
    p.add_argument(
        "--height-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add height_norm as a 4th input channel to the normal model (normal task only).",
    )
    p.add_argument(
        "--normal-variant",
        choices=list(_NORMAL_VARIANTS),
        default="v17_1_normals",
        help="Explicit normal-trainer variant contract. Use v17_1_normals for minimap->normals with height supervisor-only.",
    )
    p.add_argument(
        "--normal-weight",
        type=float,
        default=1.0,
        help="Weight of the normal loss term in the combined model (combined task only).",
    )
    p.add_argument(
        "--height-weight",
        type=float,
        default=1.0,
        help="Weight of the height loss term in the combined model (combined task only).",
    )
    return p.parse_args()


def run_task(task_name: str) -> None:
    if task_name not in TASKS:
        raise RuntimeError(f"Unknown V16.1 task: {task_name}")
    args = _parse_args(task_name)
    task = TASKS[task_name]
    if args.grad_accum_steps < 1:
        raise RuntimeError("--grad-accum-steps must be >= 1")
    if args.prefetch_factor < 1:
        raise RuntimeError("--prefetch-factor must be >= 1")
    if args.autotune_safety_factor < 0.0 or args.autotune_safety_factor > 1.0:
        raise RuntimeError("--autotune-safety-factor must be in [0.0, 1.0]")
    if args.autotune_probe_warmup_steps < 1:
        raise RuntimeError("--autotune-probe-warmup-steps must be >= 1")
    if args.autotune_probe_measure_steps < 1:
        raise RuntimeError("--autotune-probe-measure-steps must be >= 1")
    if args.train_epoch_tiles < 0:
        raise RuntimeError("--train-epoch-tiles must be >= 0")
    if args.val_epoch_tiles < 0:
        raise RuntimeError("--val-epoch-tiles must be >= 0")
    if args.refiner_probe_plateau_epochs < 1:
        raise RuntimeError("--refiner-probe-plateau-epochs must be >= 1")

    normal_variant, resolved_height_channel, resolved_refiner_enabled = _resolve_normal_variant(args, task_name)
    if task_name == "normal" and normal_variant in {"v17_hybrid", "v17_1_normals"}:
        if args.curation_manifest is None:
            raise RuntimeError(f"{normal_variant} requires --curation-manifest for curated runs")
        if float(args.curation_min_terrain_validity) <= 0.0:
            args.curation_min_terrain_validity = 0.55
        if float(args.curation_min_minimap_usefulness) <= 0.0:
            args.curation_min_minimap_usefulness = 0.45
        if not bool(args.curation_reject_what_plate):
            args.curation_reject_what_plate = True

        if normal_variant == "v17_hybrid":
            if int(args.train_max_tiles) <= 0:
                args.train_max_tiles = 80
            if int(args.val_max_tiles) <= 0:
                args.val_max_tiles = 10
        elif normal_variant == "v17_1_normals":
            if int(args.train_max_tiles) <= 0:
                args.train_max_tiles = 8000
            if int(args.val_max_tiles) <= 0:
                args.val_max_tiles = 800
            if not bool(args.rotate_val_tiles):
                args.rotate_val_tiles = True
            if int(args.val_epoch_tiles) <= 0:
                args.val_epoch_tiles = 128
            if not bool(args.autotune_batch_size):
                args.autotune_batch_size = True
            if float(args.target_vram_gb) <= 0.0:
                args.target_vram_gb = 12.0
            if float(args.height_supervision_weight) <= 0.0:
                args.height_supervision_weight = 1.0
            if float(args.invalid_neutral_weight) <= 0.0:
                args.invalid_neutral_weight = 0.20
            if int(args.num_workers) < 0:
                args.num_workers = 2
            if int(args.prefetch_factor) > 2:
                args.prefetch_factor = 2
            if args.persistent_workers is None:
                args.persistent_workers = False

    args.resolved_height_channel = bool(resolved_height_channel)
    args.resolved_refiner_enabled = bool(resolved_refiner_enabled)
    args.resolved_normal_variant = str(normal_variant)
    if task_name == "normal" and normal_variant == "v17_1_normals" and float(args.height_supervision_weight) <= 0.0:
        raise RuntimeError("v17_1_normals requires --height-supervision-weight > 0 so height supervision is active.")

    _seed_all(args.seed)
    device = _resolve_device(args.device)
    resolved_num_workers = _resolve_num_workers(int(args.num_workers), device)
    resolved_persistent_workers = _resolve_persistent_workers(args.persistent_workers, resolved_num_workers)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    height_channel_enabled = bool(resolved_height_channel)
    refiner_enabled_for_task = bool(task_name == "normal" and resolved_refiner_enabled)
    if task_name == "normal" and normal_variant == "v17_hybrid":
        if not run_name.startswith("v17_"):
            run_name = f"v17_{run_name}"
    elif task_name == "normal" and normal_variant == "v17_1_normals":
        if not run_name.startswith("v17_1_"):
            run_name = f"v17_1_{run_name}"
    elif height_channel_enabled and not run_name.startswith("v16_1_3"):
        run_name = f"v16_1_3_{run_name}"
    elif refiner_enabled_for_task and not run_name.startswith("v16_1_2"):
        run_name = f"v16_1_2_{run_name}"
    run_dir = _MODELS_ROOT / task_name / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    val_dir = run_dir / "validation"
    evidence_dir = run_dir / "evidence"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    train_ds = V161Dataset(
        args.dataset_dir,
        builds=args.builds,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=not args.no_augment,
        curation_manifest=args.curation_manifest,
        height_channel=bool(resolved_height_channel),
        curation_min_terrain_validity=float(args.curation_min_terrain_validity),
        curation_min_minimap_usefulness=float(args.curation_min_minimap_usefulness),
        curation_reject_what_plate=bool(args.curation_reject_what_plate),
    )
    val_ds = V161Dataset(
        args.dataset_dir,
        builds=args.builds,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=False,
        curation_manifest=args.curation_manifest,
        height_channel=bool(resolved_height_channel),
        curation_min_terrain_validity=float(args.curation_min_terrain_validity),
        curation_min_minimap_usefulness=float(args.curation_min_minimap_usefulness),
        curation_reject_what_plate=bool(args.curation_reject_what_plate),
    )
    curation_seed = int(args.curation_seed) if args.curation_seed is not None else int(args.seed)
    train_pool = _apply_dataset_pool(
        train_ds,
        split="train",
        max_tiles=int(args.train_max_tiles),
        seed=curation_seed + 101,
        evidence_dir=evidence_dir,
        build_balanced=True,
    )
    val_pool = _apply_dataset_pool(
        val_ds,
        split="val",
        max_tiles=int(args.val_max_tiles),
        seed=curation_seed + 202,
        evidence_dir=evidence_dir,
        build_balanced=True,
    )
    _apply_dataset_limit(train_ds, int(args.max_train_samples))
    _apply_dataset_limit(val_ds, int(args.max_val_samples))
    autotune_result = _autotune_batch_size(
        task=task,
        task_name=task_name,
        train_ds=train_ds,
        device=device,
        args=args,
        evidence_dir=evidence_dir,
    )
    train_build_labels = [
        str(train_ds._index_entries[global_idx].get("_build", "unknown"))
        for global_idx in train_ds._indices
    ]
    train_bucket_labels = [
        _normalize_bucket_label(train_ds._index_entries[global_idx].get("_curation_difficulty_bucket", ""))
        for global_idx in train_ds._indices
    ]
    train_sample_rows = [
        _pool_row(train_ds._index_entries[global_idx], subset_pos=idx, split_pos=idx)
        for idx, global_idx in enumerate(train_ds._indices)
    ]
    train_order_log = evidence_dir / "train_epoch_orders.jsonl"
    train_bucket_log = evidence_dir / "train_epoch_bucket_usage.jsonl"
    train_sampler = _DeterministicEpochSampler(
        len(train_ds),
        seed=int(args.seed),
        order_log_path=train_order_log,
        bucket_log_path=train_bucket_log,
        epoch_size=int(args.train_epoch_tiles),
        build_labels=train_build_labels,
        build_balanced=bool(args.train_epoch_build_balanced),
        bucket_labels=train_bucket_labels,
        bucket_sampling_profile=args.bucket_sampling_profile,
        sample_rows=train_sample_rows,
    )
    val_sampler: _DeterministicEpochSampler | None = None
    if bool(args.rotate_val_tiles) and int(args.val_epoch_tiles) > 0:
        val_build_labels = [
            str(val_ds._index_entries[global_idx].get("_build", "unknown"))
            for global_idx in val_ds._indices
        ]
        val_bucket_labels = [
            _normalize_bucket_label(val_ds._index_entries[global_idx].get("_curation_difficulty_bucket", ""))
            for global_idx in val_ds._indices
        ]
        val_sample_rows = [
            _pool_row(val_ds._index_entries[global_idx], subset_pos=idx, split_pos=idx)
            for idx, global_idx in enumerate(val_ds._indices)
        ]
        val_sampler = _DeterministicEpochSampler(
            len(val_ds),
            seed=int(args.seed) + 7001,
            order_log_path=evidence_dir / "val_epoch_orders.jsonl",
            bucket_log_path=evidence_dir / "val_epoch_bucket_usage.jsonl",
            epoch_size=int(args.val_epoch_tiles),
            build_labels=val_build_labels,
            build_balanced=True,
            bucket_labels=val_bucket_labels,
            bucket_sampling_profile=args.bucket_sampling_profile,
            sample_rows=val_sample_rows,
        )
    loader_kwargs: dict[str, Any] = {
        "num_workers": resolved_num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(resolved_persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, **loader_kwargs)
    if val_sampler is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, **loader_kwargs)
    else:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = task.model_factory().to(device)
    if task_name == "normal" and bool(resolved_height_channel):
        model = V161NormalHeightModel().to(device)
    can_compile = hasattr(torch, "compile") and not args.no_compile and device.type == "cuda"
    if can_compile:
        try:
            model = torch.compile(model)
            compile_status = "enabled"
        except Exception as ex:
            compile_status = f"disabled (compile failed: {ex})"
    else:
        compile_status = "disabled"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    refiner: torch.nn.Module | None = (
        V161NormalRefiner().to(device) if refiner_enabled_for_task else None
    )
    refiner_active = False
    best_val = float("inf")
    best_epoch: int | None = None
    start_epoch = 1
    log_entries: list[dict[str, Any]] = []

    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        ckpt_state = ckpt["model_state_dict"]
        model_state = model.state_dict()
        ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in ckpt_state)
        model_has_prefix = any(k.startswith("_orig_mod.") for k in model_state)
        if model_has_prefix and not ckpt_has_prefix:
            ckpt_state = {f"_orig_mod.{k}": v for k, v in ckpt_state.items()}
        elif ckpt_has_prefix and not model_has_prefix:
            ckpt_state = {k.removeprefix("_orig_mod."): v for k, v in ckpt_state.items()}
        model.load_state_dict(ckpt_state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            completed_epoch = int(ckpt["epoch"])
            loaded_t_max = int(getattr(scheduler, "T_max", max(args.epochs, 1)))
            requested_t_max = max(int(args.epochs), 1)
            if requested_t_max > loaded_t_max:
                scheduler.T_max = requested_t_max
                eta_min = float(getattr(scheduler, "eta_min", 0.0))
                resumed_lrs = [
                    eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * completed_epoch / requested_t_max)) / 2.0
                    for base_lr in scheduler.base_lrs
                ]
                for param_group, resumed_lr in zip(optimizer.param_groups, resumed_lrs):
                    param_group["lr"] = float(resumed_lr)
                scheduler._last_lr = [float(lr) for lr in resumed_lrs]
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        best_epoch = int(ckpt["best_epoch"]) if ckpt.get("best_epoch") is not None else None
        if refiner is not None and "refiner_state_dict" in ckpt:
            refiner.load_state_dict(ckpt["refiner_state_dict"])
            refiner_active = bool(ckpt.get("refiner_active", False))
            print(f"        refiner | restored from checkpoint, active={refiner_active}", flush=True)

    config = {
        "task": task_name,
        "run_name": run_name,
        "dataset_dir": str(Path(args.dataset_dir)),
        "curation_manifest": str(args.curation_manifest) if args.curation_manifest else None,
        "builds": list(args.builds),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "num_workers": args.num_workers,
        "resolved_num_workers": resolved_num_workers,
        "persistent_workers": args.persistent_workers,
        "resolved_persistent_workers": resolved_persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "resolved_curation_seed": curation_seed,
        "curation_min_terrain_validity": float(args.curation_min_terrain_validity),
        "curation_min_minimap_usefulness": float(args.curation_min_minimap_usefulness),
        "curation_reject_what_plate": bool(args.curation_reject_what_plate),
        "val_fraction": args.val_fraction,
        "target_vram_gb": args.target_vram_gb,
        "autotune_batch_size": args.autotune_batch_size,
        "autotune_batch_candidates": list(args.autotune_batch_candidates) if args.autotune_batch_candidates else None,
        "autotune_keep_epoch_steps": args.autotune_keep_epoch_steps,
        "autotune_safety_factor": args.autotune_safety_factor,
        "autotune_probe_warmup_steps": args.autotune_probe_warmup_steps,
        "autotune_probe_measure_steps": args.autotune_probe_measure_steps,
        "train_max_tiles": args.train_max_tiles,
        "train_epoch_tiles": args.train_epoch_tiles,
        "train_epoch_build_balanced": args.train_epoch_build_balanced,
        "bucket_sampling_profile": args.bucket_sampling_profile,
        "bucket_sampling_weights": _bucket_sampling_weights(args.bucket_sampling_profile),
        "val_max_tiles": args.val_max_tiles,
        "rotate_val_tiles": bool(args.rotate_val_tiles),
        "val_epoch_tiles": int(args.val_epoch_tiles),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "normal_detail_boost": args.normal_detail_boost,
        "normal_variant": normal_variant,
        "height_channel": bool(resolved_height_channel),
        "resolved_height_channel": bool(resolved_height_channel),
        "resolved_refiner_enabled": bool(resolved_refiner_enabled),
        "resolved_input_contract": ("minimap_rgb" if task_name == "normal" and normal_variant == "v17_1_normals" else "variant_defined"),
        "resolved_output_contract": ("normals_xyz" if task_name == "normal" else "task_defined"),
        "height_supervision_only": bool(task_name == "normal" and normal_variant == "v17_1_normals"),
        "model_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "refiner_enabled": refiner is not None,
        "refiner_disabled": (None if getattr(args, "refiner_disabled", None) is None else bool(getattr(args, "refiner_disabled"))),
        "refiner_distill_weight": float(getattr(args, "refiner_distill_weight", 0.25)),
        "height_supervision_weight": float(getattr(args, "height_supervision_weight", 0.0)),
        "invalid_neutral_weight": float(getattr(args, "invalid_neutral_weight", 0.0)),
        "refiner_active": bool(refiner_active) if refiner is not None else False,
        "refiner_params": refiner.count_parameters() if refiner is not None else 0,
        "train_pool": train_pool,
        "val_pool": val_pool,
        "no_compile": args.no_compile,
        "compile_status": compile_status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "best_val": best_val if np.isfinite(best_val) else None,
        "best_epoch": best_epoch,
        "autotune_result": autotune_result,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Task: {task_name}", flush=True)
    if task_name == "normal":
        print(
            f"Normal variant: {normal_variant} | "
            f"height_channel={bool(resolved_height_channel)} | "
            f"refiner_enabled={bool(resolved_refiner_enabled)} | "
            f"refiner_distill_weight={float(getattr(args, 'refiner_distill_weight', 0.25)):.3f} | "
            f"height_supervision_weight={float(getattr(args, 'height_supervision_weight', 0.0)):.3f}",
            flush=True,
        )
        if normal_variant == "v17_1_normals":
            print(
                "Normal contract: input=minimap_rgb -> output=normals_xyz | height_supervision_only=true",
                flush=True,
            )
    print(f"Device: {device}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Dataset: train={len(train_ds)} val={len(val_ds)}", flush=True)
    if args.curation_manifest is not None:
        print(f"Curation manifest: {args.curation_manifest}", flush=True)
        print(
            "Curation gates: "
            f"terrain_validity>={float(args.curation_min_terrain_validity):.2f} "
            f"minimap_usefulness>={float(args.curation_min_minimap_usefulness):.2f} "
            f"reject_what_plate={bool(args.curation_reject_what_plate)}",
            flush=True,
        )
    print(
        "Curated pools: "
        f"train={train_pool['selected_tiles']}/{train_pool['available_tiles']} "
        f"val={val_pool['selected_tiles']}/{val_pool['available_tiles']}",
        flush=True,
    )
    print(f"Curated build mix (train): {train_pool.get('build_tile_counts', {})}", flush=True)
    print(f"Curated build mix (val): {val_pool.get('build_tile_counts', {})}", flush=True)
    if train_pool.get("selected_bucket_counts"):
        print(f"Curated difficulty mix (train): {train_pool.get('selected_bucket_counts', {})}", flush=True)
        print(f"Available difficulty mix (train): {train_pool.get('available_bucket_counts', {})}", flush=True)
    if val_pool.get("selected_bucket_counts"):
        print(f"Curated difficulty mix (val): {val_pool.get('selected_bucket_counts', {})}", flush=True)
    if bool(args.rotate_val_tiles) and int(args.val_epoch_tiles) > 0:
        print(
            f"Validation rotation: val_epoch_tiles={min(int(args.val_epoch_tiles), len(val_ds))}/{len(val_ds)}",
            flush=True,
        )
    if args.train_epoch_tiles > 0:
        print(
            "Epoch sampling: "
            f"train_epoch_tiles={min(int(args.train_epoch_tiles), len(train_ds))}/{len(train_ds)} "
            f"build_balanced={bool(args.train_epoch_build_balanced)} "
            f"bucket_profile={args.bucket_sampling_profile}",
            flush=True,
        )
    print(
        f"DataLoader: workers={resolved_num_workers} "
        f"persistent_workers={resolved_persistent_workers} "
        f"prefetch_factor={(args.prefetch_factor if resolved_num_workers > 0 else 'n/a')}",
        flush=True,
    )
    print(
        f"Batching: micro={args.batch_size} accum={args.grad_accum_steps} "
        f"effective={args.batch_size * args.grad_accum_steps}",
        flush=True,
    )
    if autotune_result is not None:
        print(
            "Autotune: "
            f"target_vram_gb={autotune_result['target_vram_gb']:.2f} "
            f"chosen_batch_size={autotune_result['chosen_batch_size']} "
            f"train_epoch_tiles={autotune_result['tuned_train_epoch_tiles']}",
            flush=True,
        )
    if task_name == "normal":
        print(f"Normal detail steering: boost={args.normal_detail_boost:.2f}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    print(f"torch.compile: {compile_status}", flush=True)
    if args.resume_checkpoint is not None:
        print(f"Resume checkpoint: {args.resume_checkpoint}", flush=True)
        print(f"Resume scheduler T_max: {getattr(scheduler, 'T_max', 'n/a')}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        model.train()
        metric_sums: dict[str, float] = {}
        train_loss_sum = 0.0
        optimizer_steps = 0
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                loss, metrics, _outputs = task.loss_fn(model, batch, device, args)

            if refiner is not None and refiner_active and task_name == "normal":
                pred_raw = _outputs.get("pred")
                if pred_raw is not None:
                    h_mean = batch["height_mean"].to(device).view(-1, 1, 1, 1)
                    h_std = batch["height_std"].to(device).view(-1, 1, 1, 1)
                    norm_height = (batch["height_raw"].to(device, non_blocking=True) - h_mean) / (h_std + 1e-8)
                    with torch.no_grad():
                        if str(getattr(args, "resolved_normal_variant", "")) == "v17_1_normals":
                            teacher_input = torch.cat([torch.zeros_like(pred_raw), norm_height], dim=1)
                        else:
                            teacher_input = torch.cat([pred_raw.detach(), norm_height], dim=1)
                        teacher = refiner(teacher_input)
                        teacher_n = F.normalize(teacher, dim=1, eps=1e-6)
                    train_mask = _outputs.get("train_mask")
                    if train_mask is not None:
                        distill_cos = _masked_mean(
                            1.0 - (pred_raw * teacher_n).sum(dim=1, keepdim=True),
                            train_mask,
                        )
                        loss = loss + float(args.refiner_distill_weight) * distill_cos
                        metrics["refiner_distill"] = float(distill_cos.item() / max(float(args.refiner_distill_weight), 1e-8))

            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = (batch_idx % args.grad_accum_steps == 0) or (batch_idx == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            train_loss_sum += float(loss.item())
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        scheduler.step()

        n_train = max(len(train_loader), 1)
        peak_alloc_gb = None
        peak_reserved_gb = None
        if device.type == "cuda":
            peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024.0 ** 3)
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss_sum / n_train,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.perf_counter() - t0,
            "optimizer_steps": optimizer_steps,
        }
        if peak_alloc_gb is not None and peak_reserved_gb is not None:
            entry["cuda_peak_alloc_gb"] = peak_alloc_gb
            entry["cuda_peak_reserved_gb"] = peak_reserved_gb
        for key, value in metric_sums.items():
            entry[f"train_{key}"] = value / n_train

        print(
            f"Epoch {epoch:3d}/{args.epochs} | loss={entry['train_loss']:.4f} "
            f"lr={entry['lr']:.2e} opt_steps={optimizer_steps} {entry['elapsed_s']:.1f}s",
            flush=True,
        )
        if task_name == "normal" and "train_normal_height_sup" in entry:
            print(
                f"        height_sup | loss={entry['train_normal_height_sup']:.4f} "
                f"weight={float(getattr(args, 'height_supervision_weight', 0.0)):.3f}",
                flush=True,
            )
        if peak_reserved_gb is not None:
            print(
                f"        cuda_mem | alloc_peak={peak_alloc_gb:.2f}GB "
                f"reserved_peak={peak_reserved_gb:.2f}GB",
                flush=True,
            )
            if float(args.target_vram_gb) > 0.0:
                target = float(args.target_vram_gb)
                if peak_reserved_gb < target * 0.70:
                    suggested_bs = max(
                        int(args.batch_size) + 1,
                        int(round(int(args.batch_size) * (target / max(peak_reserved_gb, 1e-6)))),
                    )
                    print(
                        f"        tuning | below target_vram_gb={target:.2f}; "
                        f"consider batch-size ~{suggested_bs}",
                        flush=True,
                    )
                elif peak_reserved_gb > target * 1.05:
                    print(
                        f"        tuning | above target_vram_gb={target:.2f}; "
                        f"consider reducing batch-size",
                        flush=True,
                    )

        if epoch % args.val_interval == 0 and len(val_loader) > 0:
            model.eval()
            val_loss_sum = 0.0
            val_metric_sums: dict[str, float] = {}
            preview_batch = None
            preview_outputs = None
            with torch.no_grad():
                for batch in val_loader:
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                        loss, metrics, outputs = task.loss_fn(model, batch, device, args)
                    val_loss_sum += float(loss.item())
                    for key, value in metrics.items():
                        val_metric_sums[key] = val_metric_sums.get(key, 0.0) + float(value)
                    if preview_batch is None:
                        preview_batch = batch
                        preview_outputs = outputs
                        if refiner is not None and task_name == "normal" and bool(getattr(args, "preview_refiner_teacher", False)):
                            _pred = outputs.get("pred")
                            if _pred is not None:
                                _h_mean = batch["height_mean"].to(device).view(-1, 1, 1, 1)
                                _h_std = batch["height_std"].to(device).view(-1, 1, 1, 1)
                                _norm_h = (batch["height_raw"].to(device, non_blocking=True) - _h_mean) / (_h_std + 1e-8)
                                if str(getattr(args, "resolved_normal_variant", "")) == "v17_1_normals":
                                    _ref_input = torch.cat([torch.zeros_like(_pred), _norm_h], dim=1)
                                else:
                                    _ref_input = torch.cat([_pred, _norm_h], dim=1)
                                _refined = refiner(_ref_input)
                                preview_outputs["refined_normals"] = _refined
            n_val = max(len(val_loader), 1)
            entry["val_loss"] = val_loss_sum / n_val
            for key, value in val_metric_sums.items():
                entry[f"val_{key}"] = value / n_val
            print(f"        val | loss={entry['val_loss']:.4f}", flush=True)
            is_new_best = False
            if entry["val_loss"] < best_val:
                best_val = float(entry["val_loss"])
                best_epoch = int(epoch)
                is_new_best = True
                entry["is_new_best"] = True

                ckpt_payload: dict[str, Any] = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "task": task_name,
                }

                if refiner is not None:
                    refiner_loss_sum = 0.0
                    refiner_n_val = 0
                    raw_loss_sum = 0.0
                    refined_outputs = None
                    refiner.eval()
                    for ref_batch in val_loader:
                        with torch.no_grad():
                            _r_refined, _r_loss, _raw_loss, _improved = _refiner_refine_and_compare(
                                model, refiner, ref_batch, device, args,
                            )
                        refiner_loss_sum += _r_loss
                        raw_loss_sum += _raw_loss
                        refiner_n_val += 1
                        if refined_outputs is None:
                            refined_outputs = _r_refined
                    _mean_refined = refiner_loss_sum / max(refiner_n_val, 1)
                    _mean_raw = raw_loss_sum / max(refiner_n_val, 1)
                    _refiner_improved = bool(_mean_refined < _mean_raw)
                    entry["refiner_loss"] = _mean_refined
                    entry["refiner_raw_loss"] = _mean_raw
                    entry["refiner_improved"] = _refiner_improved

                    if not refiner_active:
                        if _refiner_improved:
                            refiner_active = True
                            print(
                                f"        refiner | activating distillation "
                                f"(first best-epoch, raw={_mean_raw:.4f} refined={_mean_refined:.4f})",
                                flush=True,
                            )
                        else:
                            print(
                                f"        refiner | not activated (first best-epoch not improved): "
                                f"raw={_mean_raw:.4f} refined={_mean_refined:.4f}",
                                flush=True,
                            )
                    elif _refiner_improved:
                        print(
                            f"        refiner | improved: raw={_mean_raw:.4f} refined={_mean_refined:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"        refiner | still active but not improved: "
                            f"raw={_mean_raw:.4f} refined={_mean_refined:.4f}",
                            flush=True,
                        )
                    ckpt_payload["refiner_state_dict"] = refiner.state_dict()
                    ckpt_payload["refiner_active"] = bool(refiner_active)

                torch.save(ckpt_payload, ckpt_dir / f"v16_1_{task_name}_best.pt")
                print(f"        *** new best val_loss={best_val:.4f}", flush=True)
            if (
                is_new_best
                and preview_batch is not None
                and preview_outputs is not None
                and args.val_preview_interval > 0
            ):
                task.save_preview(preview_batch, preview_outputs, val_dir / f"best_epoch_{epoch:04d}.png")

        log_entries.append(entry)
        last_ckpt: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "task": task_name,
        }
        if refiner is not None:
            last_ckpt["refiner_state_dict"] = refiner.state_dict()
            last_ckpt["refiner_active"] = bool(refiner_active)
        torch.save(last_ckpt, ckpt_dir / f"v16_1_{task_name}_last.pt")
        (run_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2), encoding="utf-8")

    config["best_val"] = best_val if np.isfinite(best_val) else None
    config["best_epoch"] = best_epoch
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    final_ckpt: dict[str, Any] = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "task": task_name,
    }
    if refiner is not None:
        final_ckpt["refiner_state_dict"] = refiner.state_dict()
        final_ckpt["refiner_active"] = bool(refiner_active)
    torch.save(final_ckpt, ckpt_dir / f"v16_1_{task_name}_final.pt")
    print(f"Done. Run dir: {run_dir}", flush=True)
