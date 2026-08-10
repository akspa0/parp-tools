"""Shared evaluator and user-owned trainer for the clean-signal v60 lane."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from harvester.v60.clean_signal_corpus import (
    COARSE_SIGNAL,
    CONFIDENCE_SIGNAL,
    DETAIL_SIGNAL,
    GRADIENT_SIGNAL,
    LUMA_SIGNAL,
    RELATIVE_HEIGHT_SIGNAL,
    load_clean_signal_manifest,
    validate_clean_signal_corpus,
)
from harvester.v60.clean_signal_losses import (
    PARITY_PROFILE,
    V7GuidanceConfig,
    clean_signal_loss,
    get_clean_signal_loss_config,
    loss_metrics,
)
from harvester.v60.clean_signal_model import (
    CleanSignalPredictions,
    build_clean_signal_model,
)

TRAIN_SCHEMA = "v7-clean-signal-training-report-v1"
CHECKPOINT_SCHEMA = "v7-clean-signal-checkpoint-v1"
VALID_TRAIN_SPLIT_MODES = frozenset({"within_family", "complete_family"})


class CleanSignalTrainError(ValueError):
    """Raised when a clean-signal training or evaluation contract is invalid."""


@dataclass(frozen=True)
class CleanSignalRow:
    """Validated manifest metadata for one clean-signal NPZ row."""

    row_id: str
    source_group_id: str
    family: str
    complexity_bucket: str
    variant: int
    split: str
    npz_path: Path


@dataclass(frozen=True)
class CleanSignalSplit:
    """Deterministic row selection shared by every architecture/loss cell."""

    train_rows: tuple[CleanSignalRow, ...]
    validation_rows: tuple[CleanSignalRow, ...]
    mode: str
    seed: int
    identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "identity": self.identity,
            "train_row_ids": [row.row_id for row in self.train_rows],
            "validation_row_ids": [row.row_id for row in self.validation_rows],
            "train_families": sorted({row.family for row in self.train_rows}),
            "validation_families": sorted({row.family for row in self.validation_rows}),
        }


@dataclass(frozen=True)
class CleanSignalTrainConfig:
    """Small shared training budget; the confirmed CLI owns the actual heavy run."""

    epochs: int = 80
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    seed: int = 7137
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise CleanSignalTrainError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise CleanSignalTrainError("learning_rate must be positive and weight_decay non-negative")
        if self.seed < 0:
            raise CleanSignalTrainError("seed must be non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise CleanSignalTrainError("device must be one of 'auto', 'cpu', or 'cuda'")


def load_clean_signal_rows(corpus_root: str | Path) -> tuple[Path, list[CleanSignalRow]]:
    """Validate and load all rows without assembling any target-derived model input."""

    root = Path(corpus_root)
    validation = validate_clean_signal_corpus(root)
    if not validation["valid"]:
        failures = "; ".join(str(item) for item in validation["failures"][:8])
        raise CleanSignalTrainError(f"invalid clean-signal corpus: {failures}")
    manifest = load_clean_signal_manifest(root)
    rows = [
        CleanSignalRow(
            row_id=str(raw["row_id"]),
            source_group_id=str(raw["source_group_id"]),
            family=str(raw["family"]),
            complexity_bucket=str(raw.get("complexity_bucket", "unknown")),
            variant=int(raw.get("variant", 0)),
            split=str(raw["split"]),
            npz_path=root / str(raw["npz"]),
        )
        for raw in manifest["rows"]
    ]
    return root, sorted(rows, key=lambda row: row.row_id)


def _split_identity(mode: str, seed: int, train_rows: Iterable[CleanSignalRow], validation_rows: Iterable[CleanSignalRow]) -> str:
    payload = {
        "mode": mode,
        "seed": seed,
        "train": [row.row_id for row in train_rows],
        "validation": [row.row_id for row in validation_rows],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_clean_signal_split(
    rows: Iterable[CleanSignalRow],
    *,
    mode: str,
    seed: int,
) -> CleanSignalSplit:
    """Build a fixed complete-family or within-family-variant split."""

    if mode not in VALID_TRAIN_SPLIT_MODES:
        raise CleanSignalTrainError(f"split mode must be one of {sorted(VALID_TRAIN_SPLIT_MODES)}")
    if seed < 0:
        raise CleanSignalTrainError("split seed must be non-negative")
    candidates = sorted((row for row in rows if row.split != "test"), key=lambda row: row.row_id)
    if mode == "complete_family":
        train_rows = [row for row in candidates if row.split == "train"]
        validation_rows = [row for row in candidates if row.split == "validation"]
        train_families = {row.family for row in train_rows}
        validation_families = {row.family for row in validation_rows}
        overlap = sorted(train_families & validation_families)
        if overlap:
            raise CleanSignalTrainError(f"complete-family split leaks families: {overlap}")
    else:
        by_family: dict[str, list[CleanSignalRow]] = {}
        for row in candidates:
            by_family.setdefault(row.family, []).append(row)
        train_rows = []
        validation_rows = []
        for family, family_rows in sorted(by_family.items()):
            if len(family_rows) < 2:
                raise CleanSignalTrainError(
                    f"within-family split requires at least two variants for family {family!r}"
                )
            digest = hashlib.sha256(f"{seed}:{family}".encode()).digest()
            validation_index = int.from_bytes(digest[:8], "little") % len(family_rows)
            validation_rows.append(family_rows[validation_index])
            train_rows.extend(row for index, row in enumerate(family_rows) if index != validation_index)

    train_rows = sorted(train_rows, key=lambda row: row.row_id)
    validation_rows = sorted(validation_rows, key=lambda row: row.row_id)
    if not train_rows or not validation_rows:
        raise CleanSignalTrainError("split must contain both train and validation rows")
    identity = _split_identity(mode, seed, train_rows, validation_rows)
    return CleanSignalSplit(tuple(train_rows), tuple(validation_rows), mode, seed, identity)


def select_clean_signal_training_rows(
    rows: Iterable[CleanSignalRow],
    *,
    count: int,
    seed: int,
) -> tuple[CleanSignalRow, ...]:
    """Select one deterministic training subset reused by every matrix cell."""

    candidates = sorted(rows, key=lambda row: row.row_id)
    if count < 1 or count > len(candidates):
        raise CleanSignalTrainError(f"train size {count} must be within 1..{len(candidates)}")
    order = np.random.default_rng(seed).permutation(len(candidates))
    return tuple(sorted((candidates[int(index)] for index in order[:count]), key=lambda row: row.row_id))


class CleanSignalDataset(Dataset[dict[str, Any]]):
    """Lazy NPZ dataset that assembles exactly four observation channels."""

    def __init__(self, rows: Iterable[CleanSignalRow]) -> None:
        self.rows = tuple(rows)
        if not self.rows:
            raise CleanSignalTrainError("clean-signal dataset cannot be empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with np.load(row.npz_path, allow_pickle=False) as payload:
            luma = np.asarray(payload[LUMA_SIGNAL], dtype=np.float32)
            gradient = np.asarray(payload[GRADIENT_SIGNAL], dtype=np.float32)
            confidence = np.asarray(payload[CONFIDENCE_SIGNAL], dtype=np.float32)
            coarse = np.asarray(payload[COARSE_SIGNAL], dtype=np.float32)
            detail = np.asarray(payload[DETAIL_SIGNAL], dtype=np.float32)
            relative = np.asarray(payload[RELATIVE_HEIGHT_SIGNAL], dtype=np.float32)
        inputs = np.ascontiguousarray(np.concatenate((luma[None], gradient, confidence[None]), axis=0))
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": {
                "relative_height_257": torch.from_numpy(relative),
                "coarse_relief_257": torch.from_numpy(coarse),
                "detail_residual_257": torch.from_numpy(detail),
            },
            "row_id": row.row_id,
            "family": row.family,
            "complexity_bucket": row.complexity_bucket,
        }


def _device(config: CleanSignalTrainConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise CleanSignalTrainError("device=cuda requested but CUDA is unavailable")
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _group_metrics(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record[key]), []).append(record)
    result: dict[str, dict[str, Any]] = {}
    for name, values in sorted(groups.items()):
        model_mae = float(np.mean([float(value["final_height_mae"]) for value in values]))
        baseline_mae = float(np.mean([float(value["tile_mean_baseline_mae"]) for value in values]))
        result[name] = {
            "row_count": len(values),
            "final_height_mae": model_mae,
            "coarse_mae": float(np.mean([float(value["coarse_mae"]) for value in values])),
            "detail_mae": float(np.mean([float(value["detail_mae"]) for value in values])),
            "tile_mean_baseline_mae": baseline_mae,
            "improvement_vs_tile_mean": None if baseline_mae <= 1e-12 else 1.0 - model_mae / baseline_mae,
        }
    return result


def evaluate_clean_signal_model(
    model: nn.Module,
    rows: Iterable[CleanSignalRow],
    *,
    profile: str = PARITY_PROFILE,
    config: V7GuidanceConfig | None = None,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Evaluate every output and loss component with family/bucket breakdowns."""

    if batch_size < 1:
        raise CleanSignalTrainError("batch_size must be positive")
    selected_config = config if config is not None else get_clean_signal_loss_config(profile)
    device_obj = torch.device(device)
    selected_rows = tuple(rows)
    dataset = CleanSignalDataset(selected_rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device_obj)
    model.eval()
    records: list[dict[str, Any]] = []
    component_sums: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device_obj)
            targets = {name: value.to(device_obj) for name, value in batch["targets"].items()}
            predictions = model(inputs)
            if not isinstance(predictions, CleanSignalPredictions):
                raise CleanSignalTrainError("clean-signal model must return CleanSignalPredictions")
            total, components = clean_signal_loss(predictions, targets, profile=profile, config=selected_config)
            batch_metrics = loss_metrics({**components, "total": total})
            for name, value in batch_metrics.items():
                component_sums[name] = component_sums.get(name, 0.0) + value * inputs.shape[0]
            target_height = targets["relative_height_257"].detach().cpu()
            target_coarse = targets["coarse_relief_257"].detach().cpu()
            target_detail = targets["detail_residual_257"].detach().cpu()
            predicted_height = predictions.height_prediction_257.detach().cpu()
            predicted_coarse = predictions.coarse_prediction_257.detach().cpu()
            predicted_detail = predictions.detail_prediction_257.detach().cpu()
            row_ids = list(batch["row_id"])
            families = list(batch["family"])
            buckets = list(batch["complexity_bucket"])
            for index, row_id in enumerate(row_ids):
                target = target_height[index]
                baseline = float((target - target.mean()).abs().mean())
                records.append(
                    {
                        "row_id": str(row_id),
                        "family": str(families[index]),
                        "complexity_bucket": str(buckets[index]),
                        "final_height_mae": float((predicted_height[index] - target).abs().mean()),
                        "coarse_mae": float((predicted_coarse[index] - target_coarse[index]).abs().mean()),
                        "detail_mae": float((predicted_detail[index] - target_detail[index]).abs().mean()),
                        "tile_mean_baseline_mae": baseline,
                    }
                )
    row_count = len(records)
    if row_count == 0:
        raise CleanSignalTrainError("evaluation produced no rows")
    return {
        "loss_profile": selected_config.as_dict(),
        "row_count": row_count,
        "final_height_mae": float(np.mean([record["final_height_mae"] for record in records])),
        "coarse_mae": float(np.mean([record["coarse_mae"] for record in records])),
        "detail_mae": float(np.mean([record["detail_mae"] for record in records])),
        "tile_mean_baseline_mae": float(np.mean([record["tile_mean_baseline_mae"] for record in records])),
        "loss_components": {name: value / row_count for name, value in sorted(component_sums.items())},
        "by_family": _group_metrics(records, "family"),
        "by_complexity_bucket": _group_metrics(records, "complexity_bucket"),
        "rows": records,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_fresh_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing training output: {output}")
    output.mkdir(parents=True, exist_ok=True)


ModelBuilder = Callable[..., tuple[nn.Module, dict[str, Any]]]


def train_clean_signal_model(
    train_rows: Iterable[CleanSignalRow],
    validation_rows: Iterable[CleanSignalRow],
    *,
    architecture: str,
    profile: str = PARITY_PROFILE,
    output: str | Path,
    config: CleanSignalTrainConfig | None = None,
    split: CleanSignalSplit | Mapping[str, Any] | None = None,
    model_builder: ModelBuilder | None = None,
    model_profile: str = "tiny",
) -> dict[str, Any]:
    """Train one architecture/profile cell and persist best/last checkpoints."""

    selected_config = config or CleanSignalTrainConfig()
    selected_loss = get_clean_signal_loss_config(profile)
    train_selection = tuple(train_rows)
    validation_selection = tuple(validation_rows)
    if not train_selection or not validation_selection:
        raise CleanSignalTrainError("training requires non-empty train and validation rows")
    output_path = Path(output)
    _require_fresh_output(output_path)
    _set_seed(selected_config.seed)
    device = _device(selected_config)
    builder = model_builder or build_clean_signal_model
    model, identity = builder(architecture, profile=model_profile)
    model = model.to(device)
    train_loader = DataLoader(
        CleanSignalDataset(train_selection),
        batch_size=min(selected_config.batch_size, len(train_selection)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(selected_config.seed),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=selected_config.learning_rate,
        weight_decay=selected_config.weight_decay,
    )
    best_score = float("inf")
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    stale_epochs = 0
    for epoch in range(1, selected_config.epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            targets = {name: value.to(device) for name, value in batch["targets"].items()}
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            total, _ = clean_signal_loss(predictions, targets, profile=profile, config=selected_loss)
            total.backward()
            optimizer.step()
            train_total += float(total.detach().cpu()) * inputs.shape[0]
        validation_metrics = evaluate_clean_signal_model(
            model,
            validation_selection,
            profile=profile,
            config=selected_loss,
            batch_size=selected_config.batch_size,
            device=device,
        )
        epoch_record = {
            "epoch": epoch,
            "train_total_loss": train_total / len(train_selection),
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        score = float(validation_metrics["final_height_mae"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = validation_metrics
            stale_epochs = 0
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "epoch": epoch,
                    "architecture": architecture,
                    "model_identity": identity,
                    "loss_profile": selected_loss.as_dict(),
                    "seed": selected_config.seed,
                    "split": split.as_dict() if isinstance(split, CleanSignalSplit) else dict(split or {}),
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                },
                output_path / "checkpoint_best.pt",
            )
        else:
            stale_epochs += 1
        if stale_epochs >= selected_config.patience:
            break
    final_metrics = evaluate_clean_signal_model(
        model,
        validation_selection,
        profile=profile,
        config=selected_loss,
        batch_size=selected_config.batch_size,
        device=device,
    )
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": len(history),
            "architecture": architecture,
            "model_identity": identity,
            "loss_profile": selected_loss.as_dict(),
            "seed": selected_config.seed,
            "split": split.as_dict() if isinstance(split, CleanSignalSplit) else dict(split or {}),
            "model_state_dict": model.state_dict(),
            "validation_metrics": final_metrics,
        },
        output_path / "checkpoint_last.pt",
    )
    if best_metrics is None:
        raise CleanSignalTrainError("training completed without a best checkpoint")
    report = {
        "schema": TRAIN_SCHEMA,
        "architecture": architecture,
        "model_profile": model_profile,
        "model_identity": identity,
        "loss_profile": selected_loss.as_dict(),
        "seed": selected_config.seed,
        "device": str(device),
        "split": split.as_dict() if isinstance(split, CleanSignalSplit) else dict(split or {}),
        "train_row_count": len(train_selection),
        "validation_row_count": len(validation_selection),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "final_validation": final_metrics,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (output_path / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "CHECKPOINT_SCHEMA",
    "TRAIN_SCHEMA",
    "VALID_TRAIN_SPLIT_MODES",
    "CleanSignalDataset",
    "CleanSignalRow",
    "CleanSignalSplit",
    "CleanSignalTrainConfig",
    "CleanSignalTrainError",
    "build_clean_signal_split",
    "evaluate_clean_signal_model",
    "load_clean_signal_rows",
    "select_clean_signal_training_rows",
    "train_clean_signal_model",
]
