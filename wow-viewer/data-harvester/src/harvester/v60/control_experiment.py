"""Control-v1 loading, fixed-family selection, and baseline evaluation.

This module is deliberately limited to the first terrain signal:
``terrain_shadow_256`` -> ``height_257``.  It reads the C#-owned NPZ corpus and
never imports object-sieve or object-marker targets.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from harvester.v50.height_relative_model import encode_relative_height
from harvester.v60.control_corpus import (
    INPUT_SHAPE,
    INPUT_SIGNAL,
    TARGET_SHAPE,
    TARGET_SIGNAL,
    load_control_manifest,
    validate_control_corpus,
)


@dataclass(frozen=True)
class ControlRow:
    """Manifest metadata for one control NPZ row."""

    row_id: str
    control_family: str
    complexity_bucket: str
    variant: int
    split: str
    npz_path: Path


def load_control_rows(corpus_root: str | Path) -> tuple[Path, list[ControlRow]]:
    """Load and structurally validate the control manifest and row paths."""

    root = Path(corpus_root)
    manifest = load_control_manifest(root)
    validation = validate_control_corpus(root)
    if not validation["valid"]:
        failures = "; ".join(validation["failures"][:8])
        raise ValueError(f"invalid control corpus: {failures}")

    rows: list[ControlRow] = []
    for raw in manifest["rows"]:
        rows.append(
            ControlRow(
                row_id=str(raw["row_id"]),
                control_family=str(raw["control_family"]),
                complexity_bucket=str(raw["complexity_bucket"]),
                variant=int(raw.get("variant", 0)),
                split=str(raw["split"]),
                npz_path=root / str(raw["npz"]),
            )
        )
    return root, rows


def select_training_rows(rows: Iterable[ControlRow], count: int, seed: int) -> list[ControlRow]:
    """Select a deterministic training subset without changing manifest holdouts."""

    return select_training_schedule(rows, [count], seed)[count]


def select_training_schedule(
    rows: Iterable[ControlRow], counts: Iterable[int], seed: int
) -> dict[int, list[ControlRow]]:
    """Select nested deterministic training subsets for an architecture bakeoff.

    One seeded permutation is shared by every requested count, so the 8-row set is contained in
    the 16-row set, and so on.  The fixed validation families are never eligible for selection.
    """

    candidates = sorted((row for row in rows if row.split == "train"), key=lambda row: row.row_id)
    requested = list(counts)
    if not requested or any(count < 1 for count in requested) or len(set(requested)) != len(requested):
        raise ValueError("training sizes must be unique positive integers")
    if any(count > len(candidates) for count in requested):
        largest = max(requested)
        raise ValueError(f"training size {largest} exceeds available train rows {len(candidates)}")
    order = np.random.default_rng(seed).permutation(len(candidates))
    shuffled = [candidates[int(index)] for index in order]
    return {
        count: sorted(shuffled[:count], key=lambda row: row.row_id)
        for count in sorted(requested)
    }


def fixed_validation_rows(rows: Iterable[ControlRow]) -> list[ControlRow]:
    """Return the manifest validation split, preserving complete-family holdouts."""

    selected = sorted((row for row in rows if row.split == "validation"), key=lambda row: row.row_id)
    if not selected:
        raise ValueError("control manifest has no validation rows")
    train_families = {row.control_family for row in rows if row.split == "train"}
    validation_families = {row.control_family for row in selected}
    overlap = sorted(train_families & validation_families)
    if overlap:
        raise ValueError(f"control family leakage between train and validation: {overlap}")
    return selected


def load_pair(row: ControlRow) -> tuple[np.ndarray, np.ndarray]:
    """Read one finite, shape-checked shadow/height pair and normalize its target."""

    with np.load(row.npz_path, allow_pickle=False) as payload:
        if INPUT_SIGNAL not in payload or TARGET_SIGNAL not in payload:
            raise ValueError(f"{row.row_id} is missing the terrain signal pair")
        shadow = np.asarray(payload[INPUT_SIGNAL], dtype=np.float32)
        height = np.asarray(payload[TARGET_SIGNAL], dtype=np.float32)
    if shadow.shape != INPUT_SHAPE or height.shape != TARGET_SHAPE:
        raise ValueError(
            f"{row.row_id} has shapes shadow={shadow.shape}, height={height.shape}; "
            f"expected {INPUT_SHAPE}/{TARGET_SHAPE}"
        )
    if not np.isfinite(shadow).all() or not np.isfinite(height).all():
        raise ValueError(f"{row.row_id} contains non-finite terrain values")
    if float(shadow.min()) < -1e-6 or float(shadow.max()) > 1.000001:
        raise ValueError(f"{row.row_id} shadow is outside [0, 1]")
    normalized, _, _ = encode_relative_height(height)
    return shadow, normalized


class ControlDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for the terrain-only control pair."""

    def __init__(self, rows: Iterable[ControlRow]) -> None:
        self.rows = list(rows)
        if not self.rows:
            raise ValueError("control dataset cannot be empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shadow, target = load_pair(self.rows[index])
        return torch.from_numpy(shadow[None, ...]), torch.from_numpy(target)


def _row_mae(row: ControlRow, prediction: np.ndarray) -> float:
    _, target = load_pair(row)
    return float(np.abs(prediction - target).mean())


def tile_mean_baseline(rows: Iterable[ControlRow]) -> dict[str, Any]:
    """Evaluate the per-tile mean-height baseline with family/variant breakdowns."""

    row_metrics: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.row_id):
        _, target = load_pair(row)
        mean = float(target.mean())
        mae = _row_mae(row, np.full_like(target, mean))
        row_metrics.append(
            {
                "row_id": row.row_id,
                "control_family": row.control_family,
                "variant": row.variant,
                "mae": mae,
                "ambiguity": bool(float(target.max() - target.min()) < 0.05),
            }
        )

    def grouped(key: str) -> dict[str, float]:
        groups: dict[str, list[float]] = {}
        for item in row_metrics:
            groups.setdefault(str(item[key]), []).append(float(item["mae"]))
        return {name: float(np.mean(values)) for name, values in sorted(groups.items())}

    return {
        "mae": float(np.mean([item["mae"] for item in row_metrics])),
        "by_family": grouped("control_family"),
        "by_variant": grouped("variant"),
        "ambiguous_rows": [item["row_id"] for item in row_metrics if item["ambiguity"]],
        "rows": row_metrics,
    }


def split_summary(train_rows: Iterable[ControlRow], validation_rows: Iterable[ControlRow]) -> dict[str, Any]:
    """Return explicit split provenance for experiment reports."""

    train = list(train_rows)
    validation = list(validation_rows)
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_families": sorted({row.control_family for row in train}),
        "validation_families": sorted({row.control_family for row in validation}),
        "family_overlap": sorted(
            {row.control_family for row in train} & {row.control_family for row in validation}
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
