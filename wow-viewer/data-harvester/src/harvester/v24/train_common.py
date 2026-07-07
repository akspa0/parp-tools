"""Shared training utilities for the V24 stages (determinism, logging, splits)."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_determinism(seed: int, strict: bool = True) -> None:
    """Seed everything and enable deterministic algorithms.

    ``strict=True`` (inference) errors on any nondeterministic op — the FR-014 /
    FR-019 contract. Training uses ``strict=False`` because some CUDA backward
    kernels (bilinear interpolate) have no deterministic implementation; the
    determinism requirement is on inference outputs, not gradient replay.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_rows(rows: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic train/val split (val is never empty when rows >= 2)."""
    rng = np.random.default_rng(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) >= 2 else 0
    return shuffled[n_val:], shuffled[:n_val]


class RunLogger:
    """Writes loss_history.jsonl + console lines for a training run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = run_dir / "loss_history.jsonl"
        self._history = self.history_path.open("a", encoding="utf-8")
        self.started = time.time()

    def log_epoch(self, epoch: int, **metrics: Any) -> None:
        record = {"epoch": epoch, "elapsed_s": round(time.time() - self.started, 2), **metrics}
        self._history.write(json.dumps(record) + "\n")
        self._history.flush()
        rendered = " ".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
        )
        print(f"epoch {epoch}: {rendered}", flush=True)

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def close(self) -> None:
        self._history.close()


class EarlyStopping:
    """Stop training when validation loss stops improving (overtraining guard).

    Tracks the best validation loss and the epoch it was set. After ``patience``
    epochs without improvement (``val_loss < best - min_delta``), ``stopped`` is
    set and the trainer should halt. Also flags ``overtraining`` when the
    train/val gap widens in the classic overfit direction: train loss keeps
    falling while val loss rises. ``patience <= 0`` disables early stopping
    (the run goes the full ``--epochs``).
    """

    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.epochs_since_best = 0
        self.stopped = False
        self.overtraining = False
        self._prev_train: float | None = None
        self._prev_val: float | None = None

    def step(self, epoch: int, val_loss: float, train_loss: float | None = None) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.epochs_since_best = 0
        else:
            self.epochs_since_best += 1

        if train_loss is not None and self._prev_train is not None and self._prev_val is not None:
            if val_loss > self._prev_val and train_loss < self._prev_train:
                self.overtraining = True
        self._prev_train = train_loss
        self._prev_val = val_loss

        if self.patience > 0 and self.epochs_since_best >= self.patience:
            self.stopped = True
        return self.stopped


def peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9
