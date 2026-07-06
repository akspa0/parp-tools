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


def peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9
