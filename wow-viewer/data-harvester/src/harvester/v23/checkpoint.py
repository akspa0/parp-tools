"""V23 checkpoint helpers for Spec 089 Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import torch


def resolve_commit_sha(repo_root: str | Path | None = None) -> str:
    """Return the current git SHA, or ``unknown`` when git metadata is unavailable."""
    cwd = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[4]
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return "unknown"
    return output.strip() or "unknown"


def path_hash(path: str | Path | None) -> str:
    if path is None:
        return ""
    return hashlib.sha256(Path(path).as_posix().encode("utf-8")).hexdigest()


@dataclass
class V23Checkpoint:
    """Serializable V23 checkpoint payload."""

    config: dict[str, Any]
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    epoch: int
    best_val: float | None = None
    scheduler_state: dict[str, Any] | None = None
    scaler_state: dict[str, Any] | None = None
    global_step: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": "v23",
            "config": dict(self.config),
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "best_val": self.best_val,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "V23Checkpoint":
        return cls(
            config=dict(payload.get("config", {})),
            model_state=dict(payload.get("model_state", {})),
            optimizer_state=dict(payload.get("optimizer_state", {})),
            scheduler_state=payload.get("scheduler_state"),
            scaler_state=payload.get("scaler_state"),
            epoch=int(payload.get("epoch", 0)),
            global_step=int(payload.get("global_step", 0)),
            best_val=payload.get("best_val"),
        )


def save_checkpoint(path: str | Path, checkpoint: V23Checkpoint) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.to_payload(), target)
    return target


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> V23Checkpoint:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary")
    return V23Checkpoint.from_payload(payload)


__all__ = [
    "V23Checkpoint",
    "load_checkpoint",
    "path_hash",
    "resolve_commit_sha",
    "save_checkpoint",
]
