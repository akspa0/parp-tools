"""Train a transformer-based roof-family identifier on curated roof exemplars.

This is the first host for the object-identification lane in spec 025.
It trains a lightweight image-classification model over roof exemplar crops.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import zarr
import zarr.storage

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except Exception as ex:  # pragma: no cover - explicit runtime guidance
    raise RuntimeError(
        "transformers is required for train_v18_object_roof_identifier.py. "
        "Install dependencies with uv sync in wow-viewer/data-harvester."
    ) from ex

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LIBRARY_ROOT = _PROJECT_ROOT / "output" / "datasets" / "object_roof_library"
_DEFAULT_MODEL_ROOT = _PROJECT_ROOT / "models" / "v18" / "object_roof_identifier"


@dataclass(frozen=True)
class SampleRef:
    sample_index: int
    label: int


class RoofExemplarDataset(Dataset):
    def __init__(
        self,
        *,
        roof_rgb: np.ndarray,
        roof_mask: np.ndarray,
        sample_indices: list[int],
        labels: list[int],
        processor,
        apply_mask: bool,
    ) -> None:
        self._roof_rgb = roof_rgb
        self._roof_mask = roof_mask
        self._sample_indices = sample_indices
        self._labels = labels
        self._processor = processor
        self._apply_mask = bool(apply_mask)

    def __len__(self) -> int:
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_idx = int(self._sample_indices[idx])
        image = self._roof_rgb[sample_idx]
        if self._apply_mask:
            mask = np.clip(self._roof_mask[sample_idx], 0.0, 1.0)
            mask3 = np.repeat(mask[:, :, None], 3, axis=2)
            image = np.clip(image.astype(np.float32) * mask3, 0.0, 255.0).astype(np.uint8)
        encoded = self._processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(int(self._labels[idx]), dtype=torch.long),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V18 object-roof identifier (transformers host).")
    parser.add_argument("--library-dir", type=Path, required=False, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.12)
    parser.add_argument("--min-family-samples", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--apply-roof-mask", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_library_dir(args: argparse.Namespace) -> Path:
    if args.library_dir is not None:
        return Path(args.library_dir)
    runs = sorted(_DEFAULT_LIBRARY_ROOT.glob("*/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError(
            "No roof-library run found. Build one first with build_v18_object_roof_library.py"
        )
    return runs[0].parent


def _seed_everything(seed: int) -> None:
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
            raise RuntimeError("Requested cuda but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_catalog_rows(library_dir: Path) -> list[dict[str, Any]]:
    path = library_dir / "roof_exemplars.parquet"
    if not path.exists():
        raise RuntimeError(f"Missing roof exemplar catalog: {path}")
    table = pq.read_table(str(path))
    rows = [{column: table.column(column)[idx].as_py() for column in table.column_names} for idx in range(table.num_rows)]
    if not rows:
        raise RuntimeError("Roof exemplar catalog is empty")
    return rows


def _load_roof_visual_arrays(library_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    zarr_path = library_dir / "object_visual.zarr"
    if not zarr_path.exists():
        raise RuntimeError(f"Missing object visual store: {zarr_path}")
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    try:
        if "roof_rgb" not in root:
            raise RuntimeError("object_visual.zarr is missing roof_rgb")
        if "roof_mask" not in root:
            raise RuntimeError("object_visual.zarr is missing roof_mask")
        rgb = root["roof_rgb"][:].astype(np.uint8)
        mask = root["roof_mask"][:].astype(np.float32)
    finally:
        store.close()
    return rgb, mask


def _build_label_space(rows: list[dict[str, Any]], min_family_samples: int) -> tuple[dict[str, int], dict[int, str]]:
    counts: dict[str, int] = {}
    for row in rows:
        family_id = str(row.get("family_id", ""))
        if not family_id:
            continue
        counts[family_id] = counts.get(family_id, 0) + 1

    kept_families = sorted([family for family, count in counts.items() if count >= int(min_family_samples)])
    if len(kept_families) < 2:
        raise RuntimeError(
            f"Need at least 2 families with >= {min_family_samples} samples; found {len(kept_families)}"
        )

    family_to_label = {family: idx for idx, family in enumerate(kept_families)}
    label_to_family = {idx: family for family, idx in family_to_label.items()}
    return family_to_label, label_to_family


def _train_val_split(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_val = int(round(float(n) * float(val_fraction)))
    n_val = max(1, min(n - 1, n_val))
    val_idx = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())
    return train_idx, val_idx


def _build_datasets(
    *,
    rows: list[dict[str, Any]],
    roof_rgb: np.ndarray,
    roof_mask: np.ndarray,
    family_to_label: dict[str, int],
    processor,
    seed: int,
    val_fraction: float,
    max_samples: int,
    apply_roof_mask: bool,
) -> tuple[RoofExemplarDataset, RoofExemplarDataset, dict[str, Any]]:
    samples: list[SampleRef] = []
    for idx, row in enumerate(rows):
        family_id = str(row.get("family_id", ""))
        label = family_to_label.get(family_id)
        if label is None:
            continue
        samples.append(SampleRef(sample_index=idx, label=label))

    if max_samples > 0 and len(samples) > int(max_samples):
        rng = np.random.RandomState(seed)
        chosen = sorted(rng.choice(np.arange(len(samples)), size=int(max_samples), replace=False).tolist())
        samples = [samples[idx] for idx in chosen]

    if len(samples) < 8:
        raise RuntimeError("Not enough samples to train object-roof identifier")

    train_positions, val_positions = _train_val_split(len(samples), val_fraction=val_fraction, seed=seed)

    train_indices = [samples[pos].sample_index for pos in train_positions]
    train_labels = [samples[pos].label for pos in train_positions]
    val_indices = [samples[pos].sample_index for pos in val_positions]
    val_labels = [samples[pos].label for pos in val_positions]

    train_ds = RoofExemplarDataset(
        roof_rgb=roof_rgb,
        roof_mask=roof_mask,
        sample_indices=train_indices,
        labels=train_labels,
        processor=processor,
        apply_mask=apply_roof_mask,
    )
    val_ds = RoofExemplarDataset(
        roof_rgb=roof_rgb,
        roof_mask=roof_mask,
        sample_indices=val_indices,
        labels=val_labels,
        processor=processor,
        apply_mask=apply_roof_mask,
    )

    evidence = {
        "samples_total": len(samples),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "apply_roof_mask": bool(apply_roof_mask),
    }
    return train_ds, val_ds, evidence


def _evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss_sum += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.shape[0])
    loss_mean = loss_sum / max(1, len(loader))
    acc = float(correct) / float(max(1, total))
    return loss_mean, acc


def main() -> None:
    args = _parse_args()
    _seed_everything(int(args.seed))
    device = _resolve_device(args.device)

    library_dir = _resolve_library_dir(args)
    rows = _load_catalog_rows(library_dir)
    roof_rgb, roof_mask = _load_roof_visual_arrays(library_dir)
    family_to_label, label_to_family = _build_label_space(rows, min_family_samples=int(args.min_family_samples))

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_ds, val_ds, split_evidence = _build_datasets(
        rows=rows,
        roof_rgb=roof_rgb,
        roof_mask=roof_mask,
        family_to_label=family_to_label,
        processor=processor,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
        max_samples=int(args.max_samples),
        apply_roof_mask=bool(args.apply_roof_mask),
    )

    run_name = args.run_name or f"roof_identifier_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = _DEFAULT_MODEL_ROOT / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(family_to_label),
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(train_loader))
        val_loss, val_acc = _evaluate(model, val_loader, device=device)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(entry)
        print(json.dumps(entry))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_name": str(args.model_name),
                    "family_to_label": family_to_label,
                    "label_to_family": label_to_family,
                    "library_dir": str(library_dir),
                    "best_val_loss": best_val_loss,
                },
                ckpt_dir / "object_roof_identifier_best.pt",
            )

    torch.save(
        {
            "epoch": int(args.epochs),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": str(args.model_name),
            "family_to_label": family_to_label,
            "label_to_family": label_to_family,
            "library_dir": str(library_dir),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        ckpt_dir / "object_roof_identifier_last.pt",
    )

    summary = {
        "run_name": run_name,
        "library_dir": str(library_dir),
        "model_name": str(args.model_name),
        "device": str(device),
        "family_count": len(family_to_label),
        "apply_roof_mask": bool(args.apply_roof_mask),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "split_evidence": split_evidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config.snapshot.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (run_dir / "label_space.json").write_text(
        json.dumps(
            {
                "family_to_label": family_to_label,
                "label_to_family": {str(k): v for k, v in label_to_family.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save HF-compatible model artifacts for downstream fallback tooling.
    model.save_pretrained(run_dir / "hf_model")
    processor.save_pretrained(run_dir / "hf_model")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
