"""Reviewable validation artifacts for direct relative-height models (Spec 114 T017a).

The evaluator keeps the model contract honest: RGB is the only inference input, while every
validation row receives numeric model-vs-baseline metrics. Sheets use a fixed [0, 1] height scale
instead of independently stretching prediction and truth, so a flat or low-contrast prediction
cannot look deceptively detailed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from harvester.v50.feature_stores import FeatureBinding, as_bindings, feature_channels_for_row
from harvester.v50.height_relative_model import encode_relative_height

DEFAULT_BORDER_WIDTH = 8


def compute_row_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    border_width: int = DEFAULT_BORDER_WIDTH,
) -> dict[str, float]:
    """Compute non-cherry-picked row metrics against both truth and the tile-mean baseline."""
    pred = np.asarray(predicted, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(f"predicted/target must be matching 2D arrays, got {pred.shape}/{truth.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("predicted/target must be finite")

    absolute = np.abs(pred - truth)
    pred_dx = pred[:, 1:] - pred[:, :-1]
    truth_dx = truth[:, 1:] - truth[:, :-1]
    pred_dy = pred[1:, :] - pred[:-1, :]
    truth_dy = truth[1:, :] - truth[:-1, :]
    gradient_mae = 0.5 * (
        float(np.abs(pred_dx - truth_dx).mean())
        + float(np.abs(pred_dy - truth_dy).mean())
    )

    width = min(max(int(border_width), 1), max(1, min(truth.shape) // 2))
    border = np.zeros(truth.shape, dtype=bool)
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True

    baseline = np.full_like(truth, float(truth.mean()))
    model_mae = float(absolute.mean())
    baseline_mae = float(np.abs(baseline - truth).mean())
    return {
        "mae": model_mae,
        "gradient_mae": gradient_mae,
        "border_mae": float(absolute[border].mean()),
        # Interior complement of the border ring: SC-002 compares held-out border error against
        # the interior error distribution's 95th percentile.
        "interior_mae": float(absolute[~border].mean()),
        "tile_mean_baseline_mae": baseline_mae,
        "mae_delta_vs_baseline": model_mae - baseline_mae,
    }


def select_error_quantile_rows(records: list[dict[str, Any]], count: int) -> list[int]:
    """Select an explicitly error-stratified review set, always including best and worst rows."""
    if count <= 0 or not records:
        return []
    ordered = sorted(records, key=lambda record: (float(record["mae"]), int(record["row_id"])))
    if len(ordered) <= count:
        return [int(record["row_id"]) for record in ordered]
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [int(ordered[position]["row_id"]) for position in positions]


def select_fixed_preview_rows(val_rows: list[int], count: int) -> list[int]:
    """Select stable validation rows before metrics exist; never depends on model quality."""
    if count <= 0 or not val_rows:
        return []
    if len(val_rows) <= count:
        return list(val_rows)
    positions = np.linspace(0, len(val_rows) - 1, count).round().astype(int)
    return [int(val_rows[position]) for position in positions]


def _gray_fixed(values: np.ndarray) -> np.ndarray:
    gray = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    gray = np.rint(gray * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _signed_error_rgb(error: np.ndarray, scale: float = 0.5) -> np.ndarray:
    normalized = np.clip(np.asarray(error, dtype=np.float32) / max(scale, 1e-6), -1.0, 1.0)
    magnitude = np.abs(normalized)
    image = np.full((*normalized.shape, 3), 255.0, dtype=np.float32)
    positive = normalized >= 0
    image[..., 1][positive] *= 1.0 - magnitude[positive]
    image[..., 2][positive] *= 1.0 - magnitude[positive]
    image[..., 0][~positive] *= 1.0 - magnitude[~positive]
    image[..., 1][~positive] *= 1.0 - magnitude[~positive]
    return np.rint(image).astype(np.uint8)


def _absolute_error_rgb(error: np.ndarray, scale: float = 0.5) -> np.ndarray:
    magnitude = np.clip(np.asarray(error, dtype=np.float32) / max(scale, 1e-6), 0.0, 1.0)
    image = np.zeros((*magnitude.shape, 3), dtype=np.float32)
    image[..., 0] = 255.0 * magnitude
    image[..., 1] = 180.0 * np.sqrt(magnitude)
    return np.rint(image).astype(np.uint8)


def render_validation_sheet(
    samples: list[dict[str, Any]],
    output: Path,
    *,
    title: str,
    panel_size: int = 192,
) -> None:
    """Write [RGB, truth, prediction, baseline, signed error, absolute error] review rows."""
    if not samples:
        raise ValueError("cannot render an empty validation sheet")
    labels = ["Minimap RGB", "Truth [0,1]", "Prediction [0,1]", "Tile-mean baseline", "Signed error", "Abs error"]
    label_width = 210
    header_height = 42
    row_height = panel_size + 4
    gap = 4
    width = label_width + (len(labels) * panel_size) + ((len(labels) - 1) * gap)
    height = header_height + (len(samples) * row_height)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 3), title, fill=(20, 20, 20), font=font)
    for column, label in enumerate(labels):
        x = label_width + column * (panel_size + gap)
        draw.text((x + 3, 23), label, fill=(30, 30, 30), font=font)

    for row_index, sample in enumerate(samples):
        y = header_height + row_index * row_height
        metrics = sample["metrics"]
        label = (
            f"{sample['label']}\n"
            f"MAE {metrics['mae']:.4f}  base {metrics['tile_mean_baseline_mae']:.4f}\n"
            f"grad {metrics['gradient_mae']:.4f}  border {metrics['border_mae']:.4f}"
        )
        draw.multiline_text((5, y + 5), label, fill=(20, 20, 20), font=font, spacing=3)
        target = np.asarray(sample["target"], dtype=np.float32)
        predicted = np.asarray(sample["predicted"], dtype=np.float32)
        baseline = np.full_like(target, float(target.mean()))
        panels = [
            np.asarray(sample["rgb"], dtype=np.uint8),
            _gray_fixed(target),
            _gray_fixed(predicted),
            _gray_fixed(baseline),
            _signed_error_rgb(predicted - target),
            _absolute_error_rgb(np.abs(predicted - target)),
        ]
        for column, panel in enumerate(panels):
            image = Image.fromarray(panel, mode="RGB").resize(
                (panel_size, panel_size), Image.Resampling.BILINEAR
            )
            x = label_width + column * (panel_size + gap)
            canvas.paste(image, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


class _ValidationDataset(Dataset):
    def __init__(
        self,
        group: Any,
        rows: list[int],
        *,
        feature_group: Any | None = None,
        feature_row_to_position: dict[int, int] | None = None,
        feature_bindings: list[FeatureBinding] | None = None,
    ) -> None:
        self.group = group
        self.rows = rows
        self.feature_bindings = as_bindings(
            feature_bindings, feature_group, feature_row_to_position
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row_id = int(self.rows[position])
        rgb = np.asarray(self.group["minimap_rgb"][row_id], dtype=np.float32) / 255.0
        channels = torch.from_numpy(rgb).permute(2, 0, 1)
        feats = feature_channels_for_row(self.feature_bindings, row_id)
        if feats is not None:
            channels = torch.cat([channels, torch.from_numpy(feats)], dim=0)
        target, _, _ = encode_relative_height(np.asarray(self.group["height_257"][row_id]))
        return channels, torch.from_numpy(target), row_id


@torch.no_grad()
def _predict_samples(
    model: torch.nn.Module,
    group: Any,
    index_rows: list[dict[str, Any]],
    row_ids: list[int],
    device: torch.device,
    *,
    use_amp: bool,
    feature_group: Any | None = None,
    feature_row_to_position: dict[int, int] | None = None,
    feature_bindings: list[FeatureBinding] | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    bindings = as_bindings(feature_bindings, feature_group, feature_row_to_position)
    model.eval()
    for row_id in row_ids:
        rgb = np.asarray(group["minimap_rgb"][row_id], dtype=np.uint8)
        target, _, _ = encode_relative_height(np.asarray(group["height_257"][row_id]))
        channels = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        feats = feature_channels_for_row(bindings, row_id)
        if feats is not None:
            channels = torch.cat([channels, torch.from_numpy(feats)], dim=0)
        tensor = channels.unsqueeze(0).to(device)
        with torch.amp.autocast(device.type, enabled=use_amp and device.type == "cuda"):
            predicted = model(tensor)[0].float().cpu().numpy()
        record = index_rows[row_id]
        samples.append(
            {
                "row_id": row_id,
                "label": f"row {row_id}  {record.get('map', '?')} {record.get('tile_x', '?')},{record.get('tile_y', '?')}",
                "rgb": rgb,
                "target": target,
                "predicted": predicted,
                "metrics": compute_row_metrics(predicted, target),
            }
        )
    return samples


@torch.no_grad()
def render_fixed_model_preview(
    model: torch.nn.Module,
    group: Any,
    index_rows: list[dict[str, Any]],
    row_ids: list[int],
    device: torch.device,
    output: Path,
    *,
    epoch: int,
    val_mae: float,
    use_amp: bool,
    feature_group: Any | None = None,
    feature_row_to_position: dict[int, int] | None = None,
    feature_bindings: list[FeatureBinding] | None = None,
) -> None:
    samples = _predict_samples(
        model, group, index_rows, row_ids, device, use_amp=use_amp,
        feature_group=feature_group, feature_row_to_position=feature_row_to_position,
        feature_bindings=feature_bindings,
    )
    render_validation_sheet(samples, output, title=f"fixed validation preview | epoch {epoch} | MAE {val_mae:.6f}")


@torch.no_grad()
def evaluate_height_model(
    model: torch.nn.Module,
    group: Any,
    index_rows: list[dict[str, Any]],
    val_rows: list[int],
    device: torch.device,
    output: Path,
    *,
    batch_size: int,
    workers: int,
    checkpoint_epoch: int,
    use_amp: bool,
    review_count: int = 8,
    feature_group: Any | None = None,
    feature_row_to_position: dict[int, int] | None = None,
    feature_bindings: list[FeatureBinding] | None = None,
) -> dict[str, Any]:
    """Evaluate every held-out row, then render honest quantile and worst-case sheets."""
    loader = DataLoader(
        _ValidationDataset(
            group, val_rows,
            feature_group=feature_group,
            feature_row_to_position=feature_row_to_position,
            feature_bindings=feature_bindings,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    records: list[dict[str, Any]] = []
    for inputs, targets, row_ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp and device.type == "cuda"):
            predictions = model(inputs).float().cpu().numpy()
        target_values = targets.numpy()
        for predicted, target, row_id_tensor in zip(predictions, target_values, row_ids, strict=True):
            row_id = int(row_id_tensor)
            index = index_rows[row_id]
            records.append(
                {
                    "row_id": row_id,
                    "map": str(index.get("map", "unknown")),
                    "tile_x": int(index.get("tile_x", -1)),
                    "tile_y": int(index.get("tile_y", -1)),
                    "minimap_source": str(index.get("minimap_source", "unknown")),
                    **compute_row_metrics(predicted, target),
                }
            )

    if not records:
        raise ValueError("validation selection contains zero rows")
    aggregate = {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in ("mae", "gradient_mae", "border_mae", "tile_mean_baseline_mae", "mae_delta_vs_baseline")
    }
    aggregate["beats_tile_mean_baseline"] = aggregate["mae"] < aggregate["tile_mean_baseline_mae"]
    aggregate["checkpoint_epoch"] = int(checkpoint_epoch)
    aggregate["val_rows"] = len(records)

    output.mkdir(parents=True, exist_ok=True)
    (output / "per_row_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    quantile_rows = select_error_quantile_rows(records, review_count)
    worst_rows = [
        int(record["row_id"])
        for record in sorted(records, key=lambda record: (-float(record["mae"]), int(record["row_id"])))[:review_count]
    ]
    render_validation_sheet(
        _predict_samples(
            model, group, index_rows, quantile_rows, device, use_amp=use_amp,
            feature_group=feature_group, feature_row_to_position=feature_row_to_position,
            feature_bindings=feature_bindings,
        ),
        output / "error_quantiles.png",
        title=f"all-validation error quantiles | checkpoint epoch {checkpoint_epoch}",
    )
    render_validation_sheet(
        _predict_samples(
            model, group, index_rows, worst_rows, device, use_amp=use_amp,
            feature_group=feature_group, feature_row_to_position=feature_row_to_position,
            feature_bindings=feature_bindings,
        ),
        output / "worst_cases.png",
        title=f"worst held-out rows | checkpoint epoch {checkpoint_epoch}",
    )
    aggregate["artifacts"] = {
        "per_row_metrics": "per_row_metrics.json",
        "error_quantiles": "error_quantiles.png",
        "worst_cases": "worst_cases.png",
    }
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main() -> int:
    """Backfill validation artifacts from an immutable direct-height checkpoint (USER runs)."""
    import argparse

    import pyarrow.parquet as pq
    import zarr

    from harvester.v50.contracts import require_store_release, validate_release
    from harvester.v50.height_relative_model import TARGET_CONTRACT_VERSION, HeightRelativeNet
    from harvester.v50.height_relative_train import (
        ARCHITECTURE_ID,
        SOURCE_CHOICES,
        TrainerContractError,
        curriculum_identity,
        require_new_output,
        select_training_rows,
        validate_curriculum_contract,
        validate_source_selection,
    )

    parser = argparse.ArgumentParser(description="v50 direct-height checkpoint evaluator (USER runs)")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_CHOICES))
    parser.add_argument("--val-key", default="split")
    parser.add_argument("--val-value", default="val")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, choices=[0], default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--release", default="v50.1", type=validate_release)
    args = parser.parse_args()

    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu only for a deliberate slow evaluation.")
    try:
        require_new_output(args.output)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index_rows = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        validate_curriculum_contract(
            attrs=dict(group.attrs),
            array_lengths={name: int(group[name].shape[0]) for name in group.array_keys()},
            index_rows=index_rows,
        )
        validate_source_selection(attrs=dict(group.attrs), source=args.source)
        selected_rows = select_training_rows(index_rows, args.source)
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc
    val_rows = [
        row_id
        for row_id in selected_rows
        if str(index_rows[row_id].get(args.val_key)) == args.val_value
    ]
    if not val_rows:
        raise SystemExit("selected source has zero validation rows")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_variant") != ARCHITECTURE_ID:
        raise SystemExit(
            f"checkpoint model_variant must be {ARCHITECTURE_ID!r}, got {checkpoint.get('model_variant')!r}"
        )
    if checkpoint.get("target_contract_version") != TARGET_CONTRACT_VERSION:
        raise SystemExit(
            "checkpoint target contract mismatch: "
            f"{checkpoint.get('target_contract_version')!r} != {TARGET_CONTRACT_VERSION!r}"
        )
    expected_identity = curriculum_identity(args.store)
    if checkpoint.get("curriculum_identity") != expected_identity:
        raise SystemExit("checkpoint curriculum identity does not match --store")

    model = HeightRelativeNet().to(device)
    model.load_state_dict(checkpoint["model"])
    use_amp = device.type == "cuda" and not args.no_amp
    summary = evaluate_height_model(
        model,
        group,
        index_rows,
        val_rows,
        device,
        args.output,
        batch_size=args.batch,
        workers=args.workers,
        checkpoint_epoch=int(checkpoint.get("epoch", 0)),
        use_amp=use_amp,
    )
    identity = {
        "schema": "v114-height-evaluation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "store": str(args.store.resolve()),
        "curriculum_identity": expected_identity,
        "source_filter": args.source,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "device": str(device),
        "amp_enabled": use_amp,
        "summary": summary,
    }
    (args.output / "evaluation_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    print(json.dumps(identity, indent=2), flush=True)
    return 0
