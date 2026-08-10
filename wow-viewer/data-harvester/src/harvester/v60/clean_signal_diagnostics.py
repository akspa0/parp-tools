"""Checkpoint loading, prediction export, and failure-focused visual diagnostics for Spec 139."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from harvester.v60.clean_signal_model import build_clean_signal_model_from_identity
from harvester.v60.clean_signal_train import (
    CHECKPOINT_SCHEMA,
    CleanSignalDataset,
    CleanSignalRow,
    load_clean_signal_rows,
)

DIAGNOSTIC_SCHEMA = "v7-clean-signal-checkpoint-diagnostic-v1"
CELL_SIZE = 256
LABEL_HEIGHT = 24
PANEL_NAMES = ("luma", "target_height", "predicted_height", "absolute_error")


class CleanSignalDiagnosticError(ValueError):
    """Raised when a checkpoint cannot be used for a clean-signal diagnosis."""


def _resolve_device(value: str) -> torch.device:
    if value not in {"cpu", "cuda", "auto"}:
        raise CleanSignalDiagnosticError("device must be one of 'cpu', 'cuda', or 'auto'")
    if value == "cuda" and not torch.cuda.is_available():
        raise CleanSignalDiagnosticError("device=cuda requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def _load_checkpoint(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not path.is_file():
        raise CleanSignalDiagnosticError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CleanSignalDiagnosticError(f"invalid checkpoint schema: {path}")
    identity = payload.get("model_identity")
    state_dict = payload.get("model_state_dict")
    if not isinstance(identity, dict) or not isinstance(state_dict, dict):
        raise CleanSignalDiagnosticError("checkpoint must contain model_identity and model_state_dict")
    try:
        model, rebuilt_identity = build_clean_signal_model_from_identity(identity)
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise CleanSignalDiagnosticError(f"checkpoint model identity/state is invalid: {exc}") from exc
    if rebuilt_identity.get("config_sha256") != identity.get("config_sha256"):
        raise CleanSignalDiagnosticError("checkpoint model identity changed during reconstruction")
    return model, payload


def _select_validation_rows(rows: list[CleanSignalRow], payload: dict[str, Any]) -> list[CleanSignalRow]:
    split = payload.get("split")
    if not isinstance(split, dict):
        raise CleanSignalDiagnosticError("checkpoint split metadata is missing")
    validation_ids = split.get("validation_row_ids")
    if not isinstance(validation_ids, list) or not validation_ids:
        raise CleanSignalDiagnosticError("checkpoint has no validation_row_ids")
    by_id = {row.row_id: row for row in rows}
    missing = sorted({str(row_id) for row_id in validation_ids} - set(by_id))
    if missing:
        raise CleanSignalDiagnosticError(f"checkpoint validation rows are missing from corpus: {missing[:8]}")
    selected = [by_id[str(row_id)] for row_id in validation_ids]
    families = {row.family for row in selected}
    if not families:
        raise CleanSignalDiagnosticError("checkpoint validation selection is empty")
    return sorted(selected, key=lambda row: row.row_id)


def _aggregate(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    result: dict[str, dict[str, Any]] = {}
    for name, values in sorted(grouped.items()):
        model_mae = float(np.mean([value["final_height_mae"] for value in values]))
        baseline_mae = float(np.mean([value["tile_mean_baseline_mae"] for value in values]))
        result[name] = {
            "row_count": len(values),
            "final_height_mae": model_mae,
            "tile_mean_baseline_mae": baseline_mae,
            "improvement_vs_tile_mean": None if baseline_mae <= 1e-12 else 1.0 - model_mae / baseline_mae,
        }
    return result


def _to_image(array: np.ndarray, *, error: bool = False) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    if error:
        values = np.clip(values * 4.0, 0.0, 1.0)
    else:
        values = np.clip(values, 0.0, 1.0)
    pixels = np.rint(values * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB").resize(
        (CELL_SIZE, CELL_SIZE), Image.Resampling.BILINEAR
    )


def _labeled_cell(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (CELL_SIZE, CELL_SIZE + LABEL_HEIGHT), (24, 24, 24))
    canvas.paste(image, (0, LABEL_HEIGHT))
    ImageDraw.Draw(canvas).text((6, 5), label, fill=(245, 245, 245))
    return canvas


def _render_sheet(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    cell_height = CELL_SIZE + LABEL_HEIGHT
    sheet = Image.new("RGB", (CELL_SIZE * len(PANEL_NAMES), cell_height * len(records)), (12, 12, 12))
    for row_index, record in enumerate(records):
        panels = {
            "luma": _to_image(record["luma_256"]),
            "target_height": _to_image(record["target_height_257"]),
            "predicted_height": _to_image(record["predicted_height_257"]),
            "absolute_error": _to_image(record["absolute_error_257"], error=True),
        }
        for panel_index, name in enumerate(PANEL_NAMES):
            label = name if name != "predicted_height" else f"predicted ({record['final_height_mae']:.3f})"
            cell = _labeled_cell(panels[name], label)
            sheet.paste(cell, (panel_index * CELL_SIZE, row_index * cell_height))
    sheet.save(path)


def diagnose_clean_signal_checkpoint(
    checkpoint: str | Path,
    corpus: str | Path,
    output: str | Path,
    *,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run image-only checkpoint inference on the checkpoint's held-out rows."""

    if batch_size < 1:
        raise CleanSignalDiagnosticError("batch_size must be positive")
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_path}")
    model, checkpoint_payload = _load_checkpoint(Path(checkpoint))
    corpus_root, rows = load_clean_signal_rows(corpus)
    validation_rows = _select_validation_rows(rows, checkpoint_payload)
    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    model.eval()
    loader = DataLoader(CleanSignalDataset(validation_rows), batch_size=batch_size, shuffle=False, num_workers=0)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch["inputs"].to(device_obj))
            predicted_height = predictions.height_prediction_257.detach().cpu().numpy()
            target_height = batch["targets"]["relative_height_257"].numpy()
            for index, row_id in enumerate(batch["row_id"]):
                target = np.asarray(target_height[index], dtype=np.float32)
                predicted = np.asarray(predicted_height[index], dtype=np.float32)
                error = np.abs(predicted - target).astype(np.float32)
                record = {
                    "row_id": str(row_id),
                    "family": str(batch["family"][index]),
                    "complexity_bucket": str(batch["complexity_bucket"][index]),
                    "final_height_mae": float(error.mean()),
                    "tile_mean_baseline_mae": float(np.abs(target - target.mean()).mean()),
                    "luma_256": batch["inputs"][index, 0].numpy().astype(np.float32),
                    "target_height_257": target,
                    "predicted_height_257": predicted,
                    "absolute_error_257": error,
                }
                records.append(record)
    if not records:
        raise CleanSignalDiagnosticError("checkpoint diagnosis produced no rows")

    output_path.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_path / "predictions"
    prediction_dir.mkdir()
    report_rows: list[dict[str, Any]] = []
    for record in records:
        prediction_path = prediction_dir / f"{record['row_id']}.npz"
        np.savez_compressed(
            prediction_path,
            luma_256=record["luma_256"],
            target_height_257=record["target_height_257"],
            predicted_height_257=record["predicted_height_257"],
            absolute_error_257=record["absolute_error_257"],
        )
        report_rows.append(
            {
                key: value
                for key, value in record.items()
                if not isinstance(value, np.ndarray)
            }
        )

    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "checkpoint": str(Path(checkpoint).resolve()),
        "corpus_root": str(corpus_root.resolve()),
        "architecture": checkpoint_payload.get("architecture"),
        "model_identity": checkpoint_payload.get("model_identity"),
        "loss_profile": checkpoint_payload.get("loss_profile"),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "device": str(device_obj),
        "split": checkpoint_payload.get("split"),
        "validation_row_count": len(report_rows),
        "validation_families": sorted({row["family"] for row in report_rows}),
        "aggregate": {
            "final_height_mae": float(np.mean([row["final_height_mae"] for row in report_rows])),
            "tile_mean_baseline_mae": float(np.mean([row["tile_mean_baseline_mae"] for row in report_rows])),
        },
        "by_family": _aggregate(report_rows, "family"),
        "by_complexity_bucket": _aggregate(report_rows, "complexity_bucket"),
        "rows": report_rows,
        "outputs": {
            "prediction_dir": prediction_dir.as_posix(),
            "validation_atlas": (output_path / "validation-diagnostic-atlas.png").as_posix(),
        },
    }
    _render_sheet(records, output_path / "validation-diagnostic-atlas.png")
    cross_tile = [record for record in records if str(record["family"]).startswith("cross_tile_")]
    if cross_tile:
        cross_path = output_path / "cross-tile-diagnostic-atlas.png"
        _render_sheet(cross_tile, cross_path)
        report["outputs"]["cross_tile_atlas"] = cross_path.as_posix()
    (output_path / "diagnostic_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["DIAGNOSTIC_SCHEMA", "CleanSignalDiagnosticError", "diagnose_clean_signal_checkpoint"]
