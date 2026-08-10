"""Image-only checkpoint evaluation on a separately prepared clean-signal corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from harvester.v60.clean_signal_corpus import load_clean_signal_manifest
from harvester.v60.clean_signal_diagnostics import (
    _aggregate,
    _load_checkpoint,
    _render_sheet,
)
from harvester.v60.clean_signal_train import CleanSignalDataset, load_clean_signal_rows

TRANSFER_SCHEMA = "v7-clean-signal-transfer-diagnostic-v1"


def evaluate_clean_signal_checkpoint(
    checkpoint: str | Path,
    corpus: str | Path,
    output: str | Path,
    *,
    batch_size: int = 8,
    device: str = "cpu",
    source_kind: str | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on all rows of a prepared corpus using image-only model inputs."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite transfer output: {output_path}")
    model, checkpoint_payload = _load_checkpoint(Path(checkpoint))
    corpus_root, rows = load_clean_signal_rows(corpus)
    manifest = load_clean_signal_manifest(corpus_root)
    source_by_id = {str(row["row_id"]): str(row.get("source_kind", "")) for row in manifest["rows"]}
    if source_kind is not None:
        rows = [row for row in rows if source_by_id.get(row.row_id) == source_kind]
    if not rows:
        raise ValueError("transfer corpus selection is empty")
    device_obj = torch.device(device)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device=cuda requested but CUDA is unavailable")
    model = model.to(device_obj)
    model.eval()
    loader = DataLoader(CleanSignalDataset(rows), batch_size=batch_size, shuffle=False, num_workers=0)
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
                records.append(
                    {
                        "row_id": str(row_id),
                        "source_kind": source_by_id.get(str(row_id), ""),
                        "family": str(batch["family"][index]),
                        "complexity_bucket": str(batch["complexity_bucket"][index]),
                        "final_height_mae": float(error.mean()),
                        "tile_mean_baseline_mae": float(np.abs(target - target.mean()).mean()),
                        "luma_256": batch["inputs"][index, 0].numpy().astype(np.float32),
                        "target_height_257": target,
                        "predicted_height_257": predicted,
                        "absolute_error_257": error,
                    }
                )
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_path / "predictions"
    prediction_dir.mkdir()
    report_rows: list[dict[str, Any]] = []
    for record in records:
        np.savez_compressed(
            prediction_dir / f"{record['row_id']}.npz",
            luma_256=record["luma_256"],
            target_height_257=record["target_height_257"],
            predicted_height_257=record["predicted_height_257"],
            absolute_error_257=record["absolute_error_257"],
        )
        report_rows.append({key: value for key, value in record.items() if not isinstance(value, np.ndarray)})
    report = {
        "schema": TRANSFER_SCHEMA,
        "checkpoint": str(Path(checkpoint).resolve()),
        "corpus_root": str(corpus_root.resolve()),
        "architecture": checkpoint_payload.get("architecture"),
        "model_identity": checkpoint_payload.get("model_identity"),
        "loss_profile": checkpoint_payload.get("loss_profile"),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "device": str(device_obj),
        "evaluation_scope": "all_selected_rows",
        "selected_source_kind": source_kind,
        "forbidden_signal_reads": [],
        "row_count": len(report_rows),
        "aggregate": {
            "final_height_mae": float(np.mean([row["final_height_mae"] for row in report_rows])),
            "tile_mean_baseline_mae": float(np.mean([row["tile_mean_baseline_mae"] for row in report_rows])),
        },
        "by_family": _aggregate(report_rows, "family"),
        "by_complexity_bucket": _aggregate(report_rows, "complexity_bucket"),
        "rows": report_rows,
        "outputs": {
            "prediction_dir": prediction_dir.as_posix(),
            "validation_atlas": (output_path / "transfer-diagnostic-atlas.png").as_posix(),
        },
    }
    _render_sheet(records, output_path / "transfer-diagnostic-atlas.png")
    (output_path / "transfer_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["TRANSFER_SCHEMA", "evaluate_clean_signal_checkpoint"]
