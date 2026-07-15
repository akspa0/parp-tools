"""Evaluate one real paired-store tile using RGB-only Spec 108 inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SRC = _SCRIPTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infer_spec103_wdl_prior import load_model, predict_rgb  # noqa: E402
from harvester.spec103.v7_inputs import wdl_lattice_from_height257  # noqa: E402


def _metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {"mae_world": float(np.abs(error).mean()), "rmse_world": float(np.sqrt(np.square(error).mean())), "max_abs_world": float(np.abs(error).max())}


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 108 real-tile RGB-only WDL evaluation")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--row", required=True, type=int, help="one real paired-store row to evaluate")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args()
    group = zarr.open_group(str(args.store), mode="r")
    if "minimap_rgb" not in group or "height_257" not in group:
        raise SystemExit("evaluation requires a paired store with minimap_rgb and height_257")
    count = int(group["minimap_rgb"].shape[0])
    if args.row < 0 or args.row >= count:
        raise SystemExit(f"row must be in 0..{count - 1}")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    rgb = np.asarray(group["minimap_rgb"][args.row], dtype=np.uint8)
    predicted_outer, predicted_inner = predict_rgb(model, rgb, device)
    truth_outer, truth_inner = wdl_lattice_from_height257(np.asarray(group["height_257"][args.row], dtype=np.float32))
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    source = index[args.row]
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(args.output / "input_minimap.png")
    with Image.open(args.output / "input_minimap.png") as exported:
        png_rgb = np.asarray(exported.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), dtype=np.uint8)
    png_outer, png_inner = predict_rgb(model, png_rgb, device)
    np.savez_compressed(args.output / "predicted_wdl_lattice.npz", outer_17=predicted_outer, inner_16=predicted_inner)
    np.savez_compressed(args.output / "ground_truth_wdl_lattice.npz", outer_17=truth_outer, inner_16=truth_inner)
    np.savez_compressed(args.output / "standalone_png_wdl_lattice.npz", outer_17=png_outer, inner_16=png_inner)
    report = {"schema": "spec108-real-tile-evaluation-v1", "source_row": args.row, "source": source,
              "checkpoint": str(args.checkpoint.resolve()), "input": "input_minimap.png", "prediction": "predicted_wdl_lattice.npz",
              "ground_truth": "ground_truth_wdl_lattice.npz", "standalone_png_prediction": "standalone_png_wdl_lattice.npz",
              "outer_17": _metrics(predicted_outer, truth_outer), "inner_16": _metrics(predicted_inner, truth_inner),
              "standalone_png_vs_store_rgb": {"outer_17": _metrics(png_outer, predicted_outer), "inner_16": _metrics(png_inner, predicted_inner)}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[REAL TILE] row={args.row} outer_mae={report['outer_17']['mae_world']:.3f} inner_mae={report['inner_16']['mae_world']:.3f} -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
