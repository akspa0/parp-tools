"""Write visible OBJ/PNG review artifacts for one Spec 108 WDL evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spec103_export_mesh import height_to_obj  # noqa: E402
from harvester.spec103.wdl_visualization import reconstruct_wdl_pair  # noqa: E402


def _heatmap(error: np.ndarray) -> np.ndarray:
    scale = max(float(np.percentile(np.abs(error), 99.0)), 1e-6)
    value = np.clip(error / scale, -1.0, 1.0)
    red = np.clip(value, 0.0, 1.0); blue = np.clip(-value, 0.0, 1.0)
    green = 1.0 - np.abs(value)
    return (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


def _load_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as lattice:
        return np.asarray(lattice["outer_17"], dtype=np.float32), np.asarray(lattice["inner_16"], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write visible OBJ/PNG artifacts for one WDL-prior evaluation")
    ap.add_argument("--evaluation", required=True, type=Path, help="evaluate_spec103_wdl_prior.py output directory")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    source = args.evaluation; output = args.output or source / "visual_review"; output.mkdir(parents=True, exist_ok=True)
    predicted_outer, predicted_inner = _load_pair(source / "predicted_wdl_lattice.npz")
    truth_outer, truth_inner = _load_pair(source / "ground_truth_wdl_lattice.npz")
    predicted = reconstruct_wdl_pair(predicted_outer, predicted_inner)
    truth_lattice = reconstruct_wdl_pair(truth_outer, truth_inner)
    truth_full = np.load(source / "ground_truth_height_257.npy")
    error = predicted - truth_full
    texture = "input_minimap.png"
    Image.open(source / texture).convert("RGB").save(output / texture)
    np.save(output / "predicted_wdl_reconstruction_257.npy", predicted)
    np.save(output / "truth_wdl_reconstruction_257.npy", truth_lattice)
    np.save(output / "error_vs_full_truth_257.npy", error)
    Image.fromarray(_heatmap(error), mode="RGB").save(output / "error_vs_full_truth_heatmap.png")
    height_to_obj(predicted, output / "predicted_wdl_reconstruction.obj", texture)
    height_to_obj(truth_lattice, output / "truth_wdl_reconstruction.obj", texture)
    height_to_obj(truth_full, output / "truth_full_resolution.obj", texture)
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    (output / "README.md").write_text(
        "# WDL visual review\n\n"
        "Open `predicted_wdl_reconstruction.obj` and `truth_wdl_reconstruction.obj` side by side. "
        "They are the same paired-WDL interpolation, so their difference shows coarse-prior quality.\n\n"
        "`truth_full_resolution.obj` is the actual 257x257 terrain. `error_vs_full_truth_heatmap.png` is signed: red = prediction above truth, blue = below.\n",
        encoding="utf-8",
    )
    print(f"[DONE] visual WDL review -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
