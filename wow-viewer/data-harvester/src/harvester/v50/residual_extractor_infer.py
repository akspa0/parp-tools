"""Canonical v50 owner for residual-extractor inference and visual review (Spec 125 US7).

Loads a trained ``ResidualExtractorNet`` checkpoint and runs it over store rows (default: the
validation split) to produce:

- a per-row contact sheet PNG: minimap RGB input, predicted residual, ground-truth residual, and the
  stripped albedo (minimap - predicted residual) so the shading-removal claim is visible, not just
  reported as a number;
- a JSON report with per-row MAE and the per-regime breakdown, so the visual and the metric agree.

The deployment input is the DXT1-degraded minimap (``minimap_rgb_dxt1``) by default, matching the
authored tiles the model is deployed on; pass ``--input-array minimap_rgb`` to review the pristine
render instead. Contract enforcement lives in small pure functions so the gates are CPU-testable
without CUDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image, ImageDraw, ImageFont

from harvester.v50.contracts import (
    release_identity,
    require_metadata_release,
    require_store_release,
    validate_release,
)
from harvester.v50.residual_extractor_model import (
    RESIDUAL_GRID,
    TARGET_CONTRACT_VERSION,
    ResidualExtractorNet,
)
from harvester.v50.residual_extractor_train import (
    ARCHITECTURE_ID,
    CURRICULUM_SCHEMA,
    DEFAULT_INPUT_ARRAY,
    REQUIRED_ARRAYS,
    REQUIRED_INDEX_FIELDS,
    TrainerContractError,
    validate_curriculum_contract,
)

CELL_SIZE = 256
HEADER_HEIGHT = 86
ROW_TITLE_HEIGHT = 28
CELL_LABEL_HEIGHT = 32
COLUMN_TITLES = (
    "Minimap RGB (input)",
    "Predicted residual",
    "Ground-truth residual",
    "Stripped albedo (minimap - pred)",
)


class ResidualExtractorInferError(ValueError):
    """Raised when inference cannot produce an honest visual review."""


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def load_model(checkpoint_path: Path, device: torch.device, release: str = "v50.1") -> ResidualExtractorNet:
    """Load a Spec 125 US7 residual-extractor checkpoint, gating on release and contract version."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    require_metadata_release(checkpoint, release, artifact="residual-extractor checkpoint")
    if checkpoint.get("model_variant") != ARCHITECTURE_ID:
        raise ResidualExtractorInferError(
            f"checkpoint is not a {ARCHITECTURE_ID!r} model; got {checkpoint.get('model_variant')!r}"
        )
    if checkpoint.get("target_contract_version") != TARGET_CONTRACT_VERSION:
        raise ResidualExtractorInferError(
            f"checkpoint target contract {checkpoint.get('target_contract_version')!r} != "
            f"{TARGET_CONTRACT_VERSION!r}"
        )
    model = ResidualExtractorNet().to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def predict_residual(model: ResidualExtractorNet, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Minimap RGB (H, W, 3) uint8 -> predicted residual (H, W) float in [0, 1]."""
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor.to(device))[0].cpu().numpy()
    return out


def _grayscale_rgb(values: np.ndarray) -> np.ndarray:
    """Float [0,1] field -> uint8 RGB grayscale image."""
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    gray = (clipped * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _rgb_image(array: np.ndarray, *, size: int = CELL_SIZE) -> Image.Image:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _stripped_albedo_rgb(rgb: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """minimap - residual, per channel, clipped to [0, 255]. The shading layer is multiplicative in
    the compositor, so this is an approximation of the unlit albedo -- good enough for a visual
    review, not a claim of exact albedo recovery."""
    r = np.asarray(rgb, dtype=np.float32)
    res = np.clip(np.asarray(residual, dtype=np.float32), 0.0, 1.0)
    if res.ndim == 2:
        res = res[..., None]
    # Subtract the shading as a fraction of the lit value, then rescale to keep brightness readable.
    stripped = r * (1.0 - res)
    return np.clip(stripped, 0, 255).astype(np.uint8)


def render_review_sheet(
    *,
    rgb: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    title: str,
    output: Path,
) -> None:
    """Render one row's four-panel contact sheet."""
    cells = (
        _rgb_image(rgb),
        _rgb_image(_grayscale_rgb(predicted)),
        _rgb_image(_grayscale_rgb(target)),
        _rgb_image(_stripped_albedo_rgb(rgb, predicted)),
    )
    width = len(COLUMN_TITLES) * CELL_SIZE
    block_height = ROW_TITLE_HEIGHT + CELL_SIZE + CELL_LABEL_HEIGHT
    sheet = Image.new("RGB", (width, HEADER_HEIGHT + block_height), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    font = _font()
    draw.text((10, 10), "v50 residual-extractor visual review (Spec 125 US7)", fill=(245, 245, 245), font=font)
    draw.text((10, 30), str(output.resolve()), fill=(178, 184, 194), font=font)
    draw.text(
        (10, 50),
        "Stripped albedo = minimap - predicted residual (approximate unlit albedo; not exact recovery).",
        fill=(255, 210, 110),
        font=font,
    )
    draw.rectangle((0, HEADER_HEIGHT, width, HEADER_HEIGHT + ROW_TITLE_HEIGHT), fill=(34, 38, 45))
    draw.text((8, HEADER_HEIGHT + 8), title, fill=(230, 232, 236), font=font)
    image_y = HEADER_HEIGHT + ROW_TITLE_HEIGHT
    for column, (cell, label) in enumerate(zip(cells, COLUMN_TITLES)):
        x = column * CELL_SIZE
        sheet.paste(cell, (x, image_y))
        draw.rectangle(
            (x, image_y + CELL_SIZE, x + CELL_SIZE, image_y + CELL_SIZE + CELL_LABEL_HEIGHT),
            fill=(28, 31, 37),
        )
        draw.text((x + 6, image_y + CELL_SIZE + 10), label, fill=(220, 222, 226), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    ap = argparse.ArgumentParser(description="v50 residual-extractor inference + visual review (Spec 125 US7)")
    ap.add_argument("--store", required=True, type=Path, help="paired minimap_rgb + residual_256 store")
    ap.add_argument("--checkpoint", required=True, type=Path, help="trained residual-extractor checkpoint")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--input-array",
        default=DEFAULT_INPUT_ARRAY,
        help=(
            "which minimap array to run on. Default is the DXT1-degraded variant, matching the "
            "authored tiles the model is deployed on; pass minimap_rgb to review the pristine render."
        ),
    )
    ap.add_argument("--val-key", default="split")
    ap.add_argument("--val-value", default="val")
    ap.add_argument("--samples", type=int, default=6, help="max rows to render as contact sheets")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")

    try:
        model = load_model(args.checkpoint, device, args.release)
    except (ValueError, ResidualExtractorInferError) as exc:
        raise SystemExit(str(exc)) from exc

    group = zarr.open_group(str(args.store), mode="r")
    try:
        require_store_release(group, args.release, store=args.store, expected_schema=CURRICULUM_SCHEMA)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    try:
        array_lengths = {name: int(group[name].shape[0]) for name in group.array_keys()}
        validate_curriculum_contract(
            attrs=dict(group.attrs), array_lengths=array_lengths, index_rows=index
        )
    except TrainerContractError as exc:
        raise SystemExit(str(exc)) from exc

    if args.input_array not in group:
        raise SystemExit(
            f"store has no {args.input_array!r} array. Available: {sorted(group.array_keys())}"
        )
    if "residual_256" not in group:
        raise SystemExit("store has no residual_256 array")

    rows = [i for i, row in enumerate(index) if str(row.get(args.val_key)) == str(args.val_value)]
    if not rows:
        raise SystemExit(f"no store rows matched {args.val_key}={args.val_value!r}")
    rows = rows[: args.samples]

    def _regime(i: int) -> str:
        return str(index[i].get("height_regime", "uncurated"))

    per_row = []
    for row in rows:
        rgb = np.asarray(group[args.input_array][row], dtype=np.uint8)
        target = np.asarray(group["residual_256"][row], dtype=np.float32) / 255.0
        if target.ndim == 3:
            target = target[..., 0]
        predicted = predict_residual(model, rgb, device)
        mae = float(np.abs(predicted - target).mean())
        metadata = index[row]
        title = (
            f"row={row} map={metadata.get('map')} tile={metadata.get('tile_x')},{metadata.get('tile_y')} "
            f"regime={_regime(row)}  MAE={mae:.4f}"
        )
        output = args.output_dir / f"row-{row:04d}-{_regime(row)}.png"
        render_review_sheet(rgb=rgb, predicted=predicted, target=target, title=title, output=output)
        per_row.append(
            {
                "row": row,
                "map": str(metadata.get("map", "")),
                "tile_x": int(metadata.get("tile_x", -1)),
                "tile_y": int(metadata.get("tile_y", -1)),
                "regime": _regime(row),
                "mae": mae,
                "image": str(output.resolve()),
            }
        )
        print(f"[row {row:04d}] {_regime(row):8s} MAE={mae:.4f} -> {output}", flush=True)

    by_regime: dict[str, list[float]] = {}
    for entry in per_row:
        by_regime.setdefault(entry["regime"], []).append(entry["mae"])
    report = {
        "schema": "v50-residual-extractor-infer-v1",
        **release_identity(args.release),
        "checkpoint": str(args.checkpoint.resolve()),
        "store": str(args.store.resolve()),
        "input_array": args.input_array,
        "trained_on_codec_degraded_input": args.input_array == DEFAULT_INPUT_ARRAY,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "rows": per_row,
        "mean_mae": float(np.mean([e["mae"] for e in per_row])),
        "mae_by_regime": {k: float(np.mean(v)) for k, v in sorted(by_regime.items())},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "infer-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] {len(per_row)} rows -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
