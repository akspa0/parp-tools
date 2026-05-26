"""Infer learned object-roof masks from minimap tiles.

This script is the learned fallback host for Spec-025 when placement metadata
is missing at inference time. It consumes:

- a trained roof-family identifier checkpoint (`hf_model`)
- roof-library canonical masks (`object_visual.zarr` + roof catalogs)

and emits per-tile learned masks that `patch_v18_object_roof_masks.py` can
consume through `--learned-mask-dir`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import zarr
import zarr.storage


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_DEFAULT_MODEL_ROOT = _PROJECT_ROOT / "models" / "v18" / "object_roof_identifier"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "tmp" / "v18_object_roof_infer"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer learned object-roof masks from minimap tiles.")
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--build", type=str, required=True)
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--tile-x", type=int, default=None)
    parser.add_argument("--tile-y", type=int, default=None)
    parser.add_argument("--max-tiles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--library-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument("--min-family-confidence", type=float, default=0.35)
    parser.add_argument("--max-panels", type=int, default=24)
    return parser.parse_args()


def _resolve_latest_model_dir() -> Path:
    runs = sorted(_DEFAULT_MODEL_ROOT.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError("No object-roof identifier run found. Train one first.")
    return runs[0].parent


def _load_label_space(model_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    label_path = model_dir / "label_space.json"
    if not label_path.exists():
        raise RuntimeError(f"Missing label-space file: {label_path}")
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    family_to_label = {str(k): int(v) for k, v in dict(payload.get("family_to_label", {})).items()}
    raw_label_to_family = dict(payload.get("label_to_family", {}))
    label_to_family = {int(k): str(v) for k, v in raw_label_to_family.items()}
    if not family_to_label or not label_to_family:
        raise RuntimeError("Label-space file is empty")
    return family_to_label, label_to_family


def _load_library_dir(model_dir: Path, explicit_library_dir: Path | None) -> Path:
    if explicit_library_dir is not None:
        return Path(explicit_library_dir)
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing model summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = str(summary.get("library_dir", "")).strip()
    if not raw:
        raise RuntimeError("Model summary does not include library_dir")
    out = Path(raw)
    if not out.exists():
        raise RuntimeError(f"Library dir from summary not found: {out}")
    return out


def _read_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(str(path))
    return [{column: table.column(column)[idx].as_py() for column in table.column_names} for idx in range(table.num_rows)]


def _load_family_templates(library_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    exemplars_path = library_dir / "roof_exemplars.parquet"
    families_path = library_dir / "roof_families.parquet"
    visual_path = library_dir / "object_visual.zarr"
    if not exemplars_path.exists() or not families_path.exists() or not visual_path.exists():
        raise RuntimeError("Library is missing required files (roof_exemplars/families/object_visual)")

    exemplar_rows = _read_rows(exemplars_path)
    family_rows = _read_rows(families_path)
    family_asset = {str(row.get("family_id", "")): str(row.get("canonical_asset_path", "")) for row in family_rows}

    store = zarr.storage.LocalStore(str(visual_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    try:
        roof_masks = root["roof_mask"][:].astype(np.float32)
    finally:
        store.close()

    by_family: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(exemplar_rows):
        fam = str(row.get("family_id", ""))
        if fam:
            by_family[fam].append((idx, row))

    templates: dict[str, np.ndarray] = {}
    for fam, members in by_family.items():
        canonical = [item for item in members if bool(item[1].get("is_canonical", False))]
        chosen_idx = int((canonical[0] if canonical else members[0])[0])
        mask = np.clip(roof_masks[chosen_idx], 0.0, 1.0)
        templates[fam] = mask.astype(np.float32)

    return templates, family_asset


def _load_index_rows(dataset_root: Path, build: str) -> list[dict[str, Any]]:
    index_path = dataset_root / f"{build}.zarr" / "index.parquet"
    if not index_path.exists():
        raise RuntimeError(f"Missing index parquet: {index_path}")
    return _read_rows(index_path)


def _select_tiles(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        if args.map is not None and str(row.get("map", "")) != str(args.map):
            continue
        if args.tile_x is not None and int(row.get("tile_x", -1) or -1) != int(args.tile_x):
            continue
        if args.tile_y is not None and int(row.get("tile_y", -1) or -1) != int(args.tile_y):
            continue
        filtered.append(row)

    rng = random.Random(int(args.seed))
    if len(filtered) > int(args.max_tiles):
        filtered = rng.sample(filtered, k=int(args.max_tiles))
    filtered.sort(key=lambda r: int(r.get("tile_id", -1)))
    return filtered


def _nearest_resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = np.linspace(0, arr.shape[0] - 1, h).astype(np.int32)
    xs = np.linspace(0, arr.shape[1] - 1, w).astype(np.int32)
    if arr.ndim == 2:
        return arr[np.ix_(ys, xs)]
    return arr[np.ix_(ys, xs, np.arange(arr.shape[2]))]


def _iter_windows(h: int, w: int, size: int, stride: int) -> list[tuple[int, int, int, int]]:
    ys = list(range(0, max(1, h - size + 1), stride))
    xs = list(range(0, max(1, w - size + 1), stride))
    if ys[-1] != h - size:
        ys.append(max(0, h - size))
    if xs[-1] != w - size:
        xs.append(max(0, w - size))
    out = []
    for y0 in ys:
        for x0 in xs:
            out.append((x0, y0, x0 + size, y0 + size))
    return out


def _infer_tile_mask(
    minimap_rgb: np.ndarray,
    *,
    model,
    processor,
    label_to_family: dict[int, str],
    family_templates: dict[str, np.ndarray],
    min_family_confidence: float,
    window_size: int,
    window_stride: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float], str, float]:
    h, w = minimap_rgb.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    family_scores: dict[str, float] = defaultdict(float)

    windows = _iter_windows(h, w, size=int(window_size), stride=int(window_stride))
    model.eval()
    for x0, y0, x1, y1 in windows:
        crop = minimap_rgb[y0:y1, x0:x1]
        if crop.shape[0] != int(window_size) or crop.shape[1] != int(window_size):
            crop = _nearest_resize(crop, int(window_size), int(window_size)).astype(np.uint8)

        encoded = processor(images=crop, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)

        label = int(np.argmax(probs))
        conf = float(probs[label])
        if conf < float(min_family_confidence):
            continue
        family_id = label_to_family.get(label)
        if not family_id:
            continue
        template = family_templates.get(family_id)
        if template is None:
            continue

        tile_template = _nearest_resize(template, int(window_size), int(window_size)).astype(np.float32)
        weighted = np.clip(tile_template * conf, 0.0, 1.0)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], weighted)
        family_scores[family_id] += conf

    if family_scores:
        top_family, top_score = max(family_scores.items(), key=lambda item: item[1])
    else:
        top_family, top_score = "", 0.0

    return np.clip(heatmap, 0.0, 1.0), dict(sorted(family_scores.items(), key=lambda item: item[1], reverse=True)), str(top_family), float(top_score)


def _panel(minimap: np.ndarray, mask: np.ndarray, text: str) -> Image.Image:
    overlay = minimap.copy()
    overlay[mask >= 0.2, 0] = 255
    overlay[mask >= 0.2, 1] = np.maximum(overlay[mask >= 0.2, 1] // 2, 32)
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_rgb = np.repeat(mask_u8[:, :, None], 3, axis=2)

    left = Image.fromarray(minimap, mode="RGB")
    mid = Image.fromarray(mask_rgb, mode="RGB")
    right = Image.fromarray(overlay, mode="RGB")

    canvas = Image.new("RGB", (256 * 3, 256 + 18), color=(0, 0, 0))
    canvas.paste(left, (0, 18))
    canvas.paste(mid, (256, 18))
    canvas.paste(right, (512, 18))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (canvas.width, 17)], fill=(18, 18, 18))
    draw.text((4, 3), text, fill=(235, 235, 235))
    return canvas


def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir) if args.model_dir is not None else _resolve_latest_model_dir()
    library_dir = _load_library_dir(model_dir, args.library_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else (_DEFAULT_OUTPUT_ROOT / f"{args.build}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    family_to_label, label_to_family = _load_label_space(model_dir)
    family_templates, family_asset = _load_family_templates(library_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_dir / "hf_model")
    model = AutoModelForImageClassification.from_pretrained(model_dir / "hf_model").to(device)

    rows = _load_index_rows(Path(args.dataset_root), str(args.build))
    selected = _select_tiles(rows, args)
    if not selected:
        raise RuntimeError("No tiles selected for inference")

    zarr_path = Path(args.dataset_root) / f"{args.build}.zarr"
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")

    out_rows: list[dict[str, Any]] = []
    panel_dir = output_dir / "review"
    panel_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, row in enumerate(selected):
            tile_id = int(row.get("tile_id", -1))
            minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
            map_name = str(row.get("map", ""))
            tile_x = int(row.get("tile_x", -1) or -1)
            tile_y = int(row.get("tile_y", -1) or -1)

            mask, family_scores, top_family, top_score = _infer_tile_mask(
                minimap,
                model=model,
                processor=processor,
                label_to_family=label_to_family,
                family_templates=family_templates,
                min_family_confidence=float(args.min_family_confidence),
                window_size=int(args.window_size),
                window_stride=int(args.window_stride),
                device=device,
            )

            stem = f"{args.build}_{map_name}_{tile_x}_{tile_y}"
            npy_path = output_dir / f"{stem}.npy"
            png_path = output_dir / f"{stem}.png"
            np.save(npy_path, mask.astype(np.float32))
            Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(png_path)

            source = "learned" if float(mask.sum()) > 0.0 else "none"
            top_asset = str(family_asset.get(top_family, "")) if top_family else ""
            panel_text = (
                f"tile={tile_id} {map_name}_{tile_x}_{tile_y} "
                f"source={source} top_family={top_family or 'none'} "
                f"score={top_score:.3f} cov={float(mask.mean()):.4f}"
            )
            panel_path = panel_dir / f"panel_{idx:03d}.png"
            if idx < int(args.max_panels):
                _panel(minimap, mask, panel_text).save(panel_path)

            out_rows.append(
                {
                    "build": str(args.build),
                    "map": map_name,
                    "tile_id": tile_id,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "source": source,
                    "mask_mean": float(mask.mean()),
                    "mask_sum": float(mask.sum()),
                    "top_family_id": top_family,
                    "top_family_score": float(top_score),
                    "top_family_asset_path": top_asset,
                    "family_scores": family_scores,
                    "mask_npy": str(npy_path),
                    "mask_png": str(png_path),
                    "panel_path": str(panel_path),
                }
            )
    finally:
        store.close()

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "build": str(args.build),
        "tiles_selected": len(selected),
        "tiles_non_empty": int(sum(1 for row in out_rows if float(row.get("mask_sum", 0.0)) > 0.0)),
        "mean_mask_coverage": float(np.mean([float(row.get("mask_mean", 0.0)) for row in out_rows])) if out_rows else 0.0,
        "model_dir": str(model_dir),
        "library_dir": str(library_dir),
        "label_count": int(len(family_to_label)),
        "family_template_count": int(len(family_templates)),
        "output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.snapshot.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

