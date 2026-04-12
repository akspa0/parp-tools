#!/usr/bin/env python3
"""
Refine CK24/PM4-driven object visibility masks using OpenCV segmentation.

This script keeps the original exported seed mask intact and writes a refined
mask beside it, then stores the new relative path into terrain_data using a
separate JSON field (default: object_visibility_mask_cv2).

Why this exists:
- CK24 seed masks are useful priors but can over-cover broad regions.
- We often do not have trustworthy per-object height for strict 3D masking.
- 2D minimap silhouette refinement still helps restoration/training without
  destroying raw source inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_INPUT_MASK_KEYS = [
    "object_visibility_mask",
    "pm4_mask",
    "pm4_object_mask",
    "collision_mask",
]

UNTRUSTED_MARKER = "__untrusted_do_not_use"


@dataclass
class TileRefineStats:
    tile: str
    seed_pixels: int
    refined_pixels: int
    shrink_ratio: float
    out_mask: str


@dataclass
class RootSummary:
    root: str
    tiles_scanned: int
    tiles_with_seed: int
    tiles_refined: int
    tiles_skipped: int
    seed_pixels_total: int
    refined_pixels_total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine CK24 object masks into tighter minimap silhouettes and store "
            "them in a separate JSON field."
        )
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="Dataset root that contains dataset/ and images/ (repeatable).",
    )
    parser.add_argument(
        "--input-mask-key",
        action="append",
        default=None,
        help=(
            "terrain_data key(s) to treat as seed masks, in priority order. "
            "Default: object_visibility_mask, pm4_mask, pm4_object_mask, collision_mask"
        ),
    )
    parser.add_argument(
        "--output-mask-key",
        default="object_visibility_mask_cv2",
        help="terrain_data JSON key to store refined mask path.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_object_visibility_mask_cv2.png",
        help="Suffix for refined mask file name.",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=16,
        help="Ignore seed connected components smaller than this size.",
    )
    parser.add_argument(
        "--roi-margin",
        type=int,
        default=14,
        help="Extra pixels around each seed component ROI before refinement.",
    )
    parser.add_argument(
        "--max-expand-pixels",
        type=int,
        default=0,
        help="Max allowed expansion outside seed component (default 0 for shrink-first behavior).",
    )
    parser.add_argument(
        "--grabcut-iters",
        type=int,
        default=2,
        help="GrabCut iterations for each component.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing refined masks/JSON field values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats but do not write files or JSON updates.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write a run summary report JSON.",
    )
    return parser.parse_args()


def safe_relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def is_quarantined_root(root: Path) -> bool:
    return any(UNTRUSTED_MARKER in part.lower() for part in root.parts)


def kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def load_binary_mask(path: Path, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    if raw.shape != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (raw > 0).astype(np.uint8)


def refine_component(
    color_roi: np.ndarray,
    seed_roi: np.ndarray,
    max_expand_pixels: int,
    grabcut_iters: int,
) -> np.ndarray:
    if int(seed_roi.sum()) == 0:
        return np.zeros_like(seed_roi, dtype=np.uint8)

    # Build conservative support masks around the seed region.
    probable_fg = cv2.dilate(seed_roi, kernel(5), iterations=1)
    definite_fg = cv2.erode(seed_roi, kernel(3), iterations=1)
    if int(definite_fg.sum()) == 0:
        definite_fg = seed_roi.copy()

    if max_expand_pixels <= 0:
        support_limit = seed_roi.copy()
    else:
        support_limit = cv2.dilate(seed_roi, kernel(2 * max_expand_pixels + 1), iterations=1)

    gc_mask = np.full(seed_roi.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
    gc_mask[definite_fg > 0] = cv2.GC_FGD

    border = np.zeros_like(seed_roi, dtype=np.uint8)
    border[[0, -1], :] = 1
    border[:, [0, -1]] = 1
    gc_mask[border > 0] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(color_roi, gc_mask, None, bg_model, fg_model, max(1, grabcut_iters), cv2.GC_INIT_WITH_MASK)
        result = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype(np.uint8)
    except cv2.error:
        # Fall back to seed if grabcut fails on low-information regions.
        result = seed_roi.copy()

    # Keep final component bounded near the CK24 seed support.
    result = np.logical_and(result > 0, support_limit > 0).astype(np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel(3), iterations=1)

    # If refinement collapses too aggressively, keep a tight dilated seed fallback.
    seed_pixels = int(seed_roi.sum())
    refined_pixels = int(result.sum())
    if seed_pixels > 0 and refined_pixels < max(4, int(seed_pixels * 0.35)):
        result = seed_roi.copy()

    return result


def refine_mask(
    minimap_bgr: np.ndarray,
    seed_mask: np.ndarray,
    min_component_pixels: int,
    roi_margin: int,
    max_expand_pixels: int,
    grabcut_iters: int,
) -> np.ndarray:
    refined = np.zeros_like(seed_mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask.astype(np.uint8), connectivity=8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        x0 = max(0, x - roi_margin)
        y0 = max(0, y - roi_margin)
        x1 = min(seed_mask.shape[1], x + w + roi_margin)
        y1 = min(seed_mask.shape[0], y + h + roi_margin)
        if x0 >= x1 or y0 >= y1:
            continue

        component_seed = (labels[y0:y1, x0:x1] == label_idx).astype(np.uint8)
        component_img = minimap_bgr[y0:y1, x0:x1]

        component_refined = refine_component(
            color_roi=component_img,
            seed_roi=component_seed,
            max_expand_pixels=max_expand_pixels,
            grabcut_iters=grabcut_iters,
        )

        existing = refined[y0:y1, x0:x1]
        refined[y0:y1, x0:x1] = np.maximum(existing, component_refined)

    return refined


def resolve_seed_mask(
    terrain: Dict[str, object],
    dataset_root: Path,
    keys: Sequence[str],
) -> Tuple[Optional[str], Optional[Path]]:
    for key in keys:
        rel = terrain.get(key)
        if not rel:
            continue
        candidate = dataset_root / str(rel)
        if candidate.exists():
            return key, candidate
    return None, None


def resolve_minimap_path(payload: Dict[str, object], terrain: Dict[str, object], dataset_root: Path, tile_name: str) -> Path:
    minimap_rel = payload.get("image") or terrain.get("image")
    if minimap_rel:
        minimap_path = dataset_root / str(minimap_rel)
        if minimap_path.exists():
            return minimap_path

    fallback = dataset_root / "images" / f"{tile_name}.png"
    if fallback.exists():
        return fallback

    # Last fallback for older schema variants.
    no_obj_rel = terrain.get("no_object_minimap")
    if no_obj_rel:
        candidate = dataset_root / str(no_obj_rel)
        if candidate.exists():
            return candidate

    return fallback


def process_dataset_root(
    dataset_root: Path,
    input_mask_keys: Sequence[str],
    output_mask_key: str,
    output_suffix: str,
    min_component_pixels: int,
    roi_margin: int,
    max_expand_pixels: int,
    grabcut_iters: int,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[RootSummary, List[TileRefineStats]]:
    dataset_dir = dataset_root / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    tile_stats: List[TileRefineStats] = []

    tiles_scanned = 0
    tiles_with_seed = 0
    tiles_refined = 0
    tiles_skipped = 0
    seed_pixels_total = 0
    refined_pixels_total = 0

    for json_path in sorted(dataset_dir.glob("*.json")):
        tiles_scanned += 1
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        terrain = payload.get("terrain_data", {})
        tile_name = str(terrain.get("adt_tile") or json_path.stem)

        if output_mask_key in terrain and terrain.get(output_mask_key) and not overwrite:
            tiles_skipped += 1
            continue

        seed_key, seed_path = resolve_seed_mask(terrain, dataset_root, input_mask_keys)
        if not seed_path:
            tiles_skipped += 1
            continue

        minimap_path = resolve_minimap_path(payload, terrain, dataset_root, tile_name)
        minimap_bgr = cv2.imread(str(minimap_path), cv2.IMREAD_COLOR)
        if minimap_bgr is None:
            tiles_skipped += 1
            continue

        seed_mask = load_binary_mask(seed_path, minimap_bgr.shape[:2])
        if seed_mask is None:
            tiles_skipped += 1
            continue
        if int(seed_mask.sum()) == 0:
            tiles_skipped += 1
            continue

        tiles_with_seed += 1

        refined = refine_mask(
            minimap_bgr=minimap_bgr,
            seed_mask=seed_mask,
            min_component_pixels=min_component_pixels,
            roi_margin=roi_margin,
            max_expand_pixels=max_expand_pixels,
            grabcut_iters=grabcut_iters,
        )

        seed_pixels = int(seed_mask.sum())
        refined_pixels = int(refined.sum())
        seed_pixels_total += seed_pixels
        refined_pixels_total += refined_pixels

        out_name = f"{tile_name}{output_suffix}"
        out_path = dataset_root / "images" / out_name
        out_rel = safe_relpath(out_path, dataset_root)

        shrink_ratio = 0.0
        if seed_pixels > 0:
            shrink_ratio = 1.0 - (float(refined_pixels) / float(seed_pixels))

        tile_stats.append(
            TileRefineStats(
                tile=tile_name,
                seed_pixels=seed_pixels,
                refined_pixels=refined_pixels,
                shrink_ratio=shrink_ratio,
                out_mask=out_rel,
            )
        )

        terrain[output_mask_key] = out_rel
        terrain[f"{output_mask_key}_source"] = seed_key

        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), (refined * 255).astype(np.uint8))
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

        tiles_refined += 1

    summary = RootSummary(
        root=str(dataset_root),
        tiles_scanned=tiles_scanned,
        tiles_with_seed=tiles_with_seed,
        tiles_refined=tiles_refined,
        tiles_skipped=tiles_skipped,
        seed_pixels_total=seed_pixels_total,
        refined_pixels_total=refined_pixels_total,
    )
    return summary, tile_stats


def main() -> int:
    args = parse_args()

    input_mask_keys = args.input_mask_key if args.input_mask_key else DEFAULT_INPUT_MASK_KEYS

    all_summaries: List[RootSummary] = []
    all_tiles: List[TileRefineStats] = []

    for root_str in args.dataset_root:
        root = Path(root_str)
        if is_quarantined_root(root):
            raise SystemExit(
                f"Refusing quarantined dataset root: {root}. "
                "Use trusted lineage roots only."
            )
        summary, tiles = process_dataset_root(
            dataset_root=root,
            input_mask_keys=input_mask_keys,
            output_mask_key=args.output_mask_key,
            output_suffix=args.output_suffix,
            min_component_pixels=args.min_component_pixels,
            roi_margin=args.roi_margin,
            max_expand_pixels=args.max_expand_pixels,
            grabcut_iters=args.grabcut_iters,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        all_summaries.append(summary)
        all_tiles.extend(tiles)

        print(f"[root] {summary.root}")
        print(
            "  "
            f"scanned={summary.tiles_scanned} seeded={summary.tiles_with_seed} "
            f"refined={summary.tiles_refined} skipped={summary.tiles_skipped}"
        )
        if summary.seed_pixels_total > 0:
            retained = summary.refined_pixels_total / float(summary.seed_pixels_total)
            print(
                "  "
                f"seed_pixels={summary.seed_pixels_total} refined_pixels={summary.refined_pixels_total} "
                f"retained={retained:.3f}"
            )

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summaries": [summary.__dict__ for summary in all_summaries],
            "tiles": [tile.__dict__ for tile in all_tiles],
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
