"""Mine paste/prefab candidates from curated V16.1 tiles.

This extracts connected high-signal regions from normal-training guidance maps
(`hard_region`, `transition`, `train_mask`) and writes candidate crops plus a
manifest to seed a reusable paste library workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.v16_1_dataset import V161Dataset  # noqa: E402


def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_map * mask).sum() / mask.sum().clamp_min(1e-8)


def _resize_weight(weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(weight.shape[-2:]) == tuple(size):
        return weight
    return F.interpolate(weight, size=size, mode="bilinear", align_corners=False)


def _gradient_magnitude_257(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt((dx * dx) + (dy * dy) + 1e-8)


def _hard_region_signals(sample: dict[str, torch.Tensor], detail_boost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_n = F.normalize(sample["normals"].unsqueeze(0).float(), dim=1, eps=1e-6)
    height_raw = sample["height_raw"].unsqueeze(0).float()
    normal_mask = sample["normal_mask"].unsqueeze(0).float()
    terrain_valid_mask = sample["terrain_valid_mask_257"].unsqueeze(0).float()
    object_weight = sample["weight_257"].unsqueeze(0).float()
    mddf_mask = sample["mddf_mask"].unsqueeze(0).float()
    modf_mask = sample["modf_mask"].unsqueeze(0).float()
    liquid_mask = sample["liquid_mask"].unsqueeze(0).float()
    alpha_painted_256 = sample["alpha_painted_256"].unsqueeze(0).float()
    mcly_any_16 = sample["mcly_any_16"].unsqueeze(0).float()
    what_plate_flag = float(sample["what_plate_flag"].item())

    liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
    object_presence = torch.maximum(mddf_mask, modf_mask)
    liquid_weight = 1.0 - (0.85 * liquid_mask_257)
    instance_weight = 1.0 - (0.75 * object_presence)
    base_mask = normal_mask * terrain_valid_mask * object_weight * liquid_weight * instance_weight
    if what_plate_flag > 0.5:
        base_mask = torch.zeros_like(base_mask)

    height_grad = _gradient_magnitude_257(height_raw)
    normal_grad = _gradient_magnitude_257(target_n).mean(dim=1, keepdim=True)
    alpha_grad = _gradient_magnitude_257(_resize_weight(alpha_painted_256, target_n.shape[-2:]))
    mcly_grad = _gradient_magnitude_257(_resize_weight(mcly_any_16, target_n.shape[-2:]))

    height_grad_n = (height_grad / _masked_mean(height_grad, base_mask).clamp_min(1e-6)).clamp(0.0, 4.0)
    normal_grad_n = (normal_grad / _masked_mean(normal_grad, base_mask).clamp_min(1e-6)).clamp(0.0, 4.0)
    alpha_grad_n = (alpha_grad / _masked_mean(alpha_grad, base_mask).clamp_min(1e-6)).clamp(0.0, 4.0)
    mcly_grad_n = (mcly_grad / _masked_mean(mcly_grad, base_mask).clamp_min(1e-6)).clamp(0.0, 4.0)

    transition = torch.maximum(alpha_grad_n, mcly_grad_n)
    hard_region = ((0.50 * height_grad_n) + (0.25 * normal_grad_n) + (0.25 * transition)).clamp(0.0, 4.0)
    hard_region = hard_region * terrain_valid_mask
    train_mask = base_mask * (1.0 + float(detail_boost) * hard_region)

    return (
        hard_region[0, 0].cpu().numpy().astype(np.float32, copy=False),
        transition[0, 0].cpu().numpy().astype(np.float32, copy=False),
        train_mask[0, 0].cpu().numpy().astype(np.float32, copy=False),
    )


def _connected_components(binary: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, int, int, int, int]] = []
    for y0 in range(h):
        for x0 in range(w):
            if not binary[y0, x0] or visited[y0, x0]:
                continue
            q: deque[tuple[int, int]] = deque()
            q.append((y0, x0))
            visited[y0, x0] = True
            min_y, max_y = y0, y0
            min_x, max_x = x0, x0
            area = 0
            while q:
                y, x = q.popleft()
                area += 1
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))
            comps.append((min_y, min_x, max_y, max_x, area))
    return comps


def _safe(x: str) -> str:
    out = []
    for ch in str(x):
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out) or "unknown"


def _cell_label(cx: int, cy: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prefix = ""
    x = cx
    while True:
        prefix = letters[x % 26] + prefix
        x = (x // 26) - 1
        if x < 0:
            break
    return f"{prefix}{cy + 1}"


def _cells_overlapping_bbox(x0: int, y0: int, x1: int, y1: int, grid_cells: int, tile_size: int = 256) -> list[str]:
    cell_w = tile_size / float(grid_cells)
    cell_h = tile_size / float(grid_cells)
    c0 = max(0, min(grid_cells - 1, int(np.floor(x0 / cell_w))))
    c1 = max(0, min(grid_cells - 1, int(np.floor(x1 / cell_w))))
    r0 = max(0, min(grid_cells - 1, int(np.floor(y0 / cell_h))))
    r1 = max(0, min(grid_cells - 1, int(np.floor(y1 / cell_h))))
    labels = []
    for ry in range(r0, r1 + 1):
        for cx in range(c0, c1 + 1):
            labels.append(_cell_label(cx, ry))
    return labels


def _extract_cell_patch(minimap: np.ndarray, cx: int, cy: int, grid_cells: int, tile_size: int = 256) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    cell_w = tile_size / float(grid_cells)
    cell_h = tile_size / float(grid_cells)
    x0 = int(round(cx * cell_w))
    y0 = int(round(cy * cell_h))
    x1 = int(round((cx + 1) * cell_w))
    y1 = int(round((cy + 1) * cell_h))
    x0 = max(0, min(tile_size - 1, x0))
    y0 = max(0, min(tile_size - 1, y0))
    x1 = max(x0 + 1, min(tile_size, x1))
    y1 = max(y0 + 1, min(tile_size, y1))
    return minimap[y0:y1, x0:x1], (x0, y0, x1 - 1, y1 - 1)


def _find_map_name(entry: dict[str, object]) -> str:
    for k in ("map", "meta_map", "world", "map_name"):
        v = entry.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return "unknown_map"


def _panel(minimap: np.ndarray, hard: np.ndarray, trans: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    y0, x0, y1, x1 = bbox
    rgb = (np.clip(minimap, 0.0, 1.0) * 255.0).astype(np.uint8)
    h_img = (np.clip(hard / max(float(np.max(hard)), 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    t_img = (np.clip(trans / max(float(np.max(trans)), 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    m_img = (np.clip(mask / max(float(np.max(mask)), 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)

    p0 = Image.fromarray(rgb)
    p1 = Image.fromarray(h_img, mode="L").convert("RGB")
    p2 = Image.fromarray(t_img, mode="L").convert("RGB")
    p3 = Image.fromarray(m_img, mode="L").convert("RGB")
    for p in (p0, p1, p2, p3):
        d = ImageDraw.Draw(p)
        d.rectangle([(x0, y0), (x1, y1)], outline=(255, 64, 64), width=2)
    canvas = Image.new("RGB", (256 * 4, 256), color=(0, 0, 0))
    canvas.paste(p0, (0, 0))
    canvas.paste(p1, (256, 0))
    canvas.paste(p2, (512, 0))
    canvas.paste(p3, (768, 0))
    return canvas


def _crop_fingerprint(rgb_u8: np.ndarray, size: int = 16) -> str:
    """Compute a compact perceptual hash for dedupe across builds/maps."""
    img = Image.fromarray(rgb_u8, mode="RGB").convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.reshape(-1).tolist())
    # binary string to fixed-width hex
    width = (len(bits) + 3) // 4
    return f"{int(bits, 2):0{width}x}"


def _dedupe_manifest(manifest: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    keep_by_hash: dict[str, dict[str, object]] = {}
    dup_count = 0
    for row in manifest:
        fp = str(row.get("fingerprint", ""))
        if not fp:
            continue
        prev = keep_by_hash.get(fp)
        if prev is None:
            keep_by_hash[fp] = row
            continue
        dup_count += 1
        # Keep the stronger candidate
        prev_score = float(prev.get("score_mean", 0.0))
        this_score = float(row.get("score_mean", 0.0))
        if this_score > prev_score:
            keep_by_hash[fp] = row

    deduped = sorted(
        keep_by_hash.values(),
        key=lambda r: (float(r.get("score_mean", 0.0)), float(r.get("score_max", 0.0)), int(r.get("component_area", 0))),
        reverse=True,
    )
    stats = {
        "input_candidates": len(manifest),
        "unique_fingerprints": len(keep_by_hash),
        "duplicates_dropped": int(dup_count),
        "dedupe_ratio": (float(len(keep_by_hash)) / float(len(manifest))) if manifest else 0.0,
    }
    return deduped, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Mine paste/prefab candidates from V16.1 curated tiles")
    p.add_argument("--dataset-dir", type=str, default="../output/datasets/v16")
    p.add_argument("--curation-manifest", type=str, default=None)
    p.add_argument("--builds", nargs="*", default=None)
    p.add_argument("--maps", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tiles", type=int, default=0)
    p.add_argument("--detail-boost", type=float, default=1.5)
    p.add_argument("--component-threshold", type=float, default=0.35)
    p.add_argument("--min-component-area", type=int, default=120)
    p.add_argument("--max-components-per-tile", type=int, default=6)
    p.add_argument("--bbox-padding", type=int, default=6)
    p.add_argument("--grid-cells", type=int, default=8, help="Per-tile grid resolution for cell anchors")
    p.add_argument("--emit-cell-library", action="store_true", help="Write cell-level library manifest and crops")
    p.add_argument("--cell-score-threshold", type=float, default=0.20, help="Min score mean for keeping a cell in the library")
    p.add_argument("--dedupe", action="store_true", help="Dedupe candidates across builds/maps via perceptual crop hash")
    p.add_argument("--dedupe-hash-size", type=int, default=16, help="Perceptual hash grid size (higher = stricter)")
    p.add_argument("--out-dir", type=str, default="../output/validation/paste_mining")
    args = p.parse_args()

    ds = V161Dataset(
        dataset_dir=args.dataset_dir,
        builds=args.builds,
        split="train",
        val_fraction=0.1,
        seed=int(args.seed),
        augment=False,
        curation_manifest=args.curation_manifest,
        height_channel=False,
    )

    include_maps = set(args.maps) if args.maps else None
    rng = np.random.RandomState(int(args.seed))
    positions = list(range(len(ds._indices)))
    if int(args.max_tiles) > 0 and len(positions) > int(args.max_tiles):
        positions = sorted(rng.choice(positions, size=int(args.max_tiles), replace=False).tolist())

    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"
    crops_dir = out_dir / "crops"
    panels_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    cell_manifest: list[dict[str, object]] = []
    candidate_id = 0

    for local_idx, ds_pos in enumerate(positions, start=1):
        global_idx = ds._indices[ds_pos]
        entry = ds._index_entries[global_idx]
        build = str(entry.get("build") or entry.get("_build") or "unknown_build")
        map_name = _find_map_name(entry)
        if include_maps is not None and map_name not in include_maps:
            continue
        tile_x = int(entry.get("tile_x") if entry.get("tile_x") is not None else -1)
        tile_y = int(entry.get("tile_y") if entry.get("tile_y") is not None else -1)
        if tile_x < 0 or tile_y < 0:
            continue

        sample = ds[ds_pos]
        minimap = sample["input"][0:3].permute(1, 2, 0).numpy().astype(np.float32, copy=False)
        hard, trans, train_mask = _hard_region_signals(sample, detail_boost=float(args.detail_boost))
        hard_n = hard / max(float(np.max(hard)), 1e-6)
        trans_n = trans / max(float(np.max(trans)), 1e-6)
        train_n = train_mask / max(float(np.max(train_mask)), 1e-6)
        score = np.maximum(hard_n, trans_n) * np.clip(train_n, 0.0, 1.0)

        binary = score >= float(args.component_threshold)
        comps = _connected_components(binary)
        comps = [c for c in comps if c[4] >= int(args.min_component_area)]
        comps.sort(key=lambda c: c[4], reverse=True)
        comps = comps[: max(1, int(args.max_components_per_tile))]

        for comp_idx, (y0, x0, y1, x1, area) in enumerate(comps, start=1):
            pad = int(args.bbox_padding)
            y0p = max(0, y0 - pad)
            x0p = max(0, x0 - pad)
            y1p = min(256, y1 + 1 + pad)
            x1p = min(256, x1 + 1 + pad)
            bbox = (y0p, x0p, y1p - 1, x1p - 1)

            safe_build = _safe(build)
            safe_map = _safe(map_name)
            stem = f"cand_{candidate_id:07d}_{safe_build}_{safe_map}_{tile_x}_{tile_y}_{comp_idx}"

            panel = _panel(minimap, hard, trans, train_mask, bbox)
            panel_path = panels_dir / f"{stem}.png"
            panel.save(panel_path)

            crop_rgb = (np.clip(minimap[y0p:y1p, x0p:x1p], 0.0, 1.0) * 255.0).astype(np.uint8)
            crop_path = crops_dir / f"{stem}.png"
            Image.fromarray(crop_rgb, mode="RGB").save(crop_path)
            fingerprint = _crop_fingerprint(crop_rgb, size=max(8, int(args.dedupe_hash_size)))

            score_crop = score[y0p:y1p, x0p:x1p]
            row = {
                "candidate_id": candidate_id,
                "build": build,
                "map": map_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "tile_id": int(entry.get("tile_id", -1)),
                "component_rank": comp_idx,
                "bbox_xyxy": [x0p, y0p, x1p - 1, y1p - 1],
                "bbox_wh": [x1p - x0p, y1p - y0p],
                "grid_cells": int(args.grid_cells),
                "source_cell_set": _cells_overlapping_bbox(x0p, y0p, x1p - 1, y1p - 1, int(args.grid_cells)),
                "component_area": int(area),
                "score_mean": float(np.mean(score_crop)),
                "score_max": float(np.max(score_crop)),
                "fingerprint": fingerprint,
                "hard_mean": float(np.mean(hard[y0p:y1p, x0p:x1p])),
                "transition_mean": float(np.mean(trans[y0p:y1p, x0p:x1p])),
                "panel_path": str(panel_path.relative_to(out_dir)).replace("\\", "/"),
                "crop_path": str(crop_path.relative_to(out_dir)).replace("\\", "/"),
            }
            manifest.append(row)
            candidate_id += 1

        if bool(args.emit_cell_library):
            grid_cells = int(args.grid_cells)
            safe_build = _safe(build)
            safe_map = _safe(map_name)
            cell_dir = crops_dir / "cells" / safe_build / safe_map / f"{tile_x}_{tile_y}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            for cy in range(grid_cells):
                for cx in range(grid_cells):
                    patch, (cx0, cy0, cx1, cy1) = _extract_cell_patch(minimap, cx, cy, grid_cells)
                    score_patch = score[cy0:cy1 + 1, cx0:cx1 + 1]
                    score_mean = float(np.mean(score_patch)) if score_patch.size > 0 else 0.0
                    if score_mean < float(args.cell_score_threshold):
                        continue
                    label = _cell_label(cx, cy)
                    cell_stem = f"cell_{safe_build}_{safe_map}_{tile_x}_{tile_y}_{label}"
                    cell_path = cell_dir / f"{cell_stem}.png"
                    Image.fromarray((np.clip(patch, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(cell_path)
                    cell_manifest.append(
                        {
                            "build": build,
                            "map": map_name,
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "grid_cells": grid_cells,
                            "cell": label,
                            "bbox_xyxy": [cx0, cy0, cx1, cy1],
                            "score_mean": score_mean,
                            "score_max": float(np.max(score_patch)) if score_patch.size > 0 else 0.0,
                            "cell_path": str(cell_path.relative_to(out_dir)).replace("\\", "/"),
                        }
                    )

        if local_idx % 200 == 0 or local_idx == len(positions):
            print(f"Processed tiles: {local_idx}/{len(positions)} | candidates={len(manifest)}")

    manifest.sort(key=lambda r: (float(r["score_mean"]), float(r["score_max"]), int(r["component_area"])), reverse=True)
    (out_dir / "paste_candidates.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (out_dir / "paste_candidates.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    dedupe_stats: dict[str, object] | None = None
    atlas_manifest = manifest
    if bool(args.dedupe):
        deduped, dedupe_stats = _dedupe_manifest(manifest)
        (out_dir / "paste_candidates_deduped.json").write_text(json.dumps(deduped, indent=2), encoding="utf-8")
        with (out_dir / "paste_candidates_deduped.jsonl").open("w", encoding="utf-8") as f:
            for row in deduped:
                f.write(json.dumps(row) + "\n")
        atlas_manifest = deduped

    brush_seed = [
        {
            "candidate_id": row["candidate_id"],
            "build": row["build"],
            "map": row["map"],
            "tile_x": row["tile_x"],
            "tile_y": row["tile_y"],
            "source_cell_set": row["source_cell_set"],
            "bbox_xyxy": row["bbox_xyxy"],
            "score_mean": row["score_mean"],
            "crop_path": row["crop_path"],
        }
        for row in manifest
    ]
    (out_dir / "brush_library_seed.json").write_text(json.dumps(brush_seed, indent=2), encoding="utf-8")
    with (out_dir / "brush_library_seed.jsonl").open("w", encoding="utf-8") as f:
        for row in brush_seed:
            f.write(json.dumps(row) + "\n")

    if bool(args.emit_cell_library):
        (out_dir / "cell_library.json").write_text(json.dumps(cell_manifest, indent=2), encoding="utf-8")
        with (out_dir / "cell_library.jsonl").open("w", encoding="utf-8") as f:
            for row in cell_manifest:
                f.write(json.dumps(row) + "\n")

    # quick top-k atlas
    top_k = min(128, len(atlas_manifest))
    if top_k > 0:
        cols = 8
        thumb = 128
        rows = (top_k + cols - 1) // cols
        atlas = Image.new("RGB", (cols * thumb, rows * thumb), color=(0, 0, 0))
        for i in range(top_k):
            crop_rel = atlas_manifest[i]["crop_path"]
            crop_img = Image.open(out_dir / str(crop_rel)).convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
            x = (i % cols) * thumb
            y = (i // cols) * thumb
            atlas.paste(crop_img, (x, y))
        atlas.save(out_dir / "paste_candidates_top128_atlas.png")

    summary = {
        "tiles_considered": len(positions),
        "candidates": len(manifest),
        "cell_library_entries": len(cell_manifest),
        "component_threshold": float(args.component_threshold),
        "min_component_area": int(args.min_component_area),
        "max_components_per_tile": int(args.max_components_per_tile),
        "grid_cells": int(args.grid_cells),
        "dedupe": bool(args.dedupe),
        "dedupe_stats": dedupe_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
