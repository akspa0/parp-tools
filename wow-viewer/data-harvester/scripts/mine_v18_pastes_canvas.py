"""Mine V18 paste candidates on stitched build/map canvases.

Phase 1: stitched candidate extraction with multi-tile coverage.
Phase 2: deterministic cross-build dedupe clusters with alpha-layer-aware keys.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _safe(x: str) -> str:
    out = []
    for ch in str(x):
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out) or "unknown"


def _find_map_name(entry: dict[str, object]) -> str:
    for key in ("map", "meta_map", "world", "map_name"):
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown_map"


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


def _connected_components_with_tiles(binary: np.ndarray, tile_size: int) -> list[dict[str, object]]:
    height, width = binary.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []
    for y0 in range(height):
        for x0 in range(width):
            if not binary[y0, x0] or visited[y0, x0]:
                continue
            q: deque[tuple[int, int]] = deque()
            q.append((y0, x0))
            visited[y0, x0] = True
            min_y, max_y = y0, y0
            min_x, max_x = x0, x0
            area = 0
            tile_bins: set[tuple[int, int]] = set()
            while q:
                y, x = q.popleft()
                area += 1
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                tile_bins.add((x // tile_size, y // tile_size))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))
            components.append({"bbox": (min_y, min_x, max_y, max_x), "area": int(area), "tile_bins": sorted(tile_bins)})
    return components


def _to_u8(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if mask is not None and np.any(mask):
        m = float(np.max(x[mask]))
    else:
        m = float(np.max(x))
    if m <= 0.0:
        return np.zeros_like(x, dtype=np.uint8)
    return (np.clip(x / m, 0.0, 1.0) * 255.0).astype(np.uint8)


def _colorize_gray(x: np.ndarray) -> np.ndarray:
    g = _to_u8(x)
    return np.stack([g, g, g], axis=-1)


def _overlay_candidates(minimap_rgb: np.ndarray, candidates: list[dict[str, object]], out_path: Path, downscale_max_side: int) -> None:
    img = Image.fromarray((np.clip(minimap_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    for idx, row in enumerate(candidates, start=1):
        x0, y0, x1, y1 = [int(v) for v in row["canvas_bbox"]]
        color = (255, 96, 64) if bool(row.get("multi_tile")) else (255, 196, 64)
        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), str(idx), fill=(255, 255, 255))
    if max(img.size) > int(downscale_max_side):
        scale = float(downscale_max_side) / float(max(img.size))
        new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
        img = img.resize(new_size, Image.Resampling.BILINEAR)
    img.save(out_path)


def _atlas_rows(rows: list[dict[str, object]], out_dir: Path, out_path: Path, cols: int = 8, thumb: int = 128) -> None:
    if not rows:
        return
    count = len(rows)
    grid_rows = (count + cols - 1) // cols
    atlas = Image.new("RGB", (cols * thumb, grid_rows * thumb), color=(0, 0, 0))
    for i, row in enumerate(rows):
        crop_rel = str(row.get("crop_path", ""))
        if not crop_rel:
            continue
        crop_abs = out_dir / crop_rel
        if not crop_abs.exists():
            continue
        crop_img = Image.open(crop_abs).convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
        x = (i % cols) * thumb
        y = (i // cols) * thumb
        atlas.paste(crop_img, (x, y))
    atlas.save(out_path)


def _selection_hash(rows: list[dict[str, object]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        key = "|".join(
            [
                str(row.get("build", "")),
                str(row.get("map", "")),
                str(row.get("candidate_id", "")),
                ",".join(str(v) for v in row.get("canvas_bbox", [])),
                ",".join(f"{t.get('tile_x', '')}:{t.get('tile_y', '')}" for t in row.get("tile_coverage", [])),
            ]
        )
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _crop_fingerprint(rgb_u8: np.ndarray, size: int = 16) -> str:
    img = Image.fromarray(rgb_u8, mode="RGB").convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.reshape(-1).tolist())
    width = (len(bits) + 3) // 4
    return f"{int(bits, 2):0{width}x}"


def _alpha_layer_signature(alpha_crop: np.ndarray) -> dict[str, object]:
    layers = alpha_crop
    if layers.ndim != 3 or layers.shape[2] != 4:
        layers = np.zeros((max(1, alpha_crop.shape[0]), max(1, alpha_crop.shape[1]), 4), dtype=np.float32)
    layers = np.clip(layers, 0.0, 1.0).astype(np.float32, copy=False)
    means = np.mean(layers, axis=(0, 1))
    coverage = np.mean(layers >= 0.05, axis=(0, 1))
    dominant = np.argsort(-means).tolist()
    dominant_layers = [int(i) for i in dominant if float(means[i]) >= 0.01][:3]
    quant = [int(round(float(v) * 1000.0)) for v in np.concatenate([means, coverage], axis=0)]
    sig_payload = ",".join(str(v) for v in quant)
    sig_hash = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:20]
    return {
        "layer_means": [float(v) for v in means.tolist()],
        "layer_coverage": [float(v) for v in coverage.tolist()],
        "dominant_layers": dominant_layers,
        "alpha_layer_signature": f"als_{sig_hash}",
    }


def _candidate_cluster_key(row: dict[str, object]) -> str:
    return "|".join(
        [
            str(row.get("rgb_fingerprint", "")),
            str(row.get("alpha_layer_signature", "")),
            str(row.get("tile_coverage_count", "")),
            str(row.get("canvas_bbox_wh", ["", ""])[0]),
            str(row.get("canvas_bbox_wh", ["", ""])[1]),
        ]
    )


def _cluster_score_key(row: dict[str, object]) -> tuple[float, float, int, int]:
    return (
        float(row.get("score_mean", 0.0)),
        float(row.get("score_max", 0.0)),
        int(row.get("component_area", 0)),
        -int(row.get("candidate_id", 0)),
    )


def _cluster_candidates(candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    by_key: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        key = _candidate_cluster_key(row)
        by_key.setdefault(key, []).append(row)

    deduped_rows: list[dict[str, object]] = []
    cluster_summaries: list[dict[str, object]] = []

    sorted_keys = sorted(by_key.keys())
    total_duplicates = 0
    for idx, key in enumerate(sorted_keys, start=1):
        members = list(by_key[key])
        members.sort(key=_cluster_score_key, reverse=True)
        canonical = members[0]
        cluster_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        cluster_id = f"cluster_{idx:06d}_{cluster_hash}"
        canonical_id = int(canonical.get("candidate_id", -1))
        total_duplicates += max(0, len(members) - 1)

        build_set = sorted({str(m.get("build", "")) for m in members})
        map_set = sorted({str(m.get("map", "")) for m in members})
        tile_coverage_hist: dict[str, int] = {}
        for m in members:
            tc = int(m.get("tile_coverage_count", 0))
            k = str(tc)
            tile_coverage_hist[k] = int(tile_coverage_hist.get(k, 0) + 1)

        cluster_summary = {
            "cluster_id": cluster_id,
            "cluster_key": key,
            "canonical_id": canonical_id,
            "size": len(members),
            "builds": build_set,
            "maps": map_set,
            "tile_coverage_hist": tile_coverage_hist,
            "score_mean_max": float(max(float(m.get("score_mean", 0.0)) for m in members)),
            "score_mean_min": float(min(float(m.get("score_mean", 0.0)) for m in members)),
            "alpha_layer_signature": str(canonical.get("alpha_layer_signature", "")),
            "rgb_fingerprint": str(canonical.get("rgb_fingerprint", "")),
        }
        cluster_summaries.append(cluster_summary)

        for variant_rank, member in enumerate(members, start=1):
            row = dict(member)
            row["cluster_id"] = cluster_id
            row["canonical_id"] = canonical_id
            row["variant_rank"] = int(variant_rank)
            row["cluster_size"] = int(len(members))
            row["is_canonical"] = bool(variant_rank == 1)
            row["cluster_key"] = key
            deduped_rows.append(row)

    deduped_rows.sort(
        key=lambda r: (
            str(r.get("cluster_id", "")),
            int(r.get("variant_rank", 0)),
            -int(r.get("candidate_id", 0)),
        )
    )
    cluster_summaries.sort(key=lambda r: (int(r.get("size", 0)), str(r.get("cluster_id", ""))), reverse=True)

    stats = {
        "input_candidates": len(candidates),
        "clusters": len(cluster_summaries),
        "canonical_count": len(cluster_summaries),
        "duplicates_dropped_if_canonical_only": int(total_duplicates),
        "canonical_ratio": (float(len(cluster_summaries)) / float(len(candidates))) if candidates else 0.0,
    }
    return deduped_rows, cluster_summaries, stats


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _cluster_atlas(
    deduped_rows: list[dict[str, object]],
    cluster_summaries: list[dict[str, object]],
    out_dir: Path,
    max_clusters: int,
    per_cluster_items: int,
    thumb: int,
) -> None:
    cluster_members: dict[str, list[dict[str, object]]] = {}
    for row in deduped_rows:
        cluster_members.setdefault(str(row.get("cluster_id", "")), []).append(row)
    for rows in cluster_members.values():
        rows.sort(key=lambda r: int(r.get("variant_rank", 0)))

    cluster_atlas_dir = out_dir / "cluster_atlas"
    cluster_atlas_dir.mkdir(parents=True, exist_ok=True)

    top_clusters = cluster_summaries[: max(0, int(max_clusters))]
    for cluster in top_clusters:
        cluster_id = str(cluster.get("cluster_id", "cluster_unknown"))
        rows = cluster_members.get(cluster_id, [])[: max(1, int(per_cluster_items))]
        if not rows:
            continue
        strip = Image.new("RGB", (thumb * len(rows), thumb), color=(0, 0, 0))
        for i, row in enumerate(rows):
            crop_rel = str(row.get("crop_path", ""))
            if not crop_rel:
                continue
            crop_abs = out_dir / crop_rel
            if not crop_abs.exists():
                continue
            img = Image.open(crop_abs).convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
            strip.paste(img, (i * thumb, 0))
        strip.save(cluster_atlas_dir / f"{cluster_id}.png")

    canonical_rows: list[dict[str, object]] = []
    for cluster in top_clusters:
        cluster_id = str(cluster.get("cluster_id", ""))
        rows = cluster_members.get(cluster_id, [])
        if rows:
            canonical_rows.append(rows[0])
    _atlas_rows(canonical_rows, out_dir, out_dir / "clusters_canonical_top_atlas.png", cols=8, thumb=thumb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine V18 paste candidates from stitched build/map canvases")
    parser.add_argument("--dataset-dir", type=str, default="../output/datasets/v16")
    parser.add_argument("--curation-manifest", type=str, default=None)
    parser.add_argument("--builds", nargs="*", default=None)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tiles", type=int, default=0, help="Global cap over sampled tiles before canvas stitch")
    parser.add_argument("--detail-boost", type=float, default=1.5)
    parser.add_argument("--component-threshold", type=float, default=0.35)
    parser.add_argument("--min-component-area", type=int, default=240)
    parser.add_argument("--max-components-per-canvas", type=int, default=24)
    parser.add_argument("--min-component-width", type=int, default=10)
    parser.add_argument("--min-component-height", type=int, default=10)
    parser.add_argument("--bbox-padding", type=int, default=8)
    parser.add_argument("--downscale-overlay-max-side", type=int, default=4096)
    parser.add_argument("--topk-atlas", type=int, default=128)
    parser.add_argument("--dedupe", action="store_true", help="Enable deterministic cross-build cluster assignment")
    parser.add_argument("--dedupe-hash-size", type=int, default=16)
    parser.add_argument("--cluster-atlas-top-clusters", type=int, default=64)
    parser.add_argument("--cluster-atlas-per-cluster", type=int, default=4)
    parser.add_argument("--cluster-atlas-thumb", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="../output/validation/v18_paste_canvas")
    args = parser.parse_args()

    ds = V161Dataset(
        dataset_dir=args.dataset_dir,
        builds=args.builds,
        split="train",
        val_fraction=0.0,
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

    grouped_tiles: dict[tuple[str, str], list[dict[str, object]]] = {}
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
        alpha = sample["alpha"].permute(1, 2, 0).numpy().astype(np.float32, copy=False)
        hard_region, transition, train_mask = _hard_region_signals(sample, detail_boost=float(args.detail_boost))
        grouped_tiles.setdefault((build, map_name), []).append(
            {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "tile_id": int(entry.get("tile_id", -1)),
                "minimap": minimap,
                "alpha": alpha,
                "hard_region": hard_region,
                "transition": transition,
                "train_mask": train_mask,
            }
        )
        if local_idx % 200 == 0 or local_idx == len(positions):
            print(f"Prepared tiles: {local_idx}/{len(positions)} | groups={len(grouped_tiles)}")

    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    overlays_dir = out_dir / "overlays"
    canvas_debug_dir = out_dir / "canvas_debug"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    canvas_debug_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, object]] = []
    candidate_id = 0
    canvas_summaries: list[dict[str, object]] = []

    for canvas_index, (canvas_key, tiles) in enumerate(sorted(grouped_tiles.items()), start=1):
        build, map_name = canvas_key
        if not tiles:
            continue
        tile_xs = [int(t["tile_x"]) for t in tiles]
        tile_ys = [int(t["tile_y"]) for t in tiles]
        min_tile_x = min(tile_xs)
        max_tile_x = max(tile_xs)
        min_tile_y = min(tile_ys)
        max_tile_y = max(tile_ys)
        tiles_w = (max_tile_x - min_tile_x + 1)
        tiles_h = (max_tile_y - min_tile_y + 1)
        canvas_w = tiles_w * 256
        canvas_h = tiles_h * 256

        minimap_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        alpha_canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
        hard_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        transition_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        train_mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        present_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        present_tile_set: set[tuple[int, int]] = set()

        for tile in tiles:
            tile_x = int(tile["tile_x"])
            tile_y = int(tile["tile_y"])
            present_tile_set.add((tile_x, tile_y))
            ox = (tile_x - min_tile_x) * 256
            oy = (tile_y - min_tile_y) * 256
            minimap_canvas[oy:oy + 256, ox:ox + 256, :] = tile["minimap"]
            alpha_canvas[oy:oy + 256, ox:ox + 256, :] = tile["alpha"][:256, :256, :]
            hard_canvas[oy:oy + 256, ox:ox + 256] = tile["hard_region"][:256, :256]
            transition_canvas[oy:oy + 256, ox:ox + 256] = tile["transition"][:256, :256]
            train_mask_canvas[oy:oy + 256, ox:ox + 256] = tile["train_mask"][:256, :256]
            present_mask[oy:oy + 256, ox:ox + 256] = True

        hard_n = np.zeros_like(hard_canvas, dtype=np.float32)
        trans_n = np.zeros_like(transition_canvas, dtype=np.float32)
        mask_n = np.zeros_like(train_mask_canvas, dtype=np.float32)
        if np.any(present_mask):
            hard_max = float(np.max(hard_canvas[present_mask]))
            trans_max = float(np.max(transition_canvas[present_mask]))
            mask_max = float(np.max(train_mask_canvas[present_mask]))
            if hard_max > 1e-6:
                hard_n[present_mask] = hard_canvas[present_mask] / hard_max
            if trans_max > 1e-6:
                trans_n[present_mask] = transition_canvas[present_mask] / trans_max
            if mask_max > 1e-6:
                mask_n[present_mask] = train_mask_canvas[present_mask] / mask_max

        score = np.maximum(hard_n, trans_n) * np.clip(mask_n, 0.0, 1.0)
        binary = (score >= float(args.component_threshold)) & present_mask
        components = _connected_components_with_tiles(binary, tile_size=256)
        components = [c for c in components if int(c["area"]) >= int(args.min_component_area)]
        components = [
            c
            for c in components
            if (int(c["bbox"][3]) - int(c["bbox"][1]) + 1) >= int(args.min_component_width)
            and (int(c["bbox"][2]) - int(c["bbox"][0]) + 1) >= int(args.min_component_height)
        ]
        components.sort(key=lambda c: int(c["area"]), reverse=True)
        components = components[: max(1, int(args.max_components_per_canvas))]

        safe_build = _safe(build)
        safe_map = _safe(map_name)
        kept_for_overlay: list[dict[str, object]] = []

        for comp_rank, comp in enumerate(components, start=1):
            min_y, min_x, max_y, max_x = [int(v) for v in comp["bbox"]]
            pad = int(args.bbox_padding)
            x0 = max(0, min_x - pad)
            y0 = max(0, min_y - pad)
            x1 = min(canvas_w - 1, max_x + pad)
            y1 = min(canvas_h - 1, max_y + pad)
            if x1 <= x0 or y1 <= y0:
                continue

            crop_rgb = (np.clip(minimap_canvas[y0:y1 + 1, x0:x1 + 1, :], 0.0, 1.0) * 255.0).astype(np.uint8)
            crop_rel = f"crops/cand_{candidate_id:08d}_{safe_build}_{safe_map}.png"
            crop_path = out_dir / crop_rel
            Image.fromarray(crop_rgb, mode="RGB").save(crop_path)
            rgb_fingerprint = _crop_fingerprint(crop_rgb, size=max(8, int(args.dedupe_hash_size)))

            tile_coverage: list[dict[str, int]] = []
            for local_tx, local_ty in comp["tile_bins"]:
                tile_x = min_tile_x + int(local_tx)
                tile_y = min_tile_y + int(local_ty)
                if (tile_x, tile_y) not in present_tile_set:
                    continue
                tile_coverage.append({"tile_x": int(tile_x), "tile_y": int(tile_y)})
            tile_coverage.sort(key=lambda t: (t["tile_y"], t["tile_x"]))

            score_crop = score[y0:y1 + 1, x0:x1 + 1]
            hard_crop = hard_canvas[y0:y1 + 1, x0:x1 + 1]
            transition_crop = transition_canvas[y0:y1 + 1, x0:x1 + 1]
            train_mask_crop = train_mask_canvas[y0:y1 + 1, x0:x1 + 1]
            alpha_crop = alpha_canvas[y0:y1 + 1, x0:x1 + 1, :]
            alpha_sig = _alpha_layer_signature(alpha_crop)

            row = {
                "candidate_id": candidate_id,
                "build": build,
                "map": map_name,
                "canvas_id": f"{safe_build}:{safe_map}",
                "canvas_origin_tile": [int(min_tile_x), int(min_tile_y)],
                "canvas_tiles_wh": [int(tiles_w), int(tiles_h)],
                "canvas_px_wh": [int(canvas_w), int(canvas_h)],
                "canvas_bbox": [int(x0), int(y0), int(x1), int(y1)],
                "canvas_bbox_wh": [int(x1 - x0 + 1), int(y1 - y0 + 1)],
                "component_rank": int(comp_rank),
                "component_area": int(comp["area"]),
                "tile_coverage": tile_coverage,
                "tile_coverage_count": len(tile_coverage),
                "multi_tile": bool(len(tile_coverage) > 1),
                "score_mean": float(np.mean(score_crop)) if score_crop.size > 0 else 0.0,
                "score_max": float(np.max(score_crop)) if score_crop.size > 0 else 0.0,
                "hard_mean": float(np.mean(hard_crop)) if hard_crop.size > 0 else 0.0,
                "transition_mean": float(np.mean(transition_crop)) if transition_crop.size > 0 else 0.0,
                "train_mask_mean": float(np.mean(train_mask_crop)) if train_mask_crop.size > 0 else 0.0,
                "rgb_fingerprint": rgb_fingerprint,
                "crop_path": crop_rel.replace("\\", "/"),
            }
            row.update(alpha_sig)
            all_candidates.append(row)
            kept_for_overlay.append(row)
            candidate_id += 1

        overlay_rel = f"overlays/{safe_build}_{safe_map}_canvas_overlay.png"
        _overlay_candidates(minimap_canvas, kept_for_overlay, out_dir / overlay_rel, downscale_max_side=int(args.downscale_overlay_max_side))

        signal_panel = np.concatenate(
            [
                (np.clip(minimap_canvas, 0.0, 1.0) * 255.0).astype(np.uint8),
                _colorize_gray(hard_canvas),
                _colorize_gray(transition_canvas),
                _colorize_gray(score),
            ],
            axis=1,
        )
        Image.fromarray(signal_panel, mode="RGB").save(canvas_debug_dir / f"{safe_build}_{safe_map}_signals.png")

        canvas_summary = {
            "canvas_index": int(canvas_index),
            "build": build,
            "map": map_name,
            "origin_tile": [int(min_tile_x), int(min_tile_y)],
            "tiles_wh": [int(tiles_w), int(tiles_h)],
            "canvas_px_wh": [int(canvas_w), int(canvas_h)],
            "tiles_present": int(len(tiles)),
            "components_total": int(len(components)),
            "candidates_emitted": int(len(kept_for_overlay)),
            "multi_tile_candidates": int(sum(1 for r in kept_for_overlay if bool(r.get("multi_tile")))),
            "overlay_path": overlay_rel.replace("\\", "/"),
        }
        canvas_summaries.append(canvas_summary)
        print(
            f"Canvas {canvas_index}/{len(grouped_tiles)} {build}/{map_name}: "
            f"tiles={len(tiles)} candidates={canvas_summary['candidates_emitted']} "
            f"multi_tile={canvas_summary['multi_tile_candidates']}"
        )

    all_candidates.sort(
        key=lambda r: (
            int(r.get("tile_coverage_count", 0)),
            float(r.get("score_mean", 0.0)),
            int(r.get("component_area", 0)),
        ),
        reverse=True,
    )

    (out_dir / "candidates.json").write_text(json.dumps(all_candidates, indent=2), encoding="utf-8")
    _write_jsonl(out_dir / "candidates.jsonl", all_candidates)

    (out_dir / "canvas_summary.json").write_text(json.dumps(canvas_summaries, indent=2), encoding="utf-8")
    _write_jsonl(out_dir / "canvas_summary.jsonl", canvas_summaries)

    top_k = min(max(0, int(args.topk_atlas)), len(all_candidates))
    if top_k > 0:
        _atlas_rows(all_candidates[:top_k], out_dir, out_dir / "candidates_top_atlas.png", cols=8, thumb=128)

    dedupe_stats: dict[str, object] | None = None
    cluster_summaries: list[dict[str, object]] = []
    deduped_rows: list[dict[str, object]] = []
    if bool(args.dedupe):
        deduped_rows, cluster_summaries, dedupe_stats = _cluster_candidates(all_candidates)
        (out_dir / "candidates_deduped.json").write_text(json.dumps(deduped_rows, indent=2), encoding="utf-8")
        _write_jsonl(out_dir / "candidates_deduped.jsonl", deduped_rows)
        (out_dir / "cluster_summary.json").write_text(json.dumps(cluster_summaries, indent=2), encoding="utf-8")
        _write_jsonl(out_dir / "cluster_summary.jsonl", cluster_summaries)
        (out_dir / "dedupe_stats.json").write_text(json.dumps(dedupe_stats, indent=2), encoding="utf-8")
        _cluster_atlas(
            deduped_rows,
            cluster_summaries,
            out_dir=out_dir,
            max_clusters=int(args.cluster_atlas_top_clusters),
            per_cluster_items=int(args.cluster_atlas_per_cluster),
            thumb=int(args.cluster_atlas_thumb),
        )

    multi_tile_count = int(sum(1 for row in all_candidates if bool(row.get("multi_tile"))))
    summary = {
        "tiles_considered": int(len(positions)),
        "canvas_groups": int(len(canvas_summaries)),
        "candidates": int(len(all_candidates)),
        "multi_tile_candidates": int(multi_tile_count),
        "multi_tile_ratio": (float(multi_tile_count) / float(len(all_candidates))) if all_candidates else 0.0,
        "component_threshold": float(args.component_threshold),
        "min_component_area": int(args.min_component_area),
        "max_components_per_canvas": int(args.max_components_per_canvas),
        "selection_hash": _selection_hash(all_candidates),
        "dedupe_enabled": bool(args.dedupe),
        "dedupe_stats": dedupe_stats,
        "cluster_hash": _selection_hash(deduped_rows) if deduped_rows else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "config.snapshot.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
