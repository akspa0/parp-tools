"""Mine V18 paste candidates via direct Zarr bulk reads + GPU batch signals.

No canvas assembly. Each tile's signal map is processed independently to find
sub-tile paste candidates. Cross-tile pastes are detected by edge-touching
components and merged via dedupe.

Output: Zarr tile->paste index + JSONL metadata (no pixel files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from scipy.ndimage import find_objects, label, sum as nd_sum

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BATCH_SIZE = 128


# ── GPU signal helpers ──────────────────────────────────────────────

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


def _hard_region_signals_batched(
    batch: dict[str, torch.Tensor], detail_boost: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched GPU signal computation. Returns (B, 257, 257) tensors on CPU."""
    target_n = F.normalize(batch["normals"].float(), dim=1, eps=1e-6)
    height_raw = batch["height_raw"].float()
    normal_mask = batch["normal_mask"].float()
    terrain_valid_mask = batch["terrain_valid_mask_257"].float()
    object_weight = batch["weight_257"].float()
    mddf_mask = batch["mddf_mask"].float()
    modf_mask = batch["modf_mask"].float()
    liquid_mask = batch["liquid_mask"].float()
    alpha_painted_256 = batch["alpha_painted_256"].float()
    mcly_any_16 = batch["mcly_any_16"].float()
    what_plate_flag = batch["what_plate_flag"].float()

    liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
    object_presence = torch.maximum(mddf_mask, modf_mask)
    base_mask = (
        normal_mask
        * terrain_valid_mask
        * object_weight
        * (1.0 - 0.85 * liquid_mask_257)
        * (1.0 - 0.75 * object_presence)
        * (1.0 - what_plate_flag.reshape(-1, 1, 1, 1).float())
    )

    height_grad = _gradient_magnitude_257(height_raw)
    normal_grad = _gradient_magnitude_257(target_n).mean(dim=1, keepdim=True)
    alpha_grad = _gradient_magnitude_257(_resize_weight(alpha_painted_256, target_n.shape[-2:]))
    mcly_grad = _gradient_magnitude_257(_resize_weight(mcly_any_16, target_n.shape[-2:]))

    def _batch_norm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = mask.shape[0]
        sum_x = (x * mask).reshape(b, -1).sum(dim=1)
        sum_m = mask.reshape(b, -1).sum(dim=1).clamp_min(1e-6)
        return (x / (sum_x / sum_m).reshape(-1, 1, 1, 1)).clamp(0.0, 4.0)

    hgn = _batch_norm(height_grad, base_mask)
    ngn = _batch_norm(normal_grad, base_mask)
    agn = _batch_norm(alpha_grad, base_mask)
    mgn = _batch_norm(mcly_grad, base_mask)

    transition = torch.maximum(agn, mgn)
    hard_region = ((0.50 * hgn) + (0.25 * ngn) + (0.25 * transition)).clamp(0.0, 4.0)
    hard_region = hard_region * terrain_valid_mask
    train_mask = base_mask * (1.0 + float(detail_boost) * hard_region)

    return (
        hard_region[:, 0].cpu(),
        transition[:, 0].cpu(),
        train_mask[:, 0].cpu(),
    )


# ── Component finding ───────────────────────────────────────────────

def _connected_components(binary: np.ndarray, _tile_size: int) -> list[dict[str, object]]:
    labeled, n = label(binary)
    if n == 0:
        return []
    objs = find_objects(labeled)
    areas = nd_sum(binary, labeled, range(1, n + 1))
    components: list[dict[str, object]] = []
    for i in range(n):
        slice_y, slice_x = objs[i]
        min_y, max_y = slice_y.start, slice_y.stop - 1
        min_x, max_x = slice_x.start, slice_x.stop - 1
        components.append({
            "bbox": (min_y, min_x, max_y, max_x),
            "area": int(areas[i]),
        })
    return components


# ── Hash / fingerprint helpers ──────────────────────────────────────

def _numpy_resize_bilinear(gray: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    src_h, src_w = gray.shape
    ys = np.linspace(0, src_h - 1, dst_h)
    xs = np.linspace(0, src_w - 1, dst_w)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    fy = (ys - y0)[:, np.newaxis]
    fx = (xs - x0)[np.newaxis, :]
    top = gray[y0][:, x0] * (1 - fx) + gray[y0][:, x1] * fx
    bot = gray[y1][:, x0] * (1 - fx) + gray[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def _crop_fingerprint(rgb_u8: np.ndarray, size: int = 16) -> str:
    h, w = rgb_u8.shape[:2]
    max_side = max(h, w)
    if max_side == 0:
        return "0" * ((size * size + 3) // 4)
    pad_h = max_side - h
    pad_w = max_side - w
    padded = np.pad(rgb_u8.astype(np.float32, copy=False), ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=128.0)
    gray = 0.299 * padded[:, :, 0] + 0.587 * padded[:, :, 1] + 0.114 * padded[:, :, 2]
    resized = _numpy_resize_bilinear(gray, size, size + 1)
    diff = resized[:, 1:] > resized[:, :-1]
    return bytes(np.packbits(diff.astype(np.uint8, copy=False))).hex()


def _average_hash(rgb_u8: np.ndarray, size: int = 8) -> str:
    h, w = rgb_u8.shape[:2]
    max_side = max(h, w)
    if max_side == 0:
        return "0" * ((size * size + 3) // 4)
    pad_h = max_side - h
    pad_w = max_side - w
    padded = np.pad(rgb_u8.astype(np.float32, copy=False), ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=128.0)
    gray = 0.299 * padded[:, :, 0] + 0.587 * padded[:, :, 1] + 0.114 * padded[:, :, 2]
    resized = _numpy_resize_bilinear(gray, size, size)
    mean = float(np.mean(resized))
    diff = resized > mean
    return bytes(np.packbits(diff.astype(np.uint8, copy=False))).hex()


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


def _hamming_distance_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        return 999
    xor = int(a, 16) ^ int(b, 16)
    return bin(xor).count("1")


def _candidate_exact_key(row: dict[str, object]) -> str:
    return "|".join([
        str(row.get("rgb_fingerprint", "")),
        str(row.get("alpha_layer_signature", "")),
        str(row.get("tile_coverage_count", "")),
    ])


def _cluster_score_key(row: dict[str, object]) -> tuple[float, float, int, int]:
    return (
        float(row.get("score_mean", 0.0)),
        float(row.get("score_max", 0.0)),
        int(row.get("component_area", 0)),
        -int(row.get("candidate_id", 0)),
    )


def _selection_hash(rows: list[dict[str, object]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        key = "|".join([
            str(row.get("build", "")),
            str(row.get("map", "")),
            str(row.get("candidate_id", "")),
            ",".join(str(v) for v in row.get("tile_local_bbox", [])),
            str(row.get("tile_id", "")),
        ])
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ── Dedupe ──────────────────────────────────────────────────────────

def _cluster_candidates(
    candidates: list[dict[str, object]],
    hamming_threshold: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if hamming_threshold <= 0:
        return _cluster_candidates_exact(candidates)
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        bucket_key = f'{row.get("alpha_layer_signature", "")}|{row.get("tile_coverage_count", "")}'
        buckets.setdefault(bucket_key, []).append(row)
    cluster_id_counter = 0
    deduped_rows: list[dict[str, object]] = []
    cluster_summaries: list[dict[str, object]] = []
    for bucket_key, bucket_rows in sorted(buckets.items()):
        bucket_rows.sort(key=_cluster_score_key, reverse=True)
        clusters_in_bucket: list[list[dict[str, object]]] = []
        for row in bucket_rows:
            row_hash = str(row.get("avg_hash", ""))
            best_idx = -1
            best_dist = 999
            for ci, cluster in enumerate(clusters_in_bucket):
                can_hash = str(cluster[0].get("avg_hash", ""))
                dist = _hamming_distance_hex(row_hash, can_hash)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = ci
            if best_idx >= 0 and best_dist <= int(hamming_threshold):
                clusters_in_bucket[best_idx].append(row)
            else:
                clusters_in_bucket.append([row])
        for members in clusters_in_bucket:
            members.sort(key=_cluster_score_key, reverse=True)
            canonical = members[0]
            cluster_id_counter += 1
            cluster_key_payload = "|".join([
                str(canonical.get("avg_hash", "")),
                str(canonical.get("alpha_layer_signature", "")),
                str(canonical.get("tile_coverage_count", "")),
            ])
            cluster_hash = hashlib.sha256(cluster_key_payload.encode("utf-8")).hexdigest()[:12]
            cluster_id = f"cluster_{cluster_id_counter:06d}_{cluster_hash}"
            canonical_id = int(canonical.get("candidate_id", -1))
            build_set = sorted({str(m.get("build", "")) for m in members})
            map_set = sorted({str(m.get("map", "")) for m in members})
            tile_coverage_hist: dict[str, int] = {}
            for m in members:
                tc = int(m.get("tile_coverage_count", 0))
                k = str(tc)
                tile_coverage_hist[k] = int(tile_coverage_hist.get(k, 0) + 1)
            cluster_summaries.append({
                "cluster_id": cluster_id,
                "cluster_key": f"approx_{cluster_key_payload}",
                "canonical_id": canonical_id,
                "size": len(members),
                "builds": build_set,
                "maps": map_set,
                "tile_coverage_hist": tile_coverage_hist,
                "score_mean_max": float(max(float(m.get("score_mean", 0.0)) for m in members)),
                "score_mean_min": float(min(float(m.get("score_mean", 0.0)) for m in members)),
                "alpha_layer_signature": str(canonical.get("alpha_layer_signature", "")),
                "avg_hash": str(canonical.get("avg_hash", "")),
                "hamming_threshold": int(hamming_threshold),
            })
            for variant_rank, member in enumerate(members, start=1):
                row = dict(member)
                row["cluster_id"] = cluster_id
                row["canonical_id"] = canonical_id
                row["variant_rank"] = int(variant_rank)
                row["cluster_size"] = int(len(members))
                row["is_canonical"] = bool(variant_rank == 1)
                row["cluster_key"] = f"approx_{cluster_key_payload}"
                deduped_rows.append(row)
    deduped_rows.sort(key=lambda r: (str(r.get("cluster_id", "")), int(r.get("variant_rank", 0)), -int(r.get("candidate_id", 0))))
    cluster_summaries.sort(key=lambda r: (int(r.get("size", 0)), str(r.get("cluster_id", ""))), reverse=True)
    total_duplicates = sum(max(0, int(cs["size"]) - 1) for cs in cluster_summaries)
    stats = {
        "input_candidates": len(candidates),
        "clusters": len(cluster_summaries),
        "canonical_count": len(cluster_summaries),
        "duplicates_dropped_if_canonical_only": int(total_duplicates),
        "canonical_ratio": (float(len(cluster_summaries)) / float(len(candidates))) if candidates else 0.0,
        "hamming_threshold": int(hamming_threshold),
    }
    return deduped_rows, cluster_summaries, stats


def _cluster_candidates_exact(candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    by_key: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        key = _candidate_exact_key(row)
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
        cluster_summaries.append({
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
        })
        for variant_rank, member in enumerate(members, start=1):
            row = dict(member)
            row["cluster_id"] = cluster_id
            row["canonical_id"] = canonical_id
            row["variant_rank"] = int(variant_rank)
            row["cluster_size"] = int(len(members))
            row["is_canonical"] = bool(variant_rank == 1)
            row["cluster_key"] = key
            deduped_rows.append(row)
    deduped_rows.sort(key=lambda r: (str(r.get("cluster_id", "")), int(r.get("variant_rank", 0)), -int(r.get("candidate_id", 0))))
    cluster_summaries.sort(key=lambda r: (int(r.get("size", 0)), str(r.get("cluster_id", ""))), reverse=True)
    stats = {
        "input_candidates": len(candidates),
        "clusters": len(cluster_summaries),
        "canonical_count": len(cluster_summaries),
        "duplicates_dropped_if_canonical_only": int(total_duplicates),
        "canonical_ratio": (float(len(cluster_summaries)) / float(len(candidates))) if candidates else 0.0,
    }
    return deduped_rows, cluster_summaries, stats


# ── I/O helpers ─────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _safe(x: str) -> str:
    out = []
    for ch in str(x):
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out) or "unknown"


def _compute_local_bboxes(
    candidates: list[dict[str, object]],
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    pairs: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for cand_idx, row in enumerate(candidates):
        tid = int(row.get("tile_id", -1))
        bbox = row.get("tile_local_bbox", None)
        if tid >= 0 and bbox and len(bbox) == 4:
            pairs.append((tid, cand_idx, tuple(bbox)))
    return pairs


def _build_zarr_index(
    zarr_path: Path,
    tile_paste_pairs: list[tuple[int, int, tuple[int, int, int, int]]],
    _deduped_rows: list[dict[str, object]],
) -> None:
    if not tile_paste_pairs:
        return
    max_tile_id = max(p[0] for p in tile_paste_pairs) + 1
    pairs_by_tile: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for tid, cidx, bbox in tile_paste_pairs:
        pairs_by_tile.setdefault(tid, []).append((cidx, bbox))

    tile_offset = np.zeros(max_tile_id + 1, dtype=np.int64)
    cand_idxs: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    total = 0
    for tid in range(max_tile_id):
        tile_offset[tid] = total
        for cidx, bbox in pairs_by_tile.get(tid, []):
            cand_idxs.append(cidx)
            bboxes.append(bbox)
            total += 1
    tile_offset[max_tile_id] = total

    z = zarr.open_group(str(zarr_path), mode="w")
    z.create_array("tile_offset", data=tile_offset)
    z.create_array("candidate_idx", data=np.array(cand_idxs, dtype=np.int64))
    bbox_arr = np.zeros((len(bboxes), 4), dtype=np.int32)
    for i, b in enumerate(bboxes):
        bbox_arr[i] = list(b)
    z.create_array("tile_local_bbox", data=bbox_arr)
    z.attrs["total_tiles"] = int(max_tile_id)
    z.attrs["total_paste_pairs"] = int(total)
    print(f"Zarr index: {max_tile_id} tiles, {total} paste overlap pairs")


def _save_checkpoint(out_dir: Path, candidates: list[dict], checkpoint_idx: int) -> None:
    cp = out_dir / f"checkpoint_{checkpoint_idx:06d}.jsonl"
    _write_jsonl(cp, candidates)
    print(f"Checkpoint saved: {cp} ({len(candidates)} candidates)")


# ── core per-tile processing ────────────────────────────────────────

def _process_tile_signals(
    hard_257: np.ndarray,
    trans_257: np.ndarray,
    mask_257: np.ndarray,
    minimap: np.ndarray,
    alpha: np.ndarray,
    component_threshold: float,
    min_component_area: int,
    min_component_width: int,
    min_component_height: int,
    max_components: int,
    bbox_padding: int,
    dedupe_hash_size: int,
) -> list[dict[str, object]]:
    """Find paste candidates within one tile's signal maps."""
    h, trans, m = hard_257[:256, :256], trans_257[:256, :256], mask_257[:256, :256]
    hm = h.max()
    tm = trans.max()
    mm = m.max()
    if hm < 1e-6 or mm < 1e-6:
        return []
    hn = h / hm
    mn = m / mm
    # Degenerate transition signal: if near-constant (single-layer terrain),
    # fall back to hard_region only
    if tm >= 1e-6 and float(np.std(trans)) > 0.05:
        tn = trans / tm
        score = np.maximum(hn, tn) * np.clip(mn, 0.0, 1.0)
    else:
        score = hn * np.clip(mn, 0.0, 1.0)
    binary = score >= float(component_threshold)
    if not binary.any():
        return []

    comps = _connected_components(binary, 256)
    comps = [c for c in comps if int(c["area"]) >= int(min_component_area)]
    comps = [c for c in comps
             if (int(c["bbox"][2]) - int(c["bbox"][0]) + 1) >= int(min_component_width)
             and (int(c["bbox"][3]) - int(c["bbox"][1]) + 1) >= int(min_component_height)]
    comps.sort(key=lambda c: int(c["area"]), reverse=True)
    comps = comps[:max(1, int(max_components))]

    results: list[dict[str, object]] = []
    for comp in comps:
        min_y, min_x, max_y, max_x = [int(v) for v in comp["bbox"]]
        pad = int(bbox_padding)
        x0 = max(0, min_x - pad)
        y0 = max(0, min_y - pad)
        x1 = min(255, max_x + pad)
        y1 = min(255, max_y + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        # Snap to ADT chunk grid (16x16 sub-cells)
        x0_a = (x0 // 16) * 16
        y0_a = (y0 // 16) * 16
        x1_a = min(255, ((x1 + 15) // 16) * 16 - 1)
        y1_a = min(255, ((y1 + 15) // 16) * 16 - 1)
        touches_edge = (x0_a == 0 or y0_a == 0 or x1_a == 255 or y1_a == 255)

        crop_rgb = (np.clip(minimap[y0_a:y1_a + 1, x0_a:x1_a + 1, :], 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb_fp = _crop_fingerprint(crop_rgb, size=max(8, int(dedupe_hash_size)))
        avg_h = _average_hash(crop_rgb, size=8)
        alpha_crop = alpha[y0_a:y1_a + 1, x0_a:x1_a + 1, :]
        alpha_sig = _alpha_layer_signature(alpha_crop)
        score_crop = score[y0_a:y1_a + 1, x0_a:x1_a + 1]

        results.append({
            "tile_local_bbox": [int(x0_a), int(y0_a), int(x1_a), int(y1_a)],
            "component_area": int(comp["area"]),
            "score_mean": float(np.mean(score_crop)) if score_crop.size > 0 else 0.0,
            "score_max": float(np.max(score_crop)) if score_crop.size > 0 else 0.0,
            "rgb_fingerprint": rgb_fp,
            "avg_hash": avg_h,
            "touches_edge": touches_edge,
            **alpha_sig,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine V18 paste candidates (direct Zarr, GPU batch, sub-tile)")
    parser.add_argument("--dataset-dir", type=str, default="../output/datasets/v16")
    parser.add_argument("--builds", nargs="*", default=None)
    parser.add_argument("--curation-manifest", type=str, default=None,
                        help="Path to kept_tiles.parquet for filtering tile_ids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--detail-boost", type=float, default=1.5)
    parser.add_argument("--component-threshold", type=float, default=0.20)
    parser.add_argument("--min-component-area", type=int, default=256)
    parser.add_argument("--max-components-per-tile", type=int, default=12)
    parser.add_argument("--min-component-width", type=int, default=16)
    parser.add_argument("--min-component-height", type=int, default=16)
    parser.add_argument("--bbox-padding", type=int, default=8)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--dedupe-hash-size", type=int, default=16)
    parser.add_argument("--dedupe-hamming-threshold", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=10000,
                        help="Save checkpoint every N tiles processed")
    parser.add_argument("--out-dir", type=str, default="../output/v18/pastes/v18_full_corpus_v2")
    args = parser.parse_args()

    global _BATCH_SIZE
    _BATCH_SIZE = max(1, int(args.batch_size))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_base = Path(args.dataset_dir)

    # ── Phase 0: load index entries ──────────────────────────────
    import pyarrow.parquet as pq

    build_dirs = args.builds or sorted(
        d.stem.replace(".zarr", "") for d in zarr_base.glob("*.zarr")
    )

    all_entries: list[dict] = []
    for build in build_dirs:
        index_path = zarr_base / f"{build}.zarr" / "index.parquet"
        if not index_path.exists():
            print(f"  Skipping {build}: no index.parquet")
            continue
        table = pq.read_table(str(index_path))
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            row["_build"] = build
            tx = int(row.get("tile_x", -1))
            ty = int(row.get("tile_y", -1))
            tid = int(row.get("tile_id", -1))
            if tx < 0 or ty < 0 or tid < 0:
                continue
            all_entries.append(row)

    print(f"Loaded {len(all_entries)} index entries from {len(build_dirs)} builds")

    curation_manifest_path = args.curation_manifest
    if curation_manifest_path:
        if Path(curation_manifest_path).is_dir():
            curation_manifest_path = str(Path(curation_manifest_path) / "kept_tiles.parquet")
        curated = pq.read_table(str(curation_manifest_path))
        curated_ids = set(int(v) for v in curated.column("tile_id").to_pylist())
        before = len(all_entries)
        all_entries = [e for e in all_entries if int(e["tile_id"]) in curated_ids]
        print(f"Curation manifest: {before} -> {len(all_entries)} (kept {len(curated_ids)} unique tile_ids)")

    rng = np.random.RandomState(int(args.seed))
    rng.shuffle(all_entries)
    if int(args.max_tiles) > 0 and len(all_entries) > int(args.max_tiles):
        all_entries = all_entries[:int(args.max_tiles)]

    # ── Prepare Zarr stores ──────────────────────────────────────
    stores: dict[str, zarr.Group] = {}
    for build in build_dirs:
        zarr_path = zarr_base / f"{build}.zarr"
        if not zarr_path.exists():
            continue
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        stores[build] = zarr.open_group(store=store, mode="r")

    # list of Zarr array names to read in bulk per build
    bulk_keys = [
        "minimap_rgb",
        "alpha_256",
        "normal_xyz",
        "normal_mask",
        "height_257",
        "mddf_mask",
        "modf_mask",
        "liquid_mask",
        "mcly_layer_mask",
        "object_precise_mask",
        "object_filtered_mask",
        "object_mask",
    ]

    def _read_build_arrays(build: str) -> dict[str, np.ndarray]:
        root = stores[build]
        arrays: dict[str, np.ndarray] = {}
        for k in bulk_keys:
            if k in root:
                arrays[k] = root[k][:]
        return arrays

    def _fill_checkerboard_mask(mask: np.ndarray) -> np.ndarray:
        """Fill checkerboard gaps (valid/invalid alternating) via cardinal propagation."""
        filled = mask.copy()
        filled[1:, :] |= mask[:-1, :]
        filled[:-1, :] |= mask[1:, :]
        filled[:, 1:] |= mask[:, :-1]
        filled[:, :-1] |= mask[:, 1:]
        return filled.astype(np.float32)


    def _resolve_object_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
        for k in ("object_precise_mask", "object_filtered_mask", "object_mask"):
            if k in arrays:
                return arrays[k]
        raise KeyError("No object mask found")

    def _compute_what_plate(height_257: np.ndarray, normal_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        terrain_frac = (normal_mask > 0.1).reshape(normal_mask.shape[0], -1).mean(axis=1)
        height_range = height_257.reshape(height_257.shape[0], -1).max(axis=1) - height_257.reshape(height_257.shape[0], -1).min(axis=1)
        flat_terrain = terrain_frac < 0.30
        blank_height = height_range < 0.5
        obj_frac = (object_mask > 0.5).reshape(object_mask.shape[0], -1).mean(axis=1)
        heavy_objects = obj_frac > 0.7
        return (flat_terrain | blank_height | heavy_objects).astype(np.float32)

    # ── Phase 1: process tiles in GPU batches ────────────────────
    all_candidates: list[dict[str, object]] = []
    candidate_id = 0
    total_tiles = len(all_entries)
    tiles_processed = 0

    # Group entries by build for bulk array reads
    by_build: dict[str, list[dict]] = {}
    for e in all_entries:
        by_build.setdefault(e["_build"], []).append(e)

    for build, entries in sorted(by_build.items()):
        if build not in stores:
            continue
        n = len(entries)
        print(f"\nBuild {build} ({n} tiles) — loading arrays...")
        arrays = _read_build_arrays(build)
        n_arr = arrays["minimap_rgb"].shape[0]
        print(f"  Loaded {len(arrays)} arrays ({n_arr} tiles in store)")

        # Pre-compute per-tile derived fields (bulk numpy, fast)
        print(f"  Pre-computing derived fields...")
        obj_mask = _resolve_object_mask(arrays)
        weight_257 = 1.0 - np.clip(obj_mask, 0.0, 1.0)
        obj_presence = np.maximum(
            arrays["mddf_mask"] if "mddf_mask" in arrays else np.zeros_like(weight_257),
            arrays["modf_mask"] if "modf_mask" in arrays else np.zeros_like(weight_257),
        )
        alpha_painted = np.clip(arrays["alpha_256"], 0.0, 1.0)
        mcly = arrays.get("mcly_layer_mask", np.zeros((n_arr, 16, 16, 4), dtype=np.float32))
        mcly_any_16 = (mcly.max(axis=3) > 0.05).astype(np.float32)
        # Fix checkerboard normal_mask (V18Dataset does this in __getitem__)
        normal_mask_contiguous = _fill_checkerboard_mask(arrays["normal_mask"])
        what_plate = _compute_what_plate(arrays["height_257"], normal_mask_contiguous, obj_mask)
        terrain_valid = normal_mask_contiguous * (1.0 - np.clip(obj_presence, 0.0, 1.0))
        liquid_resized = np.pad(arrays.get("liquid_mask", np.zeros((n_arr, 256, 256), dtype=np.float32)), ((0, 0), (0, 1), (0, 1)), mode="edge")
        terrain_valid *= (1.0 - np.clip(liquid_resized, 0.0, 1.0) * 0.85)
        what_plate_bool = what_plate > 0.5
        terrain_valid[what_plate_bool] = 0.0

        # Process in GPU batches
        for batch_start in range(0, n, _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, n)
            batch_entries = entries[batch_start:batch_end]
            bs = batch_end - batch_start

            # Look up tile indices in Zarr store (tile_id == position in arrays)
            batch_indices = [int(e["tile_id"]) for e in batch_entries]

            # Stack as torch + GPU
            def _gather(arr: np.ndarray, idx: list[int]) -> torch.Tensor:
                return torch.from_numpy(arr[idx]).to(_DEVICE)

            batch: dict[str, torch.Tensor] = {
                "normals": _gather(arrays["normal_xyz"], batch_indices).permute(0, 3, 1, 2).float(),
                "height_raw": _gather(arrays["height_257"], batch_indices).unsqueeze(1),
                "normal_mask": _gather(normal_mask_contiguous, batch_indices).unsqueeze(1),
                "terrain_valid_mask_257": _gather(terrain_valid, batch_indices).unsqueeze(1),
                "weight_257": _gather(weight_257, batch_indices).unsqueeze(1),
                "mddf_mask": _gather(arrays.get("mddf_mask", np.zeros((n_arr, 257, 257), dtype=np.float32)), batch_indices).unsqueeze(1),
                "modf_mask": _gather(arrays.get("modf_mask", np.zeros((n_arr, 257, 257), dtype=np.float32)), batch_indices).unsqueeze(1),
                "liquid_mask": _gather(arrays.get("liquid_mask", np.zeros((n_arr, 256, 256), dtype=np.float32)), batch_indices).unsqueeze(1),
                "alpha_painted_256": _gather(alpha_painted, batch_indices).permute(0, 3, 1, 2),
                "mcly_any_16": _gather(mcly_any_16, batch_indices).unsqueeze(1),
                "what_plate_flag": torch.from_numpy(what_plate[batch_indices]).to(_DEVICE),
            }

            hard_batch, trans_batch, mask_batch = _hard_region_signals_batched(batch, float(args.detail_boost))

            for j, e in enumerate(batch_entries):
                tid = int(e["tile_id"])
                tile_idx = batch_indices[j]

                minimap = arrays["minimap_rgb"][tile_idx].astype(np.float32, copy=False) / 255.0
                alpha_t = arrays["alpha_256"][tile_idx].astype(np.float32, copy=False)

                # Debug first tile of each build
                comps = _process_tile_signals(
                    hard_batch[j].numpy(),
                    trans_batch[j].numpy(),
                    mask_batch[j].numpy(),
                    minimap,
                    alpha_t,
                    component_threshold=float(args.component_threshold),
                    min_component_area=int(args.min_component_area),
                    min_component_width=int(args.min_component_width),
                    min_component_height=int(args.min_component_height),
                    max_components=int(args.max_components_per_tile),
                    bbox_padding=int(args.bbox_padding),
                    dedupe_hash_size=int(args.dedupe_hash_size),
                )

                for comp in comps:
                    row: dict[str, object] = {
                        "candidate_id": candidate_id,
                        "build": build,
                        "tile_id": tid,
                        "tile_x": int(e["tile_x"]),
                        "tile_y": int(e["tile_y"]),
                        "tile_local_bbox": comp["tile_local_bbox"],
                        "component_area": comp["component_area"],
                        "score_mean": comp["score_mean"],
                        "score_max": comp["score_max"],
                        "rgb_fingerprint": comp["rgb_fingerprint"],
                        "avg_hash": comp["avg_hash"],
                        "alpha_layer_signature": comp["alpha_layer_signature"],
                        "layer_means": comp["layer_means"],
                        "layer_coverage": comp["layer_coverage"],
                        "dominant_layers": comp["dominant_layers"],
                        "tile_coverage": [{"tile_x": int(e["tile_x"]), "tile_y": int(e["tile_y"]), "tile_id": tid}],
                        "tile_coverage_count": 1,
                        "multi_tile": False,
                        "touches_edge": comp["touches_edge"],
                    }
                    all_candidates.append(row)
                    candidate_id += 1

                tiles_processed += 1

            if tiles_processed % 500 == 0:
                pct = 100.0 * tiles_processed / total_tiles
                print(f"  {tiles_processed}/{total_tiles} tiles ({pct:.1f}%) — {candidate_id} candidates")

        # Free build arrays
        del arrays, weight_257, obj_presence, alpha_painted, mcly_any_16, what_plate, terrain_valid
        import gc
        gc.collect()

    print(f"\nTotal: {tiles_processed} tiles -> {candidate_id} candidates")

    # ── Phase 2: write metadata ──────────────────────────────────
    all_candidates.sort(key=lambda r: (
        int(r.get("tile_coverage_count", 0)),
        float(r.get("score_mean", 0.0)),
        int(r.get("component_area", 0)),
    ), reverse=True)

    _write_jsonl(out_dir / "candidates.jsonl", all_candidates)

    # ── Phase 3: dedupe ──────────────────────────────────────────
    dedupe_stats: dict[str, object] | None = None
    cluster_summaries: list[dict[str, object]] = []
    deduped_rows: list[dict[str, object]] = []
    if bool(args.dedupe):
        deduped_rows, cluster_summaries, dedupe_stats = _cluster_candidates(
            all_candidates,
            hamming_threshold=int(args.dedupe_hamming_threshold),
        )
        _write_jsonl(out_dir / "candidates_deduped.jsonl", deduped_rows)
        _write_jsonl(out_dir / "cluster_summary.jsonl", cluster_summaries)
        (out_dir / "dedupe_stats.json").write_text(json.dumps(dedupe_stats, indent=2), encoding="utf-8")

    # ── Phase 4: Zarr index ──────────────────────────────────────
    source_for_index = deduped_rows if deduped_rows else all_candidates
    tile_paste_pairs = _compute_local_bboxes(source_for_index)
    _build_zarr_index(out_dir / "tile_to_pastes.zarr", tile_paste_pairs, deduped_rows if deduped_rows else all_candidates)

    # ── Phase 5: summary ─────────────────────────────────────────
    multi_tile_count = int(sum(1 for row in all_candidates if bool(row.get("multi_tile"))))
    touches_edge_count = int(sum(1 for row in all_candidates if bool(row.get("touches_edge"))))
    summary = {
        "tiles_processed": int(tiles_processed),
        "candidates": int(len(all_candidates)),
        "multi_tile_candidates": int(multi_tile_count),
        "touches_edge_candidates": int(touches_edge_count),
        "component_threshold": float(args.component_threshold),
        "min_component_area": int(args.min_component_area),
        "max_components_per_tile": int(args.max_components_per_tile),
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
