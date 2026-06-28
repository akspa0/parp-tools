"""Mine scar candidates from V18 Zarr stores using composite signal detection.

Identifies sub-tile scar regions (authored alpha/MCLY transitions with
terrain-structure support) via connected-component analysis of a composite
signal blending alpha gradient + MCLY transition + height gradient.

Output: tile_to_scars.zarr (candidate index) + candidates.jsonl (metadata).

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/mine_v21_scars.py --builds 3_3_5_12340 4_0_0_11927
    uv run python scripts/mine_v21_scars.py --builds 0_5_3_3368 --max-tiles 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import zarr
from scipy.ndimage import find_objects, label, sum as nd_sum

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BATCH_SIZE = 128


def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt((dx * dx) + (dy * dy) + 1e-8)


def _compute_scar_signals_batched(
    batch: dict[str, torch.Tensor], detail_boost: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched GPU scar signal computation. Returns (B, 256, 256) CPU tensors."""
    target_n = F.normalize(batch["normals"].float(), dim=1, eps=1e-6)
    height_raw = batch["height_raw"].float()
    normal_mask = batch["normal_mask"].float()
    terrain_valid = batch["terrain_valid"].float()
    liquid_mask = batch["liquid_mask"].float()
    object_presence = batch["object_presence"].float()
    what_plate_flag = batch["what_plate_flag"].float()
    alpha_painted = batch["alpha_painted"].float()
    mcly_any = batch["mcly_any"].float()

    alpha_resized = F.interpolate(alpha_painted, size=(256, 256), mode="bilinear", align_corners=False)
    mcly_resized = F.interpolate(mcly_any, size=(256, 256), mode="nearest")

    base_mask = (
        normal_mask[:, :, :256, :256]
        * terrain_valid[:, :, :256, :256]
        * (1.0 - 0.85 * liquid_mask[:, :, :256, :256])
        * (1.0 - 0.75 * object_presence[:, :, :256, :256])
        * (1.0 - what_plate_flag.reshape(-1, 1, 1, 1).float())
    )

    height_grad = _gradient_magnitude(height_raw)[:, :, :256, :256]
    normal_grad = _gradient_magnitude(target_n).mean(dim=1, keepdim=True)[:, :, :256, :256]
    alpha_grad = _gradient_magnitude(alpha_resized)
    mcly_grad = _gradient_magnitude(mcly_resized)

    def _batch_norm(x, mask):
        b = mask.shape[0]
        s = (x * mask).reshape(b, -1).sum(dim=1)
        m = mask.reshape(b, -1).sum(dim=1).clamp_min(1e-6)
        return (x / (s / m).reshape(-1, 1, 1, 1)).clamp(0.0, 4.0)

    hgn = _batch_norm(height_grad, base_mask)
    ngn = _batch_norm(normal_grad, base_mask)
    agn = _batch_norm(alpha_grad, base_mask)
    mgn = _batch_norm(mcly_grad, base_mask)

    transition = torch.maximum(agn, mgn)
    scar_score = ((0.35 * hgn) + (0.15 * ngn) + (0.50 * transition)) * base_mask
    scar_mask = base_mask * (1.0 + detail_boost * scar_score)

    return (
        scar_score[:, 0].cpu(),
        transition[:, 0].cpu(),
        scar_mask[:, 0].cpu(),
    )


def _connected_components(binary: np.ndarray) -> list[dict]:
    labeled, n = label(binary)
    if n == 0:
        return []
    objs = find_objects(labeled)
    areas = nd_sum(binary, labeled, range(1, n + 1))
    components = []
    for i in range(n):
        sy, sx = objs[i]
        components.append({
            "bbox": (sy.start, sx.start, sy.stop - 1, sx.stop - 1),
            "area": int(areas[i]),
        })
    return components


def _crop_fingerprint(rgb_u8: np.ndarray, size: int = 16) -> str:
    h, w = rgb_u8.shape[:2]
    max_side = max(h, w)
    if max_side == 0:
        return "0" * ((size * size + 3) // 4)
    pad_h = max_side - h
    pad_w = max_side - w
    padded = np.pad(rgb_u8.astype(np.float32), ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=128.0)
    gray = 0.299 * padded[:, :, 0] + 0.587 * padded[:, :, 1] + 0.114 * padded[:, :, 2]
    src_h, src_w = gray.shape
    ys = np.linspace(0, src_h - 1, size)
    xs = np.linspace(0, src_w - 1, size + 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    fy = (ys - y0)[:, np.newaxis]
    fx = (xs - x0)[np.newaxis, :]
    top = gray[y0][:, x0] * (1 - fx) + gray[y0][:, x1] * fx
    bot = gray[y1][:, x0] * (1 - fx) + gray[y1][:, x1] * fx
    resized = top * (1 - fy) + bot * fy
    diff = resized[:, 1:] > resized[:, :-1]
    return bytes(np.packbits(diff.astype(np.uint8))).hex()


def _alpha_layer_signature(alpha_crop: np.ndarray) -> dict:
    layers = np.clip(alpha_crop, 0.0, 1.0).astype(np.float32)
    if layers.ndim != 3 or layers.shape[2] != 4:
        layers = np.zeros((max(1, layers.shape[0]), max(1, layers.shape[1]), 4), dtype=np.float32)
    means = np.mean(layers, axis=(0, 1))
    coverage = np.mean(layers >= 0.05, axis=(0, 1))
    dominant = np.argsort(-means).tolist()
    dominant_layers = [int(i) for i in dominant if float(means[i]) >= 0.01][:3]
    quant = [int(round(float(v) * 1000.0)) for v in np.concatenate([means, coverage])]
    sig_hash = hashlib.sha256(",".join(str(v) for v in quant).encode("utf-8")).hexdigest()[:20]
    return {
        "layer_means": [float(v) for v in means.tolist()],
        "layer_coverage": [float(v) for v in coverage.tolist()],
        "dominant_layers": dominant_layers,
        "alpha_layer_signature": f"als_{sig_hash}",
    }


def _process_tile_scars(
    scar_score_256: np.ndarray,
    scar_mask_256: np.ndarray,
    minimap: np.ndarray,
    alpha: np.ndarray,
    component_threshold: float,
    min_component_area: int,
    min_component_width: int,
    min_component_height: int,
    max_components: int,
    bbox_padding: int,
) -> list[dict]:
    s = scar_score_256
    m = scar_mask_256
    sm = s.max()
    mm = m.max()
    if sm < 1e-6 or mm < 1e-6:
        return []
    sn = s / sm
    mn = m / mm
    binary = (sn * np.clip(mn, 0.0, 1.0)) >= component_threshold
    if not binary.any():
        return []

    comps = _connected_components(binary)
    comps = [c for c in comps if c["area"] >= min_component_area]
    comps = [c for c in comps
             if (c["bbox"][2] - c["bbox"][0] + 1) >= min_component_width
             and (c["bbox"][3] - c["bbox"][1] + 1) >= min_component_height]
    comps.sort(key=lambda c: c["area"], reverse=True)
    comps = comps[:max(1, max_components)]

    results = []
    for comp in comps:
        min_y, min_x, max_y, max_x = comp["bbox"]
        pad = bbox_padding
        x0 = max(0, min_x - pad)
        y0 = max(0, min_y - pad)
        x1 = min(255, max_x + pad)
        y1 = min(255, max_y + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        x0_a = (x0 // 16) * 16
        y0_a = (y0 // 16) * 16
        x1_a = min(255, ((x1 + 15) // 16) * 16 - 1)
        y1_a = min(255, ((y1 + 15) // 16) * 16 - 1)
        touches_edge = (x0_a == 0 or y0_a == 0 or x1_a == 255 or y1_a == 255)

        crop_rgb = (np.clip(minimap[y0_a:y1_a + 1, x0_a:x1_a + 1, :], 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb_fp = _crop_fingerprint(crop_rgb, size=16)
        alpha_crop = alpha[y0_a:y1_a + 1, x0_a:x1_a + 1, :]
        alpha_sig = _alpha_layer_signature(alpha_crop)

        results.append({
            "tile_local_bbox": [int(x0_a), int(y0_a), int(x1_a), int(y1_a)],
            "component_area": comp["area"],
            "score_mean": float(np.mean(s[y0_a:y1_a + 1, x0_a:x1_a + 1])),
            "score_max": float(np.max(s[y0_a:y1_a + 1, x0_a:x1_a + 1])),
            "rgb_fingerprint": rgb_fp,
            "touches_edge": touches_edge,
            **alpha_sig,
        })
    return results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_zarr_index(zarr_path: Path, tile_scar_pairs: list[tuple[int, int, tuple[int, int, int, int]]]) -> None:
    if not tile_scar_pairs:
        return
    max_tile_id = max(p[0] for p in tile_scar_pairs) + 1
    pairs_by_tile: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for tid, cidx, bbox in tile_scar_pairs:
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
    print(f"Zarr index: {max_tile_id} tiles, {total} scar pairs")


def _compute_what_plate(height_257: np.ndarray, normal_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    terrain_frac = (normal_mask > 0.1).reshape(normal_mask.shape[0], -1).mean(axis=1)
    height_range = height_257.reshape(height_257.shape[0], -1).max(axis=1) - height_257.reshape(height_257.shape[0], -1).min(axis=1)
    flat_terrain = terrain_frac < 0.30
    blank_height = height_range < 0.5
    obj_frac = (object_mask > 0.5).reshape(object_mask.shape[0], -1).mean(axis=1)
    heavy_objects = obj_frac > 0.7
    return (flat_terrain | blank_height | heavy_objects).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine V21 scar candidates from V18 Zarr stores")
    parser.add_argument("--dataset-dir", type=str, default="../output/datasets/v18")
    parser.add_argument("--builds", nargs="*", default=None)
    parser.add_argument("--curation-manifest", type=str, default=None)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--detail-boost", type=float, default=1.5)
    parser.add_argument("--component-threshold", type=float, default=0.20)
    parser.add_argument("--min-component-area", type=int, default=256)
    parser.add_argument("--max-components-per-tile", type=int, default=12)
    parser.add_argument("--min-component-width", type=int, default=16)
    parser.add_argument("--min-component-height", type=int, default=16)
    parser.add_argument("--bbox-padding", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="../output/v21/scars/full_corpus")
    args = parser.parse_args()

    global _BATCH_SIZE
    _BATCH_SIZE = max(1, int(args.batch_size))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_base = Path(args.dataset_dir)

    build_dirs = args.builds or sorted(d.stem.replace(".zarr", "") for d in zarr_base.glob("*.zarr"))

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

    if args.curation_manifest:
        mp = Path(args.curation_manifest)
        if mp.is_dir():
            mp = mp / "kept_tiles.parquet"
        curated = pq.read_table(str(mp))
        curated_ids = set(int(v) for v in curated.column("tile_id").to_pylist())
        n_before = len(all_entries)
        all_entries = [e for e in all_entries if int(e["tile_id"]) in curated_ids]
        print(f"Curation manifest: {n_before} -> {len(all_entries)} tiles")

    rng = np.random.RandomState(42)
    rng.shuffle(all_entries)
    if int(args.max_tiles) > 0 and len(all_entries) > int(args.max_tiles):
        all_entries = all_entries[:int(args.max_tiles)]

    stores: dict[str, zarr.Group] = {}
    for build in build_dirs:
        zp = zarr_base / f"{build}.zarr"
        if not zp.exists():
            continue
        store = zarr.storage.LocalStore(str(zp), read_only=True)
        stores[build] = zarr.open_group(store=store, mode="r")

    bulk_keys = [
        "minimap_rgb", "alpha_256", "normal_xyz", "normal_mask",
        "height_257", "mddf_mask", "modf_mask", "liquid_mask",
        "mcly_layer_mask", "object_precise_mask", "object_filtered_mask",
        "object_mask",
    ]

    def _read_build_arrays(build: str) -> dict[str, np.ndarray]:
        root = stores[build]
        return {k: root[k][:] for k in bulk_keys if k in root}

    def _fill_checkerboard_mask(mask: np.ndarray) -> np.ndarray:
        filled = mask.copy()
        filled[1:, :] |= mask[:-1, :]
        filled[:-1, :] |= mask[1:, :]
        filled[:, 1:] |= mask[:, :-1]
        filled[:, :-1] |= mask[:, 1:]
        return filled.astype(np.float32)

    def _resolve_object_mask(arrays: dict) -> np.ndarray:
        for k in ("object_precise_mask", "object_filtered_mask", "object_mask"):
            if k in arrays:
                return arrays[k]
        raise KeyError("No object mask found")

    all_candidates: list[dict] = []
    candidate_id = 0
    total_tiles = len(all_entries)
    tiles_processed = 0

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
        print(f"  Loaded {len(arrays)} arrays ({n_arr} tiles)")

        obj_mask = _resolve_object_mask(arrays)
        obj_presence = np.maximum(
            arrays.get("mddf_mask", np.zeros((n_arr, 257, 257), dtype=np.float32)),
            arrays.get("modf_mask", np.zeros((n_arr, 257, 257), dtype=np.float32)),
        )
        alpha_painted = np.clip(arrays["alpha_256"], 0.0, 1.0)
        mcly = arrays.get("mcly_layer_mask", np.zeros((n_arr, 16, 16, 4), dtype=np.float32))
        mcly_any_16 = (mcly.max(axis=3) > 0.05).astype(np.float32)
        normal_mask_contiguous = _fill_checkerboard_mask(arrays["normal_mask"])
        what_plate = _compute_what_plate(arrays["height_257"], normal_mask_contiguous, obj_mask)
        terrain_valid = normal_mask_contiguous * (1.0 - np.clip(obj_presence, 0.0, 1.0))
        liquid_resized = np.pad(arrays.get("liquid_mask", np.zeros((n_arr, 256, 256), dtype=np.float32)), ((0, 0), (0, 1), (0, 1)), mode="edge")
        terrain_valid *= (1.0 - np.clip(liquid_resized, 0.0, 1.0) * 0.85)
        terrain_valid[what_plate > 0.5] = 0.0

        for batch_start in range(0, n, _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, n)
            batch_entries = entries[batch_start:batch_end]
            batch_indices = [int(e["tile_id"]) for e in batch_entries]

            def _gather(arr: np.ndarray, idx: list[int]) -> torch.Tensor:
                return torch.from_numpy(arr[idx]).to(_DEVICE)

            batch = {
                "normals": _gather(arrays["normal_xyz"], batch_indices).permute(0, 3, 1, 2).float(),
                "height_raw": _gather(arrays["height_257"], batch_indices).unsqueeze(1),
                "normal_mask": _gather(normal_mask_contiguous, batch_indices).unsqueeze(1),
                "terrain_valid": _gather(terrain_valid, batch_indices).unsqueeze(1),
                "liquid_mask": _gather(arrays.get("liquid_mask", np.zeros((n_arr, 256, 256), dtype=np.float32)), batch_indices).unsqueeze(1),
                "object_presence": _gather(obj_presence, batch_indices).unsqueeze(1),
                "alpha_painted": _gather(alpha_painted, batch_indices).permute(0, 3, 1, 2),
                "mcly_any": _gather(mcly_any_16, batch_indices).unsqueeze(1),
                "what_plate_flag": torch.from_numpy(what_plate[batch_indices]).to(_DEVICE),
            }

            scar_score_batch, _, scar_mask_batch = _compute_scar_signals_batched(batch, float(args.detail_boost))

            for j, e in enumerate(batch_entries):
                tid = int(e["tile_id"])
                tile_idx = batch_indices[j]
                minimap = arrays["minimap_rgb"][tile_idx].astype(np.float32) / 255.0
                alpha_t = arrays["alpha_256"][tile_idx].astype(np.float32)

                comps = _process_tile_scars(
                    scar_score_batch[j].numpy(),
                    scar_mask_batch[j].numpy(),
                    minimap,
                    alpha_t,
                    component_threshold=float(args.component_threshold),
                    min_component_area=int(args.min_component_area),
                    min_component_width=int(args.min_component_width),
                    min_component_height=int(args.min_component_height),
                    max_components=int(args.max_components_per_tile),
                    bbox_padding=int(args.bbox_padding),
                )

                for comp in comps:
                    all_candidates.append({
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
                        "alpha_layer_signature": comp["alpha_layer_signature"],
                        "layer_means": comp["layer_means"],
                        "layer_coverage": comp["layer_coverage"],
                        "dominant_layers": comp["dominant_layers"],
                        "touches_edge": comp["touches_edge"],
                    })
                    candidate_id += 1

                tiles_processed += 1

            if tiles_processed % 500 == 0:
                pct = 100.0 * tiles_processed / total_tiles
                print(f"  {tiles_processed}/{total_tiles} ({pct:.1f}%) — {candidate_id} scars")

        del arrays
        import gc
        gc.collect()

    print(f"\nTotal: {tiles_processed} tiles -> {candidate_id} scar candidates")

    all_candidates.sort(key=lambda r: (
        float(r.get("score_mean", 0.0)),
        int(r.get("component_area", 0)),
    ), reverse=True)

    _write_jsonl(out_dir / "candidates.jsonl", all_candidates)

    tile_scar_pairs = _compute_local_bboxes(all_candidates)
    _build_zarr_index(out_dir / "tile_to_scars.zarr", tile_scar_pairs)

    summary = {
        "tiles_processed": tiles_processed,
        "scar_candidates": len(all_candidates),
        "component_threshold": float(args.component_threshold),
        "min_component_area": int(args.min_component_area),
        "max_components_per_tile": int(args.max_components_per_tile),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _compute_local_bboxes(candidates: list[dict]) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    pairs = []
    for cand_idx, row in enumerate(candidates):
        tid = int(row.get("tile_id", -1))
        bbox = row.get("tile_local_bbox", None)
        if tid >= 0 and bbox and len(bbox) == 4:
            pairs.append((tid, cand_idx, tuple(bbox)))
    return pairs


if __name__ == "__main__":
    main()
