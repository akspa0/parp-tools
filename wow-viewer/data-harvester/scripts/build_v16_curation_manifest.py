from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
from PIL import Image as _PILImage
from PIL import ImageDraw as _PILImageDraw
import pyarrow.parquet as pq
import zarr
import zarr.storage

from harvester.v16_curation import (
    alpha_painted,
    dilate,
    edge_strength,
    f1,
    iou,
    load_curation_keys,
    minimap_grayscale,
    normal_edge_strength,
    normal_relief,
    write_rows_parquet,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_OUTPUT_ROOT = _DATASET_ROOT / "curation"
_PANEL_SIZE = 256


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a reusable V16 dataset curation manifest.")
    p.add_argument("--dataset-dir", type=Path, default=_DATASET_ROOT)
    p.add_argument("--build", type=str)
    p.add_argument("--builds", nargs="+")
    p.add_argument("--profile", choices=["basic_v1", "normal_terrain_v1"], default="normal_terrain_v1")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--max-tiles-per-build", type=int, default=0)
    p.add_argument("--min-minimap-gray-std", type=float, default=4.0)
    p.add_argument("--min-height-std", type=float, default=3.0)
    p.add_argument("--min-normal-coverage", type=float, default=0.25)
    p.add_argument("--normal-edge-threshold", type=float, default=0.08)
    p.add_argument("--minimap-edge-threshold", type=float, default=0.08)
    p.add_argument("--min-normal-edge-f1", type=float, default=0.10)
    p.add_argument("--min-edge-frac", type=float, default=0.01)
    p.add_argument("--dilate-radius", type=int, default=2)
    p.add_argument("--worst-k", type=int, default=12)
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Worker count for tile auditing. Use -1 to auto-resolve a CPU-friendly default.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Tile rows per worker task.",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _open_root(zarr_path: Path):
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    return store, root


def _sample_row_ids(table, max_tiles: int, seed: int) -> list[int]:
    ids = list(range(table.num_rows))
    if max_tiles <= 0 or len(ids) <= max_tiles:
        return ids
    rng = np.random.RandomState(seed)
    chosen = rng.choice(np.asarray(ids, dtype=np.int64), size=max_tiles, replace=False)
    return [int(v) for v in chosen.tolist()]


def _row_get(table, idx: int, key: str, default: Any = None) -> Any:
    if key not in table.column_names:
        return default
    return table.column(key)[idx].as_py()


def _resolve_workers(requested: int) -> int:
    if requested >= 0:
        return int(requested)
    cpu_count = os.cpu_count() or 4
    return max(2, min(16, cpu_count))


def _build_row_meta(table, row_idx: int) -> dict[str, Any]:
    return {
        "tile_id": int(_row_get(table, row_idx, "tile_id", row_idx)),
        "map": str(_row_get(table, row_idx, "map", "")),
        "tile_x": int(_row_get(table, row_idx, "tile_x", -1) or -1),
        "tile_y": int(_row_get(table, row_idx, "tile_y", -1) or -1),
        "has_normals": bool(_row_get(table, row_idx, "has_normal_xyz", False)),
        "has_alpha": bool(_row_get(table, row_idx, "has_alpha_256", False)),
        "has_liquid": bool(_row_get(table, row_idx, "has_liquid_mask", False)),
        "height_std": float(_row_get(table, row_idx, "height_std", 0.0) or 0.0),
        "n_mddf": int(_row_get(table, row_idx, "n_mddf", 0) or 0),
        "n_modf": int(_row_get(table, row_idx, "n_modf", 0) or 0),
    }


def _chunk_rows(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def _to_rgb_panel(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    x = arr.astype(np.float32)
    rng = hi - lo
    y = np.zeros_like(x, dtype=np.float32) if abs(rng) < 1e-8 else np.clip((x - lo) / rng, 0.0, 1.0)
    u8 = (y * 255.0).astype(np.uint8)
    if u8.ndim == 2:
        return np.repeat(u8[:, :, None], 3, axis=2)
    return u8


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    pil = _PILImage.fromarray(img, "RGB")
    drw = _PILImageDraw.Draw(pil)
    drw.rectangle([(0, 0), (pil.width, 18)], fill=(10, 10, 10))
    drw.text((4, 3), text, fill=(245, 245, 245))
    return np.asarray(pil)


def _resize(img: np.ndarray, size: int = _PANEL_SIZE) -> np.ndarray:
    return np.asarray(_PILImage.fromarray(img).resize((size, size), _PILImage.Resampling.NEAREST))


def _overlay_edges(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h, w = a.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[b, 1] = 255
    rgb[a, 0] = 255
    both = np.logical_and(a, b)
    rgb[both] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def _crop_257_to_256(x: np.ndarray) -> np.ndarray:
    return x[:256, :256]


def _compute_row(
    *,
    build: str,
    row_meta: dict[str, Any],
    root,
    args_dict: dict[str, Any],
) -> dict[str, Any]:
    tile_id = int(row_meta["tile_id"])
    minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
    gray = minimap_grayscale(minimap)
    minimap_edge = edge_strength(gray) >= float(args_dict["minimap_edge_threshold"])
    minimap_edge_d = dilate(minimap_edge, int(args_dict["dilate_radius"]))

    normals = root["normal_xyz"][tile_id].astype(np.float32) if bool(row_meta["has_normals"]) else np.zeros((257, 257, 3), dtype=np.float32)
    normal_mask = root["normal_mask"][tile_id].astype(np.float32) if "normal_mask" in root and bool(row_meta["has_normals"]) else np.zeros((257, 257), dtype=np.float32)
    relief = normal_relief(normals, normal_mask)
    relief = _crop_257_to_256(relief)
    normal_mask_256 = _crop_257_to_256(normal_mask)
    normal_edge = _crop_257_to_256(normal_edge_strength(normals, normal_mask)) >= float(args_dict["normal_edge_threshold"])
    normal_edge_d = dilate(normal_edge, int(args_dict["dilate_radius"]))

    alpha_cov = 0.0
    if bool(row_meta["has_alpha"]) and "alpha_256" in root:
        alpha = root["alpha_256"][tile_id].astype(np.float32)
        alpha_cov = float((alpha_painted(alpha) >= 0.05).mean())

    liquid_cov = 0.0
    if bool(row_meta["has_liquid"]) and "liquid_mask" in root:
        liquid_cov = float(root["liquid_mask"][tile_id].astype(np.float32).mean())

    object_cov = 0.0
    if "object_filtered_mask" in root:
        object_cov = float(root["object_filtered_mask"][tile_id].astype(np.float32).mean())
    elif "object_mask" in root:
        object_cov = float(root["object_mask"][tile_id].astype(np.float32).mean())

    metrics = {
        "build": build,
        "tile_id": tile_id,
        "map": row_meta["map"],
        "tile_x": int(row_meta["tile_x"]),
        "tile_y": int(row_meta["tile_y"]),
        "has_normals": bool(row_meta["has_normals"]),
        "minimap_std": float(minimap.astype(np.float32).std()),
        "minimap_gray_std": float(gray.std()),
        "height_std": float(row_meta["height_std"]),
        "normal_cov": float(normal_mask_256.mean()),
        "normal_relief_mean": float(relief.mean()),
        "normal_edge_frac": float(normal_edge.mean()),
        "minimap_edge_frac": float(minimap_edge.mean()),
        "normal_edge_f1": float(f1(normal_edge_d, minimap_edge_d)),
        "normal_edge_iou": float(iou(normal_edge_d, minimap_edge_d)),
        "alpha_cov": alpha_cov,
        "liquid_cov": liquid_cov,
        "object_cov": object_cov,
        "n_mddf": int(row_meta["n_mddf"]),
        "n_modf": int(row_meta["n_modf"]),
    }
    return metrics


def _evaluate_profile(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, float, str | None]:
    if row["minimap_gray_std"] < float(args.min_minimap_gray_std):
        if row["height_std"] < float(args.min_height_std) and row["alpha_cov"] < 0.01 and row["liquid_cov"] < 0.01 and row["normal_cov"] < float(args.min_normal_coverage):
            return False, 0.0, "blank_low_signal_tile"

    if args.profile == "basic_v1":
        score = 0.1
        score += min(row["minimap_gray_std"] / 18.0, 2.0)
        score += min(row["height_std"] / 18.0, 2.0)
        score += min(row["normal_cov"], 1.0)
        score += min(row["normal_edge_f1"], 1.0)
        return True, float(score), None

    if not row["has_normals"] or row["normal_cov"] < float(args.min_normal_coverage):
        return False, 0.0, "insufficient_normal_coverage"

    if row["normal_relief_mean"] < 0.015 and row["minimap_gray_std"] < float(args.min_minimap_gray_std):
        return False, 0.0, "blank_minimap_blank_normals"

    if row["normal_edge_frac"] >= float(args.min_edge_frac) and row["minimap_edge_frac"] >= float(args.min_edge_frac):
        if row["normal_edge_f1"] < float(args.min_normal_edge_f1):
            return False, 0.0, "normal_minimap_edge_mismatch"

    score = 0.1
    score += min(row["minimap_gray_std"] / 18.0, 2.0)
    score += min(row["height_std"] / 18.0, 2.0)
    score += min(row["normal_cov"] * 1.5, 1.5)
    score += min(row["normal_relief_mean"] * 6.0, 1.5)
    score += min(row["normal_edge_f1"] * 2.0, 2.0)
    return True, float(score), None


def _process_chunk(
    *,
    build: str,
    zarr_path_str: str,
    row_chunk: list[dict[str, Any]],
    args_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    store, root = _open_root(Path(zarr_path_str))
    try:
        out: list[dict[str, Any]] = []
        for row_meta in row_chunk:
            row = _compute_row(build=build, row_meta=row_meta, root=root, args_dict=args_dict)
            keep, quality_score, reject_reason = _evaluate_profile(row, argparse.Namespace(**args_dict))
            row["profile"] = args_dict["profile"]
            row["keep"] = bool(keep)
            row["quality_score"] = float(quality_score)
            row["reject_reason"] = reject_reason
            out.append(row)
        return out
    finally:
        store.close()


def main() -> None:
    args = _parse_args()
    builds = args.builds or ([args.build] if args.build else [])
    if not builds:
        raise SystemExit("Provide --build or --builds")

    run_name = args.run_name or f"{args.profile}"
    output_dir = args.output_dir or (_OUTPUT_ROOT / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_workers = _resolve_workers(int(args.workers))
    args_dict = vars(args).copy()
    print(
        f"Curation: profile={args.profile} workers={resolved_workers} chunk_size={max(1, int(args.chunk_size))}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for build_idx, build in enumerate(builds):
        zarr_path = args.dataset_dir / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: missing {zarr_path}", flush=True)
            continue
        table = pq.read_table(str(zarr_path / "index.parquet"))
        sampled_row_ids = _sample_row_ids(table, int(args.max_tiles_per_build), int(args.sample_seed) + build_idx)
        row_metas = [_build_row_meta(table, row_idx) for row_idx in sampled_row_ids]
        row_chunks = _chunk_rows(row_metas, int(args.chunk_size))
        print(f"Build {build}: tiles={len(row_metas)} chunks={len(row_chunks)}", flush=True)
        if resolved_workers <= 1 or len(row_chunks) <= 1:
            build_rows: list[dict[str, Any]] = []
            for chunk_idx, row_chunk in enumerate(row_chunks, start=1):
                build_rows.extend(
                    _process_chunk(
                        build=build,
                        zarr_path_str=str(zarr_path),
                        row_chunk=row_chunk,
                        args_dict=args_dict,
                    )
                )
                print(f"  {build}: chunk {chunk_idx}/{len(row_chunks)}", flush=True)
            rows.extend(build_rows)
            continue

        completed = 0
        build_rows = []
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    build=build,
                    zarr_path_str=str(zarr_path),
                    row_chunk=row_chunk,
                    args_dict=args_dict,
                )
                for row_chunk in row_chunks
            ]
            for fut in as_completed(futures):
                build_rows.extend(fut.result())
                completed += 1
                print(f"  {build}: chunk {completed}/{len(row_chunks)}", flush=True)
        rows.extend(build_rows)

    if not rows:
        raise SystemExit("No rows were audited.")

    kept = [row for row in rows if row["keep"]]
    rejected = [row for row in rows if not row["keep"]]
    rows.sort(key=lambda row: (bool(row["keep"]), row["normal_edge_f1"], row["minimap_gray_std"]))
    write_rows_parquet(output_dir / "tiles.parquet", rows)
    write_rows_parquet(output_dir / "kept_tiles.parquet", kept)
    (output_dir / "tiles.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = {
        "profile": args.profile,
        "builds": builds,
        "dataset_dir": str(args.dataset_dir),
        "sample_seed": int(args.sample_seed),
        "max_tiles_per_build": int(args.max_tiles_per_build),
        "tile_count": len(rows),
        "kept_tiles": len(kept),
        "rejected_tiles": len(rejected),
        "keep_ratio": float(len(kept) / max(1, len(rows))),
        "normal_edge_f1_mean_kept": float(np.mean([row["normal_edge_f1"] for row in kept])) if kept else 0.0,
        "normal_edge_f1_p10_kept": float(np.percentile([row["normal_edge_f1"] for row in kept], 10)) if kept else 0.0,
        "minimap_gray_std_mean_kept": float(np.mean([row["minimap_gray_std"] for row in kept])) if kept else 0.0,
        "reject_reason_counts": {},
        "worst_rejected_examples": rejected[: int(args.worst_k)],
    }
    reason_counts: dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("reject_reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary["reject_reason_counts"] = dict(sorted(reason_counts.items()))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    worst_rows = rejected[: int(args.worst_k)]
    if worst_rows:
        panels: list[np.ndarray] = []
        for row in worst_rows:
            zarr_path = args.dataset_dir / f"{row['build']}.zarr"
            store, root = _open_root(zarr_path)
            try:
                tile_id = int(row["tile_id"])
                minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
                gray = minimap_grayscale(minimap)
                minimap_edge = dilate(edge_strength(gray) >= float(args.minimap_edge_threshold), int(args.dilate_radius))
                normals = root["normal_xyz"][tile_id].astype(np.float32) if "normal_xyz" in root else np.zeros((257, 257, 3), dtype=np.float32)
                normal_mask_full = root["normal_mask"][tile_id].astype(np.float32) if "normal_mask" in root else np.zeros((257, 257), dtype=np.float32)
                relief = _crop_257_to_256(normal_relief(normals, normal_mask_full))
                normal_mask = _crop_257_to_256(normal_mask_full)
                normal_edge_strength_256 = _crop_257_to_256(normal_edge_strength(normals, normal_mask_full))
                normal_edge = dilate(normal_edge_strength_256 >= float(args.normal_edge_threshold), int(args.dilate_radius))
                overlay = _overlay_edges(normal_edge, minimap_edge)
                strip = np.concatenate(
                    [
                        _draw_label(_resize(minimap), "minimap"),
                        _draw_label(_resize(_to_rgb_panel(relief, 0.0, 1.0)), "normal relief"),
                        _draw_label(_resize(_to_rgb_panel(normal_mask, 0.0, 1.0)), "normal mask"),
                        _draw_label(_resize(overlay), f"edge f1={row['normal_edge_f1']:.3f}"),
                    ],
                    axis=1,
                )
                title = f"{row['build']}:{row['map']}:{row['tile_id']} reject={row['reject_reason']}"
                panels.append(_draw_label(strip, title))
            finally:
                store.close()
        canvas = np.concatenate(panels, axis=0)
        _PILImage.fromarray(canvas).save(output_dir / "worst_cases.png")

    print(f"Wrote {output_dir / 'summary.json'}", flush=True)
    print(f"Wrote {output_dir / 'kept_tiles.parquet'}", flush=True)
    print(
        f"profile={args.profile} kept={len(kept)}/{len(rows)} "
        f"keep_ratio={summary['keep_ratio']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
