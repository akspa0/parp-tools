from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
from PIL import Image as _PILImage
from PIL import ImageDraw as _PILImageDraw
import pyarrow.parquet as pq
import zarr
import zarr.storage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_OUTPUT_ROOT = _DATASET_ROOT / "validation"
_PANEL_SIZE = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V16 alpha-vs-minimap alignment.")
    parser.add_argument("--build", type=str, help="Single build key")
    parser.add_argument("--builds", nargs="+", help="Multiple build keys")
    parser.add_argument("--max-tiles-per-build", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--alpha-edge-threshold", type=float, default=0.03)
    parser.add_argument("--minimap-edge-threshold", type=float, default=0.08)
    parser.add_argument("--min-alpha-coverage", type=float, default=0.01)
    parser.add_argument("--dilate-radius", type=int, default=2)
    parser.add_argument("--worst-k", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_ROOT / "alpha_minimap_alignment",
        help="Directory for reports and worst-case audit image",
    )
    return parser.parse_args()


def _open_root(zarr_path: Path):
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    return store, root


def _alpha_painted(alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim != 3 or alpha.shape[2] <= 0:
        return np.zeros(alpha.shape[:2], dtype=np.float32)
    if alpha.shape[2] > 1:
        painted = alpha[:, :, 1:]
        if float(painted.max()) > 0.0:
            return painted.max(axis=2).astype(np.float32, copy=False)
    return alpha.max(axis=2).astype(np.float32, copy=False)


def _grayscale_minimap(minimap: np.ndarray) -> np.ndarray:
    x = minimap.astype(np.float32) / 255.0
    return (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]).astype(np.float32)


def _edge_strength(x: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    gx[:, 1:] = np.abs(x[:, 1:] - x[:, :-1])
    gy[1:, :] = np.abs(x[1:, :] - x[:-1, :])
    return np.maximum(gx, gy)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(radius))):
        acc = out.copy()
        acc[:-1, :] |= out[1:, :]
        acc[1:, :] |= out[:-1, :]
        acc[:, :-1] |= out[:, 1:]
        acc[:, 1:] |= out[:, :-1]
        acc[:-1, :-1] |= out[1:, 1:]
        acc[1:, 1:] |= out[:-1, :-1]
        acc[:-1, 1:] |= out[1:, :-1]
        acc[1:, :-1] |= out[:-1, 1:]
        out = acc
    return out


def _f1(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    denom = int(a.sum()) + int(b.sum())
    if denom <= 0:
        return 1.0
    return float((2.0 * inter) / denom)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 1.0
    return float(inter / union)


def _to_rgb_panel(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    x = arr.astype(np.float32)
    rng = hi - lo
    if abs(rng) < 1e-8:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = np.clip((x - lo) / rng, 0.0, 1.0)
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


def _overlay_edges(alpha_edge: np.ndarray, minimap_edge: np.ndarray) -> np.ndarray:
    h, w = alpha_edge.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[minimap_edge, 1] = 255
    rgb[alpha_edge, 0] = 255
    both = np.logical_and(alpha_edge, minimap_edge)
    rgb[both] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def _sample_tile_ids(table, max_tiles: int, seed: int) -> list[int]:
    ids = [i for i in range(table.num_rows) if bool(table.column("has_alpha_256")[i].as_py())]
    if not ids:
        return []
    if max_tiles <= 0 or len(ids) <= max_tiles:
        return ids
    rng = np.random.RandomState(seed)
    chosen = rng.choice(np.asarray(ids, dtype=np.int64), size=max_tiles, replace=False)
    return [int(v) for v in chosen.tolist()]


def _row_get(table, idx: int, key: str, default: Any = None) -> Any:
    if key not in table.column_names:
        return default
    return table.column(key)[idx].as_py()


def main() -> None:
    args = _parse_args()
    builds = args.builds or ([args.build] if args.build else [])
    if not builds:
        raise SystemExit("Provide --build or --builds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []

    for build_idx, build in enumerate(builds):
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: missing {zarr_path}")
            continue
        store, root = _open_root(zarr_path)
        try:
            table = pq.read_table(str(zarr_path / "index.parquet"))
            tile_ids = _sample_tile_ids(table, max_tiles=int(args.max_tiles_per_build), seed=int(args.sample_seed) + build_idx)
            for row_idx in tile_ids:
                tile_id = int(_row_get(table, row_idx, "tile_id", row_idx))
                minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
                alpha = root["alpha_256"][tile_id].astype(np.float32)
                painted = _alpha_painted(alpha)
                alpha_mask = painted >= float(args.alpha_threshold)
                alpha_cov = float(alpha_mask.mean())
                if alpha_cov < float(args.min_alpha_coverage):
                    continue

                alpha_edge = _edge_strength(painted) >= float(args.alpha_edge_threshold)
                gray = _grayscale_minimap(minimap)
                minimap_edge = _edge_strength(gray) >= float(args.minimap_edge_threshold)
                alpha_edge_d = _dilate(alpha_edge, int(args.dilate_radius))
                minimap_edge_d = _dilate(minimap_edge, int(args.dilate_radius))

                audit_rows.append(
                    {
                        "build": build,
                        "map": str(_row_get(table, row_idx, "map", "")),
                        "tile_id": tile_id,
                        "tile_x": int(_row_get(table, row_idx, "tile_x", -1) or -1),
                        "tile_y": int(_row_get(table, row_idx, "tile_y", -1) or -1),
                        "height_std": float(_row_get(table, row_idx, "height_std", 0.0) or 0.0),
                        "alpha_cov": alpha_cov,
                        "alpha_edge_frac": float(alpha_edge.mean()),
                        "minimap_edge_frac": float(minimap_edge.mean()),
                        "edge_f1": _f1(alpha_edge_d, minimap_edge_d),
                        "edge_iou": _iou(alpha_edge_d, minimap_edge_d),
                        "minimap_std": float(minimap.astype(np.float32).std()),
                    }
                )
        finally:
            store.close()

    if not audit_rows:
        raise SystemExit("No alpha-bearing tiles passed the audit filter.")

    audit_rows.sort(key=lambda row: (row["edge_f1"], row["edge_iou"], -row["alpha_cov"]))
    summary = {
        "builds": builds,
        "tile_count": len(audit_rows),
        "alpha_threshold": float(args.alpha_threshold),
        "alpha_edge_threshold": float(args.alpha_edge_threshold),
        "minimap_edge_threshold": float(args.minimap_edge_threshold),
        "min_alpha_coverage": float(args.min_alpha_coverage),
        "dilate_radius": int(args.dilate_radius),
        "edge_f1_mean": float(np.mean([row["edge_f1"] for row in audit_rows])),
        "edge_f1_median": float(np.median([row["edge_f1"] for row in audit_rows])),
        "edge_f1_p10": float(np.percentile([row["edge_f1"] for row in audit_rows], 10)),
        "edge_iou_mean": float(np.mean([row["edge_iou"] for row in audit_rows])),
        "worst_examples": audit_rows[: int(args.worst_k)],
    }

    summary_path = args.output_dir / "alpha_minimap_alignment.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    worst_rows = audit_rows[: int(args.worst_k)]
    panels: list[np.ndarray] = []
    for row in worst_rows:
        zarr_path = _DATASET_ROOT / f"{row['build']}.zarr"
        store, root = _open_root(zarr_path)
        try:
            tile_id = int(row["tile_id"])
            minimap = root["minimap_rgb"][tile_id].astype(np.uint8)
            alpha = root["alpha_256"][tile_id].astype(np.float32)
            painted = _alpha_painted(alpha)
            alpha_mask = painted >= float(args.alpha_threshold)
            alpha_edge = _edge_strength(painted) >= float(args.alpha_edge_threshold)
            gray = _grayscale_minimap(minimap)
            minimap_edge = _edge_strength(gray) >= float(args.minimap_edge_threshold)
            overlay = _overlay_edges(_dilate(alpha_edge, int(args.dilate_radius)), _dilate(minimap_edge, int(args.dilate_radius)))

            strip = np.concatenate(
                [
                    _draw_label(_resize(minimap), "minimap"),
                    _draw_label(_resize(_to_rgb_panel(painted, 0.0, 1.0)), "alpha painted"),
                    _draw_label(_resize(_to_rgb_panel(alpha_mask.astype(np.float32), 0.0, 1.0)), "alpha mask"),
                    _draw_label(_resize(_to_rgb_panel(minimap_edge.astype(np.float32), 0.0, 1.0)), "minimap edges"),
                    _draw_label(_resize(overlay), f"edge overlap f1={row['edge_f1']:.3f}"),
                ],
                axis=1,
            )
            title = (
                f"{row['build']} {row['map']} xy=({row['tile_x']},{row['tile_y']}) "
                f"tile_id={row['tile_id']} alpha_cov={row['alpha_cov']:.3f} "
                f"f1={row['edge_f1']:.3f} iou={row['edge_iou']:.3f}"
            )
            img = _PILImage.fromarray(strip, "RGB")
            drw = _PILImageDraw.Draw(img)
            drw.rectangle([(0, 0), (img.width, 18)], fill=(24, 24, 24))
            drw.text((4, 3), title, fill=(245, 245, 245))
            panels.append(np.asarray(img))
        finally:
            store.close()

    if panels:
        cols = 1
        rows = len(panels)
        strip_h, strip_w = panels[0].shape[:2]
        canvas = _PILImage.new("RGB", (cols * strip_w, rows * strip_h), (8, 8, 8))
        for idx, panel in enumerate(panels):
            canvas.paste(_PILImage.fromarray(panel, "RGB"), (0, idx * strip_h))
        canvas.save(args.output_dir / "alpha_minimap_alignment.worst_cases.png")

    print(f"Wrote {summary_path}")
    if panels:
        print(f"Wrote {args.output_dir / 'alpha_minimap_alignment.worst_cases.png'}")
    print(
        f"tiles={summary['tile_count']} "
        f"edge_f1_mean={summary['edge_f1_mean']:.4f} "
        f"edge_f1_median={summary['edge_f1_median']:.4f} "
        f"edge_f1_p10={summary['edge_f1_p10']:.4f}"
    )


if __name__ == "__main__":
    main()
