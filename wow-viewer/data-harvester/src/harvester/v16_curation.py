"""Reusable V16 dataset curation metrics and manifest filtering helpers.

This sits between the raw Zarr truth stores and the trainers so tile-quality
rules can be computed once and reused across model families.

Spec 122 note: for the v50 lane, the canonical curation entrypoint is now
``WowViewer.Tool.Harvest curate`` (``wow-viewer/src/core/WowViewer.Core.Curation``), read via
``harvester.curation_store``. This module is NOT converted to a thin shim over it: a real-caller
search (2026-07-30) found it is still imported by V16/V18/V23-era scripts
(``harvester.v16_1_dataset``, ``harvester.v16_2_dataset``, ``harvester.v23.dataset``,
``scripts/build_v16_curation_manifest.py``, ``scripts/detect_height_normal_mismatch.py``, and
others) that operate on store shapes ``curate`` was never built to read. It remains the correct
tool for that scope. Do not add new v50-scoped curation logic here -- add it to
``WowViewer.Core.Curation`` instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DIFFICULTY_BUCKETS = ("easy", "medium", "hard", "pathological")


def minimap_grayscale(minimap: np.ndarray) -> np.ndarray:
    x = minimap.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]).astype(np.float32)


def edge_strength(x: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    gx[:, 1:] = np.abs(x[:, 1:] - x[:, :-1])
    gy[1:, :] = np.abs(x[1:, :] - x[:-1, :])
    return np.maximum(gx, gy)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
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


def f1(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.logical_and(aa, bb).sum())
    denom = int(aa.sum()) + int(bb.sum())
    if denom <= 0:
        return 1.0
    return float((2.0 * inter) / denom)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    if union <= 0:
        return 1.0
    return float(inter / union)


def alpha_painted(alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim != 3 or alpha.shape[2] <= 0:
        return np.zeros(alpha.shape[:2], dtype=np.float32)
    if alpha.shape[2] > 1:
        painted = alpha[:, :, 1:]
        if float(painted.max()) > 0.0:
            return painted.max(axis=2).astype(np.float32, copy=False)
    return alpha.max(axis=2).astype(np.float32, copy=False)


def crop_257_to_256(x: np.ndarray) -> np.ndarray:
    return x[:256, :256]


def downsample_256_to_16(x: np.ndarray) -> np.ndarray:
    arr = x[:256, :256]
    reshaped = arr.reshape(16, 16, 16, 16)
    return reshaped.mean(axis=(1, 3)).astype(np.float32, copy=False)


def normal_relief(normals: np.ndarray, normal_mask: np.ndarray) -> np.ndarray:
    nx = normals[:, :, 0].astype(np.float32, copy=False)
    ny = normals[:, :, 1].astype(np.float32, copy=False)
    relief = np.sqrt(np.maximum(0.0, (nx * nx) + (ny * ny))).astype(np.float32, copy=False)
    return relief * normal_mask.astype(np.float32, copy=False)


def normal_edge_strength(normals: np.ndarray, normal_mask: np.ndarray) -> np.ndarray:
    mask = normal_mask.astype(np.float32, copy=False)
    nx = normals[:, :, 0].astype(np.float32, copy=False) * mask
    ny = normals[:, :, 1].astype(np.float32, copy=False) * mask
    nz = normals[:, :, 2].astype(np.float32, copy=False) * mask
    return np.maximum(edge_strength(nx), np.maximum(edge_strength(ny), edge_strength(nz)))


def height_gradient_strength(height_257: np.ndarray) -> np.ndarray:
    return edge_strength(height_257.astype(np.float32, copy=False))


def mcly_painted_coverage(mcly_mask: np.ndarray) -> float:
    if mcly_mask.ndim != 3 or mcly_mask.shape[2] <= 0:
        return 0.0
    painted = mcly_mask.max(axis=2).astype(np.float32, copy=False)
    return float((painted > 0.05).mean())


def is_blank_what_plate(
    *,
    height_257: np.ndarray,
    alpha_cov: float,
    mcly_cov: float,
    liquid_cov: float,
    object_cov: float,
    height_abs_max_eps: float = 1e-6,
    height_std_eps: float = 1e-6,
) -> bool:
    height = height_257.astype(np.float32, copy=False)
    height_abs_max = float(np.abs(height).max())
    height_std = float(height.std())
    return bool(
        height_abs_max <= float(height_abs_max_eps)
        and height_std <= float(height_std_eps)
        and float(alpha_cov) <= 1e-4
        and float(mcly_cov) <= 1e-4
        and float(liquid_cov) <= 1e-4
        and float(object_cov) <= 1e-4
    )


def load_curation_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if path.is_dir():
        kept_path = path / "kept_tiles.parquet"
        all_path = path / "tiles.parquet"
        if kept_path.exists():
            path = kept_path
        elif all_path.exists():
            path = all_path
        else:
            raise FileNotFoundError(f"No kept_tiles.parquet or tiles.parquet under {path}")

    if path.suffix.lower() == ".parquet":
        table = pq.read_table(str(path))
        rows = table.to_pylist()
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        else:
            rows = payload.get("rows", [])
    else:
        raise RuntimeError(f"Unsupported curation manifest format: {path}")

    return rows


def load_curation_index(manifest_path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = load_curation_rows(manifest_path)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        build = str(row.get("build", ""))
        tile_id = int(row.get("tile_id", -1))
        if build and tile_id >= 0:
            out[(build, tile_id)] = row
    return out


def load_curation_keys(manifest_path: str | Path) -> set[tuple[str, int]]:
    rows = load_curation_rows(manifest_path)
    keys: set[tuple[str, int]] = set()
    for row in rows:
        if not bool(row.get("keep", True)):
            continue
        build = str(row.get("build", ""))
        tile_id = int(row.get("tile_id", -1))
        if build and tile_id >= 0:
            keys.add((build, tile_id))
    return keys


def write_rows_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))
