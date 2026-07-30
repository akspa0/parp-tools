"""Height-normal mismatch detection for Zarr terrain stores.

Detects tiles where normal vectors encode significant terrain variation
but the height data is suspiciously flat — indicating poisoned supervision
that degrades height model training.

Spec 122 note: this exact check is now also ported to C# as
``WowViewer.Core.Curation.Mismatch.HeightNormalMismatchDetector`` (same thresholds, same reason
strings), run via ``WowViewer.Tool.Harvest curate`` against v50 stores and read via
``harvester.curation_store``. This module is NOT converted to a thin shim: a real-caller search
(2026-07-30) found it is still imported by ``scripts/detect_height_normal_mismatch.py``,
``scripts/reconstruct_heights_from_normals.py``, ``scripts/build_teacher_prior_dataset.py``, and
``harvester.test_061_mismatch_repair``, which operate on V16/V18/V23-era store shapes ``curate``
does not read. It remains the correct tool for that scope. The SC-003 side-by-side comparison
between this detector's output and the new C# port's output, on the same real tiles, is a tracked
follow-up (not yet run this session).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class MismatchTile:
    build: str
    tile_id: int
    tile_x: int
    tile_y: int
    map_name: str
    height_range: float
    height_std: float
    normal_relief_mean: float
    normal_cov: float
    normal_edge_frac: float
    minimap_gray_std: float
    mismatch_severity: str
    mismatch_reason: str
    object_cov: float = 0.0
    has_normals: bool = True


@dataclass
class MismatchReport:
    tiles: list[MismatchTile] = field(default_factory=list)

    def to_parquet(self, path: str | Path) -> None:
        if not self.tiles:
            table = pa.table({
                "build": pa.array([], type=pa.string()),
                "tile_id": pa.array([], type=pa.int64()),
                "tile_x": pa.array([], type=pa.int32()),
                "tile_y": pa.array([], type=pa.int32()),
                "map": pa.array([], type=pa.string()),
                "height_range": pa.array([], type=pa.float32()),
                "height_std": pa.array([], type=pa.float32()),
                "normal_relief_mean": pa.array([], type=pa.float32()),
                "normal_cov": pa.array([], type=pa.float32()),
                "normal_edge_frac": pa.array([], type=pa.float32()),
                "minimap_gray_std": pa.array([], type=pa.float32()),
                "mismatch_severity": pa.array([], type=pa.string()),
                "mismatch_reason": pa.array([], type=pa.string()),
                "object_cov": pa.array([], type=pa.float32()),
            })
        else:
            table = pa.table({
                "build": pa.array([t.build for t in self.tiles], type=pa.string()),
                "tile_id": pa.array([t.tile_id for t in self.tiles], type=pa.int64()),
                "tile_x": pa.array([t.tile_x for t in self.tiles], type=pa.int32()),
                "tile_y": pa.array([t.tile_y for t in self.tiles], type=pa.int32()),
                "map": pa.array([t.map_name for t in self.tiles], type=pa.string()),
                "height_range": pa.array([t.height_range for t in self.tiles], type=pa.float32()),
                "height_std": pa.array([t.height_std for t in self.tiles], type=pa.float32()),
                "normal_relief_mean": pa.array([t.normal_relief_mean for t in self.tiles], type=pa.float32()),
                "normal_cov": pa.array([t.normal_cov for t in self.tiles], type=pa.float32()),
                "normal_edge_frac": pa.array([t.normal_edge_frac for t in self.tiles], type=pa.float32()),
                "minimap_gray_std": pa.array([t.minimap_gray_std for t in self.tiles], type=pa.float32()),
                "mismatch_severity": pa.array([t.mismatch_severity for t in self.tiles], type=pa.string()),
                "mismatch_reason": pa.array([t.mismatch_reason for t in self.tiles], type=pa.string()),
                "object_cov": pa.array([t.object_cov for t in self.tiles], type=pa.float32()),
            })
        pq.write_table(table, str(path))


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

    def _edge(x: np.ndarray) -> np.ndarray:
        gx = np.zeros_like(x, dtype=np.float32)
        gy = np.zeros_like(x, dtype=np.float32)
        gx[:, 1:] = np.abs(x[:, 1:] - x[:, :-1])
        gy[1:, :] = np.abs(x[1:, :] - x[:-1, :])
        return np.maximum(gx, gy)

    return np.maximum(_edge(nx), np.maximum(_edge(ny), _edge(nz)))


def minimap_grayscale(minimap: np.ndarray) -> np.ndarray:
    x = minimap.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]).astype(np.float32)


def _severity(relief: float, height_range: float) -> str:
    if height_range < 0.001:
        ratio = relief
    else:
        ratio = relief / max(height_range, 1e-6)
    if ratio > 0.10:
        return "high"
    if ratio > 0.03:
        return "medium"
    return "low"


def _object_coverage(root: Any, row_meta: dict[str, Any]) -> float:
    if "object_mask" in root:
        mask = root["object_mask"][int(row_meta["tile_id"])].astype(np.float32)
        return float(np.clip(mask, 0.0, 1.0).mean())
    if "mddf_mask" in root and "modf_mask" in root:
        mddf = root["mddf_mask"][int(row_meta["tile_id"])].astype(np.float32)
        modf = root["modf_mask"][int(row_meta["tile_id"])].astype(np.float32)
        return float(np.clip(np.maximum(mddf, modf), 0.0, 1.0).mean())
    return 0.0


def compute_tile_mismatch_metrics(
    root: Any,
    tile_id: int,
    row_meta: dict[str, Any],
) -> dict[str, Any]:
    height = root["height_257"][tile_id].astype(np.float32)
    height_range = float(height.max() - height.min())
    height_std = float(height.std())

    has_normals = bool(row_meta.get("has_normals", True))
    if not has_normals or "normal_xyz" not in root:
        return {
            "tile_id": tile_id,
            "height_range": height_range,
            "height_std": height_std,
            "normal_relief_mean": 0.0,
            "normal_cov": 0.0,
            "normal_edge_frac": 0.0,
            "minimap_gray_std": 0.0,
            "has_normals": False,
            "is_mismatch": False,
            "mismatch_severity": "none",
            "mismatch_reason": "",
            "object_cov": 0.0,
        }

    normals = root["normal_xyz"][tile_id].astype(np.float32)
    normal_mask = np.ones((257, 257), dtype=np.float32)
    if "normal_mask" in root:
        normal_mask = root["normal_mask"][tile_id].astype(np.float32)

    normal_cov = float(normal_mask.mean())

    if normal_cov < 0.05:
        return {
            "tile_id": tile_id,
            "height_range": height_range,
            "height_std": height_std,
            "normal_relief_mean": 0.0,
            "normal_cov": normal_cov,
            "normal_edge_frac": 0.0,
            "minimap_gray_std": 0.0,
            "has_normals": True,
            "is_mismatch": False,
            "mismatch_severity": "none",
            "mismatch_reason": "insufficient_normal_coverage",
            "object_cov": 0.0,
        }

    relief = normal_relief(normals, normal_mask)
    relief_mean = float(relief.mean())

    n_edge = normal_edge_strength(normals, normal_mask)
    n_edge_frac = float((n_edge >= 0.01).mean())

    minimap_rgb = root["minimap_rgb"][tile_id].astype(np.uint8)
    gray = minimap_grayscale(minimap_rgb)
    gray_std = float(gray.std())

    obj_cov = _object_coverage(root, row_meta)

    result: dict[str, Any] = {
        "tile_id": tile_id,
        "height_range": height_range,
        "height_std": height_std,
        "normal_relief_mean": relief_mean,
        "normal_cov": normal_cov,
        "normal_edge_frac": n_edge_frac,
        "minimap_gray_std": gray_std,
        "has_normals": True,
        "is_mismatch": False,
        "mismatch_severity": "none",
        "mismatch_reason": "",
        "object_cov": obj_cov,
    }
    return result


def detect_mismatches(
    *,
    root: Any,
    metrics: dict[str, Any],
    normal_relief_threshold: float = 0.02,
    height_range_threshold: float = 3.0,
    normal_cov_threshold: float = 0.10,
) -> dict[str, Any]:
    has_normals = metrics.get("has_normals", True)
    normal_cov = metrics.get("normal_cov", 0.0)
    normal_relief_mean = metrics.get("normal_relief_mean", 0.0)
    height_range = metrics.get("height_range", 0.0)

    if not has_normals:
        return {**metrics, "is_mismatch": False, "mismatch_severity": "none",
                "mismatch_reason": "no_normal_data"}

    if normal_cov < normal_cov_threshold:
        return {**metrics, "is_mismatch": False, "mismatch_severity": "none",
                "mismatch_reason": "insufficient_normal_coverage"}

    if normal_relief_mean < normal_relief_threshold:
        return {**metrics, "is_mismatch": False, "mismatch_severity": "none",
                "mismatch_reason": "flat_normals"}

    if height_range >= height_range_threshold:
        return {**metrics, "is_mismatch": False, "mismatch_severity": "none",
                "mismatch_reason": "sufficient_height_range"}

    severity = _severity(normal_relief_mean, height_range)

    return {
        **metrics,
        "is_mismatch": True,
        "mismatch_severity": severity,
        "mismatch_reason": "height_flat_vs_normal_varied",
    }
