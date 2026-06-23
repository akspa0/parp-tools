"""Trainable terrain-art primitive library helpers for spec 076 Phase 3."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

DEFAULT_ACCEPTED_LABELS = ("accepted_candidate", "fractal_member")
REJECTED_TRAINING_LABELS = {"composite_chonker", "one_off_detail", "too_small_unique", "rejected_unknown"}


@dataclass(frozen=True, slots=True)
class TerrainArtSample:
    sample_id: str
    region_id: str
    build: str
    map_name: str
    layer_slot: int
    layer_idx: int
    curation_label: str
    split: str
    bbox_xywh: tuple[int, int, int, int]
    crop_xywh: tuple[int, int, int, int]
    crop_truncated: bool
    area: int
    tile_coverage_count: int
    tile_coverage: list[dict[str, int]]
    alpha_mean: float
    alpha_max: float
    height_mean: float | None
    height_std: float | None
    height_range: float | None
    normal_mean_xyz: tuple[float, float, float] | None
    mcly_texture_ids: list[int]
    mcly_active_layers: list[int]
    linked_component_ids: list[str]
    tensor_index: int
    tensor_store: str
    alpha_tensor: str
    height_tensor: str
    normal_tensor: str
    mcly_texture_ids_tensor: str
    mcly_layer_mask_tensor: str
    provenance: dict[str, Any]


def load_region_rows(path: str | Path) -> list[dict[str, Any]]:
    region_path = Path(path)
    if region_path.is_dir():
        region_path = region_path / "fractal_regions.parquet"
    if not region_path.exists():
        raise FileNotFoundError(f"Missing fractal regions parquet: {region_path}")
    return [_normalize_region_row(row) for row in pq.read_table(region_path).to_pylist()]


def build_trainable_library(
    *,
    canvas_dir: str | Path,
    regions_path: str | Path,
    output_dir: str | Path,
    crop_size: int = 128,
    accepted_labels: tuple[str, ...] = DEFAULT_ACCEPTED_LABELS,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Build fixed-size accepted tensors plus accepted/rejected metadata."""
    canvas = _load_canvas(canvas_dir)
    rows = load_region_rows(regions_path)
    accepted_set = {str(label) for label in accepted_labels}
    accepted_rows = [row for row in rows if str(row.get("curation_label")) in accepted_set and not row.get("rejection_reason")]
    rejected_rows = [row for row in rows if row not in accepted_rows]
    accepted_rows.sort(key=lambda row: (str(row.get("region_id", "")), int(row.get("layer_idx", -1))))
    if max_samples is not None:
        accepted_rows = accepted_rows[: max(0, int(max_samples))]

    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    tensors, samples = _materialize_samples(canvas, accepted_rows, Path(output_dir), crop_size=crop_size)
    split_rows = [{"sample_id": sample.sample_id, "region_id": sample.region_id, "split": sample.split} for sample in samples]
    rejected = [_rejected_row(row, accepted_set) for row in rejected_rows]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_table(out / "samples.parquet", [_json_ready(asdict(sample)) for sample in samples])
    _write_table(out / "rejected.parquet", rejected)
    _write_table(out / "split.parquet", split_rows)
    _write_sample_tensors(out / "samples.zarr", tensors)

    label_counts = Counter(str(row.get("curation_label", "")) for row in rows)
    split_counts = Counter(sample.split for sample in samples)
    missing_signal_counts = _missing_signal_counts(tensors)
    summary = {
        "sample_count": int(len(samples)),
        "accepted_count": int(len(samples)),
        "rejected_count": int(len(rejected)),
        "input_region_count": int(len(rows)),
        "accepted_labels": sorted(accepted_set),
        "rejected_training_labels": sorted(REJECTED_TRAINING_LABELS),
        "curation_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "crop_size": int(crop_size),
        "missing_signal_counts": missing_signal_counts,
        "outputs": {
            "samples_zarr": str(out / "samples.zarr"),
            "samples_parquet": str(out / "samples.parquet"),
            "rejected_parquet": str(out / "rejected.parquet"),
            "split_parquet": str(out / "split.parquet"),
        },
    }
    (out / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


class FractalBrushLibrary:
    """Small smoke loader for accepted spec 076 training-library samples."""

    def __init__(self, library_dir: str | Path) -> None:
        self.library_dir = Path(library_dir)
        self.samples = pq.read_table(self.library_dir / "samples.parquet").to_pylist()
        self.root = zarr.open_group(str(self.library_dir / "samples.zarr"), mode="r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[int(index)]
        tensor_index = int(row["tensor_index"])
        return {
            "metadata": row,
            "alpha": self.root["alpha"][tensor_index],
            "height": self.root["height"][tensor_index],
            "normal": self.root["normal_xyz"][tensor_index],
            "mcly_texture_ids": self.root["mcly_texture_ids"][tensor_index],
            "mcly_layer_mask": self.root["mcly_layer_mask"][tensor_index],
            "provenance": row.get("provenance", {}),
            "optional_source_blp_evidence": row.get("optional_source_blp_evidence", []),
        }


def smoke_load_library(library_dir: str | Path, *, count: int = 32) -> dict[str, Any]:
    dataset = FractalBrushLibrary(library_dir)
    requested = int(count)
    loaded = min(len(dataset), max(0, requested))
    labels: list[str] = []
    for index in range(loaded):
        sample = dataset[index]
        labels.append(str(sample["metadata"].get("curation_label", "")))
        if sample["alpha"].ndim != 3:
            raise ValueError("alpha sample must be HxWxL")
        if sample["height"].ndim != 2:
            raise ValueError("height sample must be HxW")
        if sample["normal"].ndim != 3 or sample["normal"].shape[-1] != 3:
            raise ValueError("normal sample must be HxWx3")
    rejected_loaded = sorted(set(labels) & REJECTED_TRAINING_LABELS)
    if rejected_loaded:
        raise ValueError(f"Default sample loader returned rejected labels: {rejected_loaded}")
    return {"requested": requested, "loaded": loaded, "dataset_size": len(dataset), "labels": dict(sorted(Counter(labels).items()))}


def stable_sample_id(region_id: str, bbox_xywh: tuple[int, int, int, int], layer_idx: int) -> str:
    payload = f"{region_id}|{layer_idx}|{bbox_xywh}"
    return "ta_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def split_for_sample(sample_id: str) -> str:
    bucket = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _materialize_samples(
    canvas: zarr.Group,
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    crop_size: int,
) -> tuple[dict[str, np.ndarray], list[TerrainArtSample]]:
    alpha = canvas["alpha_256"][:].astype(np.float32)
    height = canvas["height_257"][:].astype(np.float32) if "height_257" in canvas else np.zeros((alpha.shape[0] + 1, alpha.shape[1] + 1), dtype=np.float32)
    normals = canvas["normal_xyz"][:].astype(np.float32) if "normal_xyz" in canvas else np.zeros((alpha.shape[0] + 1, alpha.shape[1] + 1, 3), dtype=np.float32)
    mcly_ids = canvas["mcly_texture_ids"][:].astype(np.int32) if "mcly_texture_ids" in canvas else np.full((max(1, alpha.shape[0] // 16), max(1, alpha.shape[1] // 16), 4), -1, dtype=np.int32)
    mcly_mask = canvas["mcly_layer_mask"][:].astype(np.float32) if "mcly_layer_mask" in canvas else np.zeros(mcly_ids.shape, dtype=np.float32)
    tile_ids = canvas["tile_id_256"][:].astype(np.int32) if "tile_id_256" in canvas else np.full(alpha.shape[:2], -1, dtype=np.int32)
    layout = dict(canvas.attrs.get("layout", {}))

    n = len(rows)
    mcly_crop = max(1, (crop_size + 15) // 16)
    tensors = {
        "alpha": np.zeros((n, crop_size, crop_size, alpha.shape[2]), dtype=np.float32),
        "height": np.zeros((n, crop_size + 1, crop_size + 1), dtype=np.float32),
        "normal_xyz": np.zeros((n, crop_size + 1, crop_size + 1, 3), dtype=np.float32),
        "mcly_texture_ids": np.full((n, mcly_crop, mcly_crop, 4), -1, dtype=np.int32),
        "mcly_layer_mask": np.zeros((n, mcly_crop, mcly_crop, 4), dtype=np.float32),
        "tile_id_256": np.full((n, crop_size, crop_size), -1, dtype=np.int32),
    }
    samples: list[TerrainArtSample] = []
    for tensor_index, row in enumerate(rows):
        bbox = _bbox_tuple(row.get("bbox_xywh", (0, 0, 1, 1)))
        crop = _crop_window_for_bbox(bbox, canvas_w=alpha.shape[1], canvas_h=alpha.shape[0], crop_size=crop_size)
        x, y, w, h = crop
        _copy_2d_or_3d(alpha, tensors["alpha"][tensor_index], x, y, fill=0.0)
        _copy_2d_or_3d(height, tensors["height"][tensor_index], x, y, fill=0.0)
        _copy_2d_or_3d(normals, tensors["normal_xyz"][tensor_index], x, y, fill=0.0)
        _copy_2d_or_3d(tile_ids, tensors["tile_id_256"][tensor_index], x, y, fill=-1)
        mx, my = x // 16, y // 16
        _copy_2d_or_3d(mcly_ids, tensors["mcly_texture_ids"][tensor_index], mx, my, fill=-1)
        _copy_2d_or_3d(mcly_mask, tensors["mcly_layer_mask"][tensor_index], mx, my, fill=0.0)

        sample_id = stable_sample_id(str(row.get("region_id", "")), bbox, int(row.get("layer_idx", -1)))
        samples.append(
            TerrainArtSample(
                sample_id=sample_id,
                region_id=str(row.get("region_id", "")),
                build=str(row.get("build") or layout.get("build", "")),
                map_name=str(row.get("map_name") or layout.get("map_name", "")),
                layer_slot=int(row.get("layer_slot", -1)),
                layer_idx=int(row.get("layer_idx", -1)),
                curation_label=str(row.get("curation_label", "")),
                split=split_for_sample(sample_id),
                bbox_xywh=bbox,
                crop_xywh=crop,
                crop_truncated=bool(bbox[2] > crop_size or bbox[3] > crop_size),
                area=int(row.get("area", 0)),
                tile_coverage_count=int(row.get("tile_coverage_count", 0)),
                tile_coverage=_list_of_dicts(row.get("tile_coverage", [])),
                alpha_mean=float(row.get("alpha_mean", 0.0)),
                alpha_max=float(row.get("alpha_max", 0.0)),
                height_mean=_optional_float(row.get("height_mean")),
                height_std=_optional_float(row.get("height_std")),
                height_range=_optional_float(row.get("height_range")),
                normal_mean_xyz=_optional_xyz(row.get("normal_mean_xyz")),
                mcly_texture_ids=[int(value) for value in row.get("mcly_texture_ids", [])],
                mcly_active_layers=[int(value) for value in row.get("mcly_active_layers", [])],
                linked_component_ids=[str(value) for value in row.get("linked_component_ids", [])],
                tensor_index=int(tensor_index),
                tensor_store=str(output_dir / "samples.zarr"),
                alpha_tensor="alpha",
                height_tensor="height",
                normal_tensor="normal_xyz",
                mcly_texture_ids_tensor="mcly_texture_ids",
                mcly_layer_mask_tensor="mcly_layer_mask",
                provenance={
                    "canvas_dir": str(Path(canvas.store.root) if hasattr(canvas.store, "root") else ""),
                    "source_region_id": str(row.get("region_id", "")),
                    "source_bbox_xywh": list(bbox),
                    "crop_xywh": list(crop),
                    "tile_coverage": _list_of_dicts(row.get("tile_coverage", [])),
                },
            )
        )
    return tensors, samples


def _load_canvas(canvas_dir: str | Path) -> zarr.Group:
    path = Path(canvas_dir)
    if path.name != "canvas.zarr":
        path = path / "canvas.zarr"
    if not path.exists():
        raise FileNotFoundError(f"Canvas Zarr not found: {path}")
    return zarr.open_group(str(path), mode="r")


def _write_sample_tensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    root = zarr.open_group(str(path), mode="w")
    for name, array in tensors.items():
        root.create_array(name, data=array)


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist([_json_ready(row) for row in rows])
    else:
        table = pa.Table.from_pylist([])
    pq.write_table(table, path)


def _rejected_row(row: dict[str, Any], accepted_labels: set[str]) -> dict[str, Any]:
    label = str(row.get("curation_label", ""))
    reason = row.get("rejection_reason")
    if not reason:
        if label not in accepted_labels:
            reason = "label_excluded_from_default_training"
        else:
            reason = "bad_provenance"
    return {**_json_ready(row), "training_rejection_reason": str(reason)}


def _missing_signal_counts(tensors: dict[str, np.ndarray]) -> dict[str, int]:
    if tensors["alpha"].shape[0] == 0:
        return {"alpha": 0, "height": 0, "normal": 0, "mcly_texture_ids": 0, "mcly_layer_mask": 0}
    return {
        "alpha": int(np.count_nonzero(tensors["alpha"].reshape(tensors["alpha"].shape[0], -1).max(axis=1) <= 0.0)),
        "height": int(np.count_nonzero(np.ptp(tensors["height"].reshape(tensors["height"].shape[0], -1), axis=1) <= 0.0)),
        "normal": int(np.count_nonzero(np.abs(tensors["normal_xyz"]).reshape(tensors["normal_xyz"].shape[0], -1).max(axis=1) <= 0.0)),
        "mcly_texture_ids": int(np.count_nonzero(tensors["mcly_texture_ids"].reshape(tensors["mcly_texture_ids"].shape[0], -1).max(axis=1) < 0)),
        "mcly_layer_mask": int(np.count_nonzero(tensors["mcly_layer_mask"].reshape(tensors["mcly_layer_mask"].shape[0], -1).max(axis=1) <= 0.0)),
    }


def _crop_window_for_bbox(bbox: tuple[int, int, int, int], *, canvas_w: int, canvas_h: int, crop_size: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    out_x = min(max(0, cx - crop_size // 2), max(0, canvas_w - crop_size))
    out_y = min(max(0, cy - crop_size // 2), max(0, canvas_h - crop_size))
    return (int(out_x), int(out_y), int(crop_size), int(crop_size))


def _copy_2d_or_3d(source: np.ndarray, dest: np.ndarray, x: int, y: int, *, fill: float | int) -> None:
    dest[...] = fill
    source_h, source_w = source.shape[:2]
    dest_h, dest_w = dest.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(source_w, int(x) + dest_w), min(source_h, int(y) + dest_h)
    if x1 <= x0 or y1 <= y0:
        return
    dx0, dy0 = x0 - int(x), y0 - int(y)
    dest[dy0 : dy0 + (y1 - y0), dx0 : dx0 + (x1 - x0), ...] = source[y0:y1, x0:x1, ...]


def _normalize_region_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "bbox_xywh" in normalized:
        normalized["bbox_xywh"] = list(_bbox_tuple(normalized["bbox_xywh"]))
    for key in ("tile_coverage", "mcly_texture_ids", "mcly_active_layers", "linked_component_ids"):
        if normalized.get(key) is None:
            normalized[key] = []
    return normalized


def _bbox_tuple(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"Expected bbox_xywh with four values, got {value!r}")
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def _list_of_dicts(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, int]] = []
    for item in value:
        if isinstance(item, dict):
            out.append({str(key): int(val) for key, val in item.items()})
    return out


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_xyz(value: Any) -> tuple[float, float, float] | None:
    if value is None or not isinstance(value, list | tuple) or len(value) != 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
