"""One-shot raw component analysis and exact dedupe helpers for spec 076."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.fractal_segments import FractalRegion


@dataclass(frozen=True, slots=True)
class RawComponentFingerprint:
    region_id: str
    pattern_id: str
    build: str
    map_name: str
    layer_idx: int
    layer_slot: int
    bbox_xywh: tuple[int, int, int, int]
    area: int
    crop_w: int
    crop_h: int
    alpha_mean: float
    alpha_max: float
    tile_coverage_count: int
    tile_coverage: list[dict[str, int]]
    mcly_texture_ids: list[int]
    mcly_active_layers: list[int]


def fingerprint_raw_regions(
    canvas: zarr.Group,
    regions: list[FractalRegion],
    *,
    threshold: float = 0.05,
) -> list[RawComponentFingerprint]:
    """Fingerprint raw alpha-region crops for exact cross-build dedupe."""
    alpha = canvas["alpha_256"][:].astype(np.float32)
    rows: list[RawComponentFingerprint] = []
    for region in regions:
        x, y, w, h = region.bbox_xywh
        if w <= 0 or h <= 0:
            continue
        crop = alpha[y : y + h, x : x + w, int(region.layer_slot)]
        pattern_id = raw_component_pattern_id(crop, threshold=threshold)
        rows.append(
            RawComponentFingerprint(
                region_id=region.region_id,
                pattern_id=pattern_id,
                build=region.build,
                map_name=region.map_name,
                layer_idx=int(region.layer_idx),
                layer_slot=int(region.layer_slot),
                bbox_xywh=region.bbox_xywh,
                area=int(region.area),
                crop_w=int(w),
                crop_h=int(h),
                alpha_mean=float(region.alpha_mean),
                alpha_max=float(region.alpha_max),
                tile_coverage_count=int(region.tile_coverage_count),
                tile_coverage=region.tile_coverage,
                mcly_texture_ids=region.mcly_texture_ids,
                mcly_active_layers=region.mcly_active_layers,
            )
        )
    return rows


def raw_component_pattern_id(crop: np.ndarray, *, threshold: float = 0.05) -> str:
    """Stable exact binary-shape fingerprint for a raw alpha crop."""
    binary = np.asarray(crop, dtype=np.float32) > float(threshold)
    h, w = binary.shape[:2]
    packed = np.packbits(binary.reshape(-1).astype(np.uint8))
    digest = hashlib.sha256()
    digest.update(str(int(w)).encode("ascii"))
    digest.update(b"x")
    digest.update(str(int(h)).encode("ascii"))
    digest.update(b"|")
    digest.update(packed.tobytes())
    return "pat_" + digest.hexdigest()[:20]


def build_pattern_catalog(rows: list[RawComponentFingerprint]) -> list[dict[str, Any]]:
    by_pattern: dict[str, list[RawComponentFingerprint]] = defaultdict(list)
    for row in rows:
        by_pattern[row.pattern_id].append(row)

    catalog: list[dict[str, Any]] = []
    for pattern_id, members in by_pattern.items():
        members_sorted = sorted(members, key=lambda item: (item.build, item.map_name, item.layer_idx, item.region_id))
        first = members_sorted[0]
        builds = sorted({item.build for item in members_sorted})
        maps = sorted({item.map_name for item in members_sorted})
        layers = sorted({int(item.layer_idx) for item in members_sorted})
        catalog.append(
            {
                "pattern_id": pattern_id,
                "member_count": int(len(members_sorted)),
                "build_count": int(len(builds)),
                "map_count": int(len(maps)),
                "layer_count": int(len(layers)),
                "builds": builds,
                "maps": maps,
                "layer_indices": layers,
                "crop_w": int(first.crop_w),
                "crop_h": int(first.crop_h),
                "area": int(first.area),
                "example_region_id": first.region_id,
                "example_bbox_xywh": list(first.bbox_xywh),
                "mean_alpha_mean": float(np.mean([item.alpha_mean for item in members_sorted])),
                "max_alpha_max": float(max(item.alpha_max for item in members_sorted)),
                "region_ids": [item.region_id for item in members_sorted[:128]],
                "mcly_texture_ids": sorted({texture for item in members_sorted for texture in item.mcly_texture_ids})[:64],
                "mcly_active_layers": sorted({layer for item in members_sorted for layer in item.mcly_active_layers}),
            }
        )
    catalog.sort(key=lambda row: (-int(row["member_count"]), -int(row["area"]), str(row["pattern_id"])))
    return catalog


def write_raw_dedupe_outputs(output_dir: str | Path, rows: list[RawComponentFingerprint]) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    component_rows = [_json_ready(asdict(row)) for row in rows]
    catalog_rows = [_json_ready(row) for row in build_pattern_catalog(rows)]
    _write_table(out / "raw_components.parquet", component_rows)
    _write_jsonl(out / "raw_components.jsonl", component_rows)
    _write_table(out / "exact_patterns.parquet", catalog_rows)
    _write_jsonl(out / "exact_patterns.jsonl", catalog_rows)

    pattern_counts = Counter(row.pattern_id for row in rows)
    summary = {
        "raw_component_count": int(len(rows)),
        "exact_pattern_count": int(len(pattern_counts)),
        "duplicate_pattern_count": int(sum(1 for count in pattern_counts.values() if count > 1)),
        "max_pattern_members": int(max(pattern_counts.values(), default=0)),
        "builds": sorted({row.build for row in rows}),
        "maps": sorted({row.map_name for row in rows}),
        "outputs": {
            "raw_components_parquet": str(out / "raw_components.parquet"),
            "exact_patterns_parquet": str(out / "exact_patterns.parquet"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([])
    pq.write_table(table, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
