"""Spec 120 Metadata Sidecar Exporter (T009).

Converts detected OBB objects on minimap tiles into structured JSON and Parquet metadata sidecar files.
Calculates continuous tile and world positions, pixel and world scales, rotation angles, confidence scores,
and retrieved asset paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from harvester.spec120.obb_contract import (
    format_sidecar_item,
    tile_pixels_to_world,
    validate_sidecar_schema,
)


def export_tile_sidecar(
    detections: list[dict[str, Any]],
    tile_x: int = 32,
    tile_y: int = 32,
    output_json_path: Path | None = None,
    output_parquet_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Export list of OBB detection dicts to structured sidecar metadata records."""
    sidecar_items: list[dict[str, Any]] = []

    for idx, det in enumerate(detections, start=1):
        px = float(det["px"])
        py = float(det["py"])

        world_x, world_y = tile_pixels_to_world(px, py, tile_x, tile_y)
        world_z = float(det.get("world_z", 0.0))

        w_px = float(det["w_px"])
        h_px = float(det["h_px"])

        scale_factor = (w_px + h_px) / 32.0  # Normalized scale factor relative to 16px baseline
        coarse_class = str(det.get("coarse_class", "wmo"))
        retrieved_asset = str(det.get("asset_path", det.get("normalized_asset_path", "unretrieved_asset")))

        item = format_sidecar_item(
            instance_id=idx,
            position_px=(px, py),
            world_pos=(world_x, world_y, world_z),
            scale_px=(w_px, h_px),
            scale_factor=scale_factor,
            rotation_deg=float(det.get("angle_deg", 0.0)),
            coarse_class=coarse_class,
            retrieved_asset=retrieved_asset,
            confidence=float(det.get("conf", 0.9)),
            tile_x=tile_x,
            tile_y=tile_y,
        )
        sidecar_items.append(item)

    # Schema validation
    validate_sidecar_schema(sidecar_items)

    # Write JSON sidecar if requested
    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_items, f, indent=2)

    # Write Parquet sidecar if requested
    if output_parquet_path is not None:
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if sidecar_items:
            table = pa.Table.from_pylist(sidecar_items)
            pq.write_table(table, output_parquet_path)

    return sidecar_items
