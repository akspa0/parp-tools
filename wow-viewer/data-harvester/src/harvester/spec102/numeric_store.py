"""Minimal, identity-checked numeric store for Spec 102."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.spec102.m0 import PRECISE_MASK_KEY
from harvester.v25.dataset import DEFAULT_CODEC

SPECS = {
    "minimap_rgb": (np.uint8, (256, 256, 3)),
    PRECISE_MASK_KEY: (np.float32, (257, 257)),
    "liquid_mask_256": (np.uint8, (256, 256)),
    "liquid_height_256": (np.float32, (256, 256)),
    "mcnk_flags_16": (np.int32, (16, 16)),
    "normal_xyz_257": (np.int8, (257, 257, 3)),
    "height_257": (np.float32, (257, 257)),
}


def _u8_unit(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value)
    if data.dtype == np.uint8:
        return data
    data = data.astype(np.float32)
    if data.max(initial=0.0) <= 1.5:
        data *= 255.0
    return np.clip(np.rint(data), 0, 255).astype(np.uint8)


def _i8_normals(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value)
    if data.dtype == np.int8:
        return data
    data = data.astype(np.float32)
    if np.abs(data).max(initial=0.0) <= 1.5:
        data *= 127.0
    return np.clip(np.rint(data), -127, 127).astype(np.int8)


def build_numeric_store(
    *, selection_store: Path,
    v18_stores: list[Path],
    output: Path,
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite numeric store: {output}")
    selection_rows = pq.read_table(selection_store / "index.parquet").to_pylist()
    sources: dict[str, tuple[zarr.Group, list[dict]]] = {}
    for path in v18_stores:
        group = zarr.open_group(str(path), mode="r")
        rows = pq.read_table(path / "index.parquet").to_pylist()
        builds = {str(row["build"]) for row in rows}
        if len(builds) != 1:
            raise RuntimeError(f"V18 source must contain exactly one build: {path}")
        sources[next(iter(builds))] = (group, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open_group(str(output), mode="w")
    arrays = {
        name: out.create_array(
            name, shape=(len(selection_rows), *shape), chunks=(1, *shape),
            dtype=dtype, compressors=DEFAULT_CODEC,
        )
        for name, (dtype, shape) in SPECS.items()
    }
    copied_rows: list[dict] = []
    identity_fields = ("build", "map", "tile_id", "tile_x", "tile_y")
    for out_row, selected in enumerate(selection_rows):
        build = str(selected["build"])
        if build not in sources:
            raise RuntimeError(f"missing V18 source for selected build {build}")
        source, source_index = sources[build]
        source_row = int(selected["v18_row"])
        origin = source_index[source_row]
        mismatches = [
            field for field in identity_fields
            if str(selected[field]) != str(origin[field])
        ]
        if mismatches:
            raise RuntimeError(
                f"signal identity mismatch at output row {out_row}, V18 row {source_row}: {mismatches}"
            )
        arrays["minimap_rgb"][out_row] = np.asarray(source["minimap_rgb"][source_row], dtype=np.uint8)
        arrays[PRECISE_MASK_KEY][out_row] = np.asarray(source["object_precise_mask"][source_row], dtype=np.float32)
        arrays["liquid_mask_256"][out_row] = _u8_unit(source["liquid_mask"][source_row])
        arrays["liquid_height_256"][out_row] = np.asarray(source["liquid_height"][source_row], dtype=np.float32)
        arrays["mcnk_flags_16"][out_row] = np.asarray(source["mcnk_flags_16"][source_row], dtype=np.int32)
        arrays["normal_xyz_257"][out_row] = _i8_normals(source["normal_xyz"][source_row])
        arrays["height_257"][out_row] = np.asarray(source["height_257"][source_row], dtype=np.float32)
        row = dict(selected)
        row["row"] = out_row
        row["height_repaired"] = False
        row["identity_verified"] = True
        row["has_liquid_mask"] = bool(origin.get("has_liquid_mask", False))
        row["has_liquid_height"] = bool(origin.get("has_liquid_height", False))
        liquid_sources = [
            name for name in ("mcnk", "mh2o", "mclq", "unified", "wl")
            if bool(origin.get(f"has_liquid_source_{name}", False))
        ]
        row["liquid_source"] = liquid_sources[0] if len(liquid_sources) == 1 else None
        copied_rows.append(row)

    pq.write_table(pa.Table.from_pylist(copied_rows), output / "index.parquet")
    out.attrs.update({
        "schema": "spec102-numeric-store-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tile_count": len(copied_rows),
        "source_selection_store": str(selection_store),
        "source_v18_stores": [str(path) for path in v18_stores],
        "signals": list(SPECS),
        "prohibited_absent_signals": [
            "clean_minimap_256", "object_mask_256", "object_visibility_256",
            "wdl_height_33", "placements", "height_repair",
        ],
    })
    (output / "contract.json").write_text(json.dumps(dict(out.attrs), indent=2), encoding="utf-8")
    return output
