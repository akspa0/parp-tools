"""Spec 116 US5 bridge: adapt a derived structure store into the geometry trainer's feature-store
shape.

``structure_materialize.py`` writes ``v116-structure-store-v1`` (per-chunk 16x16 hard predicted
family + confidence). ``harvester.v50.direct_geometry_train``'s ``--feature-store`` was built for
Spec 115's ``v115-feature-map-v1`` (per-pixel 256x256 soft class probabilities) and hard-refuses
any other schema or a missing ``feature_map`` array. Rather than change the trainer's established
contract, this module upsamples the structure store into that exact shape so the quickstart 5b
paired geometry comparison (with vs without predicted structure) runs unmodified against the
existing trainer.

Each 16x16 chunk's predicted family is repeated to its 16x16 pixel footprint. The predicted
class keeps its confidence as probability mass; the remaining ``1 - confidence`` is spread
uniformly across the other classes, so every pixel is still a valid probability distribution over
``CLASS_COUNT`` -- this is the trainer's only requirement, not a claim of finer-grained certainty.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.v50.terrain_feature_labels import CLASS_COUNT, FAMILY_NAMES

FEATURE_STORE_SCHEMA = "v115-feature-map-v1"
FEATURE_ARRAY = "feature_map"
SOURCE_SCHEMA = "v116-structure-store-v1"
CHUNK_GRID = 16
PIXELS = 256
UPSAMPLE = PIXELS // CHUNK_GRID


class StructureFeatureBridgeError(ValueError):
    """Raised when a structure store cannot be adapted into a feature-map store."""


def _upsample_to_pixels(chunks: np.ndarray) -> np.ndarray:
    """(16, 16) -> (256, 256) by nearest-neighbour repeat (one value per chunk footprint)."""
    return np.repeat(np.repeat(chunks, UPSAMPLE, axis=0), UPSAMPLE, axis=1)


def structure_to_feature_map(
    *,
    structure_store: Path,
    output: Path,
    write: bool = False,
) -> dict:
    """Adapt one ``v116-structure-store-v1`` into a ``v115-feature-map-v1``-shaped store.

    Returns the materialization plan (or the written store summary if ``write=True``). The
    source structure store is never mutated.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr

    source = zarr.open_group(str(structure_store), mode="r")
    source_schema = dict(source.attrs).get("schema")
    if source_schema != SOURCE_SCHEMA:
        raise StructureFeatureBridgeError(
            f"--structure-store is not a {SOURCE_SCHEMA!r} store: {structure_store} "
            f"(schema={source_schema!r})"
        )
    for required in ("structure_family", "structure_confidence"):
        if required not in source:
            raise StructureFeatureBridgeError(f"structure store is missing {required!r}: {structure_store}")

    row_count = int(source["structure_family"].shape[0])
    index_path = structure_store / "index.parquet"
    if not index_path.exists():
        raise StructureFeatureBridgeError(f"structure store has no index.parquet: {structure_store}")
    index_rows = pq.read_table(index_path).to_pylist()
    if len(index_rows) != row_count:
        raise StructureFeatureBridgeError(
            f"index.parquet rows ({len(index_rows)}) != structure_family rows ({row_count})"
        )

    plan = {
        "schema": "v116-structure-feature-bridge-plan-v1",
        "source_store": str(structure_store),
        "source_schema": source_schema,
        "selected_rows": row_count,
        "output_array": {
            "name": FEATURE_ARRAY,
            "shape": [row_count, CLASS_COUNT, PIXELS, PIXELS],
            "dtype": "float16",
        },
    }
    if not write:
        return plan

    if output.exists() and any(output.iterdir()):
        raise StructureFeatureBridgeError(
            f"refusing to overwrite non-empty feature-map store {output}; choose a new path"
        )
    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w")
    feature_array = group.create_array(
        FEATURE_ARRAY,
        shape=(row_count, CLASS_COUNT, PIXELS, PIXELS),
        chunks=(1, CLASS_COUNT, PIXELS, PIXELS),
        dtype=np.float16,
    )

    other_count = max(CLASS_COUNT - 1, 1)
    for row in range(row_count):
        family = np.asarray(source["structure_family"][row], dtype=np.int64)  # (16, 16)
        confidence = np.asarray(source["structure_confidence"][row], dtype=np.float32)  # (16, 16)
        family_px = _upsample_to_pixels(family)  # (256, 256)
        confidence_px = np.clip(_upsample_to_pixels(confidence), 0.0, 1.0)

        probs = np.empty((CLASS_COUNT, PIXELS, PIXELS), dtype=np.float32)
        fill = (1.0 - confidence_px) / other_count
        for k in range(CLASS_COUNT):
            is_predicted = family_px == k
            probs[k] = np.where(is_predicted, confidence_px, fill)
        feature_array[row] = probs.astype(np.float16)

    derived_index = [
        {
            "source_row_index": int(index_rows[row].get("source_row_index", row)),
            "map": str(index_rows[row].get("map")),
            "tile_x": int(index_rows[row].get("tile_x", -1)),
            "tile_y": int(index_rows[row].get("tile_y", -1)),
        }
        for row in range(row_count)
    ]
    pq.write_table(pa.Table.from_pylist(derived_index), output / "index.parquet")

    created_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    group.attrs.update(
        {
            "schema": FEATURE_STORE_SCHEMA,
            "created_utc": created_utc,
            "class_count": CLASS_COUNT,
            "family_names": list(FAMILY_NAMES),
            "source_structure_store": str(structure_store),
            "bridge": "spec116_structure_to_feature_map_v1",
        }
    )

    return {**plan, "schema": FEATURE_STORE_SCHEMA, "created_utc": created_utc, "output": str(output)}


__all__ = [
    "StructureFeatureBridgeError",
    "structure_to_feature_map",
    "FEATURE_STORE_SCHEMA",
    "FEATURE_ARRAY",
    "SOURCE_SCHEMA",
]
