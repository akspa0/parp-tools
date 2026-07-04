"""V22 Zarr writer and reader.

The canonical V22 dataset is built on top of a V18 Zarr store plus a
C#-produced enrichment stream. The ``V22ZarrWriter`` reads the V18 store,
derives the V22-patched signals in pure Python, promotes V18 placements to
native V22 placement arrays, and accumulates per-build model + tileset
libraries from the enrichment stream.

The ``V22Dataset`` reader is the fixed-key consumer contract. Every tile
returns the same batch keys. No ``has_*`` branches, no optional keys.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

# ---------------------------------------------------------------------------
# Canonical layout — mirror of the Spec 086 architecture document.
# ---------------------------------------------------------------------------

V22_DATASET_VERSION = "v22"
V22_BUILD_IDS: tuple[str, ...] = (
    "0_5_3_3368",
    "3_3_5_12340",
    "4_0_0_11927",
)

V22_ROOT_ARRAYS: tuple[str, ...] = (
    "height_257",
    "normal_xyz",
    "normal_mask",
    "alpha_256",
    "holes_16",
    "liquid_mask",
    "liquid_height",
    "object_mask",
    "object_precise_mask",
    "object_instance_mask",
    "mcnk_flags_16",
    "mddf_mask",
    "modf_mask",
    "object_filtered_mask",
    "model_focus_mask",
    "model_above_terrain_mask",
    "object_roof_mask",
    "object_roof_confidence",
    "minimap_rgb",
    "shadow_mask",
    "mcly_texture_ids",
    "mcly_layer_mask",
    "mcnr_mask_257",
    "liquid_type_256",
    "ground_intent_height_257",
    "mddf_placement_offset",
    "mddf_count",
    "mddf_placement_data",
    "mddf_unique_ids",
    "mddf_model_ids",
    "modf_placement_offset",
    "modf_count",
    "modf_placement_data",
    "modf_unique_ids",
    "modf_model_ids",
    "mcly_tileset_ids",
)

V22_METADATA_KEYS: tuple[str, ...] = (
    "tile_id",
    "build",
    "map",
    "tile_x",
    "tile_y",
    "mtex_texture_paths",
    "placement_mddf_asset_paths",
    "placement_modf_asset_paths",
)

V22_MODELS_GROUP = "models"
V22_TILESETS_GROUP = "tilesets"

DEFAULT_CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")


# ---------------------------------------------------------------------------
# Shapes / dtypes / fill — mirror of the frozen V22 contract.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ArraySpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]


def _arr(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> _ArraySpec:
    return _ArraySpec(name, np.dtype(dtype), shape)


V22_PER_TILE_SPECS: tuple[_ArraySpec, ...] = (
    _arr("height_257", np.float32, (257, 257)),
    _arr("normal_xyz", np.float32, (257, 257, 3)),
    _arr("normal_mask", np.bool_, (257, 257)),
    _arr("alpha_256", np.float32, (256, 256, 4)),
    _arr("holes_16", np.bool_, (16, 16)),
    _arr("liquid_mask", np.float32, (256, 256)),
    _arr("liquid_height", np.float32, (256, 256)),
    _arr("object_mask", np.bool_, (257, 257)),
    _arr("object_precise_mask", np.float32, (257, 257)),
    _arr("object_instance_mask", np.int32, (257, 257)),
    _arr("mcnk_flags_16", np.int32, (16, 16)),
    _arr("mddf_mask", np.float32, (257, 257)),
    _arr("modf_mask", np.float32, (257, 257)),
    _arr("object_filtered_mask", np.float32, (257, 257)),
    _arr("model_focus_mask", np.float32, (257, 257)),
    _arr("model_above_terrain_mask", np.float32, (257, 257)),
    _arr("object_roof_mask", np.float32, (256, 256)),
    _arr("object_roof_confidence", np.float32, (256, 256)),
    _arr("minimap_rgb", np.uint8, (256, 256, 3)),
    _arr("shadow_mask", np.float32, (256, 256)),
    _arr("mcly_texture_ids", np.int32, (16, 16, 4)),
    _arr("mcly_layer_mask", np.float32, (16, 16, 4)),
    _arr("mcnr_mask_257", np.bool_, (257, 257)),
    _arr("liquid_type_256", np.uint8, (256, 256)),
    _arr("ground_intent_height_257", np.float32, (257, 257)),
    _arr("mddf_count", np.int32, (1,)),
    _arr("modf_count", np.int32, (1,)),
    _arr("mcly_tileset_ids", np.int32, (16, 16, 4)),
)

V22_FLAT_SPECS: tuple[_ArraySpec, ...] = (
    _arr("mddf_placement_offset", np.int64, (0,)),
    _arr("mddf_placement_data", np.float32, (0, 9)),
    _arr("mddf_unique_ids", np.int32, (0,)),
    _arr("mddf_model_ids", np.int32, (0,)),
    _arr("modf_placement_offset", np.int64, (0,)),
    _arr("modf_placement_data", np.float32, (0, 17)),
    _arr("modf_unique_ids", np.int32, (0,)),
    _arr("modf_model_ids", np.int32, (0,)),
)

_PER_TILE_CHUNK: dict[str, tuple[int, ...]] = {
    "height_257": (129, 129),
    "normal_xyz": (129, 129, 3),
    "normal_mask": (129, 129),
    "alpha_256": (128, 128, 4),
    "holes_16": (16, 16),
    "liquid_mask": (128, 128),
    "liquid_height": (128, 128),
    "object_mask": (129, 129),
    "object_precise_mask": (129, 129),
    "object_instance_mask": (129, 129),
    "mcnk_flags_16": (16, 16),
    "mddf_mask": (129, 129),
    "modf_mask": (129, 129),
    "object_filtered_mask": (129, 129),
    "model_focus_mask": (129, 129),
    "model_above_terrain_mask": (129, 129),
    "object_roof_mask": (128, 128),
    "object_roof_confidence": (128, 128),
    "minimap_rgb": (128, 128, 3),
    "shadow_mask": (128, 128),
    "mcly_texture_ids": (16, 16, 4),
    "mcly_layer_mask": (16, 16, 4),
    "mcnr_mask_257": (129, 129),
    "liquid_type_256": (128, 128),
    "ground_intent_height_257": (129, 129),
    "mcly_tileset_ids": (16, 16, 4),
}

_FLAT_CHUNK: dict[str, tuple[int, ...]] = {
    "mddf_placement_offset": (4096,),
    "mddf_placement_data": (4096, 9),
    "mddf_unique_ids": (4096,),
    "mddf_model_ids": (4096,),
    "modf_placement_offset": (4096,),
    "modf_placement_data": (4096, 17),
    "modf_unique_ids": (4096,),
    "modf_model_ids": (4096,),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _resolve_chunk(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    preset = _PER_TILE_CHUNK.get(name) or _FLAT_CHUNK.get(name)
    if preset is not None:
        return tuple(min(c, s) if s else c for c, s in zip(preset, shape, strict=False))
    if len(shape) == 1:
        return shape
    if len(shape) == 2:
        return tuple(min(64, s) if s else 64 for s in shape)
    return (min(64, shape[0]), min(64, shape[1])) + shape[2:]


def _resize_flat_to(shape0: int, n: int, dtype: np.dtype, fill: Any = 0) -> np.ndarray:
    if shape0 >= n:
        return np.zeros((shape0, *()), dtype=dtype)
    out = np.zeros((n,), dtype=dtype)
    return out  # keep zeros; data is appended at write time


def _parse_dtype(dtype_str: str) -> np.dtype:
    """Map an enrichment stream dtype string to a numpy dtype."""
    mapping = {
        "<f4": np.float32,
        "<f8": np.float64,
        "<i4": np.int32,
        "<u4": np.uint32,
        "<i2": np.int16,
        "<u2": np.uint16,
        "|u1": np.uint8,
        "|i1": np.int8,
        "|b1": np.bool_,
    }
    return mapping.get(dtype_str, np.uint8)


def _normalize_asset_path(path: str | None) -> str:
    """Normalize an asset path for storage and display."""
    if not path:
        return ""
    normalized = str(path).replace("\\", "/").strip()
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _asset_lookup_key(path: str | None) -> str:
    """Normalize an asset path for stable lookups across placements/enrichment."""
    return _normalize_asset_path(path).lower()


def _dataset_texture_key(path: str | None) -> str:
    """Normalize a decoded texture identity for the V22 dataset.

    The staged source asset can still be a BLP, but the dataset stores decoded
    texture payloads, so the in-dataset key uses a PNG-style path.
    """
    normalized = _normalize_asset_path(path)
    if normalized.lower().endswith(".blp"):
        normalized = normalized[:-4] + ".png"
    return normalized


def _source_kind(source_in_listfile: int | bool) -> str:
    return "internal_listfile" if int(source_in_listfile) else "archive_unlisted"


def _decode_string_table_blob(arr: np.ndarray | None) -> tuple[str, ...]:
    if arr is None:
        return ()
    blob = np.asarray(arr, dtype=np.uint8).reshape(-1).tobytes()
    if len(blob) < 4:
        return ()

    offset = 0
    count = int.from_bytes(blob[offset:offset + 4], "little", signed=True)
    offset += 4
    if count <= 0:
        return ()

    values: list[str] = []
    for _ in range(count):
        if offset + 4 > len(blob):
            break
        length = int.from_bytes(blob[offset:offset + 4], "little", signed=True)
        offset += 4
        if length < 0 or offset + length > len(blob):
            break
        values.append(_normalize_asset_path(blob[offset:offset + length].decode("utf-8")))
        offset += length
    return tuple(value for value in values if value)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class V22ZarrWriter:
    """Write a V22 Zarr store from a stream of decoded tile records."""

    def __init__(
        self,
        store_path: str | Path,
        *,
        codec: Any = DEFAULT_CODEC,
        overwrite: bool = True,
        embed_asset_payloads: bool = True,
    ) -> None:
        self.store_path = Path(store_path)
        if overwrite and self.store_path.exists():
            self._rmtree(self.store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._tile_records: list[dict[str, Any]] = []
        self._tile_rows: list[dict[str, np.ndarray]] = []
        self._mddf_data: list[np.ndarray] = []
        self._modf_data: list[np.ndarray] = []
        self._mddf_uids: list[np.ndarray] = []
        self._modf_uids: list[np.ndarray] = []
        self._mddf_model_ids: list[np.ndarray] = []
        self._modf_model_ids: list[np.ndarray] = []
        self._mddf_offsets: list[int] = []
        self._modf_offsets: list[int] = []
        self._models: dict[str, dict[str, Any]] = {}
        self._tilesets: dict[str, dict[str, Any]] = {}
        self._placement_rows: list[dict[str, Any]] = []
        self._source_v18_store: str | None = None
        self._codec = codec
        self._embed_asset_payloads = bool(embed_asset_payloads)

    # ── V18+enrichment → V22 ingestion ────────────────────────────
    def add_from_v18(self, v18_path: str | Path, enrichment_path: str | Path) -> None:
        """Read a V18 Zarr store and enrichment stream, populate the V22 store.

        This replaces the old ``add_tile`` API. The V22 store accumulates
        per-tile arrays from V18, derives V22-patched signals in pure Python,
        promotes placements from V18's sidecar parquet, and reads the enrichment
        stream for per-build model and tileset libraries.
        """
        v18_store_path = Path(v18_path)
        enrich_path = Path(enrichment_path)
        if not v18_store_path.exists():
            raise FileNotFoundError(v18_store_path)

        self._source_v18_store = str(v18_store_path)

        # ── 1. Read enrichment stream → populate _models, _tilesets ────
        self._ingest_enrichment_stream(enrich_path)

        # Build path → index lookups
        model_paths = sorted(self._models.keys(), key=str.casefold)
        tileset_paths = sorted(self._tilesets.keys(), key=str.casefold)
        model_path_to_idx = {_asset_lookup_key(p): i for i, p in enumerate(model_paths)}
        tileset_path_to_idx = {_asset_lookup_key(p): i for i, p in enumerate(tileset_paths)}

        # ── 2. Read V18 placements.parquet ──────────────────────────
        placements_path = v18_store_path / "placements.parquet"
        mddf_by_tile: dict[int, list[dict]] = {}
        modf_by_tile: dict[int, list[dict]] = {}
        if placements_path.exists():
            placements_table = pq.read_table(str(placements_path))
            placement_columns = {
                name: placements_table.column(name).to_pylist()
                for name in placements_table.column_names
            }
            row_count = placements_table.num_rows
            for row_index in range(row_count):
                row = {name: values[row_index] for name, values in placement_columns.items()}
                tile_id = int(row.get("tile_id", -1))
                if tile_id < 0:
                    continue
                i_type = str(row.get("instance_type", "")).lower()
                entry = {
                    "nameId": int(row.get("nameId", -1)),
                    "uniqueId": int(row.get("uniqueId", -1)),
                    "posX": float(row.get("posX", 0)),
                    "posY": float(row.get("posY", 0)),
                    "posZ": float(row.get("posZ", 0)),
                    "rotX": float(row.get("rotX", 0)),
                    "rotY": float(row.get("rotY", 0)),
                    "rotZ": float(row.get("rotZ", 0)),
                    "scale": float(row.get("scale", 0)),
                    "asset_path": _normalize_asset_path(str(row.get("asset_path", "") or "")),
                    "bbMinX": float(row.get("bbMinX", 0)),
                    "bbMinY": float(row.get("bbMinY", 0)),
                    "bbMinZ": float(row.get("bbMinZ", 0)),
                    "bbMaxX": float(row.get("bbMaxX", 0)),
                    "bbMaxY": float(row.get("bbMaxY", 0)),
                    "bbMaxZ": float(row.get("bbMaxZ", 0)),
                    "instance_idx": int(row.get("instance_idx", -1)),
                }
                if i_type == "mddf":
                    mddf_by_tile.setdefault(tile_id, []).append(entry)
                elif i_type == "modf":
                    modf_by_tile.setdefault(tile_id, []).append(entry)

        # ── 3. Read decoded V18 metadata sidecar ─────────────────
        decoded_metadata_by_tile: dict[int, dict[str, Any]] = {}
        decoded_meta_path = v18_store_path / "decoded_metadata.parquet"
        if decoded_meta_path.exists():
            metadata_table = pq.read_table(str(decoded_meta_path))
            metadata_columns = {
                name: metadata_table.column(name).to_pylist()
                for name in metadata_table.column_names
            }
            for row_index in range(metadata_table.num_rows):
                row = {name: values[row_index] for name, values in metadata_columns.items()}
                tile_id = int(row.get("tile_id", -1))
                payload = row.get("decoded_metadata_json", "")
                if tile_id < 0 or not payload:
                    continue
                try:
                    decoded_metadata_by_tile[tile_id] = json.loads(payload)
                except json.JSONDecodeError:
                    decoded_metadata_by_tile[tile_id] = {}

        # ── 4. Enrichment stream entry lookups for model_ids ──────
        def _model_index(asset_path: str) -> int:
            return model_path_to_idx.get(_asset_lookup_key(asset_path), -1)

        def _tileset_index(asset_path: str) -> int:
            return tileset_path_to_idx.get(_asset_lookup_key(_dataset_texture_key(asset_path)), -1)

        # ── 5. Open V18 store ─────────────────────────────────────
        v18_store = zarr.storage.LocalStore(str(v18_store_path), read_only=True)
        v18 = zarr.open_group(store=v18_store, mode="r")
        n_v18_tiles = v18["height_257"].shape[0]

        # Read V18 index.parquet for per-tile metadata without pandas.
        v18_index_path = v18_store_path / "index.parquet"
        v18_index_rows: list[dict[str, Any]] = []
        if v18_index_path.exists():
            index_table = pq.read_table(str(v18_index_path))
            index_columns = {
                name: index_table.column(name).to_pylist()
                for name in index_table.column_names
            }
            for row_index in range(index_table.num_rows):
                v18_index_rows.append({
                    name: values[row_index]
                    for name, values in index_columns.items()
                })

        from harvester.v22_patched_signals import (
            derive_ground_intent_height_257,
            derive_liquid_type_256,
            derive_mcnr_mask_257,
            derive_model_above_terrain_mask,
            derive_model_focus_mask,
        )

        V18_ROOT_ARRAYS = [
            "height_257", "normal_xyz", "normal_mask",
            "alpha_256", "holes_16", "liquid_mask", "liquid_height",
            "object_mask", "object_precise_mask", "object_instance_mask",
            "mcnk_flags_16", "mddf_mask", "modf_mask", "object_filtered_mask",
            "object_roof_mask", "object_roof_confidence",
            "minimap_rgb", "shadow_mask", "mcly_texture_ids", "mcly_layer_mask",
            "mcnr_mask_257", "liquid_basic_type_257",
        ]

        mddf_cursor = 0
        modf_cursor = 0

        for tile_idx in range(n_v18_tiles):
            # Build per-tile V18 dict
            tile: dict[str, np.ndarray] = {}
            for key in V18_ROOT_ARRAYS:
                if key in v18:
                    tile[key] = np.asarray(v18[key][tile_idx])

            # Read per-tile metadata from V18 index
            index_row = v18_index_rows[tile_idx] if tile_idx < len(v18_index_rows) else {}
            tile_id = int(index_row.get("tile_id", tile_idx))
            build_key = str(index_row.get("build", ""))
            map_name = str(index_row.get("map", ""))
            tile_x = int(index_row.get("tile_x", 0))
            tile_y = int(index_row.get("tile_y", 0))

            # Derive V22-patched signals
            tile["mcnr_mask_257"] = derive_mcnr_mask_257(tile)
            tile["liquid_type_256"] = derive_liquid_type_256(tile)
            tile["ground_intent_height_257"] = derive_ground_intent_height_257(tile)
            tile["model_focus_mask"] = derive_model_focus_mask(tile)
            tile["model_above_terrain_mask"] = derive_model_above_terrain_mask(
                tile,
                mddf_by_tile.get(tile_id, []),
                modf_by_tile.get(tile_id, []),
                tile_x,
                tile_y,
            )

            # Build per-tile V22 row
            per_tile: dict[str, np.ndarray] = {}
            for spec in V22_PER_TILE_SPECS:
                arr = tile.get(spec.name)
                if arr is None:
                    arr = np.zeros(spec.shape, dtype=spec.dtype)
                else:
                    arr = np.asarray(arr, dtype=spec.dtype)
                    if arr.shape != spec.shape:
                        arr = np.zeros(spec.shape, dtype=spec.dtype)
                per_tile[spec.name] = arr

            self._tile_rows.append(per_tile)

            # ── Placement promotion ───────────────────────────────
            tile_mddf = mddf_by_tile.get(tile_id, [])
            tile_modf = modf_by_tile.get(tile_id, [])
            decoded_metadata = decoded_metadata_by_tile.get(tile_id, {})

            n_mddf = len(tile_mddf)
            n_modf = len(tile_modf)

            self._mddf_offsets.append(mddf_cursor)
            self._modf_offsets.append(modf_cursor)

            mcly_tileset_ids = np.full((16, 16, 4), -1, dtype=np.int32)
            if "mcly_texture_ids" in tile and "mcly_layer_mask" in tile:
                local_texture_names = [
                    _dataset_texture_key(name)
                    for name in (decoded_metadata.get("mcly_texture_names", []) or [])
                    if _dataset_texture_key(name)
                ]
                local_texture_ids = np.asarray(tile["mcly_texture_ids"], dtype=np.int32)
                local_layer_mask = np.asarray(tile["mcly_layer_mask"], dtype=np.float32)
                for cy in range(16):
                    for cx in range(16):
                        for layer in range(4):
                            if local_layer_mask[cy, cx, layer] <= 0.0:
                                continue
                            texture_id = int(local_texture_ids[cy, cx, layer])
                            if 0 <= texture_id < len(local_texture_names):
                                mcly_tileset_ids[cy, cx, layer] = _tileset_index(local_texture_names[texture_id])

            if n_mddf > 0:
                mddf = np.zeros((n_mddf, 9), dtype=np.float32)
                uids = np.zeros(n_mddf, dtype=np.int32)
                mids = np.full(n_mddf, -1, dtype=np.int32)
                for i, row in enumerate(tile_mddf):
                    mddf[i] = [
                        row["nameId"], row["uniqueId"],
                        row["posX"], row["posY"], row["posZ"],
                        row["rotX"], row["rotY"], row["rotZ"],
                        row["scale"],
                    ]
                    uids[i] = row["uniqueId"]
                    mids[i] = _model_index(row["asset_path"])
                self._mddf_data.append(mddf)
                self._mddf_uids.append(uids)
                self._mddf_model_ids.append(mids)
                mddf_cursor += n_mddf

            if n_modf > 0:
                modf = np.zeros((n_modf, 17), dtype=np.float32)
                uids = np.zeros(n_modf, dtype=np.int32)
                mids = np.full(n_modf, -1, dtype=np.int32)
                for i, row in enumerate(tile_modf):
                    # Expand MODF from 14 → 17 columns (zero for flags/doodadSet/nameSet if missing)
                    modf[i] = [
                        row["nameId"], row["uniqueId"],
                        row["posX"], row["posY"], row["posZ"],
                        row["rotX"], row["rotY"], row["rotZ"],
                        0.0,  # flags (zero-fill from V18's 14-col layout)
                        0.0,  # doodadSet
                        0.0,  # nameSet
                        row["bbMinX"], row["bbMinY"], row["bbMinZ"],
                        row["bbMaxX"], row["bbMaxY"], row["bbMaxZ"],
                    ]
                    uids[i] = row["uniqueId"]
                    mids[i] = _model_index(row["asset_path"])
                self._modf_data.append(modf)
                self._modf_uids.append(uids)
                self._modf_model_ids.append(mids)
                modf_cursor += n_modf

            # ── Tile metadata for the index ──────────────────────
            mddf_asset_paths = [_normalize_asset_path(row["asset_path"]) for row in tile_mddf]
            modf_asset_paths = [_normalize_asset_path(row["asset_path"]) for row in tile_modf]
            mtex_paths = [
                _dataset_texture_key(path)
                for path in (decoded_metadata.get("mcly_texture_names", []) or [])
                if _dataset_texture_key(path)
            ]

            per_tile["mddf_count"] = np.asarray([n_mddf], dtype=np.int32)
            per_tile["modf_count"] = np.asarray([n_modf], dtype=np.int32)
            per_tile["mcly_tileset_ids"] = mcly_tileset_ids

            for row in tile_mddf:
                self._placement_rows.append({
                    "tile_id": tile_id,
                    "instance_type": "mddf",
                    "instance_idx": row.get("instance_idx", -1),
                    "asset_path": _normalize_asset_path(row["asset_path"]),
                    "nameId": row["nameId"],
                    "uniqueId": row["uniqueId"],
                    "posX": row["posX"],
                    "posY": row["posY"],
                    "posZ": row["posZ"],
                    "rotX": row["rotX"],
                    "rotY": row["rotY"],
                    "rotZ": row["rotZ"],
                    "scale": row["scale"],
                    "bbMinX": row["bbMinX"],
                    "bbMinY": row["bbMinY"],
                    "bbMinZ": row["bbMinZ"],
                    "bbMaxX": row["bbMaxX"],
                    "bbMaxY": row["bbMaxY"],
                    "bbMaxZ": row["bbMaxZ"],
                })
            for row in tile_modf:
                self._placement_rows.append({
                    "tile_id": tile_id,
                    "instance_type": "modf",
                    "instance_idx": row.get("instance_idx", -1),
                    "asset_path": _normalize_asset_path(row["asset_path"]),
                    "nameId": row["nameId"],
                    "uniqueId": row["uniqueId"],
                    "posX": row["posX"],
                    "posY": row["posY"],
                    "posZ": row["posZ"],
                    "rotX": row["rotX"],
                    "rotY": row["rotY"],
                    "rotZ": row["rotZ"],
                    "scale": 1.0,
                    "bbMinX": row["bbMinX"],
                    "bbMinY": row["bbMinY"],
                    "bbMinZ": row["bbMinZ"],
                    "bbMaxX": row["bbMaxX"],
                    "bbMaxY": row["bbMaxY"],
                    "bbMaxZ": row["bbMaxZ"],
                })

            self._tile_records.append({
                "tile_id": tile_id,
                "build": build_key,
                "map": map_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "mtex_texture_paths": mtex_paths,
                "placement_mddf_asset_paths": mddf_asset_paths,
                "placement_modf_asset_paths": modf_asset_paths,
            })

    def add_model(
        self,
        model_path: str,
        payload: dict[str, np.ndarray],
        *,
        load_error: int = 0,
        texture_paths: tuple[str, ...] = (),
        material_texture_paths: tuple[str, ...] = (),
        doodad_set_paths: tuple[str, ...] = (),
        source_path: str | None = None,
        source_in_listfile: int | bool = 0,
    ) -> None:
        model_path = _normalize_asset_path(model_path)
        if model_path in self._models:
            return
        normalized_source = _normalize_asset_path(source_path or model_path)
        self._models[model_path] = {
            "payload": payload,
            "load_error": int(load_error),
            "texture_paths": tuple(_normalize_asset_path(path) for path in texture_paths if _normalize_asset_path(path)),
            "material_texture_paths": tuple(_normalize_asset_path(path) for path in material_texture_paths if _normalize_asset_path(path)),
            "doodad_set_paths": tuple(_normalize_asset_path(path) for path in doodad_set_paths if _normalize_asset_path(path)),
            "source_path": normalized_source,
            "source_in_listfile": int(source_in_listfile),
            "source_kind": _source_kind(source_in_listfile),
        }

    def add_tileset(
        self,
        tileset_path: str,
        payload: dict[str, np.ndarray],
        *,
        load_error: int = 0,
        source_path: str | None = None,
        source_in_listfile: int | bool = 0,
    ) -> None:
        dataset_path = _dataset_texture_key(tileset_path)
        normalized_source = _normalize_asset_path(source_path or tileset_path)
        if dataset_path in self._tilesets:
            return
        self._tilesets[dataset_path] = {
            "payload": payload,
            "load_error": int(load_error),
            "source_path": normalized_source,
            "source_in_listfile": int(source_in_listfile),
            "source_kind": _source_kind(source_in_listfile),
        }

    # ── Enrichment stream ingestion ────────────────────────────
    def _ingest_enrichment_stream(self, enrichment_path: str | Path) -> None:
        """Read the binary enrichment stream produced by
        ``WowViewer.Tool.V22Enrich`` and populate ``_models`` and ``_tilesets``.

        Stream format (from ``EnrichmentStreamWriter``):
            HEADER: "V22E" + version uint32
            ENTRIES: "ENTRY" + path_len + path_utf8 + kind + load_error +
                     array_count + array data × count
            TERMINATOR: "ENDS"

        Each entry's arrays are flattened bytes. The Python side reconstructs
        them into the format expected by ``_write_models`` / ``_write_tilesets``.
        """
        import struct

        path = Path(enrichment_path)
        if not path.exists() or path.stat().st_size < 8:
            return  # Empty or missing enrichment stream — skip

        data = path.read_bytes()
        offset = 0

        # ── Read header ────────────────────────────────────────────
        magic = data[offset:offset + 4]
        if magic != b"V22E":
            return  # Not a valid enrichment stream — skip
        offset += 4 + 4  # magic + version uint32 (skip version for now)

        # ── Read entries ───────────────────────────────────────────
        while offset + 4 <= len(data):
            entry_magic = data[offset:offset + 4]
            offset += 4

            if entry_magic == b"ENDS":
                break
            if offset >= len(data):
                break
            entry_magic += data[offset:offset + 1]
            offset += 1
            if entry_magic != b"ENTRY":
                break

            # Path
            path_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
            entry_path = _normalize_asset_path(data[offset:offset + path_len].decode("utf-8")); offset += path_len

            # Kind + load_error
            kind_byte = data[offset]; offset += 1
            load_error = data[offset]; offset += 1

            # Array count
            array_count = struct.unpack_from("<I", data, offset)[0]; offset += 4

            arrays: dict[str, np.ndarray] = {}
            for _ in range(array_count):
                # Name
                name_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
                array_name = data[offset:offset + name_len].decode("utf-8"); offset += name_len

                # Ndim + shape
                ndim = struct.unpack_from("<I", data, offset)[0]; offset += 4
                shape = struct.unpack_from(f"<{ndim}I", data, offset); offset += ndim * 4

                # Dtype
                dtype_str = data[offset:offset + 8].rstrip(b"\x00").decode("ascii"); offset += 8

                # Data
                data_len = struct.unpack_from("<q", data, offset)[0]; offset += 8
                raw = data[offset:offset + data_len]; offset += data_len

                dt = _parse_dtype(dtype_str)
                arr = np.frombuffer(raw, dtype=dt).reshape(shape)
                arrays[array_name] = arr

            source_in_listfile_arr = arrays.pop("source_in_listfile", None)
            source_in_listfile = 0
            if source_in_listfile_arr is not None and np.asarray(source_in_listfile_arr).size > 0:
                source_in_listfile = int(np.asarray(source_in_listfile_arr).reshape(-1)[0])

            texture_paths = _decode_string_table_blob(arrays.pop("texture_paths", None))
            material_texture_paths = _decode_string_table_blob(arrays.pop("material_texture_paths", None))
            doodad_set_paths = _decode_string_table_blob(arrays.pop("doodad_set_paths", None))

            # Determine payload kind
            if kind_byte == 1:  # M2
                kind_code = np.asarray([1], dtype=np.uint8)
                self.add_model(
                    entry_path,
                    {"kind": kind_code, **arrays},
                    load_error=int(load_error),
                    texture_paths=texture_paths,
                    source_path=entry_path,
                    source_in_listfile=source_in_listfile,
                )
            elif kind_byte == 2:  # WMO
                kind_code = np.asarray([2], dtype=np.uint8)
                self.add_model(
                    entry_path,
                    {"kind": kind_code, **arrays},
                    load_error=int(load_error),
                    texture_paths=texture_paths,
                    material_texture_paths=material_texture_paths,
                    doodad_set_paths=doodad_set_paths,
                    source_path=entry_path,
                    source_in_listfile=source_in_listfile,
                )
            elif kind_byte == 3:  # BLP (tileset)
                rgb_arr = arrays.get("texture_rgb")
                shape_arr = arrays.get("texture_shape")
                payload = {
                    "texture_rgb": rgb_arr if rgb_arr is not None else np.zeros((0, 0, 3), dtype=np.uint8),
                    "texture_shape": shape_arr if shape_arr is not None else np.asarray([0, 0], dtype=np.int32),
                }
                self.add_tileset(
                    entry_path,
                    payload,
                    load_error=int(load_error),
                    source_path=entry_path,
                    source_in_listfile=source_in_listfile,
                )

        # ── Ensure all loaded entries are tracked ─────────────────
        # The stream may contain entries with kind=0 (unknown) which we skip.

    # -------------------------- finalise on disk --------------------------
    def finalize(self) -> Path:
        store = zarr.storage.LocalStore(str(self.store_path), read_only=False)
        root = zarr.group(store=store, attributes={"v22_dataset_version": V22_DATASET_VERSION})

        n_tiles = len(self._tile_records)
        index_rows: list[dict[str, Any]] = []

        for spec in V22_PER_TILE_SPECS:
            if spec.shape == (1,):
                arr = np.zeros((n_tiles, *spec.shape), dtype=spec.dtype)
            else:
                arr = np.zeros((n_tiles, *spec.shape), dtype=spec.dtype)
            for i, row in enumerate(self._tile_rows):
                arr[i] = row[spec.name]
            value_chunk = _resolve_chunk(spec.name, arr.shape[1:] or (1,))
            tile_chunk = max(1, min(arr.shape[0], 64)) if arr.shape[0] else 1
            root.create_array(
                spec.name,
                data=arr,
                chunks=(tile_chunk, *value_chunk),
                compressors=self._codec,
            )

        # Flat placement arrays.
        for name in (
            "mddf_placement_data",
            "modf_placement_data",
            "mddf_unique_ids",
            "modf_unique_ids",
            "mddf_model_ids",
            "modf_model_ids",
        ):
            spec = next(s for s in V22_FLAT_SPECS if s.name == name)
            if name == "mddf_placement_data":
                data = self._mddf_data
            elif name == "modf_placement_data":
                data = self._modf_data
            elif name == "mddf_unique_ids":
                data = self._mddf_uids
            elif name == "modf_unique_ids":
                data = self._modf_uids
            elif name == "mddf_model_ids":
                data = self._mddf_model_ids
            else:
                data = self._modf_model_ids
            if not data:
                arr = np.zeros(spec.shape, dtype=spec.dtype)
            else:
                arr = np.concatenate(data, axis=0).astype(spec.dtype, copy=False)
            self._write_flat(root, name, arr, spec)

        # Per-tile offsets.
        for name, offsets in (("mddf_placement_offset", self._mddf_offsets), ("modf_placement_offset", self._modf_offsets)):
            arr = np.asarray(offsets, dtype=np.int64)
            self._write_flat(root, name, arr, _ArraySpec(name, np.int64, arr.shape))

        # Audit metadata.
        root.attrs["tile_count"] = n_tiles
        root.attrs["builds"] = sorted({r["build"] for r in self._tile_records})
        if self._tile_records:
            root.attrs["scoped_builds"] = list(V22_BUILD_IDS)
        if self._source_v18_store:
            root.attrs["source_v18_store"] = self._source_v18_store

        # Index report.
        for r in self._tile_records:
            index_rows.append(
                {
                    "tile_id": int(r["tile_id"]),
                    "build": r["build"],
                    "map": str(r.get("map", "")),
                    "tile_x": int(r["tile_x"]),
                    "tile_y": int(r["tile_y"]),
                    "mtex_texture_paths": list(r.get("mtex_texture_paths", [])),
                    "placement_mddf_asset_paths": list(r.get("placement_mddf_asset_paths", [])),
                    "placement_modf_asset_paths": list(r.get("placement_modf_asset_paths", [])),
                }
            )
        root.attrs["tile_index"] = index_rows

        # Model library.
        self._write_models(root)
        # Tileset library.
        self._write_tilesets(root)
        # Audit sidecars.
        self._write_sidecars(index_rows)

        return self.store_path

    def _write_flat(self, root: zarr.Group, name: str, arr: np.ndarray, spec: _ArraySpec) -> None:
        chunk = _resolve_chunk(name, arr.shape)
        if arr.shape[0] == 0:
            chunk = (1,) + chunk[1:]
        root.create_array(name, data=arr, chunks=chunk, compressors=self._codec)

    def _write_models(self, root: zarr.Group) -> None:
        if not self._models:
            return
        group = root.create_group(
            V22_MODELS_GROUP,
            attributes={
                "kind": "per-build-model-library",
                "payload_mode": "embedded" if self._embed_asset_payloads else "paths_only",
            },
        )
        paths = sorted(self._models.keys())
        max_len = max(len(path) for path in paths)
        group.create_array("model_paths", data=np.asarray(paths, dtype=f"<U{max_len}"), chunks=(len(paths),))
        kinds = np.asarray(
            [int(np.asarray(self._models[p]["payload"].get("kind", 0)).reshape(-1)[0]) for p in paths],
            dtype=np.uint8,
        )
        group.create_array("model_kind", data=kinds, chunks=(len(paths),), compressors=self._codec)
        errors = np.asarray([int(self._models[p]["load_error"]) for p in paths], dtype=np.uint8)
        group.create_array("load_error", data=errors, chunks=(len(paths),), compressors=self._codec)
        if not self._embed_asset_payloads:
            return
        for p in paths:
            kind_code = int(np.asarray(self._models[p]["payload"].get("kind", 0)).reshape(-1)[0])
            kind_name = {1: "m2", 2: "wmo"}.get(kind_code, "model")
            self._write_asset_entry(group, kind_name, p, self._models[p]["payload"])

    def _write_tilesets(self, root: zarr.Group) -> None:
        if not self._tilesets:
            return
        group = root.create_group(
            V22_TILESETS_GROUP,
            attributes={
                "kind": "per-build-tileset-library",
                "payload_mode": "embedded" if self._embed_asset_payloads else "paths_only",
            },
        )
        paths = sorted(self._tilesets.keys())
        max_len = max(len(path) for path in paths)
        group.create_array("tileset_paths", data=np.asarray(paths, dtype=f"<U{max_len}"), chunks=(len(paths),))
        errors = np.asarray([int(self._tilesets[p]["load_error"]) for p in paths], dtype=np.uint8)
        group.create_array("load_error", data=errors, chunks=(len(paths),), compressors=self._codec)
        shapes = np.asarray(
            [tuple(int(s) for s in self._tilesets[p]["payload"].get("texture_shape", (0, 0))) for p in paths],
            dtype=np.int32,
        )
        group.create_array("texture_shape", data=shapes, chunks=(len(paths), 2), compressors=self._codec)
        if not self._embed_asset_payloads:
            return
        for p in paths:
            self._write_asset_entry(group, "tileset", p, self._tilesets[p]["payload"])

    def _write_asset_entry(self, root: zarr.Group, kind: str, model_path: str, payload: dict[str, np.ndarray]) -> None:
        if not payload:
            return
        entry = root.create_group(model_path.replace(".", "_").replace("\\", "/"), attributes={"kind": kind, "model_path": model_path})
        if model_path in self._models:
            model_record = self._models[model_path]
            self._write_string_values(entry, "texture_paths", model_record.get("texture_paths", ()))
            self._write_string_values(entry, "material_texture_paths", model_record.get("material_texture_paths", ()))
            self._write_string_values(entry, "doodad_set_paths", model_record.get("doodad_set_paths", ()))
        for key, arr in payload.items():
            if key in {"kind", "texture_shape"}:
                continue
            arr = np.asarray(arr)
            entry.create_array(
                key,
                data=arr,
                chunks=_resolve_chunk(key, arr.shape),
                compressors=self._codec,
            )

    def _write_string_values(self, root: zarr.Group, name: str, values: Iterable[str]) -> None:
        normalized = [value for value in values if value]
        if not normalized:
            return
        max_len = max(len(value) for value in normalized)
        root.create_array(
            name,
            data=np.asarray(normalized, dtype=f"<U{max_len}"),
            chunks=(len(normalized),),
        )

    def _write_sidecars(self, index_rows: list[dict[str, Any]]) -> None:
        self._write_index_sidecar(index_rows)
        self._write_placements_sidecar()
        self._write_asset_inventory_sidecar()
        self._write_finalization_json(index_rows)

    def _write_index_sidecar(self, index_rows: list[dict[str, Any]]) -> None:
        if index_rows:
            table = pa.Table.from_pylist(index_rows)
        else:
            table = pa.Table.from_pydict({
                "tile_id": pa.array([], type=pa.int64()),
                "build": pa.array([], type=pa.string()),
                "map": pa.array([], type=pa.string()),
                "tile_x": pa.array([], type=pa.int32()),
                "tile_y": pa.array([], type=pa.int32()),
                "mtex_texture_paths": pa.array([], type=pa.list_(pa.string())),
                "placement_mddf_asset_paths": pa.array([], type=pa.list_(pa.string())),
                "placement_modf_asset_paths": pa.array([], type=pa.list_(pa.string())),
            })
        pq.write_table(table, str(self.store_path / "index.parquet"))

    def _write_placements_sidecar(self) -> None:
        if self._placement_rows:
            table = pa.Table.from_pylist(self._placement_rows)
        else:
            table = pa.Table.from_pydict({
                "tile_id": pa.array([], type=pa.int64()),
                "instance_type": pa.array([], type=pa.string()),
                "instance_idx": pa.array([], type=pa.int32()),
                "asset_path": pa.array([], type=pa.string()),
                "nameId": pa.array([], type=pa.int32()),
                "uniqueId": pa.array([], type=pa.int32()),
                "posX": pa.array([], type=pa.float64()),
                "posY": pa.array([], type=pa.float64()),
                "posZ": pa.array([], type=pa.float64()),
                "rotX": pa.array([], type=pa.float64()),
                "rotY": pa.array([], type=pa.float64()),
                "rotZ": pa.array([], type=pa.float64()),
                "scale": pa.array([], type=pa.float64()),
                "bbMinX": pa.array([], type=pa.float64()),
                "bbMinY": pa.array([], type=pa.float64()),
                "bbMinZ": pa.array([], type=pa.float64()),
                "bbMaxX": pa.array([], type=pa.float64()),
                "bbMaxY": pa.array([], type=pa.float64()),
                "bbMaxZ": pa.array([], type=pa.float64()),
            })
        pq.write_table(table, str(self.store_path / "placements.parquet"))

    def _write_asset_inventory_sidecar(self) -> None:
        rows: list[dict[str, Any]] = []
        for path, record in sorted(self._models.items(), key=lambda item: item[0].casefold()):
            kind_code = int(np.asarray(record["payload"].get("kind", 0)).reshape(-1)[0]) if record["payload"] else 0
            rows.append({
                "asset_path": path,
                "source_path": str(record.get("source_path", path)),
                "source_in_listfile": int(record.get("source_in_listfile", 0)),
                "source_kind": str(record.get("source_kind", "archive_unlisted")),
                "kind": "m2" if kind_code == 1 else "wmo" if kind_code == 2 else "unknown",
                "load_error": int(record["load_error"]),
            })
        for path, record in sorted(self._tilesets.items(), key=lambda item: item[0].casefold()):
            rows.append({
                "asset_path": path,
                "source_path": str(record.get("source_path", "")),
                "source_in_listfile": int(record.get("source_in_listfile", 0)),
                "source_kind": str(record.get("source_kind", "archive_unlisted")),
                "kind": "texture_rgb",
                "load_error": int(record["load_error"]),
            })
        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            table = pa.Table.from_pydict({
                "asset_path": pa.array([], type=pa.string()),
                "source_path": pa.array([], type=pa.string()),
                "source_in_listfile": pa.array([], type=pa.int32()),
                "source_kind": pa.array([], type=pa.string()),
                "kind": pa.array([], type=pa.string()),
                "load_error": pa.array([], type=pa.int32()),
            })
        pq.write_table(table, str(self.store_path / "asset_inventory.parquet"))

    def _write_finalization_json(self, index_rows: list[dict[str, Any]]) -> None:
        payload = {
            "tile_count": len(index_rows),
            "builds": sorted({row["build"] for row in index_rows}),
            "root_arrays": [spec.name for spec in V22_PER_TILE_SPECS],
            "flat_arrays": [spec.name for spec in V22_FLAT_SPECS],
            "model_count": len(self._models),
            "tileset_count": len(self._tilesets),
            "source_v18_store": self._source_v18_store,
            "asset_payload_mode": "embedded" if self._embed_asset_payloads else "paths_only",
            "missing_components": [],
        }
        (self.store_path / "finalization.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _rmtree(path: Path) -> None:
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_dir():
                V22ZarrWriter._rmtree(child)
            else:
                child.unlink()
        path.rmdir()


def complete_existing_store_from_enrichment(
    store_path: str | Path,
    v18_store_path: str | Path,
    enrichment_path: str | Path,
    *,
    backup_suffix: str = ".partial-backup",
    embed_asset_payloads: bool = True,
) -> Path:
    """Finish a partially-written V22 store using only the enrichment stream.

    This is the recovery path for builds that already finished the expensive
    per-tile root arrays but timed out while writing `models/`, `tilesets/`,
    or the audit sidecars.
    """
    store_path = Path(store_path)
    v18_store_path = Path(v18_store_path)
    enrichment_path = Path(enrichment_path)

    if not store_path.exists():
        raise FileNotFoundError(store_path)
    if not v18_store_path.exists():
        raise FileNotFoundError(v18_store_path)
    if not enrichment_path.exists():
        raise FileNotFoundError(enrichment_path)

    writer = V22ZarrWriter(store_path, overwrite=False, embed_asset_payloads=embed_asset_payloads)
    writer._source_v18_store = str(v18_store_path)
    writer._ingest_enrichment_stream(enrichment_path)

    store = zarr.storage.LocalStore(str(store_path), read_only=False)
    root = zarr.open_group(store=store, mode="a")
    index_rows = list(root.attrs.get("tile_index", []))

    for group_name in ("models", "tilesets"):
        group_path = store_path / group_name
        if group_path.exists():
            backup_path = store_path.parent / f"{store_path.name}.{group_name}{backup_suffix}"
            if backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
            shutil.move(str(group_path), str(backup_path))

    writer._write_models(root)
    writer._write_tilesets(root)
    writer._write_index_sidecar(index_rows)
    shutil.copy2(v18_store_path / "placements.parquet", store_path / "placements.parquet")
    writer._write_asset_inventory_sidecar()
    writer._write_finalization_json(index_rows)

    return store_path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class V22Dataset:
    """Fixed-key V22 Zarr reader.

    ``__getitem__`` returns the same dict keys for every tile. Missing arrays
    are returned with their documented shape and dtype, filled with zeros
    (signal absence is a real input, not a missing key).
    """

    def __init__(self, store_path: str | Path, *, available_signals: Iterable[str] | None = None) -> None:
        self.store_path = Path(store_path)
        store = zarr.storage.LocalStore(str(self.store_path), read_only=True)
        self.root = zarr.open_group(store, mode="r")
        self.tile_index: list[dict[str, Any]] = list(self.root.attrs.get("tile_index", []))
        self._available = set(available_signals) if available_signals is not None else None

    def __len__(self) -> int:
        if self.tile_index:
            return len(self.tile_index)
        if "tile_index" in self.root.attrs:
            return len(self.root.attrs["tile_index"])
        return int(self.root["height_257"].shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = self.tile_index[index] if self.tile_index else {}
        out: dict[str, np.ndarray] = {}
        for spec in V22_PER_TILE_SPECS:
            if spec.name in {"mddf_count", "modf_count"}:
                out[spec.name] = self._read_per_tile_scalar(spec, index)
            else:
                out[spec.name] = self._read_per_tile(spec, index)
        out.update(self._read_flat_for_tile(index, "mddf"))
        out.update(self._read_flat_for_tile(index, "modf"))
        if row:
            out["tile_id"] = np.asarray(int(row.get("tile_id", index)), dtype=np.int64)
            out["build"] = row.get("build", "")
            out["map"] = row.get("map", "")
            out["tile_x"] = np.asarray(int(row.get("tile_x", 0)), dtype=np.int32)
            out["tile_y"] = np.asarray(int(row.get("tile_y", 0)), dtype=np.int32)
            out["mtex_texture_paths"] = list(row.get("mtex_texture_paths", []))
            out["placement_mddf_asset_paths"] = list(row.get("placement_mddf_asset_paths", []))
            out["placement_modf_asset_paths"] = list(row.get("placement_modf_asset_paths", []))
        return out

    # -------------------------- helpers --------------------------
    def _read_per_tile(self, spec: _ArraySpec, index: int) -> np.ndarray:
        if spec.name not in self.root:
            return np.zeros(spec.shape, dtype=spec.dtype)
        return np.asarray(self.root[spec.name][index], dtype=spec.dtype)

    def _read_per_tile_scalar(self, spec: _ArraySpec, index: int) -> np.ndarray:
        if spec.name not in self.root:
            return np.zeros(spec.shape, dtype=spec.dtype)
        return np.asarray(self.root[spec.name][index], dtype=spec.dtype)

    def _read_flat_for_tile(self, index: int, kind: str) -> dict[str, np.ndarray]:
        prefix = kind  # "mddf" or "modf"
        out: dict[str, np.ndarray] = {}
        offset_name = f"{prefix}_placement_offset"
        count_name = f"{prefix}_count"
        if offset_name not in self.root:
            for spec in V22_FLAT_SPECS:
                if spec.name.startswith(prefix + "_"):
                    out[spec.name] = np.zeros((0, *spec.shape[1:]), dtype=spec.dtype)
            return out
        offsets = self.root[offset_name][:]
        cur = int(offsets[index])
        if count_name in self.root:
            length = int(np.asarray(self.root[count_name][index]).reshape(-1)[0])
        else:
            nxt = int(offsets[index + 1]) if index + 1 < offsets.shape[0] else int(offsets[-1])
            length = max(0, nxt - cur)
        for spec in V22_FLAT_SPECS:
            if not spec.name.startswith(prefix + "_"):
                continue
            if length == 0 or spec.name not in self.root:
                out[spec.name] = np.zeros((0, *spec.shape[1:]), dtype=spec.dtype)
                continue
            data = self.root[spec.name][cur:cur + length]
            out[spec.name] = np.asarray(data, dtype=spec.dtype)
        return out

    def tile_ids(self) -> np.ndarray:
        if self.tile_index:
            return np.asarray([int(r.get("tile_id", i)) for i, r in enumerate(self.tile_index)], dtype=np.int64)
        return np.arange(len(self), dtype=np.int64)


__all__ = [
    "V22_DATASET_VERSION",
    "V22_BUILD_IDS",
    "V22_ROOT_ARRAYS",
    "V22_METADATA_KEYS",
    "V22_MODELS_GROUP",
    "V22_TILESETS_GROUP",
    "V22_PER_TILE_SPECS",
    "V22_FLAT_SPECS",
    "V22ZarrWriter",
    "V22Dataset",
    "complete_existing_store_from_enrichment",
    "DEFAULT_CODEC",
]
