"""V22 Zarr writer and reader.

The C# harvester pre-decodes every V22 tile signal into the binary V22 stream
(``RawArraySerializer.StreamProfile.V22``). This module is the canonical
Python-side writer/reader contract for the Zarr dataset.

The writer consumes parsed tile records (one record per tile) and writes the
canonical V22 Zarr store. The reader loads tiles from that store with the
fixed-key contract required by downstream consumers.

No game client reparse, no Python-side patch derivation. The decoded payloads
arrive from the C# stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
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
# Tile record contract — what the C# stream yields per tile.
# ---------------------------------------------------------------------------
@dataclass
class V22TileRecord:
    """One decoded tile coming from the C# V22 stream."""

    tile_id: int
    build: str
    map: str
    tile_x: int
    tile_y: int

    per_tile: dict[str, np.ndarray] = field(default_factory=dict)
    placement_mddf: np.ndarray | None = None  # (n, 9) float32
    placement_modf: np.ndarray | None = None  # (n, 17) float32
    mddf_asset_paths: tuple[str, ...] = field(default_factory=tuple)
    modf_asset_paths: tuple[str, ...] = field(default_factory=tuple)
    mtex_texture_paths: tuple[str, ...] = field(default_factory=tuple)


def empty_tile(shape_template: dict[str, tuple[int, ...]] | None = None) -> dict[str, np.ndarray]:
    """Return fill arrays for every V22 per-tile spec."""
    out: dict[str, np.ndarray] = {}
    for spec in V22_PER_TILE_SPECS:
        if spec.shape == (1,):
            out[spec.name] = np.zeros(spec.shape, dtype=spec.dtype)
        else:
            out[spec.name] = np.zeros(spec.shape, dtype=spec.dtype)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
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
    ) -> None:
        self.store_path = Path(store_path)
        if overwrite and self.store_path.exists():
            self._rmtree(self.store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._tile_records: list[V22TileRecord] = []
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
        self._codec = codec

    # -------------------------- record ingestion --------------------------
    def add_tile(self, record: V22TileRecord) -> None:
        per_tile: dict[str, np.ndarray] = {}
        for spec in V22_PER_TILE_SPECS:
            arr = record.per_tile.get(spec.name)
            if arr is None:
                arr = np.zeros(spec.shape, dtype=spec.dtype)
            else:
                arr = np.asarray(arr, dtype=spec.dtype)
            if arr.shape != spec.shape:
                raise ValueError(
                    f"V22 tile {record.tile_id} array {spec.name} shape {arr.shape} != spec {spec.shape}"
                )
            per_tile[spec.name] = arr

        self._tile_records.append(record)
        self._tile_rows.append(per_tile)

        mddf = record.placement_mddf
        modf = record.placement_modf
        n_mddf = int(mddf.shape[0]) if mddf is not None else 0
        n_modf = int(modf.shape[0]) if modf is not None else 0

        self._mddf_offsets.append(sum(int(a.shape[0]) for a in self._mddf_data))
        self._modf_offsets.append(sum(int(a.shape[0]) for a in self._modf_data))

        if n_mddf > 0:
            mddf = np.asarray(mddf, dtype=np.float32)
            self._mddf_data.append(mddf)
            self._mddf_uids.append(np.rint(mddf[:, 1]).astype(np.int32))
            self._mddf_model_ids.append(np.rint(mddf[:, 0]).astype(np.int32))
        if n_modf > 0:
            modf = np.asarray(modf, dtype=np.float32)
            self._modf_data.append(modf)
            self._modf_uids.append(np.rint(modf[:, 1]).astype(np.int32))
            self._modf_model_ids.append(np.rint(modf[:, 0]).astype(np.int32))

    def add_model(self, model_path: str, payload: dict[str, np.ndarray], *, load_error: int = 0) -> None:
        if model_path in self._models:
            return
        self._models[model_path] = {"payload": payload, "load_error": int(load_error)}

    def add_tileset(self, tileset_path: str, payload: dict[str, np.ndarray], *, load_error: int = 0) -> None:
        if tileset_path in self._tilesets:
            return
        self._tilesets[tileset_path] = {"payload": payload, "load_error": int(load_error)}

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
            chunk = _resolve_chunk(spec.name, arr.shape[1:] or (1,))
            root.create_array(spec.name, data=arr, chunks=(min(chunk[0], max(arr.shape[0], 1)),) + chunk[1:] if arr.shape[0] else chunk, compressors=self._codec)

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
        root.attrs["builds"] = sorted({r.build for r in self._tile_records})
        if self._tile_records:
            root.attrs["scoped_builds"] = list(V22_BUILD_IDS)

        # Index report.
        for r in self._tile_records:
            index_rows.append(
                {
                    "tile_id": int(r.tile_id),
                    "build": r.build,
                    "map": r.map,
                    "tile_x": int(r.tile_x),
                    "tile_y": int(r.tile_y),
                    "mtex_texture_paths": list(r.mtex_texture_paths),
                    "placement_mddf_asset_paths": list(r.mddf_asset_paths),
                    "placement_modf_asset_paths": list(r.modf_asset_paths),
                }
            )
        root.attrs["tile_index"] = index_rows

        # Model library.
        self._write_models(root)
        # Tileset library.
        self._write_tilesets(root)

        return self.store_path

    def _write_flat(self, root: zarr.Group, name: str, arr: np.ndarray, spec: _ArraySpec) -> None:
        chunk = _resolve_chunk(name, arr.shape)
        if arr.shape[0] == 0:
            chunk = (1,) + chunk[1:]
        root.create_array(name, data=arr, chunks=chunk, compressors=self._codec)

    def _write_models(self, root: zarr.Group) -> None:
        if not self._models:
            return
        group = root.create_group(V22_MODELS_GROUP, attributes={"kind": "per-build-model-library"})
        paths = sorted(self._models.keys())
        group.create_array("model_paths", data=np.asarray(paths, dtype=object), object_codec=zarr.codecs.MsgPack(), chunks=(len(paths),))
        kinds = np.asarray([int(self._models[p]["payload"].get("kind", 0)) for p in paths], dtype=np.uint8)
        group.create_array("model_kind", data=kinds, chunks=(len(paths),), compressors=self._codec)
        errors = np.asarray([int(self._models[p]["load_error"]) for p in paths], dtype=np.uint8)
        group.create_array("load_error", data=errors, chunks=(len(paths),), compressors=self._codec)
        for p in paths:
            self._write_asset_entry(group, "m2", p, self._models[p]["payload"])
            self._write_asset_entry(group, "wmo", p, self._models[p]["payload"])

    def _write_tilesets(self, root: zarr.Group) -> None:
        if not self._tilesets:
            return
        group = root.create_group(V22_TILESETS_GROUP, attributes={"kind": "per-build-tileset-library"})
        paths = sorted(self._tilesets.keys())
        group.create_array("tileset_paths", data=np.asarray(paths, dtype=object), object_codec=zarr.codecs.MsgPack(), chunks=(len(paths),))
        errors = np.asarray([int(self._tilesets[p]["load_error"]) for p in paths], dtype=np.uint8)
        group.create_array("load_error", data=errors, chunks=(len(paths),), compressors=self._codec)
        shapes = np.asarray(
            [tuple(int(s) for s in self._tilesets[p]["payload"].get("texture_shape", (0, 0))) for p in paths],
            dtype=np.int32,
        )
        group.create_array("texture_shape", data=shapes, chunks=(len(paths), 2), compressors=self._codec)

    def _write_asset_entry(self, root: zarr.Group, kind: str, model_path: str, payload: dict[str, np.ndarray]) -> None:
        if not payload:
            return
        entry = root.create_group(model_path.replace(".", "_").replace("\\", "/"), attributes={"kind": kind, "model_path": model_path})
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
        if offset_name not in self.root:
            for spec in V22_FLAT_SPECS:
                if spec.name.startswith(prefix + "_"):
                    out[spec.name] = np.zeros((0, *spec.shape[1:]), dtype=spec.dtype)
            return out
        offsets = self.root[offset_name][:]
        nxt = int(offsets[index + 1]) if index + 1 < offsets.shape[0] else int(offsets[-1])
        cur = int(offsets[index])
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
    "V22TileRecord",
    "V22ZarrWriter",
    "V22Dataset",
    "empty_tile",
    "DEFAULT_CODEC",
]
