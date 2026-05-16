"""Zarr I/O for shard tensors — chunked, compressed, metadata-rich.

Zarr v3 stores each array as an independent chunked dataset with configurable
compression.  This module provides round-trip read/write for NPZ-compatible
shard tensors using ``zarr.storage.LocalStore`` (directory store).

Shard layout (Zarr v3 LocalStore):

    tile.zarr/
        zarr.json                          # group metadata
        height_257/
            zarr.json                      # array metadata (v3)
            c/0/0  c/0/1  ...             # chunk binary blobs
        minimap_rgb_256/ ...

Default compression: zarr native BloscCodec (zstd level 5, bitshuffle).
String/binary metadata (e.g. ``metadata.json``) is stored in group attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zarr.codecs
import zarr.storage

# ---------------------------------------------------------------------------
# Default Zarr encoding config
# ---------------------------------------------------------------------------
DEFAULT_CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")

# Chunk shape per common key — tuned for 256×256 tiles.
_CHUNK_PRESETS: dict[str, tuple[int, ...]] = {
    "height_257": (129, 129),
    "height_65": (65, 65),
    "height_17": (17, 17),
    "mcnr_normal_xyz": (129, 129, 3),
    "mclq_surface_height": (129, 129),
    "mclq_type_mask": (16, 16),
    "mcly_texture_ids": (16, 16, 4),
    "mcly_layer_mask": (16, 16, 4),
    "mcal_alpha_pack": (128, 128, 4),
    "mcal_alpha_pack_256": (128, 128, 4),
    "mcsh_shadow_mask_256": (128, 128),
    "shadow_residual_mask_256": (128, 128),
    "minimap_rgb_256": (128, 128, 3),
    "hole_mask_16": (16, 16),
    "object_mask_257": (129, 129),
    "object_precise_mask_257": (129, 129),
    "mcmt_material_ids": (16, 16, 4),
}

# Keys whose values are stored as group-level UTF-8 string attributes.
_ATTR_KEYS = frozenset({"metadata.json"})


def _resolve_chunks(key: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return a chunk shape for *key* no larger than *shape*."""
    preset = _CHUNK_PRESETS.get(key, None)
    if preset is not None:
        return tuple(min(c, s) for c, s in zip(preset, shape, strict=False))
    if len(shape) == 1:
        return shape
    if len(shape) == 2:
        return (min(64, shape[0]), min(64, shape[1]))
    return (min(64, shape[0]), min(64, shape[1])) + shape[2:]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def write_zarr_shard(
    zarr_path: str | Path,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
    codec: Any = DEFAULT_CODEC,
    overwrite: bool = False,
) -> Path:
    """Write a collection of named numpy arrays to a Zarr LocalStore (v3).

    Args:
        zarr_path: Output directory (e.g. ``tile.zarr``).
        arrays: Mapping of key → numpy array.
        attrs: Optional dict serialised as group attributes (JSON-safe).
        codec: Zarr codec (default BloscCodec zstd bitshuffle).
        overwrite: If True, remove existing store before writing.

    Returns:
        Path to the written Zarr store.
    """
    zarr_path = Path(zarr_path)
    if overwrite and zarr_path.exists():
        _rmtree(zarr_path)

    store = zarr.storage.LocalStore(str(zarr_path), read_only=False)
    root = zarr.group(store=store)

    for key, arr in sorted(arrays.items()):
        if key in _ATTR_KEYS:
            continue
        chunks = _resolve_chunks(key, arr.shape)
        root.create_array(
            key,
            data=arr,
            chunks=chunks,
            compressors=codec,
        )

    # Persist string/binary metadata in group attributes
    if attrs:
        root.attrs.update(dict(attrs.items()))

    for key in sorted(arrays):
        if key in _ATTR_KEYS:
            root.attrs[key] = _bytes_to_str(arrays[key])

    return zarr_path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def read_zarr_shard(
    zarr_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a Zarr shard directory, returning (arrays, attrs).

    Keys in *_ATTR_KEYS* (e.g. ``metadata.json``) are returned in *attrs*
    rather than in the *arrays* dict.
    """
    zarr_path = Path(zarr_path)
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store, mode="r")

    arrays: dict[str, np.ndarray] = {}
    for key in sorted(root.array_keys()):
        arrays[key] = root[key][:]

    attrs: dict[str, Any] = dict(root.attrs)
    for key in _ATTR_KEYS:
        if key in attrs:
            val = attrs.pop(key)
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            attrs[key] = val

    return arrays, attrs


def read_zarr_array(zarr_path: str | Path, key: str) -> np.ndarray:
    """Read a single named array from a Zarr store."""
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    return root[key][:]


# ---------------------------------------------------------------------------
# NPZ → Zarr conversion
# ---------------------------------------------------------------------------
def convert_npz_to_zarr(
    npz_path: str | Path,
    zarr_path: str | Path,
    codec: Any = DEFAULT_CODEC,
    overwrite: bool = False,
) -> Path:
    """Convert a single NPZ shard to a Zarr LocalStore (v3).

    Returns the path to the Zarr store.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays: dict[str, np.ndarray] = {}
        raw_attrs: dict[str, Any] = {}
        for key in data.files:
            val = data[key]
            if key in _ATTR_KEYS:
                raw_attrs[key] = _bytes_to_str(val)
            elif isinstance(val, np.ndarray):
                arrays[key] = val

    return write_zarr_shard(
        zarr_path,
        arrays,
        attrs=raw_attrs,
        codec=codec,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bytes_to_str(val: Any) -> str | Any:
    """Decode bytes to UTF-8 string if possible."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray) and val.dtype.kind == "S":
        return val.item().decode("utf-8") if val.ndim == 0 else str(val)
    return val


def _rmtree(path: Path) -> None:
    """Recursively remove a directory tree."""
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink()
    path.rmdir()
