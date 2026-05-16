"""Zarr ZipStore reader/writer for NPZ shard conversion.

Each shard becomes a single .zarr.zip file containing chunked arrays with
blosc compression.  Arrays are read lazily — only touched chunks are
decompressed — and metadata survives in Zarr .attrs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import zarr
import zarr.storage
from zarr.codecs import BloscCodec

# Compression codec used for all arrays in the store.
_COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")

# Chunk shape for each array size category.  We bias toward power-of-two
# chunks that stay well under the blosc block size so random access is fast.
_CHUNK_PRESETS: dict[tuple[int, ...], tuple[int, ...]] = {
    (256, 256): (128, 128),
    (256, 256, 3): (64, 64, 3),
    (256, 256, 4): (64, 64, 4),
    (257, 257): (129, 129),
    (257, 257, 3): (65, 65, 3),
    (65, 65): (65, 65),
    (17, 17): (17, 17),
    (16, 16): (16, 16),
    (16, 16, 4): (16, 16, 4),
}


def _guess_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return a sensible chunk shape for the given array shape."""
    if shape in _CHUNK_PRESETS:
        return _CHUNK_PRESETS[shape]
    if len(shape) == 2:
        return (min(64, shape[0]), min(64, shape[1]))
    if len(shape) >= 3:
        return (min(64, shape[0]), min(64, shape[1]), *shape[2:])
    return shape


def _open_zipstore(path: str | Path, mode: str) -> zarr.storage.ZipStore:
    """Open a ZipStore with the _lock workaround for Zarr v3."""
    store = zarr.storage.ZipStore(str(path), mode=mode)
    store._lock = threading.RLock()
    return store


def npz_to_zarr_zipstore(
    npz_path: Path, zarr_path: Path | None = None, overwrite: bool = False
) -> Path:
    """Convert an .npz shard into a Zarr ZipStore at *zarr_path*.

    If *zarr_path* is None, the output is *npz_path* with ``.npz`` replaced
    by ``.zarr.zip``.
    """
    npz_path = Path(npz_path)
    if zarr_path is None:
        zarr_path = npz_path.with_suffix("").with_suffix(".zarr.zip")
    else:
        zarr_path = Path(zarr_path)

    if zarr_path.exists():
        if overwrite:
            zarr_path.unlink(missing_ok=True)
        else:
            return zarr_path

    with np.load(npz_path, allow_pickle=False) as data:
        store = _open_zipstore(zarr_path, "w")
        root = zarr.open_group(store=store, mode="w")

        for key in sorted(data.files):
            arr = data[key]
            if key == "metadata.json":
                meta_bytes = arr.tobytes() if hasattr(arr, "tobytes") else arr
                meta_str = meta_bytes.decode("utf-8") if isinstance(meta_bytes, bytes) else str(meta_bytes)
                root.attrs["metadata.json"] = meta_str
                continue

            shape = tuple(int(s) for s in arr.shape)
            chunks = _guess_chunks(shape)
            dtype = arr.dtype

            za = root.create_array(
                key,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressors=[_COMPRESSOR],
                fill_value=None,
            )
            za[:] = arr

        store.close()

    return zarr_path


class ZarrShardReader:
    """Lazy reader for a single Zarr ZipStore shard.

    Usage::

        reader = ZarrShardReader(path)
        minimap = reader["minimap_rgb_256"][:]  # read full array
        # or slice: alpha = reader["mcal_alpha_pack_256"][64:128, 64:128, :]
    """

    def __init__(self, zarr_path: str | Path) -> None:
        self._path = Path(zarr_path)
        self._store: zarr.storage.ZipStore | None = None
        self._root: zarr.Group | None = None

    def _ensure_open(self) -> zarr.Group:
        if self._root is None:
            self._store = _open_zipstore(self._path, "r")
            self._root = zarr.open_group(store=self._store, mode="r")
        return self._root

    def keys(self) -> list[str]:
        root = self._ensure_open()
        return list(root.keys())

    def metadata(self) -> dict | None:
        root = self._ensure_open()
        raw = root.attrs.get("metadata.json")
        if raw is None:
            return None
        return json.loads(raw) if isinstance(raw, str) else raw

    def __getitem__(self, key: str) -> zarr.Array:
        return self._ensure_open()[key]

    def read_array(self, key: str) -> np.ndarray:
        return self[key][:]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
            self._root = None

    def __enter__(self) -> ZarrShardReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
