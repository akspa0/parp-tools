"""Raw binary array reader — zero-compression replacement for NPZ.

Format per stream:
  [4 bytes: "ARRY" magic]
  [4 bytes: metadata JSON length (LE int32)]
  [N bytes: metadata JSON UTF-8]
  [for each array:]
    [4 bytes: name length (LE int32)]
    [N bytes: name UTF-8]
    [4 bytes: ndim (LE int32)]
    [4*ndim bytes: shape (LE int32 each)]
    [8 bytes: dtype ASCII null-padded]
    [8 bytes: data byte length (LE int64)]
    [N bytes: raw data]
  [4 bytes: "ENDS" magic]
  [4 bytes: 0x00000000]
"""

from __future__ import annotations

import json
import struct
from io import BytesIO

import numpy as np

ARRY_MAGIC = b"ARRY"
ENDS_MAGIC = b"ENDS"

_DTYPE_MAP: dict[bytes, np.dtype] = {
    b"<f4": np.dtype("<f4"),
    b"<f8": np.dtype("<f8"),
    b"<i4": np.dtype("<i4"),
    b"<u4": np.dtype("<u4"),
    b"<i2": np.dtype("<i2"),
    b"<u2": np.dtype("<u2"),
    b"|u1": np.dtype("uint8"),
    b"|i1": np.dtype("int8"),
    b"|b1": np.dtype("bool"),
}


def read_tile_blob(blob: bytes) -> dict[str, np.ndarray]:
    """Read one raw binary tile blob. Returns dict mapping array names to numpy arrays."""
    stream = BytesIO(blob)

    magic = stream.read(4)
    if magic != ARRY_MAGIC:
        raise ValueError(f"Expected ARRY magic, got {magic!r}")

    # Metadata JSON
    meta_len = struct.unpack("<I", stream.read(4))[0]
    meta_bytes = stream.read(meta_len)
    metadata = json.loads(meta_bytes.decode("utf-8"))

    result: dict[str, np.ndarray] = {}
    result["metadata.json"] = meta_bytes
    result["_metadata"] = metadata

    while stream.tell() < len(blob):
        pos = stream.tell()
        peek = stream.read(4)
        if not peek or len(peek) < 4:
            break
        if peek == ENDS_MAGIC:
            break
        stream.seek(pos)

        # Name
        name_len = struct.unpack("<I", stream.read(4))[0]
        name = stream.read(name_len).decode("utf-8")

        # Ndim
        ndim = struct.unpack("<I", stream.read(4))[0]

        # Shape
        shape = struct.unpack(f"<{ndim}I", stream.read(ndim * 4))

        # Dtype
        dtype_raw = stream.read(8).rstrip(b"\x00")
        dtype = _DTYPE_MAP.get(dtype_raw, np.dtype("float32"))

        # Data
        data_len = struct.unpack("<Q", stream.read(8))[0]
        data = stream.read(data_len)

        arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        result[name] = arr

    return result


def is_raw_format(blob: bytes) -> bool:
    """Check if a blob uses the new raw binary format (ARRY magic) vs legacy NPZ (PK magic)."""
    return blob[:4] == ARRY_MAGIC
