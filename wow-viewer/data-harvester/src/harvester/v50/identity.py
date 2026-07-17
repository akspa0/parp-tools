"""Deterministic content identities for v50 artifacts (Spec 109 T009).

Every identity produced here is formatted ``sha256:<64 lowercase hex chars>`` -- exactly the
``hash`` pattern in ``specs/109-v50-clean-room-audit/contracts/v50-provenance.schema.json`` and
``v50-cleanup-plan.schema.json``, and exactly what ``harvester.v50.contracts`` validates.

Four identity kinds, matching data-model.md's ``ArtifactRecord.content_identity`` /
``DatasetSignal.content_identity`` usage:

- ``hash_file``: exact byte-for-byte identity of one file (executables, checkpoints, raw blobs).
- ``hash_metadata_tree``: identity of a directory's *structure and file identities*, not raw bytes
  -- used for artifacts (client installs, store directories) where walking every byte is wasteful
  and the metadata/child-hash tree is what a manifest actually needs to reproduce.
- ``hash_parquet_table``: identity of a Parquet table's *logical content* (column names, dtypes,
  and row values), independent of physical file layout (row-group sizing, compression codec,
  column order on disk) -- two Parquet files with the same data but different physical layout must
  produce the same identity, or a resave/recompress would spuriously invalidate lineage.
- ``hash_manifest``: identity of an arbitrary JSON-serializable manifest dict, via canonical
  (sorted-key, no-whitespace-ambiguity) serialization so the same logical manifest always hashes
  the same regardless of dict insertion order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024


def _format_digest(digest: "hashlib._Hash") -> str:
    return f"sha256:{digest.hexdigest()}"


def hash_file(path: Path) -> str:
    """Exact byte-for-byte identity of one file. Streams in chunks; never loads whole large files
    (checkpoints, client archives) into memory at once."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return _format_digest(digest)


def hash_metadata_tree(root: Path) -> str:
    """Identity of a directory's structure plus per-file identity, without hashing every file's
    full content roundtrip through a second full read (each leaf is hashed once via ``hash_file``,
    then the tree's relative-path -> hash mapping is itself hashed deterministically).

    Does not follow symlinks -- a link either resolves to a real path already covered by walking
    the tree, or it points outside the tree entirely, which ``path_policy`` is responsible for
    rejecting rather than this identity function silently following.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"hash_metadata_tree requires a directory, got {root}")

    entries: dict[str, str] = {}
    for child in sorted(root.rglob("*")):
        if child.is_symlink() or not child.is_file():
            continue
        relative = child.relative_to(root).as_posix()
        entries[relative] = hash_file(child)

    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8"))
    return _format_digest(digest)


def hash_parquet_table(path: Path) -> str:
    """Identity of a Parquet file's logical content: column names/dtypes and row values, in a
    fixed row order, independent of physical layout (row groups, compression, on-disk column
    order). Requires ``pyarrow`` (an existing project dependency)."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    # Deterministic column order regardless of how the file physically stored them.
    column_names = sorted(table.column_names)
    ordered = table.select(column_names)
    schema_repr = "|".join(f"{name}:{ordered.schema.field(name).type}" for name in column_names)

    digest = hashlib.sha256()
    digest.update(schema_repr.encode("utf-8"))
    for batch in ordered.to_batches():
        # A batch's Arrow IPC serialization is a deterministic byte encoding of its exact values.
        digest.update(batch.serialize().to_pybytes())
    return _format_digest(digest)


def hash_manifest(payload: dict[str, Any]) -> str:
    """Identity of an arbitrary JSON-serializable manifest dict via canonical serialization, so
    the same logical content hashes identically regardless of key insertion order."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8"))
    return _format_digest(digest)


def hash_array(array: Any) -> str:
    """Identity of an array's actual values (a written Zarr signal array, or any array-like),
    independent of chunking/compression -- two arrays with the same shape/dtype/values hash
    identically regardless of how they are physically stored. Used by the v50 store writer/
    finalizer to prove a signal's declared ``content_identity`` matches what was actually written.
    """
    import numpy as np

    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(values.tobytes())
    return _format_digest(digest)
