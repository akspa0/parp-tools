"""Metadata-only artifact discovery (Spec 109 T016, FR-001/FR-002/FR-003/FR-009).

Walks configured roots and produces one ``ArtifactRecord`` per discovered artifact, using only
path names, sizes, and (for directories) a metadata-tree identity -- never opening dataset/model
payloads. FR-001/FR-002 are the load-bearing rule this module exists to enforce: nothing here ever
sets ``trust_state`` to anything but ``UNVERIFIED``, regardless of what the artifact is named,
where it lives, or what attributes it claims. A directory literally named
``v50_verified_dataset`` still comes out unverified -- promotion is a separate, evidence-gated step
(``verify_store.py``) that this module does not perform and cannot be asked to skip.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from harvester.v50.contracts import ArtifactRecord, Disposition, ProofLevel, TrustState
from harvester.v50.identity import hash_file, hash_manifest, hash_metadata_tree

# A reasonable default classifier. This is an internal inventory heuristic, not a client-path
# assumption (FR-014 is about never hardcoding *client roots*); operators may override or extend
# it per call. Order matters -- first matching suffix/name wins.
DEFAULT_KIND_PATTERNS: Mapping[str, str] = {
    ".zarr": "dataset",
    ".pt": "checkpoint",
    ".pth": "checkpoint",
    ".npz": "prior archive",
}
DEFAULT_MANIFEST_NAME_HINTS: tuple[str, ...] = ("manifest", "index")
DEFAULT_REPORT_NAME_HINTS: tuple[str, ...] = ("report",)


def _classify_kind(path: Path, *, kind_patterns: Mapping[str, str]) -> str:
    suffix = path.suffix.lower()
    if suffix in kind_patterns:
        return kind_patterns[suffix]
    stem = path.stem.lower()
    if suffix == ".json":
        if any(hint in stem for hint in DEFAULT_MANIFEST_NAME_HINTS):
            return "manifest"
        if any(hint in stem for hint in DEFAULT_REPORT_NAME_HINTS):
            return "report"
        return "manifest"
    return "unknown"


def _artifact_identity(path: Path) -> tuple[str, int]:
    """Return (content_identity, observed_bytes) without reading array payloads: a directory
    (e.g. a .zarr store) gets a metadata-tree identity over its file structure; a plain file gets
    its own byte hash. Neither path opens/decodes any array chunk as data."""
    if path.is_dir():
        total_bytes = sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
        return hash_metadata_tree(path), total_bytes
    return hash_file(path), path.stat().st_size


@dataclass(frozen=True)
class InventoryRoot:
    """One configured root to walk, with the artifact kind it represents (e.g. a datasets root
    scans one level deep for named stores; a checkpoints root scans for individual files)."""

    path: Path
    default_owner: str = "unknown"
    kind_patterns: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kind_patterns is None:
            object.__setattr__(self, "kind_patterns", DEFAULT_KIND_PATTERNS)


def discover_artifacts(roots: list[InventoryRoot]) -> list[ArtifactRecord]:
    """Metadata-only discovery across every configured root. Every record starts
    ``trust_state=UNVERIFIED`` and ``disposition=QUARANTINE`` -- this function inventories, it does
    not judge. A later, separate verification/cleanup pass assigns any other state."""
    records: list[ArtifactRecord] = []
    seen_paths: set[Path] = set()

    for root in roots:
        if not root.path.exists():
            continue
        for entry in sorted(root.path.iterdir()):
            resolved = entry.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            kind = _classify_kind(entry, kind_patterns=root.kind_patterns)
            content_identity, observed_bytes = _artifact_identity(entry)
            # artifact_id identifies *this filesystem entry* (kind + location + content); two
            # byte-identical stores at different paths legitimately share content_identity but
            # must not collide as the same artifact record.
            artifact_id = hash_manifest(
                {"kind": kind, "resolved_path": str(resolved), "content_identity": content_identity}
            )

            records.append(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    kind=kind,
                    resolved_path=str(resolved),
                    observed_bytes=observed_bytes,
                    content_identity=content_identity,
                    owner=root.default_owner,
                    proof_level=ProofLevel.INVENTORY,
                    trust_state=TrustState.UNVERIFIED,
                    disposition=Disposition.QUARANTINE,
                )
            )

    return records
