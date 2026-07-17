"""Immutable row-selection curricula over canonical v50 stores (Spec 109 T033,
research.md Decision 3).

A curriculum is a manifest of *references* into one or more already-written canonical stores --
``(store_id, row_id)`` pairs plus their split assignment -- never a copy of the underlying array
payloads. This is what makes repeated experiments cheap: a new curriculum for a different
train/val/test split, or a different row subset, costs one small manifest, not another full store
copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harvester.spec103.prefab_curation import validate_source_group_split
from harvester.v50.contracts import HASH_PATTERN, validate_release
from harvester.v50.identity import hash_manifest

VALID_SPLITS = frozenset({"train", "val", "test"})


@dataclass(frozen=True)
class CurriculumRowRef:
    store_id: str
    row_id: int
    source_group: str
    split: str

    def __post_init__(self) -> None:
        if not HASH_PATTERN.fullmatch(self.store_id):
            raise ValueError(f"store_id must be a sha256:<hex> identity, got {self.store_id!r}")
        if self.row_id < 0:
            raise ValueError("row_id must be non-negative")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {self.split!r}")


@dataclass(frozen=True)
class CurriculumManifest:
    manifest_id: str
    release: str
    store_ids: tuple[str, ...]
    rows: tuple[CurriculumRowRef, ...]
    selection_reason: str
    policy_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "v50-curriculum-manifest-v1",
            "manifest_id": self.manifest_id,
            "release": self.release,
            "store_ids": list(self.store_ids),
            "rows": [
                {
                    "store_id": row.store_id,
                    "row_id": row.row_id,
                    "source_group": row.source_group,
                    "split": row.split,
                }
                for row in self.rows
            ],
            "selection_reason": self.selection_reason,
            "policy_identity": self.policy_identity,
        }


def build_curriculum(
    *,
    release: str,
    rows: list[CurriculumRowRef],
    selection_reason: str,
    policy_identity: str,
) -> CurriculumManifest:
    """Build an immutable curriculum manifest. Raises before returning anything if any row is
    duplicated (same store_id+row_id selected twice, possibly into different splits -- always an
    authoring mistake) or if a source_group crosses train/val/test (reuses the existing
    ``validate_source_group_split`` leak check rather than reimplementing it)."""
    validate_release(release)
    if not selection_reason:
        raise ValueError("selection_reason is required -- a curriculum must state why these rows were chosen")

    seen_keys: set[tuple[str, int]] = set()
    duplicates: list[tuple[str, int]] = []
    for row in rows:
        key = (row.store_id, row.row_id)
        if key in seen_keys:
            duplicates.append(key)
        seen_keys.add(key)
    if duplicates:
        raise ValueError(f"duplicate (store_id, row_id) selections: {duplicates}")

    index_rows = [{"source_group_id": f"{row.store_id}:{row.source_group}"} for row in rows]
    train_positions = [i for i, row in enumerate(rows) if row.split == "train"]
    val_positions = [i for i, row in enumerate(rows) if row.split == "val"]
    validate_source_group_split(index_rows, train_positions, val_positions)
    test_positions = [i for i, row in enumerate(rows) if row.split == "test"]
    if test_positions:
        # Also check test against both other splits pairwise, since the helper only compares two
        # partitions at a time.
        validate_source_group_split(index_rows, train_positions, test_positions)
        validate_source_group_split(index_rows, val_positions, test_positions)

    store_ids = tuple(sorted({row.store_id for row in rows}))
    manifest_id = hash_manifest(
        {
            "release": release,
            "store_ids": list(store_ids),
            "rows": [(row.store_id, row.row_id, row.split) for row in rows],
            "selection_reason": selection_reason,
            "policy_identity": policy_identity,
        }
    )

    return CurriculumManifest(
        manifest_id=manifest_id,
        release=release,
        store_ids=store_ids,
        rows=tuple(rows),
        selection_reason=selection_reason,
        policy_identity=policy_identity,
    )
