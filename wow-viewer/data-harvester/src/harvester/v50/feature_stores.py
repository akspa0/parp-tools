"""Shared multi-feature-store binding + channel concatenation for the geometry chain.

Spec 118 US3 turned the geometry trainers' single ``--feature-store`` into a *repeatable* input:
the coarse trainer, the detailer, and the coarse materializer all accept MORE THAN ONE
``v115-feature-map-v1`` prior (e.g. the Spec 115 terrain-feature map AND the Spec 118
object-segmentation map) and concatenate their generated channels onto RGB in CLI order. This makes
a later prior AUGMENT the deconfounding rather than EVICT it — the earlier single-store contract
could only replace the terrain-feature map, so an object prior could never sit alongside the
roads-as-slopes deconfounding it is meant to complement.

Every store is validated the same way the old single-store path validated its one store (schema
const, ``class_count`` >= 1, a ``feature_map`` array present, full coverage of the selected
curriculum rows via ``source_row_index``), so no trainer loses any guarantee. Channel order is the
CLI order of the stores; ``in_channels`` becomes ``3 + sum(class_counts)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_STORE_SCHEMA = "v115-feature-map-v1"
FEATURE_ARRAY = "feature_map"


class FeatureStoreError(ValueError):
    """Raised when a ``--feature-store`` prior cannot be bound to the selected curriculum rows."""


@dataclass
class FeatureBinding:
    """One validated ``v115-feature-map-v1`` prior bound to a curriculum's selected rows."""

    path: Path
    group: Any
    class_count: int
    row_to_position: dict[int, int]
    attrs: dict


def load_feature_stores(
    paths: list[Path] | None, *, selected_rows: list[int]
) -> list[FeatureBinding]:
    """Open and validate every ``--feature-store`` prior, in CLI order.

    Returns ``[]`` for ``None``/empty input so callers can treat "no feature store" uniformly.
    Raises :class:`FeatureStoreError` if any store is the wrong schema, has no usable
    ``feature_map``, or is missing any selected curriculum row.
    """
    if not paths:
        return []
    import pyarrow.parquet as pq
    import zarr

    bindings: list[FeatureBinding] = []
    for path in paths:
        path = Path(path)
        group = zarr.open_group(str(path), mode="r")
        attrs = dict(group.attrs)
        if attrs.get("schema") != FEATURE_STORE_SCHEMA:
            raise FeatureStoreError(
                f"--feature-store is not a {FEATURE_STORE_SCHEMA} store: {path}"
            )
        class_count = int(attrs.get("class_count", 0))
        if class_count < 1 or FEATURE_ARRAY not in group:
            raise FeatureStoreError(f"--feature-store has no usable feature_map array: {path}")
        feature_index = pq.read_table(path / "index.parquet").to_pylist()
        row_to_position = {int(r["source_row_index"]): pos for pos, r in enumerate(feature_index)}
        missing = [i for i in selected_rows if i not in row_to_position]
        if missing:
            raise FeatureStoreError(
                f"--feature-store {path} is missing {len(missing)} selected curriculum rows "
                f"(e.g. {missing[:5]}); materialize it with the same --source"
            )
        bindings.append(
            FeatureBinding(
                path=path,
                group=group,
                class_count=class_count,
                row_to_position=row_to_position,
                attrs=attrs,
            )
        )
    return bindings


def total_class_count(bindings: list[FeatureBinding]) -> int:
    """Total concatenated feature-channel count across all bound priors."""
    return sum(b.class_count for b in bindings)


def feature_channels_for_row(
    bindings: list[FeatureBinding], source_row: int
) -> np.ndarray | None:
    """Concatenate every prior's ``feature_map`` channels for ``source_row`` (CLI order).

    Returns ``None`` when there are no bindings, or when any binding does not cover the row (so
    the eval helpers' soft "skip if absent" behaviour is preserved rather than raising mid-batch).
    """
    if not bindings:
        return None
    parts: list[np.ndarray] = []
    for binding in bindings:
        position = binding.row_to_position.get(int(source_row))
        if position is None:
            return None
        parts.append(np.asarray(binding.group[FEATURE_ARRAY][position], dtype=np.float32))
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def plan_entries(bindings: list[FeatureBinding]) -> list[dict]:
    """Per-store records for a trainer/materializer plan and run record (one dict per prior)."""
    entries: list[dict] = []
    for binding in bindings:
        entry = {
            "path": str(binding.path.resolve()),
            "class_count": binding.class_count,
            "source_signal": binding.attrs.get("source_signal"),
            "checkpoint_sha256": binding.attrs.get("checkpoint_sha256"),
        }
        # Only the Spec 115 terrain-feature classifier store carries a taxonomy; object/lattice/
        # structure bridges do not. Record it when present rather than emitting a null everywhere.
        if binding.attrs.get("taxonomy_revision") is not None:
            entry["taxonomy_revision"] = binding.attrs.get("taxonomy_revision")
        entries.append(entry)
    return entries


def road_feature_binding(bindings: list[FeatureBinding]) -> FeatureBinding | None:
    """The Spec 115 terrain-feature prior, identified by its ``taxonomy_revision`` attr.

    The road-region MAE diagnostic (FR-008) argmaxes a store's own channels against the ``road``
    class id, so it is only meaningful against the terrain-feature classifier's map — not the
    concatenation, and not an object/lattice prior. Returns the first such binding, or ``None``.
    """
    for binding in bindings:
        if binding.attrs.get("taxonomy_revision") is not None:
            return binding
    return None


def as_bindings(
    feature_bindings: list[FeatureBinding] | None,
    feature_group: Any | None,
    feature_row_to_position: dict[int, int] | None,
) -> list[FeatureBinding]:
    """Normalize an eval helper's inputs to a binding list (backward-compat shim).

    New callers pass ``feature_bindings``; legacy callers passed a single ``feature_group`` +
    ``feature_row_to_position``, which is wrapped as a one-element list so the shared
    concatenation path serves both without duplicating logic.
    """
    if feature_bindings is not None:
        return feature_bindings
    if feature_group is not None:
        return [
            FeatureBinding(
                path=Path("."),
                group=feature_group,
                class_count=0,
                row_to_position=feature_row_to_position or {},
                attrs={},
            )
        ]
    return []


__all__ = [
    "FEATURE_STORE_SCHEMA",
    "FEATURE_ARRAY",
    "FeatureStoreError",
    "FeatureBinding",
    "load_feature_stores",
    "total_class_count",
    "feature_channels_for_row",
    "plan_entries",
    "road_feature_binding",
    "as_bindings",
]
