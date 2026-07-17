"""Complete per-build v50 Zarr writer and finalization checks (Spec 109 T030).

Writing a store here never fabricates a manifest around whatever arrays happen to be handed in --
``write_v50_store`` validates the manifest and requires every required, non-unavailable signal to
have a real array of the declared row count before it writes anything. ``finalize_store`` then
reads the store back and recomputes every signal's ``content_identity`` from the bytes actually on
disk (``identity.hash_array``), so a manifest's declared hash cannot silently drift from what was
written -- finalization can only reach ``FinalizationState.COMPLETE`` when every required signal's
declared identity matches its observed identity and every row has lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from harvester.v50.contracts import DatasetStoreManifest, FinalizationState, RowLineage
from harvester.v50.identity import hash_array


class StoreWriteError(ValueError):
    """Raised when a store cannot be written as declared -- never partially written silently."""


def write_v50_store(
    output_path: Path,
    manifest: DatasetStoreManifest,
    signal_arrays: Mapping[str, np.ndarray],
) -> None:
    """Write a complete per-build v50 store. Fails closed before writing anything if any required,
    non-unavailable signal is missing its array or the array's row count disagrees with the
    manifest."""
    manifest.validate()

    unavailable_names = {signal.name for signal in manifest.unavailable_signals}
    missing: list[str] = []
    row_count_mismatches: list[str] = []
    for signal in manifest.signals:
        if signal.name in unavailable_names:
            continue
        array = signal_arrays.get(signal.name)
        if array is None:
            if signal.required:
                missing.append(signal.name)
            continue
        if array.shape[0] != manifest.row_count:
            row_count_mismatches.append(
                f"{signal.name}: array has {array.shape[0]} rows, manifest declares {manifest.row_count}"
            )

    if missing:
        raise StoreWriteError(f"missing arrays for required signals: {sorted(missing)}")
    if row_count_mismatches:
        raise StoreWriteError(f"row count mismatches: {row_count_mismatches}")

    import zarr

    root = zarr.open_group(str(output_path), mode="w")
    root.attrs.update(manifest.to_dict())
    for signal in manifest.signals:
        array = signal_arrays.get(signal.name)
        if array is None:
            continue
        root.create_array(signal.name, data=np.ascontiguousarray(array))


def read_v50_manifest(store_path: Path) -> Mapping[str, object]:
    import zarr

    root = zarr.open_group(str(store_path), mode="r")
    return dict(root.attrs)


def finalize_store(
    store_path: Path,
    manifest: DatasetStoreManifest,
    row_lineages: Sequence[RowLineage],
) -> DatasetStoreManifest:
    """Read a written store back, recompute every signal's observed content identity from the
    actual on-disk values, and return an updated manifest. Only returns
    ``finalization_state=COMPLETE`` when every declared identity matches the observed one and every
    row has lineage for every required signal; otherwise returns the manifest unchanged except for
    ``finalization_state=INCOMPLETE`` (never silently promotes a mismatched store to complete)."""
    import zarr

    root = zarr.open_group(str(store_path), mode="r")
    unavailable_names = {signal.name for signal in manifest.unavailable_signals}

    mismatches: list[str] = []
    for signal in manifest.signals:
        if signal.name in unavailable_names:
            continue
        if signal.name not in root:
            mismatches.append(f"{signal.name}: declared but not present in store")
            continue
        observed = hash_array(root[signal.name][:])
        if observed != signal.content_identity:
            mismatches.append(
                f"{signal.name}: declared {signal.content_identity!r} != observed {observed!r}"
            )

    if len(row_lineages) != manifest.row_count:
        mismatches.append(
            f"row_lineage count {len(row_lineages)} != manifest row_count {manifest.row_count}"
        )
    for signal in manifest.signals:
        if not signal.required or signal.name in unavailable_names:
            continue
        for lineage in row_lineages:
            if lineage.signal_actions.get(signal.name) in (None, "unavailable"):
                mismatches.append(
                    f"row {lineage.store_row}: required signal {signal.name!r} has no real lineage action"
                )
                break  # one representative failure per signal is enough to keep this bounded

    if mismatches:
        return replace(manifest, finalization_state=FinalizationState.INCOMPLETE)

    return replace(manifest, finalization_state=FinalizationState.COMPLETE)
