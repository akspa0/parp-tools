"""Complete per-build v50 Zarr writer and finalization checks (Spec 109 T030).

Writing a store here never fabricates a manifest around whatever arrays happen to be handed in --
``write_v50_store`` validates the manifest and requires every required, non-unavailable signal to
have a real array of the declared row count before it writes anything. ``finalize_store`` then
reads the store back and recomputes every signal's ``content_identity`` from the bytes actually on
disk (``identity.hash_array``), so a manifest's declared hash cannot silently drift from what was
written -- finalization can only reach ``FinalizationState.COMPLETE`` when every required signal's
declared identity matches its observed identity and every row has lineage.

``write_v50_store`` writes to a sibling staging directory first and only replaces ``output_path``
after every array has been written successfully (Spec 109 T053 -- a prior build that already wrote
a real, good store at this path must not be destroyed by a build that fails or is killed partway
through; see ``docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`` for the incident
this closes).
"""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from harvester.v50.contracts import DatasetStoreManifest, FinalizationState, RowLineage
from harvester.v50.identity import hash_array


class StoreWriteError(ValueError):
    """Raised when a store cannot be written as declared -- never partially written silently."""


def _replace_directory(staging_path: Path, output_path: Path) -> None:
    """Move ``staging_path`` onto ``output_path``, retrying transient failures.

    On Windows, a directory that a virus scanner or search indexer has just finished touching can
    briefly refuse ``rename``/``rmtree`` with ``PermissionError`` (WinError 5) even though nothing
    in this process still holds it open. A short retry-with-backoff clears that without giving up
    and leaving a good store deleted but its replacement not yet in place.
    """
    last_error: OSError | None = None
    for attempt in range(6):
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            staging_path.rename(output_path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2 * (2**attempt))
    raise StoreWriteError(
        f"could not replace {output_path} with the newly written store at {staging_path} "
        f"after retrying: {last_error}"
    )


def write_v50_store(
    output_path: Path,
    manifest: DatasetStoreManifest,
    signal_arrays: Mapping[str, np.ndarray],
) -> None:
    """Write a complete per-build v50 store. Fails closed before writing anything if any required,
    non-unavailable signal is missing its array or the array's row count disagrees with the
    manifest. Writes to a staging directory beside ``output_path`` and only replaces any existing
    store there once every array has been written without error -- a failed or interrupted write
    leaves a prior good store at ``output_path`` untouched and only the staging directory behind."""
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

    output_path = Path(output_path)
    staging_path = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    staging_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        root = zarr.open_group(str(staging_path), mode="w")
        root.attrs.update(manifest.to_dict())
        for signal in manifest.signals:
            array = signal_arrays.get(signal.name)
            if array is None:
                continue
            root.create_array(signal.name, data=np.ascontiguousarray(array))
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    _replace_directory(staging_path, output_path)


def read_v50_manifest(store_path: Path) -> Mapping[str, object]:
    import zarr

    root = zarr.open_group(str(store_path), mode="r")
    return dict(root.attrs)


@dataclass(frozen=True)
class FinalizeReport:
    """``finalize_store_report``'s full result: the (possibly still-incomplete) manifest plus every
    reason it isn't ``COMPLETE`` -- unlike ``finalize_store``, callers that need to tell a human (or
    a script) *why* a store didn't finalize don't have to recompute this themselves."""

    manifest: DatasetStoreManifest
    mismatches: tuple[str, ...]


def finalize_store_report(
    store_path: Path,
    manifest: DatasetStoreManifest,
    row_lineages: Sequence[RowLineage],
) -> FinalizeReport:
    """Read a written store back, recompute every signal's observed content identity from the
    actual on-disk values, and return an updated manifest plus every concrete reason it stayed
    incomplete. Only reaches ``finalization_state=COMPLETE`` when every declared identity matches
    the observed one and every row has lineage for every required signal; otherwise the manifest is
    unchanged except for ``finalization_state=INCOMPLETE`` (never silently promotes a mismatched
    store to complete) and ``mismatches`` names every signal/row responsible.

    A required signal missing lineage on some rows is a common, legitimate real-client outcome (a
    tile that genuinely lacks the source data for that signal, e.g. no texture data to synthesize a
    minimap from) -- it is reported per row, not just as a single opaque failure, so a caller can
    decide whether to drop those specific rows (e.g. via curation) rather than discard the whole
    store."""
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
        bad_rows = [
            lineage.store_row
            for lineage in row_lineages
            if lineage.signal_actions.get(signal.name) in (None, "unavailable")
        ]
        if bad_rows:
            sample = bad_rows[:5]
            suffix = f" (+{len(bad_rows) - 5} more)" if len(bad_rows) > 5 else ""
            mismatches.append(
                f"required signal {signal.name!r} has no real lineage action for "
                f"{len(bad_rows)} row(s): {sample}{suffix}"
            )

    finalization_state = FinalizationState.INCOMPLETE if mismatches else FinalizationState.COMPLETE
    return FinalizeReport(
        manifest=replace(manifest, finalization_state=finalization_state),
        mismatches=tuple(mismatches),
    )


def finalize_store(
    store_path: Path,
    manifest: DatasetStoreManifest,
    row_lineages: Sequence[RowLineage],
) -> DatasetStoreManifest:
    """Same as ``finalize_store_report`` but returns only the manifest, for callers that don't need
    the mismatch reasons."""
    return finalize_store_report(store_path, manifest, row_lineages).manifest
