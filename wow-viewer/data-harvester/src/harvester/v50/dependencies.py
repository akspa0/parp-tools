"""Dependency discovery across manifests, checkpoints, reports, and known output layouts
(Spec 109 T023, FR-020).

A cleanup candidate must never be approved for deletion while something else still points at it.
Since no concrete v50 manifest schema exists yet to hardcode against (Spec 109 T002 is not frozen),
this module works generically: for every manifest/report-kind JSON artifact already discovered by
``inventory.py``, it recursively scans the JSON for string values that reference another artifact
-- by that artifact's resolved path (as an absolute path or a path suffix, since manifests may
record relative or differently-rooted paths) or by its content/artifact identity hash. Anything
referenced this way is marked as depended-upon and is not a safe-to-delete candidate on its own.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from harvester.v50.contracts import ArtifactRecord

_MANIFEST_LIKE_KINDS = frozenset({"manifest", "report"})


def _iter_strings(payload: Any):
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_strings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_strings(item)


def _references_artifact(candidate_string: str, target: ArtifactRecord) -> bool:
    if candidate_string == target.artifact_id or candidate_string == target.content_identity:
        return True
    target_path = Path(target.resolved_path)
    if candidate_string == target.resolved_path:
        return True
    # Manifests commonly record a relative or differently-rooted path to the same artifact; a
    # suffix match on the path (e.g. ".../v50.1/azeroth.zarr") is a reasonable, conservative signal
    # without requiring every manifest to agree on an absolute path convention.
    try:
        candidate_path = Path(candidate_string)
    except (OSError, ValueError):
        return False
    return len(candidate_path.parts) > 1 and target_path.as_posix().endswith(candidate_path.as_posix())


def discover_dependencies(records: list[ArtifactRecord]) -> list[ArtifactRecord]:
    """Return a new list of records with ``dependencies`` populated: for each artifact, the
    artifact_ids of every manifest/report that references it. Does not mutate the input list
    (ArtifactRecord is frozen) and never opens a non-manifest/report artifact's content -- only
    JSON manifests/reports are scanned, so this still respects the metadata-only discipline for
    dataset/checkpoint payloads themselves."""
    dependents_by_target: dict[str, set[str]] = {record.artifact_id: set() for record in records}

    for referrer in records:
        if referrer.kind not in _MANIFEST_LIKE_KINDS:
            continue
        referrer_path = Path(referrer.resolved_path)
        if not referrer_path.is_file() or referrer_path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(referrer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        strings = list(_iter_strings(payload))
        for target in records:
            if target.artifact_id == referrer.artifact_id:
                continue
            if any(_references_artifact(value, target) for value in strings):
                dependents_by_target[target.artifact_id].add(referrer.artifact_id)

    return [
        replace(record, dependencies=tuple(sorted(dependents_by_target[record.artifact_id])))
        for record in records
    ]
