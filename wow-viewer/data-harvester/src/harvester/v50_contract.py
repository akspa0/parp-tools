"""Release identity gates for the current terrain reconstruction family.

``v50.N`` is intentionally a *data-and-model* contract, not a marketing
label.  A store, WDL checkpoint, generated-prior archive, and terrain
checkpoint may only be used together when their release values agree.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

MODEL_FAMILY = "v50"
DEFAULT_RELEASE = "v50.1"
MIXED_STORE_SCHEMA = "v50-mixed-curriculum-v1"
WDL_ARCHIVE_SCHEMA = "v50-generated-wdl-v1"
WDL_CHECKPOINT_VARIANT = "v50.1-spatial-wdl-prior"
TERRAIN_CHECKPOINT_VARIANT = "v50.1-terrain-refiner-convnextv2"
WL_LIQUID_SURFACE_QUADS_SIGNAL = "wl_liquid_surface_quads_v1"
WL_LIQUID_ABOVE_TERRAIN_SIGNAL = "wl_liquid_above_terrain_v1"
WL_LIQUID_BASIC_TYPE_SIGNAL = "wl_liquid_basic_type_header_v1"
WL_LIQUID_REQUIRED_PROVENANCE = frozenset(
    {
        WL_LIQUID_SURFACE_QUADS_SIGNAL,
        WL_LIQUID_ABOVE_TERRAIN_SIGNAL,
        WL_LIQUID_BASIC_TYPE_SIGNAL,
    }
)
WL_LIQUID_SOURCE_SIGNALS = frozenset({"wl_liquid_mask", "wl_liquid_height"})
V50_FRESH_ONLY_SIGNALS = frozenset({"liquid_mask", "liquid_height"})

_RELEASE = re.compile(r"^v50\.[1-9][0-9]*$")


def validate_release(value: str) -> str:
    release = str(value).strip().lower()
    if not _RELEASE.fullmatch(release):
        raise ValueError("--release must be v50.N (for example v50.1)")
    return release


def require_store_release(group, expected_release: str, *, store: Path) -> None:
    """Fail before a run can consume an unversioned or mismatched store."""
    release = validate_release(expected_release)
    actual_family = str(group.attrs.get("model_family", ""))
    actual_release = str(group.attrs.get("release", ""))
    actual_schema = str(group.attrs.get("schema", ""))
    if (actual_family, actual_release, actual_schema) != (MODEL_FAMILY, release, MIXED_STORE_SCHEMA):
        raise ValueError(
            "store is not the requested v50 release: "
            f"expected family={MODEL_FAMILY!r}, release={release!r}, schema={MIXED_STORE_SCHEMA!r}; "
            f"got family={actual_family!r}, release={actual_release!r}, schema={actual_schema!r} at {store}"
        )


def require_metadata_release(metadata: Mapping[str, object], expected_release: str, *, artifact: str) -> None:
    """Reject a checkpoint/archive whose declared release cannot feed this run."""
    release = validate_release(expected_release)
    family = str(metadata.get("model_family", ""))
    actual = str(metadata.get("release", ""))
    if family != MODEL_FAMILY or actual != release:
        raise ValueError(
            f"{artifact} is not compatible with {release}: "
            f"family={family!r}, release={actual!r}"
        )


def release_identity(release: str) -> dict[str, str]:
    release = validate_release(release)
    return {"model_family": MODEL_FAMILY, "release": release}


def migration_policy_for_signal(signal_name: str) -> str:
    """Return the frozen V50 migration policy for a canonical signal.

    Historic liquid payloads cannot prove whether their WL fallback was the former
    sparse-stamp projection. They are therefore never eligible for bit-preserving
    V18 migration: V50 obtains them only from a corrected client-backed extraction.
    """
    return "fresh-only" if str(signal_name) in V50_FRESH_ONLY_SIGNALS else "copy-if-verified"


def require_liquid_source_provenance(available_signals: Iterable[object] | object, *, artifact: str) -> None:
    """Reject WL liquid arrays that lack complete visibility/type provenance.

    Non-WL liquid readers (for example MH2O/MCLQ) do not carry the WL marker. The
    caller must still record their authoritative reader in the V50 row lineage.
    """
    if isinstance(available_signals, str):
        signals = {available_signals}
    else:
        try:
            signals = {str(signal) for signal in available_signals}
        except TypeError:
            signals = set()
    if signals.intersection(WL_LIQUID_SOURCE_SIGNALS) and not WL_LIQUID_REQUIRED_PROVENANCE.issubset(signals):
        missing = ", ".join(sorted(WL_LIQUID_REQUIRED_PROVENANCE.difference(signals)))
        raise ValueError(
            f"{artifact} contains WL liquid data without required provenance ({missing}); "
            "reject it and re-extract from an approved client build"
        )
