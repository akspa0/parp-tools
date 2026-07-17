"""Compatibility shim: the v50 release-identity gates now live in ``harvester.v50.contracts``
(Spec 109 T012). This module re-exports the same names so existing Spec 103/108 callers and
``tests/test_v50_contract.py`` keep working unchanged during the ownership migration
(research.md Decision 6). Do not add new behavior here -- add it to ``harvester.v50.contracts``.
"""

from __future__ import annotations

from harvester.v50.contracts import (
    DEFAULT_RELEASE,
    MIXED_STORE_SCHEMA,
    MODEL_FAMILY,
    TERRAIN_CHECKPOINT_VARIANT,
    V50_FRESH_ONLY_SIGNALS,
    WDL_ARCHIVE_SCHEMA,
    WDL_CHECKPOINT_VARIANT,
    WL_LIQUID_ABOVE_TERRAIN_SIGNAL,
    WL_LIQUID_BASIC_TYPE_SIGNAL,
    WL_LIQUID_REQUIRED_PROVENANCE,
    WL_LIQUID_SOURCE_SIGNALS,
    WL_LIQUID_SURFACE_QUADS_SIGNAL,
    migration_policy_for_signal,
    release_identity,
    require_liquid_source_provenance,
    require_metadata_release,
    require_store_release,
    validate_release,
)

__all__ = [
    "DEFAULT_RELEASE",
    "MIXED_STORE_SCHEMA",
    "MODEL_FAMILY",
    "TERRAIN_CHECKPOINT_VARIANT",
    "V50_FRESH_ONLY_SIGNALS",
    "WDL_ARCHIVE_SCHEMA",
    "WDL_CHECKPOINT_VARIANT",
    "WL_LIQUID_ABOVE_TERRAIN_SIGNAL",
    "WL_LIQUID_BASIC_TYPE_SIGNAL",
    "WL_LIQUID_REQUIRED_PROVENANCE",
    "WL_LIQUID_SOURCE_SIGNALS",
    "WL_LIQUID_SURFACE_QUADS_SIGNAL",
    "migration_policy_for_signal",
    "release_identity",
    "require_liquid_source_provenance",
    "require_metadata_release",
    "require_store_release",
    "validate_release",
]
