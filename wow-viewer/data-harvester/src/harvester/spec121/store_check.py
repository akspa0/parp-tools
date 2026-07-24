"""Spec 121 T003: read-only store prerequisite check.

The lane consumes the v50 Full-profile store as-is (FR-013: no new C# harvest). This helper
reports which required arrays a store actually carries so dry-runs and quickstart step 0 can
fail (or degrade) with a precise message instead of a KeyError deep inside a trainer.
"""

from __future__ import annotations

# Stage A supervision target (Spec 117 catalog amendment).
LATTICE_ARRAYS = ("wdl_outer_17", "wdl_inner_16", "wdl_outer_present", "wdl_inner_present")
# Stage A input.
INPUT_ARRAYS = ("minimap_rgb",)
# Loss-side object signal (Spec 118; optional — weighted runs degrade gracefully without it).
OBJECT_MASK_ARRAYS = ("object_geometry_visible_mask_257",)

REQUIRED_ARRAYS = INPUT_ARRAYS + LATTICE_ARRAYS


def check_store_arrays(group) -> dict[str, bool]:
    """Return ``{array_name: present}`` for every array this lane knows about."""
    names = set(group.array_keys()) if hasattr(group, "array_keys") else set(group.keys())
    return {name: (name in names) for name in REQUIRED_ARRAYS + OBJECT_MASK_ARRAYS}


def missing_required(group) -> list[str]:
    """Required arrays the store does not carry (empty list = store is usable)."""
    presence = check_store_arrays(group)
    return [name for name in REQUIRED_ARRAYS if not presence[name]]


def object_mask_present(group) -> bool:
    """True when the Spec 118 object-mask signal is available for loss-side weighting."""
    return check_store_arrays(group)[OBJECT_MASK_ARRAYS[0]]


def report(group) -> dict:
    """Machine-readable prerequisite summary for dry-run plans and quickstart step 0."""
    presence = check_store_arrays(group)
    return {
        "schema": "v121-store-check-v1",
        "arrays": presence,
        "missing_required": [name for name in REQUIRED_ARRAYS if not presence[name]],
        "object_mask_signal_present": presence[OBJECT_MASK_ARRAYS[0]],
        "usable": all(presence[name] for name in REQUIRED_ARRAYS),
    }


__all__ = [
    "INPUT_ARRAYS",
    "LATTICE_ARRAYS",
    "OBJECT_MASK_ARRAYS",
    "REQUIRED_ARRAYS",
    "check_store_arrays",
    "missing_required",
    "object_mask_present",
    "report",
]
