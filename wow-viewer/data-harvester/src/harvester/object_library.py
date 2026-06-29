"""Per-object capture library contract (spec 077 §1.1, §1.2, §1.3).

This module is the canonical owner of the per-object library schema for
spec 077. It mirrors the C# records in
``WowViewer.Core.Maps.ObjectLibraryEntry`` and
``WowViewer.Core.Maps.ObjectCaptureVariant`` so both the C# capture lane and
the Python training/inference side agree on field names, ID rules, and
default values.

Spec 077 mandates that one library entry exists per normalized asset path and
that multiple variants per asset are allowed when rotation, scale, or
visibility class materially change the top-down appearance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

import numpy as np

# ---- Enum-like string sets ------------------------------------------------
#
# Spec 077 uses lowercase string enums for portability between the C# and
# Python sides. Use these constants instead of free-form strings.

ASSET_TYPES: tuple[str, ...] = ("m2", "mdx", "wmo")

CAPTURE_STATUSES: tuple[str, ...] = (
    "captured",
    "failed",
    "not_attempted",
    "partial",
)

VISIBILITY_CLASSES: tuple[str, ...] = (
    "roof_visible",
    "likely_visible",
    "likely_hidden",
    "clutter_filtered",
    "unknown",
)

REVIEW_STATES: tuple[str, ...] = (
    "unreviewed",
    "accepted",
    "rejected",
    "needs_followup",
)

CAPTURE_MODES: tuple[str, ...] = (
    "orthographic_topdown",
    "geometry_projection",
    "hybrid",
    "unknown",
)


def normalize_asset_path(path: str) -> str:
    """Lowercase, slash-normalized, deduplicated-separator asset path.

    Mirrors the C# ``AssetPathTaxonomy.Normalize`` for cross-side stability.
    """
    text = (path or "").replace("\\", "/").strip().lower()
    while "//" in text:
        text = text.replace("//", "/")
    return text.strip("/")


def detect_asset_type(path: str) -> str:
    """Return one of ``"m2" | "mdx" | "wmo" | "unknown"`` for a path.

    Detection is purely suffix-based because the source placement tables do
    not carry an explicit kind column. WMO files use ``.wmo`` or
    ``.wmo.mpq`` (legacy archive form); M2 uses ``.m2``; MDX uses ``.mdx``.
    """
    if not path:
        return "unknown"
    suffixes = "".join(PurePosixPath(path.lower()).suffixes)
    if suffixes in (".wmo", ".wmo.mpq"):
        return "wmo"
    if suffixes == ".m2":
        return "m2"
    if suffixes == ".mdx":
        return "mdx"
    return "unknown"


def library_id_from_asset_path(path: str) -> str:
    """Deterministic library id from a normalized asset path.

    Mirrors ``ObjectLibraryEntry.ComputeLibraryId`` in C#: SHA1 prefix of
    the lowercase normalized path, prefixed with ``objlib_``.
    """
    normalized = normalize_asset_path(path)
    if not normalized:
        return ""
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"objlib_{digest[:14]}"


def variant_id_from_parts(parts: Iterable[object]) -> str:
    """Deterministic variant id from a tuple of pose/source parts.

    Mirrors ``ObjectCaptureVariant.ComputeVariantId`` in C#: SHA1 of the
    joined payload, prefixed with ``objvar_`` and truncated to 16 hex
    characters.
    """
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"objvar_{digest[:16]}"


def _format_single_g9(value: float) -> str:
    """Format a value like C# ``float.ToString("G9", InvariantCulture)``."""
    return format(float(np.float32(value)), ".9g")


@dataclass(frozen=True)
class ObjectLibraryEntry:
    library_id: str
    original_asset_path: str
    normalized_asset_path: str
    asset_type: str = "unknown"
    capture_status: str = "not_attempted"
    visibility_class: str = "unknown"
    review_state: str = "unreviewed"
    source_builds: tuple[str, ...] = ()
    source_maps: tuple[str, ...] = ()
    placement_observation_count: int = 0
    preferred_variant_id: str | None = None

    def __post_init__(self) -> None:
        if self.asset_type not in ASSET_TYPES + ("unknown",):
            raise ValueError(f"Unknown asset_type: {self.asset_type!r}")
        if self.capture_status not in CAPTURE_STATUSES:
            raise ValueError(f"Unknown capture_status: {self.capture_status!r}")
        if self.visibility_class not in VISIBILITY_CLASSES:
            raise ValueError(f"Unknown visibility_class: {self.visibility_class!r}")
        if self.review_state not in REVIEW_STATES:
            raise ValueError(f"Unknown review_state: {self.review_state!r}")


@dataclass(frozen=True)
class ObjectCaptureVariant:
    variant_id: str
    library_id: str
    capture_build: str
    capture_mode: str = "unknown"
    asset_type: str = "unknown"
    image_key: str = ""
    mask_key: str = ""
    bbox_x0: int = 0
    bbox_y0: int = 0
    bbox_x1: int = 0
    bbox_y1: int = 0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0
    scale: float = 1.0
    capture_notes: str = ""
    capture_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.capture_mode not in CAPTURE_MODES:
            raise ValueError(f"Unknown capture_mode: {self.capture_mode!r}")
        if self.asset_type not in ASSET_TYPES + ("unknown",):
            raise ValueError(f"Unknown asset_type: {self.asset_type!r}")
        if not 0.0 <= self.capture_confidence <= 1.0:
            raise ValueError(
                f"capture_confidence must be in [0,1]; got {self.capture_confidence!r}"
            )

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.bbox_x0), int(self.bbox_y0), int(self.bbox_x1), int(self.bbox_y1))

    @property
    def bbox_wh(self) -> tuple[int, int]:
        return (max(0, int(self.bbox_x1) - int(self.bbox_x0)),
                max(0, int(self.bbox_y1) - int(self.bbox_y0)))


def make_entry_from_path(asset_path: str) -> ObjectLibraryEntry:
    """Build a default :class:`ObjectLibraryEntry` for one asset path.

    The entry starts in :data:`not_attempted` / :data:`unknown` /
    :data:`unreviewed`; downstream tools (capture, review) update those
    fields in place.
    """
    normalized = normalize_asset_path(asset_path)
    return ObjectLibraryEntry(
        library_id=library_id_from_asset_path(normalized),
        original_asset_path=asset_path,
        normalized_asset_path=normalized,
        asset_type=detect_asset_type(normalized),
    )


def make_variant_id(
    *,
    library_id: str,
    capture_build: str,
    capture_mode: str,
    rot_x: float,
    rot_y: float,
    rot_z: float,
    scale: float,
) -> str:
    """Build a deterministic variant id matching the C# contract.

    The pose floats are formatted as single-precision ``G9`` values to match
    the C# ``ObjectCaptureVariant.ComputeVariantId`` payload exactly.
    """
    if not library_id:
        return ""
    return variant_id_from_parts(
        (
            library_id,
            capture_build or "",
            capture_mode,
            _format_single_g9(rot_x),
            _format_single_g9(rot_y),
            _format_single_g9(rot_z),
            _format_single_g9(scale),
        )
    )


def is_clutter_asset(asset_path: str) -> bool:
    """Heuristic visibility-classifier used only for ``ClutterFiltered``.

    This intentionally mirrors the C# object-mask filter logic at
    ``AdtTensorPackBuilder.ShouldIncludeDoodadInFilteredMask`` and the
    Python counterpart in :data:`object_roof.is_probable_roof_asset`. It is
    a *hint*, not authoritative — the review state still wins.
    """
    normalized = normalize_asset_path(asset_path)
    if not normalized:
        return False
    clutter_tokens = (
        "/trees/",
        "/bush",
        "/shrub",
        "/fern",
        "/plant",
        "/palm",
        "/flower",
        "/grass",
        "/twig",
        "/root",
        "/leaf",
        "/mushroom",
    )
    return any(token in normalized for token in clutter_tokens)
