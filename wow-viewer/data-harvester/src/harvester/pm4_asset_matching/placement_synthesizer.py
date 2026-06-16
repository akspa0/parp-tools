"""Python port of C# Pm4ReplacementPlacementSynthesizer.

Produces placement proposals from match results, with SHA256-based proposal IDs
that match the C# implementation.
"""

from __future__ import annotations

import hashlib
import math

from .models import (
    Pm4AssetMatchStatus,
    Pm4AssetReferenceSignalRecord,
    Pm4ReplacementPlacementProposal,
    Pm4SegmentMatchResult,
)


def _build_proposal_id(
    segment_id: str,
    asset_id: str,
    target_tiles: list[str],
) -> str:
    """Build a deterministic proposal ID matching the C# implementation.

    C# appends a trailing comma after every tile, so we build the string
    the same way to produce identical SHA256 digests.
    """
    # Match C#: foreach tile { builder.Append(tile); builder.Append(','); }
    tiles_buf = "".join(f"{t}," for t in target_tiles)
    content = f"{segment_id}|{asset_id}|{tiles_buf}"
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    return f"proposal-{digest[:8].hex()}"


def _build_fallback_rotation(
    segment: Pm4SegmentMatchResult,
) -> tuple[float, float, float] | None:
    """Build fallback rotation from anchor heading signals.

    Returns (yaw, pitch, roll) matching the Python model convention.
    C# stores in-memory as Vector3(Pitch, Roll, Yaw) and converts to
    (Yaw, Pitch, Roll) on serialization — we store directly as (yaw, pitch, roll).
    """
    yaw = segment.segment.anchor_signals.heading_mean_degrees
    if yaw is None or not math.isfinite(yaw):
        return None
    return (yaw, 0.0, 0.0)


def _format_status(status: Pm4AssetMatchStatus) -> str:
    return status.value


def synthesize_placements(
    match_results: list[Pm4SegmentMatchResult],
    asset_references: list[Pm4AssetReferenceSignalRecord],
    target_tile_coordinates: set[str] | None = None,
) -> list[Pm4ReplacementPlacementProposal]:
    """Synthesize placement proposals from match results.

    Mirrors the C# Pm4ReplacementPlacementSynthesizer.Synthesize logic.
    """
    assets_by_id = {a.asset_id: a for a in asset_references}
    proposals: list[Pm4ReplacementPlacementProposal] = []

    for result in match_results:
        if result.status in (Pm4AssetMatchStatus.INELIGIBLE, Pm4AssetMatchStatus.UNRESOLVED):
            continue
        if not result.candidates:
            continue

        # Filter target tiles
        target_tiles = _filter_target_tiles(
            _get_segment_tiles(result),
            target_tile_coordinates,
        )
        if not target_tiles:
            continue

        selected = result.candidates[0]
        asset_ref = assets_by_id.get(selected.asset_id)
        if asset_ref is None:
            continue

        used_fallback_position = asset_ref.reference_position is None
        used_fallback_rotation = asset_ref.reference_rotation is None
        used_fallback_scale = asset_ref.reference_scale is None

        world_position = asset_ref.reference_position or _get_segment_center(result)
        world_rotation = asset_ref.reference_rotation or _build_fallback_rotation(result)
        world_scale = asset_ref.reference_scale if asset_ref.reference_scale is not None else 1.0
        confidence = max(0.0, min(1.0, selected.overall_score))
        review_required = (
            result.review_required
            or result.status != Pm4AssetMatchStatus.MATCHED
            or used_fallback_position
            or used_fallback_rotation
            or used_fallback_scale
        )

        provenance = [
            f"segment:{result.segment.segment_id}",
            f"asset:{selected.asset_id}",
            f"match-status:{_format_status(result.status)}",
            f"candidate-rank:{selected.rank}",
            f"candidate-score:{selected.overall_score:.4f}",
            "position:pm4-center-fallback" if used_fallback_position else "position:asset-reference",
            "rotation:pm4-heading-fallback" if used_fallback_rotation else "rotation:asset-reference",
            "scale:unit-fallback" if used_fallback_scale else "scale:asset-reference",
        ]

        proposal_id = _build_proposal_id(
            result.segment.segment_id,
            selected.asset_id,
            target_tiles,
        )

        proposals.append(Pm4ReplacementPlacementProposal(
            proposal_id=proposal_id,
            segment_id=result.segment.segment_id,
            asset_id=selected.asset_id,
            target_tile_coordinates=target_tiles,
            world_position=world_position,
            world_rotation=world_rotation,
            world_scale=world_scale,
            confidence=confidence,
            review_required=review_required,
            provenance=provenance,
        ))

    return proposals


def _filter_target_tiles(
    segment_tiles: list[str],
    normalized_target_tiles: set[str] | None,
) -> list[str]:
    if not normalized_target_tiles:
        return segment_tiles
    return [t for t in segment_tiles if t in normalized_target_tiles]


def _get_segment_tiles(result: Pm4SegmentMatchResult) -> list[str]:
    """Extract tile coordinates from a match result."""
    return result.segment.tile_coordinates


def _get_segment_center(result: Pm4SegmentMatchResult) -> tuple[float, float, float] | None:
    """Extract segment center from bounds."""
    if result.segment.bounds is None:
        return None
    b = result.segment.bounds
    return (
        (b.min[0] + b.max[0]) / 2.0,
        (b.min[1] + b.max[1]) / 2.0,
        (b.min[2] + b.max[2]) / 2.0,
    )
