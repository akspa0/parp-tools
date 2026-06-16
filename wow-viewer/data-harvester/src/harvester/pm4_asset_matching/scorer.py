"""Python port of C# Pm4AssetMatchScorer.

Implements the same scoring algorithm: typed overlap (35%) + type profile (15%)
+ shape (50%).  Produces identical scores to the C# implementation for the
same inputs.
"""

from __future__ import annotations

import json
import math

from .models import (
    Pm4AssetMatchCandidate,
    Pm4AssetMatchStatus,
    Pm4AssetReferenceSignalRecord,
    Pm4Bounds3,
    Pm4SegmentMatchResult,
    Pm4SegmentSignalRecord,
)

MINIMUM_MATCHED_SCORE = 0.45
AMBIGUOUS_SCORE_WINDOW = 0.03

# Known MSLK.TypeFlags values
_TYPE_FLAG_M2_TOP = 0x03
_TYPE_FLAG_INTERIOR_FLOOR = 0x10
_TYPE_FLAG_EXTERIOR_SOLID = 0x12


def resolve_expected_asset_kind(ck24_type: int) -> str | None:
    """Map ck24Type to expected asset kind. Returns None if not matchable."""
    if ck24_type in (0x42, 0x43):
        return "wmo"
    if ck24_type in (0x40, 0x41, 0xC0, 0xC1, 0xC2, 0xC3):
        return "m2"
    return None


def compute_bounds_overlap_ratio(left: Pm4Bounds3, right: Pm4Bounds3) -> float:
    """Axis-aligned bounding box overlap in XY (footprint plane).

    Returns Jaccard-like ratio: intersection / union.
    """
    overlap_x = max(0.0, min(left.max[0], right.max[0]) - max(left.min[0], right.min[0]))
    overlap_y = max(0.0, min(left.max[1], right.max[1]) - max(left.min[1], right.min[1]))
    left_area = (left.max[0] - left.min[0]) * (left.max[1] - left.min[1])
    right_area = (right.max[0] - right.min[0]) * (right.max[1] - right.min[1])
    intersection_area = overlap_x * overlap_y

    if left_area <= 0 or right_area <= 0:
        return 0.0

    union_area = left_area + right_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def score_ratio(left: float, right: float) -> float:
    """Score ratio: min/max. Returns 0 if either is non-positive."""
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0 or right <= 0:
        return 0.0
    mn = min(left, right)
    mx = max(left, right)
    return mn / mx if mx > 0 else 0.0


def score_distance(distance: float, scale: float) -> float:
    """Distance-based score: 1 / (1 + distance/scale)."""
    if not math.isfinite(distance):
        return 0.0
    if distance <= 0.0:
        return 1.0
    return 1.0 / (1.0 + distance / max(0.001, scale))


def _compute_footprint_area(points: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _describe_type_profile(typed_bounds: dict[int, Pm4Bounds3]) -> str:
    if not typed_bounds:
        return "none"
    parts = []
    for flag in sorted(typed_bounds.keys()):
        label = {
            _TYPE_FLAG_M2_TOP: "m2-top(0x03)",
            _TYPE_FLAG_INTERIOR_FLOOR: "interior-floor(0x10)",
            _TYPE_FLAG_EXTERIOR_SOLID: "exterior-solid(0x12)",
        }.get(flag, f"0x{flag:02X}")
        parts.append(label)
    return ", ".join(parts)


def _parse_sub_part_bounds(asset: Pm4AssetReferenceSignalRecord) -> list[dict] | None:
    """Parse sub-part bounds from asset signal store row."""
    if not asset.signal_store_row:
        return None
    prefix = "subPartBounds:"
    if not asset.signal_store_row.startswith(prefix):
        return None
    json_str = asset.signal_store_row[len(prefix):]
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def _evaluate_typed_candidate(
    segment: Pm4SegmentSignalRecord,
    asset: Pm4AssetReferenceSignalRecord,
    typed_bounds: dict[int, Pm4Bounds3],
    profile_matches_expected_kind: bool,
    has_type_flags_data: bool,
    segment_center: tuple[float, float, float] | None,
    segment_tiles: list[str],
) -> dict | None:
    """Evaluate a single asset candidate against a segment.

    Returns a dict with scores and breakdown, or None if not scorable.
    """
    if asset.bounds is None or segment.bounds is None:
        return None

    # 1. TypeFlags profile match score
    profile_score = 1.0 if profile_matches_expected_kind else (0.3 if has_type_flags_data else 0.5)

    # 2. Per-type-class overlap
    typed_overlap_score = 0.0
    typed_count = 0
    sub_parts = _parse_sub_part_bounds(asset)

    for _type_flag, seg_bounds in typed_bounds.items():
        if seg_bounds.min == seg_bounds.max:
            continue

        overlap = compute_bounds_overlap_ratio(seg_bounds, asset.bounds)

        if sub_parts:
            best_part_overlap = 0.0
            for part in sub_parts:
                part_bounds = Pm4Bounds3(
                    min=(part["MinX"], part["MinY"], part["MinZ"]),
                    max=(part["MaxX"], part["MaxY"], part["MaxZ"]),
                )
                part_overlap = compute_bounds_overlap_ratio(seg_bounds, part_bounds)
                best_part_overlap = max(best_part_overlap, part_overlap)
            overlap = overlap * 0.4 + best_part_overlap * 0.6

        typed_overlap_score += overlap
        typed_count += 1

    typed_overlap_score = typed_overlap_score / typed_count if typed_count > 0 else 0.5

    # 3. Shape similarity
    seg_span = tuple(segment.bounds.max[i] - segment.bounds.min[i] for i in range(3))
    asset_span = tuple(asset.bounds.max[i] - asset.bounds.min[i] for i in range(3))

    sorted_seg = sorted(seg_span, reverse=True)
    sorted_asset = sorted(asset_span, reverse=True)

    span_score0 = score_ratio(sorted_seg[0], sorted_asset[0])
    span_score1 = score_ratio(sorted_seg[1], sorted_asset[1])
    span_score2 = score_ratio(sorted_seg[2], sorted_asset[2])
    sorted_span_score = (span_score0 + span_score1 + span_score2) / 3.0

    # Same-tile bonus
    same_tile_bonus = 0.0
    if asset.tile_coordinates and asset.reference_position and segment_center:
        shares_tile = any(
            t in segment_tiles for t in asset.tile_coordinates
        )
        if shares_tile:
            cx = segment_center[0] - asset.center[0]
            cy = segment_center[1] - asset.center[1]
            cz = segment_center[2] - asset.center[2]
            center_dist = math.sqrt(cx * cx + cy * cy + cz * cz)
            same_tile_bonus = score_distance(center_dist, 64.0)

    seg_footprint = _compute_footprint_area(segment.footprint_hull)
    asset_footprint = asset.footprint_area
    footprint_score = score_ratio(max(0.0, seg_footprint), max(0.0, asset_footprint))

    seg_volume = max(0.0, seg_span[0]) * max(0.0, seg_span[1]) * max(0.0, seg_span[2])
    asset_volume = max(0.0, asset_span[0]) * max(0.0, asset_span[1]) * max(0.0, asset_span[2])
    volume_score = score_ratio(seg_volume, asset_volume)

    seg_diag = math.sqrt(seg_span[0] ** 2 + seg_span[1] ** 2)
    asset_diag = math.sqrt(asset_span[0] ** 2 + asset_span[1] ** 2)
    diagonal_score = score_ratio(seg_diag, asset_diag)

    height_score = score_ratio(seg_span[2], asset_span[2])

    seg_aspect = seg_span[0] / seg_span[1] if seg_span[1] > 0 else 0.0
    asset_aspect = asset_span[0] / asset_span[1] if asset_span[1] > 0 else 0.0
    aspect_score = score_ratio(seg_aspect, asset_aspect)

    shape_score = (
        sorted_span_score * 0.25
        + footprint_score * 0.15
        + volume_score * 0.15
        + diagonal_score * 0.12
        + height_score * 0.10
        + aspect_score * 0.08
        + same_tile_bonus * 0.15
    )

    # 4. Combined score
    type_weight = 0.35 if has_type_flags_data else 0.0
    profile_weight = 0.15 if has_type_flags_data else 0.0
    shape_weight = 1.0 - type_weight - profile_weight

    overall_score = (
        typed_overlap_score * type_weight
        + profile_score * profile_weight
        + shape_score * shape_weight
    )

    score_breakdown = {
        "typeProfileScore": profile_score,
        "typedOverlapScore": typed_overlap_score,
        "sortedSpanScore": sorted_span_score,
        "footprintAreaScore": footprint_score,
        "volumeScore": volume_score,
        "diagonalScore": diagonal_score,
        "heightScore": height_score,
        "aspectScore": aspect_score,
        "shapeScore": shape_score,
        "typeWeight": type_weight,
        "profileWeight": profile_weight,
        "shapeWeight": shape_weight,
    }

    rationale = [
        f"typed overlap {typed_overlap_score:.3f} (typeWeight={type_weight:.2f})",
        f"shape score {shape_score:.3f} (shapeWeight={shape_weight:.2f})",
        f"type profile {profile_score:.3f} (profileWeight={profile_weight:.2f})",
    ]

    return {
        "asset": asset,
        "typed_overlap_score": typed_overlap_score,
        "shape_score": shape_score,
        "overall_score": overall_score,
        "score_breakdown": score_breakdown,
        "rationale": rationale,
    }


def _build_segment_rationale(
    segment: Pm4SegmentSignalRecord,
    ck24_type: int | None = None,
) -> list[str]:
    """Build rationale strings for a segment."""
    rationale = [
        f"segment family ck24Type=0x{ck24_type or 0:02X}",
        f"surfaces={segment.topology_stats.surface_count} indices={segment.topology_stats.total_index_count}",
    ]
    return rationale


def _resolve_candidate_status(
    segment_status: Pm4AssetMatchStatus,
    index: int,
    overall_score: float,
    top_score: float,
) -> Pm4AssetMatchStatus:
    if segment_status == Pm4AssetMatchStatus.UNRESOLVED:
        return Pm4AssetMatchStatus.UNRESOLVED
    if segment_status == Pm4AssetMatchStatus.AMBIGUOUS:
        if abs(top_score - overall_score) <= AMBIGUOUS_SCORE_WINDOW:
            return Pm4AssetMatchStatus.AMBIGUOUS
        return Pm4AssetMatchStatus.UNRESOLVED
    return Pm4AssetMatchStatus.MATCHED if index == 0 else Pm4AssetMatchStatus.UNRESOLVED


def score_segment(
    segment: Pm4SegmentSignalRecord,
    asset_references: list[Pm4AssetReferenceSignalRecord],
    max_candidates: int = 10,
    ck24_type: int = 0,
    ck24_object_id: int = 0,
    segment_tiles: list[str] | None = None,
    segment_center: tuple[float, float, float] | None = None,
) -> Pm4SegmentMatchResult:
    """Score a single segment against all asset references."""
    max_candidates = max(1, max_candidates)
    rationale = _build_segment_rationale(segment, ck24_type)
    expected_kind = resolve_expected_asset_kind(ck24_type)

    if expected_kind is None:
        rationale.append(f"ck24Type 0x{ck24_type:02X} is not currently treated as WMO/M2-matchable.")
        return Pm4SegmentMatchResult(
            segment=segment,
            expected_asset_kind=None,
            status=Pm4AssetMatchStatus.INELIGIBLE,
            review_required=True,
            rationale=rationale,
            candidates=[],
        )

    if segment.bounds is None:
        rationale.append("segment has no usable bounds, so geometry scoring is not possible.")
        return Pm4SegmentMatchResult(
            segment=segment,
            expected_asset_kind=expected_kind,
            status=Pm4AssetMatchStatus.UNRESOLVED,
            review_required=True,
            rationale=rationale,
            candidates=[],
        )

    # Build TypeFlags profile
    typed_bounds = dict(segment.typed_bounds)
    has_type_flags_data = len(typed_bounds) > 0
    has_exterior_solid = _TYPE_FLAG_EXTERIOR_SOLID in typed_bounds
    has_interior_floor = _TYPE_FLAG_INTERIOR_FLOOR in typed_bounds
    has_m2_top = _TYPE_FLAG_M2_TOP in typed_bounds

    type_profile = _describe_type_profile(typed_bounds)
    rationale.append(f"TypeFlags profile: {type_profile}")

    profile_matches = {
        "wmo": has_exterior_solid or has_interior_floor,
        "m2": has_m2_top,
    }.get(expected_kind, False)

    if profile_matches:
        rationale.append(f"TypeFlags profile is consistent with {expected_kind} expectation.")
    elif has_type_flags_data:
        rationale.append(f"TypeFlags profile does not match typical {expected_kind} pattern — scoring with reduced weight.")
    else:
        rationale.append("no TypeFlags surface classification available — scoring on shape only.")

    tiles = segment_tiles or []

    # Evaluate candidates
    evaluations = []
    for asset in asset_references:
        if asset.asset_kind.lower() != expected_kind.lower():
            continue
        result = _evaluate_typed_candidate(
            segment, asset, typed_bounds, profile_matches, has_type_flags_data,
            segment_center, tiles,
        )
        if result is not None:
            evaluations.append(result)

    evaluations.sort(key=lambda e: (-e["overall_score"], -e["typed_overlap_score"], -e["shape_score"]))
    evaluations = evaluations[:max_candidates]

    if not evaluations:
        rationale.append(f"no {expected_kind} validation references were available to score against this segment.")
        return Pm4SegmentMatchResult(
            segment=segment,
            expected_asset_kind=expected_kind,
            status=Pm4AssetMatchStatus.UNRESOLVED,
            review_required=True,
            rationale=rationale,
            candidates=[],
        )

    top_score = evaluations[0]["overall_score"]
    second_score = evaluations[1]["overall_score"] if len(evaluations) > 1 else float("-inf")

    if top_score < MINIMUM_MATCHED_SCORE:
        status = Pm4AssetMatchStatus.UNRESOLVED
    elif abs(top_score - second_score) <= AMBIGUOUS_SCORE_WINDOW:
        status = Pm4AssetMatchStatus.AMBIGUOUS
    else:
        status = Pm4AssetMatchStatus.MATCHED

    if status == Pm4AssetMatchStatus.MATCHED:
        rationale.append(f"top {expected_kind} candidate '{evaluations[0]['asset'].asset_path}' cleared the score floor at {top_score:.3f}.")
    elif status == Pm4AssetMatchStatus.AMBIGUOUS:
        rationale.append(f"top {expected_kind} candidates are too close to separate confidently ({top_score:.3f} vs {second_score:.3f}).")
    else:
        rationale.append(f"best {expected_kind} candidate scored only {top_score:.3f}, below the {MINIMUM_MATCHED_SCORE:.2f} acceptance floor.")

    candidates = []
    for idx, ev in enumerate(evaluations):
        cand_status = _resolve_candidate_status(status, idx, ev["overall_score"], top_score)
        candidates.append(Pm4AssetMatchCandidate(
            asset_id=ev["asset"].asset_id,
            asset_path=ev["asset"].asset_path,
            asset_kind=ev["asset"].asset_kind,
            rank=idx + 1,
            overall_score=ev["overall_score"],
            status=cand_status,
            score_breakdown=ev["score_breakdown"],
            rationale=ev["rationale"],
        ))

    review_required = status != Pm4AssetMatchStatus.MATCHED
    return Pm4SegmentMatchResult(
        segment=segment,
        expected_asset_kind=expected_kind,
        status=status,
        review_required=review_required,
        rationale=rationale,
        candidates=candidates,
    )


def score_segments(
    segments: list[Pm4SegmentSignalRecord],
    asset_references: list[Pm4AssetReferenceSignalRecord],
    max_candidates: int = 10,
    segment_ck24_types: list[int] | None = None,
    segment_tiles: list[list[str]] | None = None,
    segment_centers: list[tuple[float, float, float] | None] | None = None,
) -> list[Pm4SegmentMatchResult]:
    """Score all segments against asset references."""
    n = len(segments)
    ck24_types = segment_ck24_types or [0] * n
    tiles = segment_tiles or [[] for _ in range(n)]
    centers = segment_centers or [None] * n

    return [
        score_segment(
            seg, asset_references, max_candidates,
            ck24_type=ck24_types[i],
            segment_tiles=tiles[i],
            segment_center=centers[i],
        )
        for i, seg in enumerate(segments)
    ]
