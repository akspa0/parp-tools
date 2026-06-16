"""Import C# PM4 matching JSON exports into Python models.

C# System.Text.Json default serializer outputs PascalCase property names.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    Pm4AssetMatchCandidate,
    Pm4AssetMatchStatus,
    Pm4AssetReferenceSignalRecord,
    Pm4Bounds3,
    Pm4ReplacementPlacementProposal,
    Pm4SegmentAnchorSignals,
    Pm4SegmentHeightStats,
    Pm4SegmentMatchResult,
    Pm4SegmentSignalRecord,
    Pm4SegmentTopologyStats,
)


def _vec3(d: dict | None) -> tuple[float, float, float] | None:
    if d is None:
        return None
    return (
        float(d["X"]) if "X" in d else float(d["x"]),
        float(d["Y"]) if "Y" in d else float(d["y"]),
        float(d["Z"]) if "Z" in d else float(d["z"]),
    )


def _vec2(d: dict) -> tuple[float, float]:
    return (
        float(d["X"]) if "X" in d else float(d["x"]),
        float(d["Y"]) if "Y" in d else float(d["y"]),
    )


def _bounds3(d: dict | None) -> Pm4Bounds3 | None:
    if d is None:
        return None
    return Pm4Bounds3(min=_vec3(d["Min"]), max=_vec3(d["Max"]))


def _parse_height_stats(d: dict) -> Pm4SegmentHeightStats:
    return Pm4SegmentHeightStats(
        minimum_plane_distance=float(d["MinimumPlaneDistance"]),
        maximum_plane_distance=float(d["MaximumPlaneDistance"]),
        average_plane_distance=float(d["AveragePlaneDistance"]),
    )


def _parse_topology_stats(d: dict) -> Pm4SegmentTopologyStats:
    return Pm4SegmentTopologyStats(
        surface_count=int(d["SurfaceCount"]),
        total_index_count=int(d["TotalIndexCount"]),
        anchor_point_count=int(d["AnchorPointCount"]),
        anchor_normal_count=int(d["AnchorNormalCount"]),
    )


def _parse_anchor_signals(d: dict) -> Pm4SegmentAnchorSignals:
    return Pm4SegmentAnchorSignals(
        linked_position_ref_count=int(d["LinkedPositionRefCount"]),
        normal_heading_count=int(d["NormalHeadingCount"]),
        terminator_count=int(d["TerminatorCount"]),
        floor_minimum=int(d["FloorMinimum"]),
        floor_maximum=int(d["FloorMaximum"]),
        heading_minimum_degrees=d.get("HeadingMinimumDegrees"),
        heading_maximum_degrees=d.get("HeadingMaximumDegrees"),
        heading_mean_degrees=d.get("HeadingMeanDegrees"),
    )


def _parse_typed_bounds(d: dict | None) -> dict[int, Pm4Bounds3]:
    if d is None:
        return {}
    result: dict[int, Pm4Bounds3] = {}
    for key, val in d.items():
        flag = int(key, 16) if key.startswith("0x") else int(key)
        result[flag] = _bounds3(val)
    return result


def _parse_segment_signal(d: dict) -> Pm4SegmentSignalRecord:
    return Pm4SegmentSignalRecord(
        segment_id=d["SegmentId"],
        bounds=_bounds3(d.get("Bounds")),
        footprint_hull=[_vec2(p) for p in d.get("FootprintHull", [])],
        height_stats=_parse_height_stats(d["HeightStats"]),
        surface_family_histogram=dict(d.get("SurfaceFamilyHistogram", {})),
        topology_stats=_parse_topology_stats(d["TopologyStats"]),
        anchor_signals=_parse_anchor_signals(d["AnchorSignals"]),
        signal_version=d.get("SignalVersion", ""),
        signal_store_row=d.get("SignalStoreRow"),
        typed_bounds=_parse_typed_bounds(d.get("TypedBounds")),
        tile_coordinates=list(d.get("TileCoordinates", [])),
    )


def _parse_asset_reference(d: dict) -> Pm4AssetReferenceSignalRecord:
    return Pm4AssetReferenceSignalRecord(
        asset_id=d["AssetId"],
        asset_path=d["AssetPath"],
        asset_kind=d["AssetKind"],
        client_build=d.get("ClientBuild"),
        tile_coordinates=list(d.get("TileCoordinates", [])),
        bounds=_bounds3(d.get("Bounds")),
        center=_vec3(d.get("Center")) or (0.0, 0.0, 0.0),
        footprint_hull=[_vec2(p) for p in d.get("FootprintHull", [])],
        footprint_area=float(d.get("FootprintArea", 0)),
        reference_position=_vec3(d.get("ReferencePosition")),
        reference_rotation=_vec3(d.get("ReferenceRotation")),
        reference_scale=d.get("ReferenceScale"),
        surface_family_histogram=dict(d.get("SurfaceFamilyHistogram", {})),
        render_or_collision_signals=dict(d.get("RenderOrCollisionSignals", {})),
        signal_version=d.get("SignalVersion", ""),
        signal_store_row=d.get("SignalStoreRow"),
        validation_tags=list(d.get("ValidationTags", [])),
    )


def _parse_status(s: str | None) -> Pm4AssetMatchStatus:
    if s is None:
        return Pm4AssetMatchStatus.UNRESOLVED
    return Pm4AssetMatchStatus(s.lower())


def _parse_candidate(d: dict) -> Pm4AssetMatchCandidate:
    return Pm4AssetMatchCandidate(
        asset_id=d["AssetId"],
        asset_path=d["AssetPath"],
        asset_kind=d["AssetKind"],
        rank=int(d["Rank"]),
        overall_score=float(d["OverallScore"]),
        status=_parse_status(d.get("Status")),
        score_breakdown=dict(d.get("ScoreBreakdown", {})),
        rationale=list(d.get("Rationale", [])),
    )


def _parse_segment_from_report(d: dict) -> Pm4SegmentSignalRecord:
    height_stats_raw = d.get("HeightStats")
    topology_stats_raw = d.get("TopologyStats")
    anchor_signals_raw = d.get("AnchorSignals")

    height_stats = (
        _parse_height_stats(height_stats_raw)
        if height_stats_raw
        else Pm4SegmentHeightStats(0, 0, 0)
    )
    topology_stats = (
        _parse_topology_stats(topology_stats_raw)
        if topology_stats_raw
        else Pm4SegmentTopologyStats(0, 0, 0, 0)
    )
    anchor_signals = (
        _parse_anchor_signals(anchor_signals_raw)
        if anchor_signals_raw
        else Pm4SegmentAnchorSignals(0, 0, 0, 0, 0, None, None, None)
    )

    return Pm4SegmentSignalRecord(
        segment_id=d["SegmentId"],
        bounds=_bounds3(d.get("Bounds")),
        footprint_hull=[_vec2(p) for p in d.get("FootprintHull", [])],
        height_stats=height_stats,
        surface_family_histogram=dict(d.get("SurfaceFamilyHistogram", {})),
        topology_stats=topology_stats,
        anchor_signals=anchor_signals,
        signal_version="",
        signal_store_row=None,
        typed_bounds=_parse_typed_bounds(d.get("TypedBounds")),
        tile_coordinates=list(d.get("TileCoordinates", [])),
    )


def _parse_match_result(d: dict, segment: Pm4SegmentSignalRecord) -> Pm4SegmentMatchResult:
    candidates = [_parse_candidate(c) for c in d.get("Candidates", [])]
    return Pm4SegmentMatchResult(
        segment=segment,
        expected_asset_kind=d.get("ExpectedAssetKind"),
        status=_parse_status(d.get("Status")),
        review_required=bool(d.get("ReviewRequired", False)),
        rationale=list(d.get("Rationale", [])),
        candidates=candidates,
    )


def _parse_proposal(d: dict) -> Pm4ReplacementPlacementProposal:
    rot = d.get("WorldRotation")
    rotation = None
    if rot is not None:
        rotation = (
            float(rot.get("Yaw", 0) or 0),
            float(rot.get("Pitch", 0) or 0),
            float(rot.get("Roll", 0) or 0),
        )
    return Pm4ReplacementPlacementProposal(
        proposal_id=d["ProposalId"],
        segment_id=d.get("SegmentId", ""),
        asset_id=d.get("AssetId", ""),
        target_tile_coordinates=list(d.get("TargetTileCoordinates", [])),
        world_position=_vec3(d.get("WorldPosition")),
        world_rotation=rotation,
        world_scale=d.get("WorldScale"),
        confidence=float(d.get("Confidence", 0)),
        review_required=bool(d.get("ReviewRequired", False)),
        provenance=list(d.get("Provenance", [])),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_segment_export(json_path: str | Path) -> list[Pm4SegmentSignalRecord]:
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments_raw = data.get("Segments", [])
    return [_parse_segment_from_report(seg) for seg in segments_raw]


def import_asset_corpus(json_path: str | Path) -> list[Pm4AssetReferenceSignalRecord]:
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assets_raw = data.get("Assets", [])
    return [_parse_asset_reference(a) for a in assets_raw]


def import_match_report(
    json_path: str | Path,
) -> tuple[list[Pm4SegmentMatchResult], list[Pm4ReplacementPlacementProposal]]:
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments_raw = data.get("Segments", [])
    match_results: list[Pm4SegmentMatchResult] = []
    proposals: list[Pm4ReplacementPlacementProposal] = []

    for seg_raw in segments_raw:
        segment = _parse_segment_from_report(seg_raw)
        match_result = _parse_match_result(seg_raw, segment)
        match_results.append(match_result)

        proposal_raw = seg_raw.get("PlacementProposal")
        if proposal_raw is not None:
            proposal = _parse_proposal(proposal_raw)
            proposals.append(proposal)

    return match_results, proposals
