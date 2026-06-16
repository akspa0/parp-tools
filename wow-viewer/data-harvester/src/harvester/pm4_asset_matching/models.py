"""Python data models mirroring C# Pm4MatchingModels records.

These dataclasses match the JSON schema exported by the C# PM4 matching tools
(camelCase keys, nested Vector2/Vector3 as {x,y}/{x,y,z}).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Pm4AssetMatchStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True)
class Pm4Bounds3:
    min: tuple[float, float, float]
    max: tuple[float, float, float]


@dataclass(frozen=True)
class Pm4SegmentHeightStats:
    minimum_plane_distance: float
    maximum_plane_distance: float
    average_plane_distance: float


@dataclass(frozen=True)
class Pm4SegmentTopologyStats:
    surface_count: int
    total_index_count: int
    anchor_point_count: int
    anchor_normal_count: int


@dataclass(frozen=True)
class Pm4SegmentAnchorSignals:
    linked_position_ref_count: int
    normal_heading_count: int
    terminator_count: int
    floor_minimum: int
    floor_maximum: int
    heading_minimum_degrees: float | None
    heading_maximum_degrees: float | None
    heading_mean_degrees: float | None


@dataclass
class Pm4SegmentSignalRecord:
    segment_id: str
    bounds: Pm4Bounds3 | None
    footprint_hull: list[tuple[float, float]]
    height_stats: Pm4SegmentHeightStats
    surface_family_histogram: dict[str, int]
    topology_stats: Pm4SegmentTopologyStats
    anchor_signals: Pm4SegmentAnchorSignals
    signal_version: str
    signal_store_row: str | None
    typed_bounds: dict[int, Pm4Bounds3] = field(default_factory=dict)
    tile_coordinates: list[str] = field(default_factory=list)


@dataclass
class Pm4AssetReferenceSignalRecord:
    asset_id: str
    asset_path: str
    asset_kind: str
    client_build: str | None
    tile_coordinates: list[str]
    bounds: Pm4Bounds3 | None
    center: tuple[float, float, float]
    footprint_hull: list[tuple[float, float]]
    footprint_area: float
    reference_position: tuple[float, float, float] | None
    reference_rotation: tuple[float, float, float] | None
    reference_scale: float | None
    surface_family_histogram: dict[str, int]
    render_or_collision_signals: dict[str, float]
    signal_version: str
    signal_store_row: str | None
    validation_tags: list[str] = field(default_factory=list)


@dataclass
class Pm4AssetMatchCandidate:
    asset_id: str
    asset_path: str
    asset_kind: str
    rank: int
    overall_score: float
    status: Pm4AssetMatchStatus
    score_breakdown: dict[str, float]
    rationale: list[str]


@dataclass
class Pm4SegmentMatchResult:
    segment: Pm4SegmentSignalRecord
    expected_asset_kind: str | None
    status: Pm4AssetMatchStatus
    review_required: bool
    rationale: list[str]
    candidates: list[Pm4AssetMatchCandidate]


@dataclass
class Pm4ReplacementPlacementProposal:
    proposal_id: str
    segment_id: str
    asset_id: str
    target_tile_coordinates: list[str]
    world_position: tuple[float, float, float] | None
    world_rotation: tuple[float, float, float] | None  # (yaw, pitch, roll) in degrees
    world_scale: float | None
    confidence: float
    review_required: bool
    provenance: list[str]
