"""Tests for PM4 asset matching JSON import."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvester.pm4_asset_matching import (
    Pm4AssetMatchCandidate,
    Pm4AssetMatchStatus,
    Pm4AssetReferenceSignalRecord,
    Pm4Bounds3,
    Pm4SegmentAnchorSignals,
    Pm4SegmentHeightStats,
    Pm4SegmentMatchResult,
    Pm4SegmentSignalRecord,
    Pm4SegmentTopologyStats,
    import_asset_corpus,
    import_match_report,
    import_segment_export,
    read_asset_references_zarr,
    read_segment_signals_zarr,
    write_asset_references_zarr,
    write_segment_signals_zarr,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic JSON matching C# export format
# ---------------------------------------------------------------------------

SEGMENT_EXPORT_JSON = {
    "RunId": "pm4-export-test123",
    "InputPm4Root": "/test/pm4",
    "SegmentCount": 2,
    "Segments": [
        {
            "SegmentId": "seg-ck24:0x424242-0",
            "Ck24": "0x424242",
            "Ck24Type": 66,
            "Ck24ObjectId": 1,
            "TileCoordinates": ["33_32"],
            "Field04Values": [1, 2],
            "ExpectedAssetKind": "wmo",
            "Status": None,
            "ReviewRequired": False,
            "Rationale": [],
            "ConfidenceFlags": None,
            "SurfaceCount": 5,
            "TotalIndexCount": 120,
            "LinkGroupIds": ["0x1"],
            "DominantLinkGroupId": "0x1",
            "CoordinateMode": "Pm4",
            "AxisConvention": "Standard",
            "FrameYawDegrees": 0.0,
            "Bounds": {
                "Min": {"X": 100.0, "Y": 200.0, "Z": 10.0},
                "Max": {"X": 150.0, "Y": 250.0, "Z": 30.0},
            },
            "Center": {"X": 125.0, "Y": 225.0, "Z": 20.0},
            "FootprintHull": [
                {"X": 100.0, "Y": 200.0},
                {"X": 150.0, "Y": 200.0},
                {"X": 150.0, "Y": 250.0},
                {"X": 100.0, "Y": 250.0},
            ],
            "FootprintArea": 2500.0,
            "HeightStats": {
                "MinimumPlaneDistance": 10.0,
                "MaximumPlaneDistance": 30.0,
                "AveragePlaneDistance": 20.0,
            },
            "TopologyStats": {
                "SurfaceCount": 5,
                "TotalIndexCount": 120,
                "AnchorPointCount": 3,
                "AnchorNormalCount": 2,
            },
            "AnchorSignals": {
                "LinkedPositionRefCount": 3,
                "NormalHeadingCount": 2,
                "TerminatorCount": 1,
                "FloorMinimum": 10,
                "FloorMaximum": 30,
                "HeadingMinimumDegrees": 0.0,
                "HeadingMaximumDegrees": 90.0,
                "HeadingMeanDegrees": 45.0,
            },
            "SurfaceFamilyHistogram": {
                "ck24Type:0x42": 5,
                "groupKey:0x01": 3,
                "groupKey:0x02": 2,
            },
            "Candidates": [],
            "PlacementProposal": None,
        },
        {
            "SegmentId": "seg-ck24:0x404040-0",
            "Ck24": "0x404040",
            "Ck24Type": 64,
            "Ck24ObjectId": 2,
            "TileCoordinates": ["33_32"],
            "Field04Values": [3],
            "ExpectedAssetKind": "m2",
            "Status": None,
            "ReviewRequired": False,
            "Rationale": [],
            "ConfidenceFlags": None,
            "SurfaceCount": 2,
            "TotalIndexCount": 48,
            "LinkGroupIds": ["0x2"],
            "DominantLinkGroupId": "0x2",
            "CoordinateMode": "Pm4",
            "AxisConvention": "Standard",
            "FrameYawDegrees": 45.0,
            "Bounds": {
                "Min": {"X": 50.0, "Y": 50.0, "Z": 5.0},
                "Max": {"X": 60.0, "Y": 70.0, "Z": 15.0},
            },
            "Center": {"X": 55.0, "Y": 60.0, "Z": 10.0},
            "FootprintHull": [
                {"X": 50.0, "Y": 50.0},
                {"X": 60.0, "Y": 50.0},
                {"X": 60.0, "Y": 70.0},
                {"X": 50.0, "Y": 70.0},
            ],
            "FootprintArea": 200.0,
            "HeightStats": {
                "MinimumPlaneDistance": 5.0,
                "MaximumPlaneDistance": 15.0,
                "AveragePlaneDistance": 10.0,
            },
            "TopologyStats": {
                "SurfaceCount": 2,
                "TotalIndexCount": 48,
                "AnchorPointCount": 1,
                "AnchorNormalCount": 1,
            },
            "AnchorSignals": {
                "LinkedPositionRefCount": 1,
                "NormalHeadingCount": 1,
                "TerminatorCount": 0,
                "FloorMinimum": 5,
                "FloorMaximum": 15,
                "HeadingMinimumDegrees": 10.0,
                "HeadingMaximumDegrees": 80.0,
                "HeadingMeanDegrees": 45.0,
            },
            "SurfaceFamilyHistogram": {
                "ck24Type:0x40": 2,
            },
            "Candidates": [],
            "PlacementProposal": None,
        },
    ],
    "AssetReferenceCorpus": None,
    "SegmentSignalCorpus": "pm4-segment-signal-v2",
    "MatchedCount": 0,
    "AmbiguousCount": 0,
    "UnresolvedCount": 0,
    "IneligibleCount": 0,
    "Warnings": [],
}

ASSET_CORPUS_JSON = {
    "RunId": "pm4-asset-corpus-test-build-abc123",
    "ArchiveRoot": "/test/client",
    "ClientBuild": "3.3.5.12340",
    "AssetCount": 2,
    "Assets": [
        {
            "AssetId": "wmo:3.3.5.12340:world/wmo/azeroth/buildings/human_farm/farm.wmo",
            "AssetPath": "World/WMO/Azeroth/Buildings/Human_Farm/Farm.wmo",
            "AssetKind": "wmo",
            "ClientBuild": "3.3.5.12340",
            "TileCoordinates": [],
            "Bounds": {
                "Min": {"X": -10.0, "Y": -15.0, "Z": 0.0},
                "Max": {"X": 10.0, "Y": 15.0, "Z": 8.0},
            },
            "Center": {"X": 0.0, "Y": 0.0, "Z": 4.0},
            "FootprintHull": [
                {"X": -10.0, "Y": -15.0},
                {"X": 10.0, "Y": -15.0},
                {"X": 10.0, "Y": 15.0},
                {"X": -10.0, "Y": 15.0},
            ],
            "FootprintArea": 600.0,
            "ReferencePosition": None,
            "ReferenceRotation": None,
            "ReferenceScale": None,
            "SurfaceFamilyHistogram": {"assetKind:wmo": 1},
            "RenderOrCollisionSignals": {
                "boundsSpanX": 20.0,
                "boundsSpanY": 30.0,
                "boundsSpanZ": 8.0,
                "boundsVolume": 4800.0,
                "footprintDiagonalXY": 36.06,
            },
            "SignalVersion": "pm4-asset-reference-signal-v1",
            "SignalStoreRow": None,
            "ValidationTags": ["durable-asset-corpus"],
        },
        {
            "AssetId": "m2:3.3.5.12340:world/kalimdor/durotar/tree01.m2",
            "AssetPath": "World/Kalimdor/Durotar/Tree01.m2",
            "AssetKind": "m2",
            "ClientBuild": "3.3.5.12340",
            "TileCoordinates": [],
            "Bounds": {
                "Min": {"X": -3.0, "Y": -3.0, "Z": 0.0},
                "Max": {"X": 3.0, "Y": 3.0, "Z": 12.0},
            },
            "Center": {"X": 0.0, "Y": 0.0, "Z": 6.0},
            "FootprintHull": [
                {"X": -3.0, "Y": -3.0},
                {"X": 3.0, "Y": -3.0},
                {"X": 3.0, "Y": 3.0},
                {"X": -3.0, "Y": 3.0},
            ],
            "FootprintArea": 36.0,
            "ReferencePosition": None,
            "ReferenceRotation": None,
            "ReferenceScale": None,
            "SurfaceFamilyHistogram": {"assetKind:m2": 1},
            "RenderOrCollisionSignals": {
                "boundsSpanX": 6.0,
                "boundsSpanY": 6.0,
                "boundsSpanZ": 12.0,
                "boundsVolume": 432.0,
                "footprintDiagonalXY": 8.49,
            },
            "SignalVersion": "pm4-asset-reference-signal-v1",
            "SignalStoreRow": None,
            "ValidationTags": ["durable-asset-corpus"],
        },
    ],
    "Warnings": [],
}

MATCH_REPORT_JSON = {
    "RunId": "pm4-export-test123:match-assets:corpus",
    "InputPm4Root": "/test/pm4",
    "SegmentCount": 1,
    "Segments": [
        {
            "SegmentId": "seg-ck24:0x424242-0",
            "Ck24": "0x424242",
            "Ck24Type": 66,
            "Ck24ObjectId": 1,
            "TileCoordinates": ["33_32"],
            "Field04Values": [1, 2],
            "ExpectedAssetKind": "wmo",
            "Status": "matched",
            "ReviewRequired": False,
            "Rationale": [
                "segment family ck24Type=0x42",
                "surfaces=5 indices=120",
                "TypeFlags profile is consistent with wmo expectation.",
                "top wmo candidate 'World/WMO/Azeroth/Buildings/Human_Farm/Farm.wmo' cleared the score floor at 0.523.",
            ],
            "ConfidenceFlags": None,
            "SurfaceCount": 5,
            "TotalIndexCount": 120,
            "LinkGroupIds": ["0x1"],
            "DominantLinkGroupId": "0x1",
            "CoordinateMode": "Pm4",
            "AxisConvention": "Standard",
            "FrameYawDegrees": 0.0,
            "Bounds": {
                "Min": {"X": 100.0, "Y": 200.0, "Z": 10.0},
                "Max": {"X": 150.0, "Y": 250.0, "Z": 30.0},
            },
            "Center": {"X": 125.0, "Y": 225.0, "Z": 20.0},
            "FootprintHull": [
                {"X": 100.0, "Y": 200.0},
                {"X": 150.0, "Y": 200.0},
                {"X": 150.0, "Y": 250.0},
                {"X": 100.0, "Y": 250.0},
            ],
            "FootprintArea": 2500.0,
            "HeightStats": {
                "MinimumPlaneDistance": 10.0,
                "MaximumPlaneDistance": 30.0,
                "AveragePlaneDistance": 20.0,
            },
            "TopologyStats": {
                "SurfaceCount": 5,
                "TotalIndexCount": 120,
                "AnchorPointCount": 3,
                "AnchorNormalCount": 2,
            },
            "AnchorSignals": {
                "LinkedPositionRefCount": 3,
                "NormalHeadingCount": 2,
                "TerminatorCount": 1,
                "FloorMinimum": 10,
                "FloorMaximum": 30,
                "HeadingMinimumDegrees": 0.0,
                "HeadingMaximumDegrees": 90.0,
                "HeadingMeanDegrees": 45.0,
            },
            "SurfaceFamilyHistogram": {
                "ck24Type:0x42": 5,
            },
            "Candidates": [
                {
                    "AssetId": "wmo:3.3.5.12340:world/wmo/azeroth/buildings/human_farm/farm.wmo",
                    "AssetPath": "World/WMO/Azeroth/Buildings/Human_Farm/Farm.wmo",
                    "AssetKind": "wmo",
                    "Rank": 1,
                    "OverallScore": 0.523,
                    "Status": "matched",
                    "ScoreBreakdown": {
                        "typeProfileScore": 1.0,
                        "typedOverlapScore": 0.45,
                        "shapeScore": 0.38,
                    },
                    "Rationale": [
                        "typed overlap 0.450 (typeWeight=0.35)",
                        "shape score 0.380 (shapeWeight=0.50)",
                        "type profile 1.000 (profileWeight=0.15)",
                    ],
                },
            ],
            "PlacementProposal": {
                "ProposalId": "proposal-abcdef1234567890",
                "SegmentId": "seg-ck24:0x424242-0",
                "AssetId": "wmo:3.3.5.12340:world/wmo/azeroth/buildings/human_farm/farm.wmo",
                "TargetTileCoordinates": ["33_32"],
                "WorldPosition": {"X": 125.0, "Y": 225.0, "Z": 20.0},
                "WorldRotation": {"Yaw": 45.0, "Pitch": 0.0, "Roll": 0.0},
                "WorldScale": 1.0,
                "Confidence": 0.523,
                "ReviewRequired": False,
                "Provenance": [
                    "segment:seg-ck24:0x424242-0",
                    "asset:wmo:3.3.5.12340:world/wmo/azeroth/buildings/human_farm/farm.wmo",
                    "match-status:matched",
                ],
            },
        },
    ],
    "AssetReferenceCorpus": "/test/corpus.json",
    "SegmentSignalCorpus": "pm4-segment-signal-v2",
    "MatchedCount": 1,
    "AmbiguousCount": 0,
    "UnresolvedCount": 0,
    "IneligibleCount": 0,
    "Warnings": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(data: dict, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests: Segment Export Import
# ---------------------------------------------------------------------------


class TestImportSegmentExport:
    def test_imports_all_segments(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        assert len(segments) == 2

    def test_parses_segment_id(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        assert segments[0].segment_id == "seg-ck24:0x424242-0"
        assert segments[1].segment_id == "seg-ck24:0x404040-0"

    def test_parses_bounds(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        bounds = segments[0].bounds
        assert bounds is not None
        assert bounds.min == (100.0, 200.0, 10.0)
        assert bounds.max == (150.0, 250.0, 30.0)

    def test_parses_footprint_hull(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        hull = segments[0].footprint_hull
        assert len(hull) == 4
        assert hull[0] == (100.0, 200.0)

    def test_parses_height_stats(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        hs = segments[0].height_stats
        assert hs.minimum_plane_distance == 10.0
        assert hs.maximum_plane_distance == 30.0
        assert hs.average_plane_distance == 20.0

    def test_parses_topology_stats(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        ts = segments[0].topology_stats
        assert ts.surface_count == 5
        assert ts.total_index_count == 120
        assert ts.anchor_point_count == 3
        assert ts.anchor_normal_count == 2

    def test_parses_anchor_signals(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        ans = segments[0].anchor_signals
        assert ans.linked_position_ref_count == 3
        assert ans.heading_mean_degrees == 45.0

    def test_parses_surface_family_histogram(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        hist = segments[0].surface_family_histogram
        assert hist["ck24Type:0x42"] == 5


# ---------------------------------------------------------------------------
# Tests: Asset Corpus Import
# ---------------------------------------------------------------------------


class TestImportAssetCorpus:
    def test_imports_all_assets(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        assert len(assets) == 2

    def test_parses_asset_id(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        assert assets[0].asset_id.startswith("wmo:")
        assert assets[1].asset_id.startswith("m2:")

    def test_parses_asset_kind(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        assert assets[0].asset_kind == "wmo"
        assert assets[1].asset_kind == "m2"

    def test_parses_center(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        assert assets[0].center == (0.0, 0.0, 4.0)

    def test_parses_footprint_area(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        assert assets[0].footprint_area == 600.0
        assert assets[1].footprint_area == 36.0

    def test_parses_render_signals(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        signals = assets[0].render_or_collision_signals
        assert signals["boundsSpanX"] == 20.0


# ---------------------------------------------------------------------------
# Tests: Match Report Import
# ---------------------------------------------------------------------------


class TestImportMatchReport:
    def test_imports_match_results(self, tmp_path: Path) -> None:
        json_file = _write_json(MATCH_REPORT_JSON, tmp_path / "report.json")
        results, proposals = import_match_report(json_file)
        assert len(results) == 1

    def test_parses_match_status(self, tmp_path: Path) -> None:
        json_file = _write_json(MATCH_REPORT_JSON, tmp_path / "report.json")
        results, _ = import_match_report(json_file)
        assert results[0].status == Pm4AssetMatchStatus.MATCHED

    def test_parses_candidates(self, tmp_path: Path) -> None:
        json_file = _write_json(MATCH_REPORT_JSON, tmp_path / "report.json")
        results, _ = import_match_report(json_file)
        candidates = results[0].candidates
        assert len(candidates) == 1
        assert candidates[0].overall_score == 0.523
        assert candidates[0].rank == 1

    def test_parses_proposals(self, tmp_path: Path) -> None:
        json_file = _write_json(MATCH_REPORT_JSON, tmp_path / "report.json")
        _, proposals = import_match_report(json_file)
        assert len(proposals) == 1
        assert proposals[0].proposal_id == "proposal-abcdef1234567890"
        assert proposals[0].confidence == 0.523

    def test_parses_proposal_rotation(self, tmp_path: Path) -> None:
        json_file = _write_json(MATCH_REPORT_JSON, tmp_path / "report.json")
        _, proposals = import_match_report(json_file)
        rot = proposals[0].world_rotation
        assert rot is not None
        assert rot[0] == 45.0  # yaw


# ---------------------------------------------------------------------------
# Tests: Zarr Signal Store Round-Trip
# ---------------------------------------------------------------------------


class TestSegmentSignalStore:
    def test_round_trip_segment_count(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        zarr_path = tmp_path / "segments.zarr"
        write_segment_signals_zarr(zarr_path, segments)
        loaded = read_segment_signals_zarr(zarr_path)
        assert len(loaded) == len(segments)

    def test_round_trip_segment_ids(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        zarr_path = tmp_path / "segments.zarr"
        write_segment_signals_zarr(zarr_path, segments)
        loaded = read_segment_signals_zarr(zarr_path)
        for orig, load in zip(segments, loaded, strict=True):
            assert load.segment_id == orig.segment_id

    def test_round_trip_bounds(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        zarr_path = tmp_path / "segments.zarr"
        write_segment_signals_zarr(zarr_path, segments)
        loaded = read_segment_signals_zarr(zarr_path)
        for orig, load in zip(segments, loaded, strict=True):
            if orig.bounds is not None:
                assert load.bounds is not None
                assert load.bounds.min == pytest.approx(orig.bounds.min, abs=1e-4)
                assert load.bounds.max == pytest.approx(orig.bounds.max, abs=1e-4)

    def test_round_trip_height_stats(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        zarr_path = tmp_path / "segments.zarr"
        write_segment_signals_zarr(zarr_path, segments)
        loaded = read_segment_signals_zarr(zarr_path)
        for orig, load in zip(segments, loaded, strict=True):
            assert load.height_stats.minimum_plane_distance == pytest.approx(
                orig.height_stats.minimum_plane_distance, abs=1e-4
            )
            assert load.height_stats.maximum_plane_distance == pytest.approx(
                orig.height_stats.maximum_plane_distance, abs=1e-4
            )

    def test_round_trip_topology_stats(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        zarr_path = tmp_path / "segments.zarr"
        write_segment_signals_zarr(zarr_path, segments)
        loaded = read_segment_signals_zarr(zarr_path)
        for orig, load in zip(segments, loaded, strict=True):
            assert load.topology_stats.surface_count == orig.topology_stats.surface_count
            assert load.topology_stats.total_index_count == orig.topology_stats.total_index_count

    def test_round_trip_footprint_hull(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        zarr_path = tmp_path / "segments.zarr"
        write_segment_signals_zarr(zarr_path, segments)
        loaded = read_segment_signals_zarr(zarr_path)
        for orig, load in zip(segments, loaded, strict=True):
            assert len(load.footprint_hull) == len(orig.footprint_hull)
            for op, lp in zip(orig.footprint_hull, load.footprint_hull, strict=True):
                assert lp == pytest.approx(op, abs=1e-4)

    def test_round_trip_histogram(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        zarr_path = tmp_path / "segments.zarr"
        write_segment_signals_zarr(zarr_path, segments)
        loaded = read_segment_signals_zarr(zarr_path)
        for orig, load in zip(segments, loaded, strict=True):
            assert load.surface_family_histogram == orig.surface_family_histogram


class TestAssetReferenceStore:
    def test_round_trip_asset_count(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        zarr_path = tmp_path / "assets.zarr"
        write_asset_references_zarr(zarr_path, assets)
        loaded = read_asset_references_zarr(zarr_path)
        assert len(loaded) == len(assets)

    def test_round_trip_asset_ids(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        zarr_path = tmp_path / "assets.zarr"
        write_asset_references_zarr(zarr_path, assets)
        loaded = read_asset_references_zarr(zarr_path)
        for orig, load in zip(assets, loaded, strict=True):
            assert load.asset_id == orig.asset_id

    def test_round_trip_asset_kind(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        zarr_path = tmp_path / "assets.zarr"
        write_asset_references_zarr(zarr_path, assets)
        loaded = read_asset_references_zarr(zarr_path)
        for orig, load in zip(assets, loaded, strict=True):
            assert load.asset_kind == orig.asset_kind

    def test_round_trip_bounds(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        zarr_path = tmp_path / "assets.zarr"
        write_asset_references_zarr(zarr_path, assets)
        loaded = read_asset_references_zarr(zarr_path)
        for orig, load in zip(assets, loaded, strict=True):
            if orig.bounds is not None:
                assert load.bounds is not None
                assert load.bounds.min == pytest.approx(orig.bounds.min, abs=1e-4)
                assert load.bounds.max == pytest.approx(orig.bounds.max, abs=1e-4)

    def test_round_trip_footprint_area(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        zarr_path = tmp_path / "assets.zarr"
        write_asset_references_zarr(zarr_path, assets)
        loaded = read_asset_references_zarr(zarr_path)
        for orig, load in zip(assets, loaded, strict=True):
            assert load.footprint_area == pytest.approx(orig.footprint_area, abs=1e-2)

    def test_round_trip_render_signals(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        zarr_path = tmp_path / "assets.zarr"
        write_asset_references_zarr(zarr_path, assets)
        loaded = read_asset_references_zarr(zarr_path)
        for orig, load in zip(assets, loaded, strict=True):
            assert load.render_or_collision_signals == orig.render_or_collision_signals

    def test_overwrite_flag(self, tmp_path: Path) -> None:
        json_file = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file)
        zarr_path = tmp_path / "assets.zarr"
        write_asset_references_zarr(zarr_path, assets)
        # Write again with overwrite
        write_asset_references_zarr(zarr_path, assets[:1], overwrite=True)
        loaded = read_asset_references_zarr(zarr_path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Tests: Scorer
# ---------------------------------------------------------------------------


class TestScorer:
    def test_ineligible_for_unknown_ck24_type(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        json_file2 = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file2)
        # Use a ck24Type that's not WMO or M2
        from harvester.pm4_asset_matching.scorer import score_segment

        result = score_segment(segments[0], assets, ck24_type=0xFF)
        assert result.status == Pm4AssetMatchStatus.INELIGIBLE
        assert result.expected_asset_kind is None

    def test_wmo_segment_matches_wmo_asset(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        json_file2 = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file2)
        from harvester.pm4_asset_matching.scorer import score_segment

        # ck24Type 0x42 = WMO
        result = score_segment(
            segments[0], assets, ck24_type=0x42,
            segment_tiles=["33_32"],
            segment_center=(125.0, 225.0, 20.0),
        )
        assert result.expected_asset_kind == "wmo"
        assert len(result.candidates) > 0
        assert result.candidates[0].asset_kind == "wmo"

    def test_m2_segment_matches_m2_asset(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        json_file2 = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file2)
        from harvester.pm4_asset_matching.scorer import score_segment

        # ck24Type 0x40 = M2
        result = score_segment(
            segments[1], assets, ck24_type=0x40,
            segment_tiles=["33_32"],
            segment_center=(55.0, 60.0, 10.0),
        )
        assert result.expected_asset_kind == "m2"
        assert len(result.candidates) > 0
        assert result.candidates[0].asset_kind == "m2"

    def test_score_between_zero_and_one(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        json_file2 = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file2)
        from harvester.pm4_asset_matching.scorer import score_segment

        result = score_segment(
            segments[0], assets, ck24_type=0x42,
            segment_tiles=["33_32"],
            segment_center=(125.0, 225.0, 20.0),
        )
        for cand in result.candidates:
            assert 0.0 <= cand.overall_score <= 1.0

    def test_score_breakdown_keys(self, tmp_path: Path) -> None:
        json_file = _write_json(SEGMENT_EXPORT_JSON, tmp_path / "segments.json")
        segments = import_segment_export(json_file)
        json_file2 = _write_json(ASSET_CORPUS_JSON, tmp_path / "corpus.json")
        assets = import_asset_corpus(json_file2)
        from harvester.pm4_asset_matching.scorer import score_segment

        result = score_segment(
            segments[0], assets, ck24_type=0x42,
            segment_tiles=["33_32"],
            segment_center=(125.0, 225.0, 20.0),
        )
        if result.candidates:
            breakdown = result.candidates[0].score_breakdown
            assert "shapeScore" in breakdown
            assert "typedOverlapScore" in breakdown
            assert "typeProfileScore" in breakdown


class TestScorerHelpers:
    def test_score_ratio_equal(self) -> None:
        from harvester.pm4_asset_matching.scorer import score_ratio

        assert score_ratio(10.0, 10.0) == 1.0

    def test_score_ratio_different(self) -> None:
        from harvester.pm4_asset_matching.scorer import score_ratio

        assert score_ratio(5.0, 10.0) == 0.5

    def test_score_ratio_zero(self) -> None:
        from harvester.pm4_asset_matching.scorer import score_ratio

        assert score_ratio(0.0, 10.0) == 0.0

    def test_compute_bounds_overlap_identical(self) -> None:
        from harvester.pm4_asset_matching.scorer import compute_bounds_overlap_ratio

        b = Pm4Bounds3(min=(0.0, 0.0, 0.0), max=(10.0, 10.0, 10.0))
        assert compute_bounds_overlap_ratio(b, b) == pytest.approx(1.0)

    def test_compute_bounds_overlap_disjoint(self) -> None:
        from harvester.pm4_asset_matching.scorer import compute_bounds_overlap_ratio

        a = Pm4Bounds3(min=(0.0, 0.0, 0.0), max=(10.0, 10.0, 10.0))
        b = Pm4Bounds3(min=(20.0, 20.0, 20.0), max=(30.0, 30.0, 30.0))
        assert compute_bounds_overlap_ratio(a, b) == pytest.approx(0.0)

    def test_score_distance_zero(self) -> None:
        from harvester.pm4_asset_matching.scorer import score_distance

        assert score_distance(0.0, 64.0) == 1.0

    def test_score_distance_large(self) -> None:
        from harvester.pm4_asset_matching.scorer import score_distance

        assert score_distance(1000.0, 64.0) < 0.1


# ---------------------------------------------------------------------------
# Tests: Placement Synthesizer
# ---------------------------------------------------------------------------


class TestPlacementSynthesizer:
    def test_synthesizes_proposals_for_matched(self) -> None:
        from harvester.pm4_asset_matching.placement_synthesizer import synthesize_placements

        segment = Pm4SegmentSignalRecord(
            segment_id="seg-0",
            bounds=Pm4Bounds3(min=(0.0, 0.0, 0.0), max=(10.0, 10.0, 10.0)),
            footprint_hull=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            height_stats=Pm4SegmentHeightStats(0, 10, 5),
            surface_family_histogram={},
            topology_stats=Pm4SegmentTopologyStats(1, 24, 0, 0),
            anchor_signals=Pm4SegmentAnchorSignals(0, 0, 0, 0, 10, None, None, None),
            signal_version="v2",
            signal_store_row=None,
            tile_coordinates=["33_32"],
        )
        candidate = Pm4AssetMatchCandidate(
            asset_id="asset-1",
            asset_path="test.wmo",
            asset_kind="wmo",
            rank=1,
            overall_score=0.6,
            status=Pm4AssetMatchStatus.MATCHED,
            score_breakdown={},
            rationale=[],
        )
        match_result = Pm4SegmentMatchResult(
            segment=segment,
            expected_asset_kind="wmo",
            status=Pm4AssetMatchStatus.MATCHED,
            review_required=False,
            rationale=[],
            candidates=[candidate],
        )
        asset_ref = Pm4AssetReferenceSignalRecord(
            asset_id="asset-1",
            asset_path="test.wmo",
            asset_kind="wmo",
            client_build=None,
            tile_coordinates=["33_32"],
            bounds=Pm4Bounds3(min=(0.0, 0.0, 0.0), max=(10.0, 10.0, 10.0)),
            center=(5.0, 5.0, 5.0),
            footprint_hull=[],
            footprint_area=100.0,
            reference_position=(100.0, 200.0, 10.0),
            reference_rotation=None,
            reference_scale=None,
            surface_family_histogram={},
            render_or_collision_signals={},
            signal_version="v1",
            signal_store_row=None,
        )

        proposals = synthesize_placements([match_result], [asset_ref])
        assert len(proposals) == 1
        assert proposals[0].asset_id == "asset-1"

    def test_skips_ineligible(self) -> None:
        from harvester.pm4_asset_matching.placement_synthesizer import synthesize_placements

        segment = Pm4SegmentSignalRecord(
            segment_id="seg-0",
            bounds=Pm4Bounds3(min=(0.0, 0.0, 0.0), max=(10.0, 10.0, 10.0)),
            footprint_hull=[],
            height_stats=Pm4SegmentHeightStats(0, 0, 0),
            surface_family_histogram={},
            topology_stats=Pm4SegmentTopologyStats(0, 0, 0, 0),
            anchor_signals=Pm4SegmentAnchorSignals(0, 0, 0, 0, 0, None, None, None),
            signal_version="v2",
            signal_store_row=None,
        )
        match_result = Pm4SegmentMatchResult(
            segment=segment,
            expected_asset_kind=None,
            status=Pm4AssetMatchStatus.INELIGIBLE,
            review_required=True,
            rationale=[],
            candidates=[],
        )

        proposals = synthesize_placements([match_result], [])
        assert len(proposals) == 0

    def test_proposal_id_deterministic(self) -> None:
        from harvester.pm4_asset_matching.placement_synthesizer import _build_proposal_id

        id1 = _build_proposal_id("seg-0", "asset-1", ["33_32"])
        id2 = _build_proposal_id("seg-0", "asset-1", ["33_32"])
        assert id1 == id2
        assert id1.startswith("proposal-")
