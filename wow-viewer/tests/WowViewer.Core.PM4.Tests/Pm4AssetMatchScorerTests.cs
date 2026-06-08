using System.Numerics;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4AssetMatchScorerTests
{
    [Fact]
    public void ScoreSegment_WmoSegmentPrefersBestOverlappingWmoCandidate()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(
            ck24Type: 0x42,
            boundsMin: new Vector3(0f, 0f, 0f),
            boundsMax: new Vector3(10f, 10f, 4f),
            anchorPlanarPoint: new Vector2(5f, 5f));

        IReadOnlyList<Pm4AssetReferenceSignalRecord> assets =
        [
            CreateAsset("wmo-best", "world/wmo/best.wmo", "wmo", new Vector3(0f, 0f, 0f), new Vector3(10f, 10f, 4f), new Vector3(5f, 5f, 0f)),
            CreateAsset("wmo-far", "world/wmo/far.wmo", "wmo", new Vector3(50f, 50f, 0f), new Vector3(60f, 60f, 4f), new Vector3(55f, 55f, 0f)),
            CreateAsset("m2-wrong-kind", "world/model/wrong.m2", "m2", new Vector3(0f, 0f, 0f), new Vector3(10f, 10f, 4f), new Vector3(5f, 5f, 0f)),
        ];

        Pm4SegmentMatchResult result = Pm4AssetMatchScorer.ScoreSegment(segment, assets);

        Assert.Equal("wmo", result.ExpectedAssetKind);
        Assert.Equal(Pm4AssetMatchStatus.Matched, result.Status);
        Assert.False(result.ReviewRequired);
        Assert.Equal(2, result.Candidates.Count);
        Assert.Equal("wmo-best", result.Candidates[0].AssetId);
        Assert.Equal(Pm4AssetMatchStatus.Matched, result.Candidates[0].Status);
        Assert.True(result.Candidates[0].OverallScore > result.Candidates[1].OverallScore);
    }

    [Fact]
    public void ScoreSegment_NonMatchableCk24TypeReturnsIneligible()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(
            ck24Type: 0x00,
            boundsMin: new Vector3(0f, 0f, 0f),
            boundsMax: new Vector3(4f, 4f, 2f),
            anchorPlanarPoint: new Vector2(2f, 2f));

        Pm4SegmentMatchResult result = Pm4AssetMatchScorer.ScoreSegment(segment, [CreateAsset("wmo", "world/wmo/test.wmo", "wmo", Vector3.Zero, new Vector3(4f, 4f, 2f), new Vector3(2f, 2f, 0f))]);

        Assert.Equal(Pm4AssetMatchStatus.Ineligible, result.Status);
        Assert.Null(result.ExpectedAssetKind);
        Assert.Empty(result.Candidates);
    }

    [Fact]
    public void ScoreSegment_CloseTopCandidatesReturnsAmbiguous()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(
            ck24Type: 0x42,
            boundsMin: new Vector3(0f, 0f, 0f),
            boundsMax: new Vector3(10f, 10f, 4f),
            anchorPlanarPoint: new Vector2(5f, 5f),
            confidenceFlags: Pm4SegmentConfidenceFlags.MultipleLinkGroupIds);

        IReadOnlyList<Pm4AssetReferenceSignalRecord> assets =
        [
            CreateAsset("wmo-a", "world/wmo/a.wmo", "wmo", new Vector3(0f, 0f, 0f), new Vector3(10f, 10f, 4f), new Vector3(5f, 5f, 0f)),
            CreateAsset("wmo-b", "world/wmo/b.wmo", "wmo", new Vector3(0.25f, 0.25f, 0f), new Vector3(10.25f, 10.25f, 4f), new Vector3(5.25f, 5.25f, 0f)),
        ];

        Pm4SegmentMatchResult result = Pm4AssetMatchScorer.ScoreSegment(segment, assets);

        Assert.Equal(Pm4AssetMatchStatus.Ambiguous, result.Status);
        Assert.True(result.ReviewRequired);
        Assert.Equal(Pm4AssetMatchStatus.Ambiguous, result.Candidates[0].Status);
        Assert.Equal(Pm4AssetMatchStatus.Ambiguous, result.Candidates[1].Status);
    }

    [Fact]
    public void ScoreSegment_DurableAssetCorpusPrefersClosestShapeMatch()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(
            ck24Type: 0x42,
            boundsMin: new Vector3(0f, 0f, 0f),
            boundsMax: new Vector3(12f, 6f, 8f),
            anchorPlanarPoint: new Vector2(6f, 3f));

        IReadOnlyList<Pm4AssetReferenceSignalRecord> assets =
        [
            CreateAsset(
                "wmo:build:world/wmo/best.wmo",
                "world/wmo/best.wmo",
                "wmo",
                new Vector3(-6f, -3f, 0f),
                new Vector3(6f, 3f, 8f),
                referencePosition: null,
                tileCoordinates: []),
            CreateAsset(
                "wmo:build:world/wmo/bad.wmo",
                "world/wmo/bad.wmo",
                "wmo",
                new Vector3(-20f, -4f, 0f),
                new Vector3(20f, 4f, 30f),
                referencePosition: null,
                tileCoordinates: []),
        ];

        Pm4SegmentMatchResult result = Pm4AssetMatchScorer.ScoreSegment(segment, assets);

        Assert.Equal(Pm4AssetMatchStatus.Matched, result.Status);
        Assert.Equal("wmo:build:world/wmo/best.wmo", result.Candidates[0].AssetId);
        Assert.True(result.Candidates[0].OverallScore > result.Candidates[1].OverallScore);
        Assert.Contains("sortedSpanScore", result.Candidates[0].ScoreBreakdown.Keys);
    }

    private static Pm4BuiltObjectSegment CreateSegment(
        byte ck24Type,
        Vector3 boundsMin,
        Vector3 boundsMax,
        Vector2 anchorPlanarPoint,
        Pm4SegmentConfidenceFlags confidenceFlags = Pm4SegmentConfidenceFlags.None)
    {
        uint ck24 = ((uint)ck24Type << 16) | 0x1234u;
        Pm4ObjectSegment segment = new(
            $"segment-{ck24Type:X2}",
            ck24,
            ck24Type,
            0x1234,
            ["30_48"],
            [3262u],
            1,
            3,
            [1779u],
            1779u,
            confidenceFlags);

        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        IReadOnlyList<Vector2> hull =
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];

        Pm4CorrelationObjectState correlationState = new(
            30,
            48,
            new Pm4ObjectGroupKey(30, 48, ck24),
            new Pm4CorrelationObjectDescriptor(ck24, ck24Type, 0, 1779u, 1, 1, 3, 0x10, 0u, center.Z),
            boundsMin,
            boundsMax,
            center,
            hull,
            (boundsMax.X - boundsMin.X) * (boundsMax.Y - boundsMin.Y));

        Pm4LinkedPositionRefSummary anchorSummary = new(1, 1, 0, 0, 0, 0f, 0f, 0f);
        IReadOnlyList<Pm4ObjectSegmentSurface> surfaces =
        [
            new Pm4ObjectSegmentSurface(0, 3, 0x10, 3, center.Z, 0u, 0u, ck24 << 8, ck24, ck24Type, 0x1234, Vector3.UnitZ),
        ];

        Pm4SegmentSignalRecord signal = Pm4SegmentSignalExtractor.Extract(segment, correlationState, anchorSummary, surfaces);
        return new Pm4BuiltObjectSegment(
            segment,
            signal,
            correlationState,
            anchorSummary,
            [anchorPlanarPoint],
            surfaces,
            Pm4CoordinateMode.TileLocal,
            Pm4AxisConvention.XYPlaneZUp,
            new Pm4PlanarTransform(false, false, false),
            0f);
    }

    private static Pm4AssetReferenceSignalRecord CreateAsset(
        string assetId,
        string assetPath,
        string assetKind,
        Vector3 boundsMin,
        Vector3 boundsMax,
        Vector3? referencePosition,
        IReadOnlyList<string>? tileCoordinates = null)
    {
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        IReadOnlyList<Vector2> hull =
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];

        return new Pm4AssetReferenceSignalRecord(
            assetId,
            assetPath,
            assetKind,
            "validation-build",
            tileCoordinates ?? ["30_48"],
            new Pm4Bounds3(boundsMin, boundsMax),
            center,
            hull,
            (boundsMax.X - boundsMin.X) * (boundsMax.Y - boundsMin.Y),
            referencePosition,
            Vector3.Zero,
            1f,
            new Dictionary<string, int>(StringComparer.Ordinal) { [$"assetKind:{assetKind}"] = 1 },
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["boundsSpanX"] = boundsMax.X - boundsMin.X,
                ["boundsSpanY"] = boundsMax.Y - boundsMin.Y,
                ["boundsSpanZ"] = boundsMax.Z - boundsMin.Z,
                ["boundsVolume"] = (boundsMax.X - boundsMin.X) * (boundsMax.Y - boundsMin.Y) * (boundsMax.Z - boundsMin.Z),
                ["footprintDiagonalXY"] = Math.Sqrt(
                    Math.Pow(boundsMax.X - boundsMin.X, 2d) +
                    Math.Pow(boundsMax.Y - boundsMin.Y, 2d)),
            },
            Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
            null,
            referencePosition is null
                ? ["durable-asset-corpus"]
                : ["validation-placement"]);
    }
}
