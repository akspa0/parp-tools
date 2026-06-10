using System.Numerics;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4ReplacementPlacementSynthesizerTests
{
    [Fact]
    public void Synthesize_MatchedValidationReference_ReusesReferenceTransform()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(["30_48"], frameYawDegrees: 12f);
        Pm4AssetReferenceSignalRecord asset = CreateAsset(
            "wmo:gate",
            "world/wmo/gate.wmo",
            "wmo",
            referencePosition: new Vector3(100f, 200f, 20f),
            referenceRotation: new Vector3(1f, 2f, 3f),
            referenceScale: 1.5f);
        Pm4SegmentMatchResult matchResult = CreateMatchResult(segment, Pm4AssetMatchStatus.Matched, false, asset.AssetId, asset.AssetPath, asset.AssetKind, 0.92);

        IReadOnlyList<Pm4ReplacementPlacementProposal> proposals = Pm4ReplacementPlacementSynthesizer.Synthesize([matchResult], [asset]);

        Pm4ReplacementPlacementProposal proposal = Assert.Single(proposals);
        Assert.Equal(asset.AssetId, proposal.AssetId);
        Assert.Equal(asset.ReferencePosition, proposal.WorldPosition);
        Assert.Equal(asset.ReferenceRotation, proposal.WorldRotation);
        Assert.Equal(asset.ReferenceScale, proposal.WorldScale);
        Assert.False(proposal.ReviewRequired);
        Assert.Contains("position:asset-reference", proposal.Provenance);
        Assert.Contains("rotation:asset-reference", proposal.Provenance);
        Assert.Contains("scale:asset-reference", proposal.Provenance);
    }

    [Fact]
    public void Synthesize_DurableCorpusFallbacks_UsesPm4CenterAndYawAndMarksReviewRequired()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(["30_48"], frameYawDegrees: 33f);
        Pm4AssetReferenceSignalRecord asset = CreateAsset(
            "wmo:durable:world/wmo/gate.wmo",
            "world/wmo/gate.wmo",
            "wmo",
            referencePosition: null,
            referenceRotation: null,
            referenceScale: null);
        Pm4SegmentMatchResult matchResult = CreateMatchResult(segment, Pm4AssetMatchStatus.Matched, false, asset.AssetId, asset.AssetPath, asset.AssetKind, 0.81);

        IReadOnlyList<Pm4ReplacementPlacementProposal> proposals = Pm4ReplacementPlacementSynthesizer.Synthesize([matchResult], [asset]);

        Pm4ReplacementPlacementProposal proposal = Assert.Single(proposals);
        Assert.Equal(segment.CorrelationState.Center, proposal.WorldPosition);
        Assert.Equal(new Vector3(0f, 0f, 30f), proposal.WorldRotation);
        Assert.Equal(1f, proposal.WorldScale);
        Assert.True(proposal.ReviewRequired);
        Assert.Contains("position:pm4-center-fallback", proposal.Provenance);
        Assert.Contains("rotation:pm4-heading-fallback", proposal.Provenance);
        Assert.Contains("scale:unit-fallback", proposal.Provenance);
    }

    [Fact]
    public void Synthesize_TargetTileFilter_OnlyEmitsIntersectingTiles()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(["30_48", "30_49"], frameYawDegrees: 0f);
        Pm4AssetReferenceSignalRecord asset = CreateAsset(
            "m2:crate",
            "world/model/crate.m2",
            "m2",
            referencePosition: new Vector3(10f, 20f, 30f),
            referenceRotation: Vector3.Zero,
            referenceScale: 1f);
        Pm4SegmentMatchResult matchResult = CreateMatchResult(segment, Pm4AssetMatchStatus.Matched, false, asset.AssetId, asset.AssetPath, asset.AssetKind, 0.77);

        IReadOnlyList<Pm4ReplacementPlacementProposal> proposals = Pm4ReplacementPlacementSynthesizer.Synthesize([matchResult], [asset], ["30_49"]);

        Pm4ReplacementPlacementProposal proposal = Assert.Single(proposals);
        Assert.Equal(["30_49"], proposal.TargetTileCoordinates);
    }

    private static Pm4BuiltObjectSegment CreateSegment(IReadOnlyList<string> tiles, float frameYawDegrees)
    {
        Pm4ObjectSegment segment = new(
            "segment-test",
            0x421234,
            0x42,
            0x1234,
            tiles,
            [3262u],
            1,
            3,
            [1779u],
            1779u,
            Pm4SegmentConfidenceFlags.None);

        Vector3 boundsMin = new(0f, 0f, 0f);
        Vector3 boundsMax = new(10f, 6f, 4f);
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
            new Pm4ObjectGroupKey(30, 48, 0x421234),
            new Pm4CorrelationObjectDescriptor(0x421234, 0x42, 0, 1779u, 1, 1, 3, 0x10, 0u, center.Z),
            boundsMin,
            boundsMax,
            center,
            hull,
            60f);

        Pm4SegmentSignalRecord signal = new(
            segment.SegmentId,
            new Pm4Bounds3(boundsMin, boundsMax),
            hull,
            new Pm4SegmentHeightStats(0f, 4f, 2f),
            new Dictionary<string, int>(StringComparer.Ordinal) { ["wmo-surface"] = 1 },
            new Pm4SegmentTopologyStats(1, 3, 1, 1),
            new Pm4SegmentAnchorSignals(1, 1, 0, 0, 0, 30f, 30f, 30f),
            "pm4-segment-signals/v1",
            null);

        return new Pm4BuiltObjectSegment(
            segment,
            signal,
            correlationState,
            new Pm4LinkedPositionRefSummary(1, 1, 0, 0, 0, 30f, 30f, 30f),
            [new Vector2(5f, 3f)],
            [new Pm4ObjectSegmentSurface(0, 3, 0x10, 3, 2f, 0u, 0u, 0x42123400, 0x421234, 0x42, 0x1234, Vector3.UnitZ)],
            Pm4CoordinateMode.TileLocal,
            Pm4AxisConvention.XYPlaneZUp,
            new Pm4PlanarTransform(false, false, false),
            frameYawDegrees);
    }

    private static Pm4AssetReferenceSignalRecord CreateAsset(
        string assetId,
        string assetPath,
        string assetKind,
        Vector3? referencePosition,
        Vector3? referenceRotation,
        float? referenceScale)
    {
        Vector3 boundsMin = new(-5f, -3f, 0f);
        Vector3 boundsMax = new(5f, 3f, 4f);
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
            ["30_48"],
            new Pm4Bounds3(boundsMin, boundsMax),
            (boundsMin + boundsMax) * 0.5f,
            hull,
            60f,
            referencePosition,
            referenceRotation,
            referenceScale,
            new Dictionary<string, int>(StringComparer.Ordinal) { [$"assetKind:{assetKind}"] = 1 },
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["boundsSpanX"] = 10d,
                ["boundsSpanY"] = 6d,
                ["boundsSpanZ"] = 4d,
            },
            Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
            null,
            ["test"]);
    }

    private static Pm4SegmentMatchResult CreateMatchResult(
        Pm4BuiltObjectSegment segment,
        Pm4AssetMatchStatus status,
        bool reviewRequired,
        string assetId,
        string assetPath,
        string assetKind,
        double score)
    {
        return new Pm4SegmentMatchResult(
            segment,
            assetKind,
            status,
            reviewRequired,
            ["test-match"],
            [
                new Pm4AssetMatchCandidate(
                    assetId,
                    assetPath,
                    assetKind,
                    1,
                    score,
                    status,
                    new Dictionary<string, double>(StringComparer.Ordinal) { ["overall"] = score },
                    ["candidate"])
            ]);
    }
}
