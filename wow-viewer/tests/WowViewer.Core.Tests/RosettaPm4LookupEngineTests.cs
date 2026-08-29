using System.Numerics;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Reconciliation;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class RosettaPm4LookupEngineTests
{
    private static RosettaReferenceLibrary CreateTestLibrary()
    {
        var assets = new List<RosettaReferenceAsset>
        {
            CreateAsset("world/doodads/small_barrel.m2", "m2", new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2.5f)),
            CreateAsset("world/doodads/medium_crate.m2", "m2", new Vector3(-2f, -2f, 0f), new Vector3(2f, 2f, 4f)),
            CreateAsset("world/trees/oak_tree_a.m2", "m2", new Vector3(-5f, -5f, 0f), new Vector3(5f, 5f, 18f)),
            CreateAsset("world/trees/oak_tree_b.m2", "m2", new Vector3(-5f, -5f, 0f), new Vector3(5f, 5f, 18f)), // Near twin for ambiguity test
            CreateAsset("world/buildings/inn.wmo", "wmo", new Vector3(-20f, -15f, 0f), new Vector3(20f, 15f, 16f)),
            CreateAsset("world/buildings/barracks.wmo", "wmo", new Vector3(-35f, -25f, 0f), new Vector3(35f, 25f, 22f)),
        };

        return new RosettaReferenceLibrary(
            "test-lib-01",
            "3.3.5.12340",
            assets);
    }

    private static RosettaReferenceAsset CreateAsset(
        string path,
        string kind,
        Vector3 min,
        Vector3 max)
    {
        Vector3 span = max - min;
        float[] sortedSpans = [span.X, span.Y, span.Z];
        Array.Sort(sortedSpans);
        Array.Reverse(sortedSpans);

        return new RosettaReferenceAsset(
            AssetId: $"test:{path.ToLowerInvariant()}",
            AssetPath: path,
            NormalizedPath: path.Replace('/', '\\').ToLowerInvariant(),
            AssetKind: kind,
            ClientBuild: "3.3.5.12340",
            TileCoordinates: ["24_24"],
            Bounds: new Pm4Bounds3(min, max),
            Center: (min + max) * 0.5f,
            Span: span,
            DiagonalXY: MathF.Sqrt(span.X * span.X + span.Y * span.Y),
            Volume: span.X * span.Y * span.Z,
            FootprintArea: span.X * span.Y,
            FootprintHull: [new Vector2(min.X, min.Y), new Vector2(max.X, max.Y)],
            AspectRatioXY: span.X / MathF.Max(0.001f, span.Y),
            AspectRatioZMaxXY: span.Z / MathF.Max(0.001f, MathF.Max(span.X, span.Y)),
            SubPartBounds: [],
            Signals: new Dictionary<string, double>
            {
                ["boundsSpanX"] = span.X,
                ["boundsSpanY"] = span.Y,
                ["boundsSpanZ"] = span.Z,
            });
    }

    private static Pm4BuiltObjectSegment CreateSegment(
        string segmentId,
        byte ck24Type,
        Vector3 min,
        Vector3 max,
        Dictionary<byte, Pm4Bounds3>? typedBounds = null)
    {
        uint ck24 = 0x123456;
        Vector3 center = (min + max) * 0.5f;

        var objSeg = new Pm4ObjectSegment(
            SegmentId: segmentId,
            Ck24: ck24,
            Ck24Type: ck24Type,
            Ck24ObjectId: 1,
            TileCoordinates: ["24_24"],
            Field04Values: [1],
            SurfaceCount: 10,
            TotalIndexCount: 30,
            LinkGroupIds: [1],
            DominantLinkGroupId: 1,
            ConfidenceFlags: Pm4SegmentConfidenceFlags.None);

        var signal = new Pm4SegmentSignalRecord(
            SegmentId: segmentId,
            Bounds: new Pm4Bounds3(min, max),
            FootprintHull: [new Vector2(min.X, min.Y), new Vector2(max.X, max.Y)],
            HeightStats: new Pm4SegmentHeightStats(min.Z, max.Z, (min.Z + max.Z) * 0.5f),
            SurfaceFamilyHistogram: new Dictionary<string, int>(),
            TopologyStats: new Pm4SegmentTopologyStats(10, 30, 4, 4),
            AnchorSignals: new Pm4SegmentAnchorSignals(0, 0, 0, 0, 0, null, null, null),
            SignalVersion: "v1",
            SignalStoreRow: null,
            TypedBounds: typedBounds ?? new Dictionary<byte, Pm4Bounds3>());

        var corrState = new Pm4CorrelationObjectState(
            30,
            48,
            new Pm4ObjectGroupKey(30, 48, ck24),
            new Pm4CorrelationObjectDescriptor(ck24, ck24Type, 0, 1779u, 1, 1, 3, 0x10, 0u, center.Z),
            min,
            max,
            center,
            [new Vector2(min.X, min.Y), new Vector2(max.X, max.Y)],
            (max.X - min.X) * (max.Y - min.Y));

        return new Pm4BuiltObjectSegment(
            objSeg,
            signal,
            corrState,
            new Pm4LinkedPositionRefSummary(1, 1, 0, 0, 0, 0f, 0f, 0f),
            [Vector2.Zero],
            [],
            Pm4CoordinateMode.TileLocal,
            Pm4AxisConvention.XYPlaneZUp,
            new Pm4PlanarTransform(false, false, false),
            FrameYawDegrees: 0f);
    }

    [Fact]
    public void LookupSegment_Identified_ReturnsCorrectAssetAndHighConfidence()
    {
        RosettaReferenceLibrary library = CreateTestLibrary();
        // Segment matching small_barrel (min: -1, -1, 0; max: 1, 1, 2.5) with M2Top type flag
        var typed = new Dictionary<byte, Pm4Bounds3>
        {
            [0x03] = new(new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2.5f))
        };
        Pm4BuiltObjectSegment segment = CreateSegment("seg_barrel", 0x40, new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2.5f), typed);

        RosettaPm4LookupResult result = RosettaPm4LookupEngine.LookupSegment(segment, library);

        Assert.True(result.IsIdentified);
        Assert.Equal(Pm4AssetMatchStatus.Matched, result.Status);
        Assert.NotNull(result.TopCandidate);
        Assert.Equal("world/doodads/small_barrel.m2", result.TopCandidate!.AssetPath);
        Assert.True(result.TopCandidate.OverallScore >= 0.85d);
        Assert.NotEmpty(result.SignalAgreements);
        Assert.Contains("AspectRatio", result.SignalAgreements.Keys);
        Assert.Contains("MajorSpan", result.SignalAgreements.Keys);
    }

    [Fact]
    public void LookupSegment_Ambiguous_NamesAllCompetingCandidates()
    {
        RosettaReferenceLibrary library = CreateTestLibrary();
        // Segment matching oak_tree_a and oak_tree_b (identical spans -5..5, -5..5, 0..18)
        var typed = new Dictionary<byte, Pm4Bounds3>
        {
            [0x03] = new(new Vector3(-5f, -5f, 0f), new Vector3(5f, 5f, 18f))
        };
        Pm4BuiltObjectSegment segment = CreateSegment("seg_tree", 0x40, new Vector3(-5f, -5f, 0f), new Vector3(5f, 5f, 18f), typed);

        RosettaPm4LookupResult result = RosettaPm4LookupEngine.LookupSegment(segment, library);

        Assert.True(result.IsAmbiguous);
        Assert.Equal(Pm4AssetMatchStatus.Ambiguous, result.Status);
        Assert.True(result.Candidates.Count >= 2);
        var candidatePaths = result.Candidates.Select(c => c.AssetPath).ToList();
        Assert.Contains("world/trees/oak_tree_a.m2", candidatePaths);
        Assert.Contains("world/trees/oak_tree_b.m2", candidatePaths);
        Assert.True(Math.Abs(result.Candidates[0].OverallScore - result.Candidates[1].OverallScore) <= Pm4AssetMatchScorer.AmbiguousScoreWindow);
    }

    [Fact]
    public void LookupSegment_NoReference_ReturnsExplicitUnresolved()
    {
        RosettaReferenceLibrary library = CreateTestLibrary();
        // Segment with unusual extreme dimensions not present in library (e.g. 500m antenna)
        Pm4BuiltObjectSegment segment = CreateSegment("seg_alien", 0x40, new Vector3(-0.5f, -0.5f, 0f), new Vector3(0.5f, 0.5f, 500f));

        RosettaPm4LookupResult result = RosettaPm4LookupEngine.LookupSegment(segment, library);

        Assert.True(result.IsNoReference);
        Assert.Equal(Pm4AssetMatchStatus.Unresolved, result.Status);
        Assert.True(result.ReviewRequired);
        if (result.TopCandidate is not null)
        {
            Assert.True(result.TopCandidate.OverallScore < Pm4AssetMatchScorer.MinimumMatchedScore);
        }
    }

    [Fact]
    public void LookupSegment_Ineligible_ReturnsIneligibleForNonObjectCk24()
    {
        RosettaReferenceLibrary library = CreateTestLibrary();
        // Segment with CK24 type 0x00 (terrain/water)
        Pm4BuiltObjectSegment segment = CreateSegment("seg_water", 0x00, new Vector3(-10f, -10f, 0f), new Vector3(10f, 10f, 0f));

        RosettaPm4LookupResult result = RosettaPm4LookupEngine.LookupSegment(segment, library);

        Assert.True(result.IsIneligible);
        Assert.Equal(Pm4AssetMatchStatus.Ineligible, result.Status);
        Assert.True(result.ReviewRequired);
        Assert.Empty(result.Candidates);
    }

    [Fact]
    public void CompareWithLegacyScorer_DetectsAgreementsAndDisagreements()
    {
        RosettaReferenceLibrary library = CreateTestLibrary();
        var typed = new Dictionary<byte, Pm4Bounds3>
        {
            [0x03] = new(new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2.5f))
        };
        Pm4BuiltObjectSegment segment = CreateSegment("seg_barrel", 0x40, new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2.5f), typed);

        // 1. Legacy corpus containing the true barrel
        var agreeingLegacyCorpus = library.Assets
            .Select(static a => a.ToAssetReferenceSignalRecord())
            .ToList();

        RosettaPm4LookupResult agreeResult = RosettaPm4LookupEngine.CompareWithLegacyScorer(
            segment, agreeingLegacyCorpus, library);

        Assert.NotNull(agreeResult.LegacyScorerResult);
        Assert.True(agreeResult.AgreesWithLegacy == true);

        // 2. Legacy corpus containing only wrong assets (e.g. crate only)
        var disagreeingLegacyCorpus = library.Assets
            .Where(static a => a.AssetPath.Contains("crate"))
            .Select(static a => a.ToAssetReferenceSignalRecord())
            .ToList();

        RosettaPm4LookupResult disagreeResult = RosettaPm4LookupEngine.CompareWithLegacyScorer(
            segment, disagreeingLegacyCorpus, library);

        Assert.NotNull(disagreeResult.LegacyScorerResult);
        Assert.True(disagreeResult.AgreesWithLegacy == false);
    }

    [Fact]
    public void LookupSegments_BatchProcessesMultipleSegments()
    {
        RosettaReferenceLibrary library = CreateTestLibrary();
        var typedM2 = new Dictionary<byte, Pm4Bounds3>
        {
            [0x03] = new(new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2.5f))
        };
        var typedWmo = new Dictionary<byte, Pm4Bounds3>
        {
            [0x12] = new(new Vector3(-20f, -15f, 0f), new Vector3(20f, 15f, 16f))
        };

        var segments = new List<Pm4BuiltObjectSegment>
        {
            CreateSegment("seg_1", 0x40, new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2.5f), typedM2),
            CreateSegment("seg_2", 0x42, new Vector3(-20f, -15f, 0f), new Vector3(20f, 15f, 16f), typedWmo),
            CreateSegment("seg_3", 0x00, new Vector3(-10f, -10f, 0f), new Vector3(10f, 10f, 0f)),
        };

        IReadOnlyList<RosettaPm4LookupResult> batchResults = RosettaPm4LookupEngine.LookupSegments(segments, library);

        Assert.Equal(3, batchResults.Count);
        Assert.True(batchResults[0].IsIdentified);
        Assert.Equal("world/doodads/small_barrel.m2", batchResults[0].TopCandidate!.AssetPath);
        Assert.True(batchResults[1].IsIdentified);
        Assert.Equal("world/buildings/inn.wmo", batchResults[1].TopCandidate!.AssetPath);
        Assert.True(batchResults[2].IsIneligible);
    }

    [Fact]
    public void Pm4ReconciliationInputAdapter_BuildRosettaCorpusReferences_ConvertsLibrary()
    {
        RosettaReferenceLibrary library = CreateTestLibrary();
        IReadOnlyList<Pm4AssetReferenceSignalRecord> signals = Pm4ReconciliationInputAdapter.BuildRosettaCorpusReferences(library);

        Assert.Equal(library.TotalAssets, signals.Count);
        Assert.All(signals, static s =>
        {
            Assert.NotNull(s.AssetId);
            Assert.NotNull(s.AssetPath);
            Assert.NotNull(s.Bounds);
            Assert.True(s.SurfaceFamilyHistogram.Count >= 0);
        });
    }
}
