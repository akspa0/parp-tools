using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Reconciliation;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.Editor.Tests.Reconciliation;

public class Pm4ReconciliationInputAdapterTests
{
    private const float MapOrigin = Pm4CoordinateService.MapOrigin;

    [Fact]
    public void World_to_placement_matches_canonical_composition()
    {
        Vector3 world = new(778.9f, 41.0f, 312.5f);

        Vector3 placement = Pm4ReconciliationInputAdapter.WorldToPlacementSpace(world);

        Assert.Equal(MapOrigin - world.Y, placement.X, 3);
        Assert.Equal(MapOrigin - world.X, placement.Y, 3);
        Assert.Equal(world.Z, placement.Z, 3);
    }

    [Fact]
    public void Guide_observation_converts_bounds_and_resolves_height_signal()
    {
        // World-space segment spanning Z 250..350 with one surface whose _0x1C bits encode 300f.
        Pm4BuiltObjectSegment segment = CreateSegment(
            worldBoundsMin: new Vector3(0f, 0f, 250f),
            worldBoundsMax: new Vector3(10f, 6f, 350f),
            surfaceHeightBits: (uint)BitConverter.SingleToInt32Bits(300f));
        var match = new Pm4SegmentMatchResult(
            segment, "wmo", Pm4AssetMatchStatus.Matched, false, ["test"], []);

        IReadOnlyList<Pm4GuideObservation> guides = Pm4ReconciliationInputAdapter.BuildGuideObservations(
            [match], "dev_01_00.pm4", "3.3.5.12340", "development", 0, 1);

        Pm4GuideObservation guide = Assert.Single(guides);
        Assert.Equal(ExpectedAssetKind.WorldModel, guide.ExpectedAssetKind);

        // The reflection reverses axis ordering; min/max are recomputed after conversion.
        Assert.Equal(MapOrigin - 6f, guide.BoundsMin.X, 3);
        Assert.Equal(MapOrigin - 10f, guide.BoundsMin.Y, 3);
        Assert.Equal(MapOrigin - 0f, guide.BoundsMax.X, 3);
        Assert.Equal(MapOrigin - 0f, guide.BoundsMax.Y, 3);
        Assert.Equal(250f, guide.BoundsMin.Z, 3);
        Assert.Equal(350f, guide.BoundsMax.Z, 3);

        Assert.NotNull(guide.HeightSignal);
        Assert.Equal(300.0, guide.HeightSignal!.Value, 3);

        Assert.Contains(guide.Evidence, e => e.Signal == "pm4-height-signal");
        Assert.Contains(guide.Evidence, e => e.Signal == "pm4-segment-surfaces");
        Assert.Equal("segment-test", guide.Guide.GuideId);
    }

    [Fact]
    public void Height_signal_is_null_when_bits_are_outside_segment_span()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(
            worldBoundsMin: new Vector3(0f, 0f, 250f),
            worldBoundsMax: new Vector3(10f, 6f, 350f),
            surfaceHeightBits: (uint)BitConverter.SingleToInt32Bits(5000f));
        var match = new Pm4SegmentMatchResult(
            segment, "wmo", Pm4AssetMatchStatus.Unresolved, true, ["test"], []);

        IReadOnlyList<Pm4GuideObservation> guides = Pm4ReconciliationInputAdapter.BuildGuideObservations(
            [match], "dev_01_00.pm4", "x", "development", 0, 1);

        Pm4GuideObservation guide = Assert.Single(guides);
        Assert.Null(guide.HeightSignal);
        Assert.DoesNotContain(guide.Evidence, e => e.Signal == "pm4-height-signal");
    }

    [Fact]
    public void Snapshots_map_catalog_rows_with_entry_indices_and_kinds()
    {
        var catalog = new AdtPlacementCatalog(
            "synthetic_1_0_obj0.adt",
            MapFileKind.AdtObj,
            ["foo.mdx"],
            ["a.wmo"],
            [
                new AdtModelPlacement(0, "foo.mdx", 77, new Vector3(16000f, 15000f, 300f), Vector3.Zero, 1f),
            ],
            [
                new AdtWorldModelPlacement(0, "a.wmo", 99, new Vector3(16100f, 15100f, 310f), Vector3.Zero,
                    new Vector3(16090f, 15090f, 290f), new Vector3(16120f, 15120f, 330f), Flags: 0),
            ]);

        IReadOnlyList<PlacementSnapshot> snapshots = Pm4ReconciliationInputAdapter.BuildPlacementSnapshots(
            catalog, "development", 1, 0, "3.3.5.12340");

        Assert.Equal(2, snapshots.Count);

        Assert.Equal(ExpectedAssetKind.Model, snapshots[0].Identity.Kind);
        Assert.Equal(0, snapshots[0].Identity.EntryIndex);
        Assert.Equal(77, snapshots[0].Identity.UniqueId);
        Assert.Equal("foo.mdx", snapshots[0].Identity.AssetPath);

        Assert.Equal(ExpectedAssetKind.WorldModel, snapshots[1].Identity.Kind);
        Assert.Equal(0, snapshots[1].Identity.EntryIndex);
        Assert.Equal(99, snapshots[1].Identity.UniqueId);
    }

    [Fact]
    public void Self_corpus_references_are_labelled_and_bounded()
    {
        var catalog = new AdtPlacementCatalog(
            "synthetic_1_0_obj0.adt",
            MapFileKind.AdtObj,
            ["foo.mdx"],
            ["a.wmo"],
            [
                new AdtModelPlacement(0, "foo.mdx", 77, new Vector3(16000f, 15000f, 300f), Vector3.Zero, 1f),
            ],
            [
                new AdtWorldModelPlacement(0, "a.wmo", 99, new Vector3(16100f, 15100f, 310f), Vector3.Zero,
                    new Vector3(16090f, 15090f, 290f), new Vector3(16120f, 15120f, 330f), Flags: 0),
            ]);

        IReadOnlyList<Pm4AssetReferenceSignalRecord> corpus = Pm4ReconciliationInputAdapter.BuildSelfCorpusReferences(
            catalog, "1_0", "test-build");

        Assert.Equal(2, corpus.Count);

        Pm4AssetReferenceSignalRecord wmo = Assert.Single(corpus, a => a.AssetKind == "wmo");
        Assert.Equal("wmo:99", wmo.AssetId);
        Assert.Equal("a.wmo", wmo.AssetPath);
        Assert.Contains("museum-self-corpus", wmo.ValidationTags!);
        Assert.Contains("fallback-placement-bounds", wmo.ValidationTags!);
        Assert.Equal(new Vector3(16090f, 15090f, 290f), wmo.Bounds!.Min);
        Assert.Equal(new Vector3(16120f, 15120f, 330f), wmo.Bounds!.Max);

        Pm4AssetReferenceSignalRecord m2 = Assert.Single(corpus, a => a.AssetKind == "m2");
        Assert.Equal("m2:77", m2.AssetId);
        Assert.Contains("fallback-placement-bounds", m2.ValidationTags!);
    }

    [Fact]
    public void Candidates_map_status_scores_and_breakdowns()
    {
        Pm4BuiltObjectSegment segment = CreateSegment(
            new Vector3(0f, 0f, 250f), new Vector3(10f, 6f, 350f), 0u);
        var match = new Pm4SegmentMatchResult(
            segment,
            "wmo",
            Pm4AssetMatchStatus.Ambiguous,
            ReviewRequired: true,
            ["top candidates too close"],
            [
                new("wmo:a", "a.wmo", "wmo", 0, 0.91d, Pm4AssetMatchStatus.Matched, new Dictionary<string, double> { ["geom"] = 0.91d }, ["first"]),
                new("wmo:b", "b.wmo", "wmo", 1, 0.90d, Pm4AssetMatchStatus.Matched, new Dictionary<string, double> { ["geom"] = 0.90d }, ["second"]),
            ]);

        IReadOnlyDictionary<string, IReadOnlyList<ReconciliationCandidate>> byGuide =
            Pm4ReconciliationInputAdapter.BuildCandidatesByGuideId([match]);

        Assert.True(byGuide.ContainsKey("segment-test"));
        IReadOnlyList<ReconciliationCandidate> candidates = byGuide["segment-test"];
        Assert.Equal(2, candidates.Count);
        Assert.Equal(CandidateStatus.Matched, candidates[0].Status);
        Assert.Equal(ExpectedAssetKind.WorldModel, candidates[0].Kind);
        Assert.Equal(0.91d, candidates[0].Score, 3);
        Assert.Equal("a.wmo", candidates[0].AssetPath);
    }

    private static Pm4BuiltObjectSegment CreateSegment(Vector3 worldBoundsMin, Vector3 worldBoundsMax, uint surfaceHeightBits)
    {
        var segment = new Pm4ObjectSegment(
            "segment-test",
            0x421234,
            0x42,
            0x1234,
            ["0_1"],
            [3262u],
            1,
            3,
            [1779u],
            1779u,
            Pm4SegmentConfidenceFlags.None);

        Vector3 center = (worldBoundsMin + worldBoundsMax) * 0.5f;
        Vector2[] hull =
        [
            new(worldBoundsMin.X, worldBoundsMin.Y),
            new(worldBoundsMax.X, worldBoundsMin.Y),
            new(worldBoundsMax.X, worldBoundsMax.Y),
            new(worldBoundsMin.X, worldBoundsMax.Y),
        ];

        var correlationState = new Pm4CorrelationObjectState(
            0,
            1,
            new Pm4ObjectGroupKey(0, 1, 0x421234),
            new Pm4CorrelationObjectDescriptor(0x421234, 0x42, 0, 1779u, 1, 1, 3, 0x10, 0u, center.Z),
            worldBoundsMin,
            worldBoundsMax,
            center,
            hull,
            60f);

        var signal = new Pm4SegmentSignalRecord(
            segment.SegmentId,
            new Pm4Bounds3(worldBoundsMin, worldBoundsMax),
            hull,
            new Pm4SegmentHeightStats(0f, 100f, 50f),
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
            [new Pm4ObjectSegmentSurface(0, 3, 0x10, 3, 2f, 0u, 0u, surfaceHeightBits, 0x421234, 0x42, 0x1234, Vector3.UnitZ)],
            Pm4CoordinateMode.WorldSpace,
            Pm4AxisConvention.XYPlaneZUp,
            new Pm4PlanarTransform(false, false, false),
            0f);
    }
}
