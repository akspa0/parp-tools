using System.Numerics;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 151 group-admission instrumentation. These prove the counters can distinguish the rules
/// that admit WMO geometry — the detector has to be able to see the thing before a null or a
/// dominant reading from it means anything.
/// </summary>
public sealed class WmoAdmissionTallyTests
{
    [Fact]
    public void RecordGroup_SeparatesEachAdmissionRule()
    {
        WmoAdmissionTally tally = default;
        tally.RecordGroup(WmoGroupAdmissionRule.Portal);
        tally.RecordGroup(WmoGroupAdmissionRule.Frustum);
        tally.RecordGroup(WmoGroupAdmissionRule.Frustum);
        tally.RecordGroup(WmoGroupAdmissionRule.PortalAndFrustum);
        tally.RecordGroup(WmoGroupAdmissionRule.PortalFallback);
        tally.RecordGroup(WmoGroupAdmissionRule.None);

        WmoAdmissionStats stats = tally.ToStats();
        Assert.Equal(6, stats.GroupsConsidered);
        Assert.Equal(5, stats.GroupsAdmitted);
        Assert.Equal(1, stats.GroupsRejected);
        Assert.Equal(1, stats.AdmittedByPortal);
        Assert.Equal(2, stats.AdmittedByFrustum);
        Assert.Equal(1, stats.AdmittedByPortalAndFrustum);
        Assert.Equal(1, stats.AdmittedByPortalFallback);
    }

    [Fact]
    public void DominantGroupAdmissionRule_NamesTheRuleThatAdmittedMost()
    {
        WmoAdmissionTally tally = default;
        for (int i = 0; i < 3; i++)
            tally.RecordGroup(WmoGroupAdmissionRule.Portal);
        for (int i = 0; i < 9; i++)
            tally.RecordGroup(WmoGroupAdmissionRule.Frustum);

        Assert.Equal(WmoGroupAdmissionRule.Frustum, tally.ToStats().DominantGroupAdmissionRule);
    }

    [Fact]
    public void DominantGroupAdmissionRule_IsNoneWhenNothingWasAdmitted()
    {
        WmoAdmissionTally tally = default;
        tally.RecordGroup(WmoGroupAdmissionRule.None);
        tally.RecordGroup(WmoGroupAdmissionRule.None);

        Assert.Equal(WmoGroupAdmissionRule.None, tally.ToStats().DominantGroupAdmissionRule);
    }

    [Fact]
    public void RecordGroupPlacementEvaluation_KeepsWorstPlacementAndFirstFallbackReason()
    {
        WmoAdmissionTally tally = default;
        tally.RecordGroupPlacementEvaluation(12, "wmo://small", null);
        tally.RecordGroupPlacementEvaluation(400, "wmo://stormwind", "portal_data_absent");
        tally.RecordGroupPlacementEvaluation(30, "wmo://medium", "portal_edges_absent");

        WmoAdmissionStats stats = tally.ToStats();
        Assert.Equal(3, stats.GroupPlacementEvaluations);
        Assert.Equal(400, stats.MaxGroupsAdmittedInOnePlacement);
        Assert.Equal("wmo://stormwind", stats.WorstPlacementModelKey);
        Assert.Equal(2, stats.PortalFallbackEvaluations);
        Assert.Equal("portal_data_absent", stats.FirstPortalFallbackReason);
    }

    [Fact]
    public void MeanGroupsAdmittedPerPlacement_SeparatesOneHugeWmoFromManyOrdinaryOnes()
    {
        WmoAdmissionTally oneHuge = default;
        for (int i = 0; i < 300; i++)
            oneHuge.RecordGroup(WmoGroupAdmissionRule.Frustum);
        oneHuge.RecordGroupPlacementEvaluation(300, "wmo://huge", null);

        WmoAdmissionTally manySmall = default;
        for (int placement = 0; placement < 100; placement++)
        {
            for (int i = 0; i < 3; i++)
                manySmall.RecordGroup(WmoGroupAdmissionRule.Frustum);
            manySmall.RecordGroupPlacementEvaluation(3, "wmo://small", null);
        }

        Assert.Equal(300d, oneHuge.ToStats().MeanGroupsAdmittedPerPlacement);
        Assert.Equal(3d, manySmall.ToStats().MeanGroupsAdmittedPerPlacement);
        Assert.Equal(
            oneHuge.ToStats().GroupsAdmitted,
            manySmall.ToStats().GroupsAdmitted);
    }

    [Fact]
    public void Add_MergesCountersAndKeepsTheWorstPlacementAcrossTallies()
    {
        WmoAdmissionTally first = default;
        first.RecordGroup(WmoGroupAdmissionRule.Portal);
        first.RecordPlacement(WmoPlacementAdmissionRule.Admitted);
        first.RecordGroupPlacementEvaluation(40, "wmo://first", null);

        WmoAdmissionTally second = default;
        second.RecordGroup(WmoGroupAdmissionRule.Frustum);
        second.RecordPlacement(WmoPlacementAdmissionRule.RejectedDistance);
        second.RecordGroupPlacementEvaluation(900, "wmo://second", "portal_data_absent");

        first.Add(second);

        WmoAdmissionStats stats = first.ToStats();
        Assert.Equal(2, stats.GroupsConsidered);
        Assert.Equal(2, stats.GroupsAdmitted);
        Assert.Equal(1, stats.AdmittedByPortal);
        Assert.Equal(1, stats.AdmittedByFrustum);
        Assert.Equal(2, stats.PlacementsConsidered);
        Assert.Equal(1, stats.PlacementsAdmitted);
        Assert.Equal(1, stats.PlacementsRejectedDistance);
        Assert.Equal(2, stats.GroupPlacementEvaluations);
        Assert.Equal(900, stats.MaxGroupsAdmittedInOnePlacement);
        Assert.Equal("wmo://second", stats.WorstPlacementModelKey);
        Assert.Equal("portal_data_absent", stats.FirstPortalFallbackReason);
    }

    [Fact]
    public void Reset_ClearsEveryCounterIncludingTheStringFields()
    {
        WmoAdmissionTally tally = default;
        tally.RecordGroup(WmoGroupAdmissionRule.Frustum);
        tally.RecordPlacement(WmoPlacementAdmissionRule.Admitted);
        tally.RecordGroupPlacementEvaluation(7, "wmo://any", "portal_data_absent");

        tally.Reset();

        Assert.Equal(default, tally.ToStats());
    }

    [Fact]
    public void CollectVisibleWmos_RecordsAdmittedPlacement()
    {
        WorldVisibilityFrame frame = new();
        WmoAdmissionTally tally = default;

        int culled = CollectOne(frame, ref tally,
            CreateInstance("wmo://near", new Vector3(0f, 200f, 0f), halfExtent: 40f),
            frustumVisible: true);

        Assert.Equal(0, culled);
        Assert.Single(frame.VisibleWmos);
        WmoAdmissionStats stats = tally.ToStats();
        Assert.Equal(1, stats.PlacementsConsidered);
        Assert.Equal(1, stats.PlacementsAdmitted);
    }

    [Fact]
    public void CollectVisibleWmos_RecordsDistanceRejection()
    {
        WorldVisibilityFrame frame = new();
        WmoAdmissionTally tally = default;

        int culled = CollectOne(frame, ref tally,
            CreateInstance("wmo://far", new Vector3(0f, 7000f, 0f), halfExtent: 10f),
            frustumVisible: true,
            fogEnd: 500f);

        Assert.Equal(1, culled);
        WmoAdmissionStats stats = tally.ToStats();
        Assert.Equal(1, stats.PlacementsConsidered);
        Assert.Equal(0, stats.PlacementsAdmitted);
        Assert.Equal(1, stats.PlacementsRejectedDistance);
    }

    /// <summary>
    /// Hidden and not-yet-resident placements never reached the returned cull count, so the two
    /// rules that the old counter could not see are the ones most worth asserting.
    /// </summary>
    [Fact]
    public void CollectVisibleWmos_RecordsRulesTheCullCountNeverReported()
    {
        WorldVisibilityFrame frame = new();
        WmoAdmissionTally tally = default;
        WorldObjectInstance hidden = CreateInstance("wmo://hidden", new Vector3(0f, 200f, 0f), halfExtent: 40f);
        WorldObjectInstance notResident = CreateInstance("wmo://pending", new Vector3(0f, 200f, 0f), halfExtent: 40f);

        int culled = WorldObjectVisibilityCollector.CollectVisibleWmos(
            frame,
            [hidden, notResident],
            CreateWmoContext(fogEnd: 1200f),
            inst => inst.ModelKey == "wmo://hidden",
            static (_, _) => true,
            static key => key != "wmo://pending",
            static (_, _) => { },
            ref tally);

        Assert.Equal(0, culled);
        Assert.Empty(frame.VisibleWmos);
        WmoAdmissionStats stats = tally.ToStats();
        Assert.Equal(2, stats.PlacementsConsidered);
        Assert.Equal(1, stats.PlacementsRejectedHidden);
        Assert.Equal(1, stats.PlacementsRejectedAssetNotReady);
    }

    [Fact]
    public void CollectVisibleWmos_TallyOverloadMakesNoAdmissionDifference()
    {
        WorldObjectInstance[] instances =
        [
            CreateInstance("wmo://a", new Vector3(0f, 200f, 0f), halfExtent: 40f),
            CreateInstance("wmo://b", new Vector3(0f, 7000f, 0f), halfExtent: 10f),
            CreateInstance("wmo://c", new Vector3(0f, 900f, 0f), halfExtent: 60f),
        ];
        WorldObjectVisibilityContext context = CreateWmoContext(fogEnd: 1200f);

        WorldVisibilityFrame withoutTally = new();
        int culledWithout = WorldObjectVisibilityCollector.CollectVisibleWmos(
            withoutTally, instances, context,
            static _ => false, static (_, _) => true, static _ => true, static (_, _) => { });

        WorldVisibilityFrame withTally = new();
        WmoAdmissionTally tally = default;
        int culledWith = WorldObjectVisibilityCollector.CollectVisibleWmos(
            withTally, instances, context,
            static _ => false, static (_, _) => true, static _ => true, static (_, _) => { },
            ref tally);

        Assert.Equal(culledWithout, culledWith);
        Assert.Equal(
            withoutTally.VisibleWmos.Select(static entry => entry.Instance.ModelKey),
            withTally.VisibleWmos.Select(static entry => entry.Instance.ModelKey));
        Assert.Equal(instances.Length, tally.ToStats().PlacementsConsidered);
    }

    private static int CollectOne(
        WorldVisibilityFrame frame,
        ref WmoAdmissionTally tally,
        WorldObjectInstance instance,
        bool frustumVisible,
        float fogEnd = 1200f)
    {
        return WorldObjectVisibilityCollector.CollectVisibleWmos(
            frame,
            [instance],
            CreateWmoContext(fogEnd),
            static _ => false,
            (_, _) => frustumVisible,
            static _ => true,
            static (_, _) => { },
            ref tally);
    }

    // Mirrors the WorldScene WMO call site, which disables vision-cone culling for building-sized
    // objects so they do not flash as the camera turns.
    private static WorldObjectVisibilityContext CreateWmoContext(float fogEnd)
        => new(
            CameraPosition: Vector3.Zero,
            CameraForward: Vector3.UnitY,
            FogEnd: fogEnd,
            ObjectStreamingRangeMultiplier: 1.0f,
            CullSmallDoodadsOnly: false,
            CountAsTaxiActor: false,
            VerticalFieldOfViewRadians: MathF.PI / 3f,
            VisibilityProfile: WorldObjectVisibilityProfile.Quality,
            IgnoreVisionConeCulling: true);

    private static WorldObjectInstance CreateInstance(string modelKey, Vector3 center, float halfExtent)
    {
        Vector3 extent = new(halfExtent, halfExtent, halfExtent);
        return new WorldObjectInstance
        {
            ModelKey = modelKey,
            ModelName = modelKey,
            ModelPath = modelKey,
            Transform = Matrix4x4.CreateTranslation(center),
            PlacementPosition = center,
            PlacementScale = 1.0f,
            BoundsMin = center - extent,
            BoundsMax = center + extent,
            LocalBoundsMin = -extent,
            LocalBoundsMax = extent,
            BoundsResolved = true,
        };
    }
}
