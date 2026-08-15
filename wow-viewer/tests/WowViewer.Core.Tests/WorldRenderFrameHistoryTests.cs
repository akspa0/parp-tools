using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;
using Xunit;

namespace WowViewer.Core.Tests;

/// <summary>
/// Detector-power proof for the rolling frame history (Spec 152, US1 / FR-004).
///
/// The point of these tests is not that the ring buffer stores numbers. It is that the history can
/// actually SEE a hitch of known size at a known frame. The previous measurement path could not,
/// because it sampled a stationary camera for 12 frames and reported no distribution at all, so
/// every "no hitch found" result it produced was a false null. A detector whose power is untested
/// is not evidence.
/// </summary>
public class WorldRenderFrameHistoryTests
{
    private static WorldRenderFrameStats FrameOf(double totalMs)
        => WorldRenderFrameStats.Empty with { TotalCpuMs = totalMs };

    private static WorldRenderFrameStats FrameOf(double totalMs, WorldRenderStage stage, double stageMs)
    {
        WorldRenderFrameStats stats = WorldRenderFrameStats.Empty with { TotalCpuMs = totalMs };
        var stageStats = new WorldRenderStageStats(stageMs);
        return stage switch
        {
            WorldRenderStage.Terrain => stats with { Terrain = stageStats },
            WorldRenderStage.WmoSubmission => stats with { WmoSubmission = stageStats },
            WorldRenderStage.WmoTransparentSubmission => stats with { WmoTransparentSubmission = stageStats },
            WorldRenderStage.WmoVisibility => stats with { WmoVisibility = stageStats },
            WorldRenderStage.MdxOpaqueSubmission => stats with { MdxOpaqueSubmission = stageStats },
            WorldRenderStage.MdxVisibility => stats with { MdxVisibility = stageStats },
            WorldRenderStage.DeferredAssetLoads => stats with { DeferredAssetLoads = stageStats },
            WorldRenderStage.SceneMaintenance => stats with { SceneMaintenance = stageStats },
            WorldRenderStage.Overlay => stats with { Overlay = stageStats },
            // A silent fall-through would set no stage at all and quietly invalidate any test that
            // relied on it, so fail loudly instead.
            _ => throw new ArgumentOutOfRangeException(
                nameof(stage), stage, "Add this stage to the test helper before using it."),
        };
    }

    [Fact]
    public void InjectedHitch_IsFlaggedAtCorrectFrameWithCorrectMagnitude()
    {
        var history = new WorldRenderFrameHistory(capacity: 256);
        const double steadyMs = 4.0;
        const double hitchMs = 90.0;
        const int hitchAt = 37;

        for (int i = 0; i < 120; i++)
            history.Record(FrameOf(i == hitchAt ? hitchMs : steadyMs), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        WorldRenderHitch hitch = Assert.Single(snapshot.Hitches);
        Assert.Equal(hitchAt, hitch.FrameIndex);
        Assert.Equal(hitchMs, hitch.TotalCpuMs, precision: 6);
    }

    [Fact]
    public void HitchWithNoStageCost_IsReportedAsUnaccounted_NotAsATinyStage()
    {
        // The real defect: ~350ms frames whose largest instrumented stage was 0.2ms. Naming that
        // stage points at the wrong place; the honest answer is that no timer covers the cost.
        var history = new WorldRenderFrameHistory(capacity: 64);
        for (int i = 0; i < 30; i++)
            history.Record(FrameOf(1.0, WorldRenderStage.MdxVisibility, 0.2), cameraMoved: true);
        history.Record(FrameOf(350.0, WorldRenderStage.MdxVisibility, 0.2), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        WorldRenderHitch hitch = Assert.Single(snapshot.Hitches);
        Assert.True(hitch.IsDominatedByUnaccountedTime);
        Assert.Equal(349.8, hitch.UnaccountedMs, precision: 4);
        Assert.Contains("UNACCOUNTED", hitch.DominantCause);
        Assert.Equal(349.8, snapshot.Unaccounted.MaxMs, precision: 4);
    }

    [Fact]
    public void StagesByMaxDescending_SurfacesRareButHugeStage_ThatP99Hides()
    {
        // DeferredAssetLoads fires once in 200 frames at 46ms. Its p99 is ~0, so p99 ordering buries
        // it below stages that are merely steadily small - which is how the old view hid it.
        var history = new WorldRenderFrameHistory(capacity: 256);
        for (int i = 0; i < 200; i++)
            history.Record(FrameOf(1.0, WorldRenderStage.MdxVisibility, 0.3), cameraMoved: true);
        history.Record(FrameOf(47.0, WorldRenderStage.DeferredAssetLoads, 46.5), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        WorldRenderStage worstByMax = snapshot.StagesByMaxDescending().First().Key;
        WorldRenderStage worstByP99 = snapshot.StagesByP99Descending().First().Key;

        Assert.Equal(WorldRenderStage.DeferredAssetLoads, worstByMax);
        Assert.NotEqual(WorldRenderStage.DeferredAssetLoads, worstByP99);
    }

    [Fact]
    public void InjectedHitch_AttributesTheDominantStage()
    {
        var history = new WorldRenderFrameHistory(capacity: 128);
        for (int i = 0; i < 60; i++)
            history.Record(FrameOf(4.0, WorldRenderStage.Terrain, 1.0), cameraMoved: true);

        // One frame where terrain upload dominates, which is the shape a streaming hitch takes.
        history.Record(FrameOf(80.0, WorldRenderStage.Terrain, 74.0), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        WorldRenderHitch hitch = Assert.Single(snapshot.Hitches);
        Assert.Equal(WorldRenderStage.Terrain, hitch.DominantStage);
        Assert.Equal(74.0, hitch.DominantStageMs, precision: 6);
    }

    [Fact]
    public void SteadyFrames_ProduceNoFalseHitches()
    {
        var history = new WorldRenderFrameHistory(capacity: 256);
        for (int i = 0; i < 200; i++)
            history.Record(FrameOf(6.0), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        Assert.Empty(snapshot.Hitches);
        Assert.Equal(0, snapshot.Total.OverThresholdCount);
    }

    [Fact]
    public void RareHitch_IsCarriedByMaxAndCount_NotByP99()
    {
        // 99 frames at 5ms, 1 frame at 100ms. A mean-only or last-frame-only view hides this
        // completely; that is precisely the gallop's signature.
        //
        // Note what p99 does here: under nearest-rank over 100 samples it selects the 99th value,
        // which is still 5ms. A 1-in-100 hitch is invisible at p99 BY CONSTRUCTION. That is not a
        // bug, it is why this type also reports Max and OverThresholdCount — for rare hitches those
        // carry the signal and percentiles do not. Reading p99 alone would reproduce the same class
        // of false null the old stationary harness produced.
        var history = new WorldRenderFrameHistory(capacity: 128);
        for (int i = 0; i < 99; i++)
            history.Record(FrameOf(5.0), cameraMoved: true);
        history.Record(FrameOf(100.0), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        Assert.Equal(5.0, snapshot.Total.MedianMs, precision: 6);
        Assert.Equal(5.0, snapshot.Total.P95Ms, precision: 6);
        Assert.Equal(5.0, snapshot.Total.P99Ms, precision: 6);

        // These two are what actually surface the hitch.
        Assert.Equal(100.0, snapshot.Total.MaxMs, precision: 6);
        Assert.Equal(1, snapshot.Total.OverThresholdCount);
        Assert.Single(snapshot.Hitches);
    }

    [Fact]
    public void SustainedHitchRate_IsVisibleAtP99()
    {
        // Once hitching is frequent enough (here 5%), the percentile tail does move, so p95/p99 are
        // the right instrument for a persistent gallop while Max/Count cover the rare spike.
        var history = new WorldRenderFrameHistory(capacity: 256);
        for (int i = 0; i < 200; i++)
            history.Record(FrameOf(i % 20 == 0 ? 60.0 : 5.0), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        Assert.Equal(5.0, snapshot.Total.MedianMs, precision: 6);
        Assert.Equal(60.0, snapshot.Total.P99Ms, precision: 6);
        Assert.Equal(10, snapshot.Total.OverThresholdCount);
    }

    [Fact]
    public void StationaryWindow_IsLabelledAsUnableToDemonstrateMovementBehavior()
    {
        var history = new WorldRenderFrameHistory(capacity: 64);
        for (int i = 0; i < 30; i++)
            history.Record(FrameOf(5.0), cameraMoved: false);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        Assert.False(snapshot.CameraMovedDuringWindow);
        Assert.False(snapshot.CanDemonstrateMovementBehavior);
    }

    [Fact]
    public void MovingWindow_IsLabelledAsAbleToDemonstrateMovementBehavior()
    {
        var history = new WorldRenderFrameHistory(capacity: 64);
        for (int i = 0; i < 30; i++)
            history.Record(FrameOf(5.0), cameraMoved: i > 10);

        Assert.True(history.Snapshot(hitchThresholdMs: 33.0).CanDemonstrateMovementBehavior);
    }

    [Fact]
    public void RingBuffer_EvictsOldestAndKeepsAbsoluteFrameIndices()
    {
        var history = new WorldRenderFrameHistory(capacity: 8);
        for (int i = 0; i < 20; i++)
            history.Record(FrameOf(i), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 1000.0);

        Assert.Equal(8, snapshot.FrameCount);
        Assert.Equal(12, snapshot.FirstFrameIndex);
        Assert.Equal(19, snapshot.LastFrameIndex);
        Assert.Equal(20, history.TotalFramesRecorded);
    }

    [Fact]
    public void HitchSurvivesWraparound_SoALongSessionStillSeesIt()
    {
        // The hitch must remain findable after the buffer has wrapped, otherwise a long session
        // silently loses the very event being hunted.
        var history = new WorldRenderFrameHistory(capacity: 16);
        for (int i = 0; i < 40; i++)
            history.Record(FrameOf(4.0), cameraMoved: true);
        history.Record(FrameOf(75.0), cameraMoved: true);
        for (int i = 0; i < 5; i++)
            history.Record(FrameOf(4.0), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        WorldRenderHitch hitch = Assert.Single(snapshot.Hitches);
        Assert.Equal(40, hitch.FrameIndex);
        Assert.Equal(75.0, hitch.TotalCpuMs, precision: 6);
    }

    [Fact]
    public void EmptyHistory_ReportsEmptyRatherThanThrowing()
    {
        WorldRenderFrameHistorySnapshot snapshot =
            new WorldRenderFrameHistory(capacity: 32).Snapshot(hitchThresholdMs: 33.0);

        Assert.Equal(0, snapshot.FrameCount);
        Assert.Empty(snapshot.Hitches);
        Assert.False(snapshot.CanDemonstrateMovementBehavior);
    }

    [Fact]
    public void PerStageDistributions_AreTrackedIndependently()
    {
        var history = new WorldRenderFrameHistory(capacity: 64);
        for (int i = 0; i < 50; i++)
            history.Record(FrameOf(10.0, WorldRenderStage.WmoSubmission, 7.0), cameraMoved: true);

        WorldRenderFrameHistorySnapshot snapshot = history.Snapshot(hitchThresholdMs: 33.0);

        Assert.Equal(7.0, snapshot.Stages[WorldRenderStage.WmoSubmission].MedianMs, precision: 6);
        Assert.Equal(0.0, snapshot.Stages[WorldRenderStage.Liquid].MedianMs, precision: 6);
    }

    [Fact]
    public void TraverseInto_DoesNotAllocateOnSteadyStateFrames()
    {
        // C1: the allocating Traverse overload builds two lists, a diagnostics object holding four
        // dictionaries, and a result record PER GRAPH PER FRAME. ADT tiles are isolated into
        // independent graphs, so the hot path paid that in proportion to the resident tile set.
        WorldSceneGraph graph = SyntheticWorldWorkloadBuilder.Build(
            new SyntheticWorldWorkloadDefinition
            {
                ResidentRegionCount = 4,
                ChunksPerRegion = 16,
                M2Placements = 32,
            }).Graph;

        var visible = new List<WorldSceneNode>();
        var rejected = new List<WorldSceneNode>();
        var diagnostics = new WorldSceneTraversalDiagnostics();

        void RunBatch()
        {
            for (int i = 0; i < 200; i++)
            {
                WorldSceneTraversal.TraverseInto(
                    graph, static _ => true, visible, rejected, diagnostics,
                    collectDetailedDiagnostics: false);
            }
        }

        // First batch absorbs one-time costs (buffer growth, first-call JIT/delegate caching).
        RunBatch();

        // Steady state must be exactly zero: any residue here would be per-frame churn.
        long before = GC.GetAllocatedBytesForCurrentThread();
        RunBatch();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }

    [Fact]
    public void CopyRecentTotalMs_ReturnsRealSeriesOldestFirst_AcrossWraparound()
    {
        var history = new WorldRenderFrameHistory(capacity: 8);
        for (int i = 0; i < 12; i++)
            history.Record(FrameOf(i), cameraMoved: true);

        var buffer = new float[8];
        int written = history.CopyRecentTotalMs(buffer);

        // Newest 8 of 12 recorded, oldest first: 4,5,6,7,8,9,10,11.
        Assert.Equal(8, written);
        Assert.Equal(new float[] { 4, 5, 6, 7, 8, 9, 10, 11 }, buffer);
    }

    [Fact]
    public void CopyRecentTotalMs_HandlesBufferSmallerThanWindow()
    {
        var history = new WorldRenderFrameHistory(capacity: 16);
        for (int i = 0; i < 10; i++)
            history.Record(FrameOf(i), cameraMoved: true);

        var buffer = new float[3];
        int written = history.CopyRecentTotalMs(buffer);

        Assert.Equal(3, written);
        Assert.Equal(new float[] { 7, 8, 9 }, buffer);
    }

    [Fact]
    public void CopyRecentTotalMs_DoesNotAllocate()
    {
        var history = new WorldRenderFrameHistory(capacity: 256);
        for (int i = 0; i < 256; i++)
            history.Record(FrameOf(i), cameraMoved: true);
        var buffer = new float[240];
        history.CopyRecentTotalMs(buffer);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 500; i++)
            history.CopyRecentTotalMs(buffer);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }

    [Fact]
    public void Recording_DoesNotAllocate()
    {
        // FR-002: the recorder must not become the kind of per-frame churn it exists to measure.
        var history = new WorldRenderFrameHistory(capacity: 512);
        WorldRenderFrameStats stats = FrameOf(5.0, WorldRenderStage.Terrain, 2.0);

        for (int i = 0; i < 64; i++)
            history.Record(stats, cameraMoved: true);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 5_000; i++)
            history.Record(stats, cameraMoved: true);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }
}
