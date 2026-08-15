using WowViewer.Core.Runtime.World;
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
            WorldRenderStage.MdxOpaqueSubmission => stats with { MdxOpaqueSubmission = stageStats },
            WorldRenderStage.DeferredAssetLoads => stats with { DeferredAssetLoads = stageStats },
            WorldRenderStage.SceneMaintenance => stats with { SceneMaintenance = stageStats },
            _ => stats,
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
