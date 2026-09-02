using WowViewer.Core.Runtime.World.Passes;
using Xunit;

namespace WowViewer.Core.Tests;

/// <summary>
/// Specs 201 and 202 Phase 0. These pin the decomposition itself, not any rendering decision:
/// the point of the phase is that the existing numbers become readable before anything is
/// optimised on the strength of them.
/// </summary>
public class ModelSubmissionAccountingTests
{
    /// <summary>
    /// Spec 201 FR-005. Per-path counts must sum exactly to the aggregate totals, which is what
    /// proves the change is a decomposition of the old counters and not a redefinition of them.
    /// </summary>
    [Fact]
    public void PerPathCounts_SumExactlyToAggregateTotals()
    {
        var tally = default(WorldModelSubmissionTally);

        tally.Record(WorldModelRenderPath.AdapterSkin, WorldModelSubmissionOutcome.Instanced, WorldModelBatchGate.None);
        tally.Record(WorldModelRenderPath.AdapterSkin, WorldModelSubmissionOutcome.Instanced, WorldModelBatchGate.None);
        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.StateHoisted, WorldModelBatchGate.GpuInstancingUnsupported);
        tally.Record(WorldModelRenderPath.NativeEmbeddedProfile, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.RouteRequiresUnbatchedRender);
        tally.Record(WorldModelRenderPath.NativeEmbeddedProfile, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.RouteRequiresUnbatchedRender);
        tally.Record(WorldModelRenderPath.ConversionFallback, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.BatchingDisabled);

        WorldModelSubmissionStats stats = tally.ToStats();

        int summedPerPath =
            stats.AdapterSkin.Total
            + stats.AdapterEmbeddedProfile.Total
            + stats.NativeEmbeddedProfile.Total
            + stats.ConversionFallback.Total
            + stats.MdxDirect.Total
            + stats.Unknown.Total;

        Assert.Equal(6, stats.Total);
        Assert.Equal(stats.Total, summedPerPath);
        Assert.Equal(2, stats.Instanced);
        Assert.Equal(1, stats.StateHoisted);
        Assert.Equal(3, stats.Unbatched);

        // The old aggregate "batched" was instanced + state-hoisted, and the two totals it
        // decomposes into must still add back up to it.
        Assert.Equal(3, stats.LegacyBatchedEquivalent);
        Assert.Equal(stats.Instanced + stats.StateHoisted, stats.LegacyBatchedEquivalent);
    }

    /// <summary>
    /// Spec 202 research R1, the defect this phase exists to fix. State hoisting still issues one
    /// draw per instance; only instancing collapses draws. A test that let these share a counter
    /// would be re-encoding the bug.
    /// </summary>
    [Fact]
    public void InstancedAndStateHoisted_AreCountedSeparately()
    {
        var tally = default(WorldModelSubmissionTally);

        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.Instanced, WorldModelBatchGate.None);
        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.StateHoisted, WorldModelBatchGate.OpaqueFadeBelowInstancingThreshold);

        WorldModelSubmissionStats stats = tally.ToStats();

        Assert.Equal(1, stats.Instanced);
        Assert.Equal(1, stats.StateHoisted);
        Assert.Equal(1, stats.MdxDirect.Instanced);
        Assert.Equal(1, stats.MdxDirect.StateHoisted);
        Assert.NotEqual(stats.Instanced, stats.LegacyBatchedEquivalent);
    }

    /// <summary>
    /// Spec 202 T004. Every instance short of GPU instancing carries the gate that stopped it, and
    /// the three gates are independent with different fixes.
    /// </summary>
    [Theory]
    [InlineData(WorldModelBatchGate.BatchingDisabled)]
    [InlineData(WorldModelBatchGate.RouteRequiresUnbatchedRender)]
    [InlineData(WorldModelBatchGate.GpuInstancingUnsupported)]
    [InlineData(WorldModelBatchGate.OpaqueFadeBelowInstancingThreshold)]
    [InlineData(WorldModelBatchGate.PassHasNoBatchPath)]
    public void EveryGate_IsReachableAndReported(WorldModelBatchGate gate)
    {
        var tally = default(WorldModelSubmissionTally);
        tally.Record(WorldModelRenderPath.AdapterSkin, WorldModelSubmissionOutcome.Unbatched, gate);

        WorldModelSubmissionStats stats = tally.ToStats();

        Assert.Equal(gate, stats.DominantGate);
    }

    [Fact]
    public void NoGate_IsReportedWhenEverythingInstanced()
    {
        var tally = default(WorldModelSubmissionTally);
        tally.Record(WorldModelRenderPath.AdapterSkin, WorldModelSubmissionOutcome.Instanced, WorldModelBatchGate.None);

        Assert.Equal(WorldModelBatchGate.None, tally.ToStats().DominantGate);
    }

    /// <summary>
    /// The largest gate is the one worth fixing first, so it has to survive being recorded
    /// alongside smaller ones.
    /// </summary>
    [Fact]
    public void DominantGate_IsTheLargest_NotTheFirstOrLastRecorded()
    {
        var tally = default(WorldModelSubmissionTally);

        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.RouteRequiresUnbatchedRender);
        for (int i = 0; i < 5; i++)
            tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.StateHoisted, WorldModelBatchGate.OpaqueFadeBelowInstancingThreshold);
        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.StateHoisted, WorldModelBatchGate.GpuInstancingUnsupported);

        Assert.Equal(WorldModelBatchGate.OpaqueFadeBelowInstancingThreshold, tally.ToStats().DominantGate);
    }

    /// <summary>
    /// An instance that resolves no renderer never reaches the GPU. It must stay visible as its own
    /// number rather than being counted as a submission or silently dropped.
    /// </summary>
    [Fact]
    public void RendererUnavailable_IsCountedWithoutAddingASubmission()
    {
        var tally = default(WorldModelSubmissionTally);
        tally.RecordRendererUnavailable();

        WorldModelSubmissionStats stats = tally.ToStats();

        Assert.Equal(1, stats.GatedRendererUnavailable);
        Assert.Equal(0, stats.Total);
    }

    /// <summary>
    /// Spec 202 T003. Draw calls are not derivable from instance counts — a model draws once per
    /// geoset or section — so the tally must carry them independently of every other counter.
    /// </summary>
    [Fact]
    public void DrawCalls_AreIndependentOfInstanceCounts()
    {
        var tally = default(WorldModelSubmissionTally);

        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.RouteRequiresUnbatchedRender);
        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.RouteRequiresUnbatchedRender);
        tally.RecordDrawCalls(7);

        WorldModelSubmissionStats stats = tally.ToStats();

        Assert.Equal(2, stats.Total);
        Assert.Equal(7, stats.DrawCalls);
        Assert.Equal(3.5d, stats.DrawCallsPerInstance, 6);
    }

    [Fact]
    public void DrawCallsPerInstance_IsZeroWhenNothingWasSubmitted()
        => Assert.Equal(0d, default(WorldModelSubmissionTally).ToStats().DrawCallsPerInstance);

    /// <summary>
    /// An unrecognised route must land in <see cref="WorldModelRenderPath.Unknown"/> and still be
    /// counted, so a path added later cannot silently vanish from the totals.
    /// </summary>
    [Fact]
    public void UnrecognisedPath_FallsToUnknownAndStaysInTheTotal()
    {
        var tally = default(WorldModelSubmissionTally);
        tally.Record((WorldModelRenderPath)999, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.BatchingDisabled);

        WorldModelSubmissionStats stats = tally.ToStats();

        Assert.Equal(1, stats.Unknown.Unbatched);
        Assert.Equal(1, stats.Total);
    }

    [Fact]
    public void Add_MergesEveryCounter()
    {
        var left = default(WorldModelSubmissionTally);
        left.Record(WorldModelRenderPath.AdapterSkin, WorldModelSubmissionOutcome.Instanced, WorldModelBatchGate.None);
        left.RecordDrawCalls(3);
        left.DistinctModelCount = 1;

        var right = default(WorldModelSubmissionTally);
        right.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.BatchingDisabled);
        right.RecordRendererUnavailable();
        right.RecordDrawCalls(4);
        right.DistinctModelCount = 2;

        left.Add(right);
        WorldModelSubmissionStats stats = left.ToStats();

        Assert.Equal(2, stats.Total);
        Assert.Equal(1, stats.AdapterSkin.Instanced);
        Assert.Equal(1, stats.MdxDirect.Unbatched);
        Assert.Equal(1, stats.GatedBatchingDisabled);
        Assert.Equal(1, stats.GatedRendererUnavailable);
        Assert.Equal(7, stats.DrawCalls);
        Assert.Equal(3, stats.DistinctModelCount);
    }

    [Fact]
    public void Reset_ClearsEveryCounter()
    {
        var tally = default(WorldModelSubmissionTally);
        tally.Record(WorldModelRenderPath.AdapterSkin, WorldModelSubmissionOutcome.Instanced, WorldModelBatchGate.None);
        tally.RecordRendererUnavailable();
        tally.RecordDrawCalls(5);
        tally.DistinctModelCount = 9;

        tally.Reset();

        Assert.Equal(default, tally.ToStats());
        Assert.Equal(WorldModelSubmissionStats.Empty, tally.ToStats());
    }

    /// <summary>
    /// <see cref="WorldModelSubmissionStats.ForPath"/> is what the panel reads; it must return the
    /// field the enum names rather than a neighbouring one.
    /// </summary>
    [Fact]
    public void ForPath_ReturnsTheNamedPath()
    {
        var tally = default(WorldModelSubmissionTally);
        tally.Record(WorldModelRenderPath.AdapterSkin, WorldModelSubmissionOutcome.Instanced, WorldModelBatchGate.None);
        tally.Record(WorldModelRenderPath.AdapterEmbeddedProfile, WorldModelSubmissionOutcome.StateHoisted, WorldModelBatchGate.GpuInstancingUnsupported);
        tally.Record(WorldModelRenderPath.NativeEmbeddedProfile, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.RouteRequiresUnbatchedRender);
        tally.Record(WorldModelRenderPath.ConversionFallback, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.BatchingDisabled);
        tally.Record(WorldModelRenderPath.MdxDirect, WorldModelSubmissionOutcome.Unbatched, WorldModelBatchGate.BatchingDisabled);

        WorldModelSubmissionStats stats = tally.ToStats();

        Assert.Equal(1, stats.ForPath(WorldModelRenderPath.AdapterSkin).Instanced);
        Assert.Equal(1, stats.ForPath(WorldModelRenderPath.AdapterEmbeddedProfile).StateHoisted);
        Assert.Equal(1, stats.ForPath(WorldModelRenderPath.NativeEmbeddedProfile).Unbatched);
        Assert.Equal(1, stats.ForPath(WorldModelRenderPath.ConversionFallback).Unbatched);
        Assert.Equal(1, stats.ForPath(WorldModelRenderPath.MdxDirect).Unbatched);
        Assert.Equal(0, stats.ForPath(WorldModelRenderPath.Unknown).Total);
    }
}
