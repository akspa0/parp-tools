using WowViewer.Core.Runtime.World.Passes;

namespace WowViewer.Core.Tests;

/// <summary>Spec 207 US1 — the batch-splitting decision that replaces the fade instancing gate.</summary>
public sealed class GpuInstanceBatchPolicyTests
{
    [Theory]
    [InlineData(1.0f)]
    [InlineData(0.999f)]
    public void Classify_FullyOpaque_GoesToTheOpaqueBatch(float fade)
        => Assert.Equal(GpuInstanceBatchKind.Opaque, GpuInstanceBatchPolicy.Classify(fade));

    [Theory]
    [InlineData(0.998f)]
    [InlineData(0.5f)]
    [InlineData(0.01f)]
    public void Classify_PartiallyFaded_GoesToTheFadedBatch(float fade)
        => Assert.Equal(GpuInstanceBatchKind.Faded, GpuInstanceBatchPolicy.Classify(fade));

    [Theory]
    [InlineData(0f)]
    [InlineData(0.001f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    public void Classify_BelowVisibility_IsSkipped(float fade)
        => Assert.Equal(GpuInstanceBatchKind.Skip, GpuInstanceBatchPolicy.Classify(fade));

    [Fact]
    public void FadedInstances_AreNoLongerExcludedFromInstancing()
    {
        // The defect this spec exists for: a faded instance used to be refused by the instanced path
        // and drawn one draw call at a time, precisely because it was faded.
        Assert.NotEqual(GpuInstanceBatchKind.Skip, GpuInstanceBatchPolicy.Classify(0.5f));
    }

    [Fact]
    public void DrawCalls_AreBoundedByTwoRegardlessOfInstanceCount()
    {
        Assert.Equal(2, GpuInstanceBatchPolicy.DrawCallsForModel(opaqueInstances: 5000, fadedInstances: 3000));
        Assert.Equal(1, GpuInstanceBatchPolicy.DrawCallsForModel(opaqueInstances: 5000, fadedInstances: 0));
        Assert.Equal(1, GpuInstanceBatchPolicy.DrawCallsForModel(opaqueInstances: 0, fadedInstances: 3000));
        Assert.Equal(0, GpuInstanceBatchPolicy.DrawCallsForModel(opaqueInstances: 0, fadedInstances: 0));
    }

    [Fact]
    public void TheSavingIsTheFadedInstanceCount()
    {
        // 36% of the visible disc sits in the fade band, so on a scene of ~8,989 submissions the
        // faded share is the population that used to cost one draw call each.
        const int opaque = 5750;
        const int faded = 3239;

        int before = GpuInstanceBatchPolicy.LegacyDrawCallsForModel(opaque, faded);
        int after = GpuInstanceBatchPolicy.DrawCallsForModel(opaque, faded);

        Assert.Equal(1 + faded, before);
        Assert.Equal(2, after);
        Assert.True(before > after * 1000);
    }

    [Fact]
    public void Thresholds_MatchTheRendererAndTheCollector()
    {
        // The collector drops below 1/255 and the renderer splits at 0.999; if these drift apart,
        // instances fall between the two rules and are silently lost.
        Assert.Equal(1.0f / 255.0f, GpuInstanceBatchPolicy.MinimumVisibleFade);
        Assert.Equal(0.999f, GpuInstanceBatchPolicy.OpaqueFadeThreshold);
        Assert.True(GpuInstanceBatchPolicy.MinimumVisibleFade < GpuInstanceBatchPolicy.OpaqueFadeThreshold);
    }
}
