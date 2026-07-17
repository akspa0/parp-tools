using System.Numerics;
using WowViewer.Core.Renderer.Terrain;

namespace WowViewer.Core.Tests;

public sealed class Native0533368WorldLightDirectionTests
{
    [Fact]
    public void EvaluateNativeRay_MatchesRecoveredRuntimeSample()
    {
        Vector3 ray = Native0533368WorldLightDirection.EvaluateNativeRay(0.6976439f);

        Assert.Equal(-0.6481626f, ray.X, 4);
        Assert.Equal(-0.6481628f, ray.Y, 4);
        Assert.Equal(-0.3997127f, ray.Z, 4);
    }

    [Fact]
    public void ProvisionalViewerSource_UsesFixedNativeAzimuthAndNeverVerticalNoon()
    {
        Assert.True(Native0533368WorldLightDirection.TryEvaluateProvisionalViewerSource(
            Native0533368WorldLightDirection.BuildIdentity,
            0.5f,
            out NativeWorldLightDirectionSample noon));

        Vector2 minimapRasterSource = new(-noon.ViewerSourceDirection.Y, -noon.ViewerSourceDirection.X);
        Assert.True(noon.ViewerSourceDirection.X > 0f);
        Assert.True(noon.ViewerSourceDirection.Y > 0f);
        Assert.True(noon.ViewerSourceDirection.Z > 0f);
        Assert.True(minimapRasterSource.X < 0f);
        Assert.True(minimapRasterSource.Y < 0f);
        Assert.NotEqual(1f, noon.ViewerSourceDirection.Z);
        Assert.Equal("native_0533368_ray_recovered_viewer_transform_unproven", noon.EvidenceState);
    }

    [Fact]
    public void EvaluateNativeRay_WrapsThePeriodicTable()
    {
        Vector3 start = Native0533368WorldLightDirection.EvaluateNativeRay(0f);
        Vector3 wrapped = Native0533368WorldLightDirection.EvaluateNativeRay(1f);

        Assert.Equal(start.X, wrapped.X, 6);
        Assert.Equal(start.Y, wrapped.Y, 6);
        Assert.Equal(start.Z, wrapped.Z, 6);
        Assert.False(Native0533368WorldLightDirection.TryEvaluateProvisionalViewerSource(
            "1.12.1.5875",
            0.5f,
            out _));
    }
}
