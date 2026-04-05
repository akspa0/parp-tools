using System.Numerics;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Tests;

public sealed class WorldObjectPassCoordinatorTests
{
    [Fact]
    public void ExecuteVisibleMdxAnimation_DeduplicatesByModelKey()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateMdx("a", 10f));
        visibility.VisibleMdx.Add(CreateMdx("a", 20f));
        visibility.VisibleMdx.Add(CreateMdx("b", 30f));
        List<string> updated = new();

        int count = WorldObjectPassCoordinator.ExecuteVisibleMdxAnimation(passFrame, visibility, visible =>
            updated.Add(visible.Instance.ModelKey));

        Assert.Equal(2, count);
        Assert.Equal(["a", "b"], updated);
    }

    [Fact]
    public void PlanOpaqueMdxRoutes_CapturesFirstBatchedEntry()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateMdx("u", 10f));
        visibility.VisibleMdx.Add(CreateMdx("b", 20f));

        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(
            passFrame,
            visibility,
            visible => visible.Instance.ModelKey == "u");

        Assert.Equal(1, passFrame.FirstOpaqueBatchedVisibleMdxIndex);
    }

    [Fact]
    public void ExecutePlannedOpaqueMdx_SplitsBatchedAndUnbatchedCounts()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateMdx("u1", 10f));
        visibility.VisibleMdx.Add(CreateMdx("b1", 20f));
        visibility.VisibleMdx.Add(CreateMdx("u2", 30f));
        List<string> batched = new();
        List<string> unbatched = new();

        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(
            passFrame,
            visibility,
            visible => visible.Instance.ModelKey.StartsWith("u", StringComparison.Ordinal));

        var counts = WorldObjectPassCoordinator.ExecutePlannedOpaqueMdx(
            passFrame,
            visibility,
            visible => unbatched.Add(visible.Instance.ModelKey),
            visible => batched.Add(visible.Instance.ModelKey));

        Assert.Equal(1, counts.batchedCount);
        Assert.Equal(2, counts.unbatchedCount);
        Assert.Equal(["b1"], batched);
        Assert.Equal(["u1", "u2"], unbatched);
    }

    [Fact]
    public void PlanTransparentMdxRoutes_SortsBackToFrontByDistance()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateMdx("near", 10f));
        visibility.VisibleMdx.Add(CreateMdx("far", 100f));
        visibility.VisibleMdx.Add(CreateMdx("mid", 50f));

        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(passFrame, visibility, static _ => false);
        WorldObjectPassCoordinator.PlanTransparentMdxRoutes(passFrame, visibility);

        Assert.Equal(["far", "mid", "near"], passFrame.TransparentVisibleMdxRoutes.Select(route => visibility.VisibleMdx[route.VisibleMdxIndex].Instance.ModelKey).ToArray());
    }

    [Fact]
    public void PlanTransparentMdxRoutes_SkipsOpaqueOnlyEntries()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateMdx("opaque", 125f));
        visibility.VisibleMdx.Add(CreateMdx("transparent-far", 100f));
        visibility.VisibleMdx.Add(CreateMdx("transparent-near", 10f));

        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(passFrame, visibility, static _ => false);
        WorldObjectPassCoordinator.PlanTransparentMdxRoutes(
            passFrame,
            visibility,
            visible => !string.Equals(visible.Instance.ModelKey, "opaque", StringComparison.Ordinal));

        Assert.Equal(["transparent-far", "transparent-near"], passFrame.TransparentVisibleMdxRoutes.Select(route => visibility.VisibleMdx[route.VisibleMdxIndex].Instance.ModelKey).ToArray());
    }

    [Fact]
    public void ExecutePlannedTransparentMdx_UsesPreparedOrder()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateMdx("near", 10f));
        visibility.VisibleMdx.Add(CreateMdx("far", 100f));
        visibility.VisibleMdx.Add(CreateMdx("mid", 50f));
        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(passFrame, visibility, static _ => false);
        WorldObjectPassCoordinator.PlanTransparentMdxRoutes(passFrame, visibility);
        List<string> drawn = new();

        var counts = WorldObjectPassCoordinator.ExecutePlannedTransparentMdx(
            passFrame,
            visibility,
            static _ => { },
            visible => drawn.Add(visible.Instance.ModelKey));

        Assert.Equal(3, counts.batchedCount);
        Assert.Equal(0, counts.unbatchedCount);
        Assert.Equal(["far", "mid", "near"], drawn);
    }

    [Fact]
    public void PlanTransparentMdxRoutes_PreservesOpaqueRouteClassification()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateMdx("unbatched-far", 100f));
        visibility.VisibleMdx.Add(CreateMdx("batched-near", 10f));

        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(
            passFrame,
            visibility,
            visible => visible.Instance.ModelKey.StartsWith("unbatched", StringComparison.Ordinal));
        WorldObjectPassCoordinator.PlanTransparentMdxRoutes(passFrame, visibility);

        Assert.Equal([true, false], passFrame.TransparentVisibleMdxRoutes.Select(route => route.RequiresUnbatchedRender).ToArray());
    }

    [Fact]
    public void ExecuteVisibleWmoOpaque_VisitsAllVisibleEntries()
    {
        WorldVisibilityFrame visibility = new();
        visibility.VisibleWmos.Add(new WorldVisibleWmoEntry(CreateInstance("w1"), 10f));
        visibility.VisibleWmos.Add(new WorldVisibleWmoEntry(CreateInstance("w2"), 20f));
        List<string> drawn = new();

        int rendered = WorldObjectPassCoordinator.ExecuteVisibleWmoOpaque(visibility, visible =>
            drawn.Add(visible.Instance.ModelKey));

        Assert.Equal(2, rendered);
        Assert.Equal(["w1", "w2"], drawn);
    }

    private static WorldVisibleMdxEntry CreateMdx(string modelKey, float distanceSq)
    {
        return new WorldVisibleMdxEntry(CreateInstance(modelKey), distanceSq, 1.0f, 1.0f, false);
    }

    private static WorldObjectInstance CreateInstance(string modelKey)
    {
        return new WorldObjectInstance
        {
            ModelKey = modelKey,
            ModelName = modelKey,
            ModelPath = modelKey,
            Transform = Matrix4x4.Identity,
        };
    }
}