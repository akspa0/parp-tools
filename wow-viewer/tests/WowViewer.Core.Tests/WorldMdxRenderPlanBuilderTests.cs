using System.Numerics;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Tests;

public sealed class WorldMdxRenderPlanBuilderTests
{
    [Fact]
    public void Build_GroupsConsecutiveOpaqueRoutesByModelKey()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateVisibleMdx("tree"));
        visibility.VisibleMdx.Add(CreateVisibleMdx("tree"));
        visibility.VisibleMdx.Add(CreateVisibleMdx("rock"));

        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(passFrame, visibility, static _ => false);

        WorldMdxRenderPlan plan = WorldMdxRenderPlanBuilder.Build(passFrame, visibility);

        Assert.Equal(2, plan.OpaqueBatchCount);
        Assert.Equal([2, 1], plan.OpaqueBatches.Select(static batch => batch.InstanceCount).ToArray());
        Assert.Equal(["tree", "rock"], plan.OpaqueBatches.Select(static batch => batch.ModelKey).ToArray());
    }

    [Fact]
    public void Build_SplitsTransparentBatchesWhenUnbatchedClassificationChanges()
    {
        WorldVisibilityFrame visibility = new();
        WorldObjectPassFrame passFrame = new();
        visibility.VisibleMdx.Add(CreateVisibleMdx("tree-unbatched", centerDistanceSq: 10f));
        visibility.VisibleMdx.Add(CreateVisibleMdx("rock", centerDistanceSq: 100f));
        visibility.VisibleMdx.Add(CreateVisibleMdx("tree-batched", centerDistanceSq: 50f));

        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(
            passFrame,
            visibility,
            static visible => visible.Instance.ModelKey.Contains("unbatched", StringComparison.Ordinal));
        WorldObjectPassCoordinator.PlanTransparentMdxRoutes(passFrame, visibility);

        WorldMdxRenderPlan plan = WorldMdxRenderPlanBuilder.Build(passFrame, visibility);

        Assert.Equal(3, plan.TransparentBatchCount);
        Assert.Equal(["rock", "tree-batched", "tree-unbatched"], plan.TransparentBatches.Select(static batch => batch.ModelKey).ToArray());
        Assert.Equal([1, 1, 1], plan.TransparentBatches.Select(static batch => batch.InstanceCount).ToArray());
        Assert.Equal([false, false, true], plan.TransparentBatches.Select(static batch => batch.RequiresUnbatchedRender).ToArray());
    }

    private static WorldVisibleMdxEntry CreateVisibleMdx(string modelKey, float centerDistanceSq = 1f)
    {
        return new WorldVisibleMdxEntry(
            new WorldObjectInstance
            {
                ModelKey = modelKey,
                ModelName = modelKey,
                ModelPath = modelKey,
                Transform = Matrix4x4.Identity,
                HasOpaqueRenderContent = true,
                HasTransparentRenderContent = true,
            },
            centerDistanceSq,
            1.0f,
            1.0f,
            false);
    }
}