using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Runtime.World;

public enum WorldMdxRenderPassKind
{
    Opaque,
    Transparent,
}

public sealed class WorldMdxRenderBatch
{
    public WorldMdxRenderBatch(
        WorldMdxRenderPassKind passKind,
        string modelKey,
        bool requiresUnbatchedRender,
        IReadOnlyList<int> visibleMdxIndices)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);
        ArgumentNullException.ThrowIfNull(visibleMdxIndices);

        PassKind = passKind;
        ModelKey = modelKey;
        RequiresUnbatchedRender = requiresUnbatchedRender;
        VisibleMdxIndices = visibleMdxIndices;
    }

    public WorldMdxRenderPassKind PassKind { get; }

    public string ModelKey { get; }

    public bool RequiresUnbatchedRender { get; }

    public IReadOnlyList<int> VisibleMdxIndices { get; }

    public int InstanceCount => VisibleMdxIndices.Count;
}

public sealed class WorldMdxRenderPlan
{
    public WorldMdxRenderPlan(
        IReadOnlyList<WorldMdxRenderBatch> opaqueBatches,
        IReadOnlyList<WorldMdxRenderBatch> transparentBatches)
    {
        ArgumentNullException.ThrowIfNull(opaqueBatches);
        ArgumentNullException.ThrowIfNull(transparentBatches);

        OpaqueBatches = opaqueBatches;
        TransparentBatches = transparentBatches;
    }

    public IReadOnlyList<WorldMdxRenderBatch> OpaqueBatches { get; }

    public IReadOnlyList<WorldMdxRenderBatch> TransparentBatches { get; }

    public int OpaqueBatchCount => OpaqueBatches.Count;

    public int TransparentBatchCount => TransparentBatches.Count;

    public int OpaqueInstanceCount => OpaqueBatches.Sum(static batch => batch.InstanceCount);

    public int TransparentInstanceCount => TransparentBatches.Sum(static batch => batch.InstanceCount);
}

public static class WorldMdxRenderPlanBuilder
{
    public static WorldMdxRenderPlan Build(WorldObjectPassFrame passFrame, WorldVisibilityFrame visibility)
    {
        ArgumentNullException.ThrowIfNull(passFrame);
        ArgumentNullException.ThrowIfNull(visibility);

        return new WorldMdxRenderPlan(
            BuildBatches(WorldMdxRenderPassKind.Opaque, passFrame.OpaqueVisibleMdxRoutes, visibility),
            BuildBatches(WorldMdxRenderPassKind.Transparent, passFrame.TransparentVisibleMdxRoutes, visibility));
    }

    private static IReadOnlyList<WorldMdxRenderBatch> BuildBatches(
        WorldMdxRenderPassKind passKind,
        IReadOnlyList<WorldVisibleMdxPassRoute> routes,
        WorldVisibilityFrame visibility)
    {
        if (routes.Count == 0)
            return [];

        List<WorldMdxRenderBatch> batches = new();
        string? currentModelKey = null;
        bool currentRequiresUnbatched = false;
        List<int> currentVisibleIndices = [];

        for (int index = 0; index < routes.Count; index++)
        {
            WorldVisibleMdxPassRoute route = routes[index];
            WorldVisibleMdxEntry visible = visibility.VisibleMdx[route.VisibleMdxIndex];
            string modelKey = visible.Instance.ModelKey;

            if (currentModelKey is not null
                && string.Equals(currentModelKey, modelKey, StringComparison.OrdinalIgnoreCase)
                && currentRequiresUnbatched == route.RequiresUnbatchedRender)
            {
                currentVisibleIndices.Add(route.VisibleMdxIndex);
                continue;
            }

            FlushBatch(passKind, batches, currentModelKey, currentRequiresUnbatched, currentVisibleIndices);
            currentModelKey = modelKey;
            currentRequiresUnbatched = route.RequiresUnbatchedRender;
            currentVisibleIndices = [route.VisibleMdxIndex];
        }

        FlushBatch(passKind, batches, currentModelKey, currentRequiresUnbatched, currentVisibleIndices);
        return batches;
    }

    private static void FlushBatch(
        WorldMdxRenderPassKind passKind,
        List<WorldMdxRenderBatch> batches,
        string? modelKey,
        bool requiresUnbatchedRender,
        List<int> visibleMdxIndices)
    {
        if (string.IsNullOrWhiteSpace(modelKey) || visibleMdxIndices.Count == 0)
            return;

        batches.Add(new WorldMdxRenderBatch(passKind, modelKey, requiresUnbatchedRender, visibleMdxIndices.ToArray()));
    }
}