using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Runtime.World.Passes;

public static class WorldObjectPassCoordinator
{
    public readonly record struct WorldWmoOpaqueBatchCandidate(string ModelKey, bool CanBatch, int VisibleIndex);

    public readonly record struct WorldWmoOpaqueBatch(string ModelKey, IReadOnlyList<int> VisibleIndices);

    public readonly record struct WorldWmoOpaqueBatchPlan(
        IReadOnlyList<WorldWmoOpaqueBatch> Batches,
        IReadOnlyList<int> FallbackVisibleIndices);

    public static WorldWmoOpaqueBatchPlan PlanOpaqueWmoBatches(
        IReadOnlyList<WorldWmoOpaqueBatchCandidate> candidates)
    {
        var batchIndicesByModel = new Dictionary<string, List<int>>(StringComparer.OrdinalIgnoreCase);
        var fallbackVisibleIndices = new List<int>();

        for (int i = 0; i < candidates.Count; i++)
        {
            WorldWmoOpaqueBatchCandidate candidate = candidates[i];
            if (!candidate.CanBatch || string.IsNullOrWhiteSpace(candidate.ModelKey))
            {
                fallbackVisibleIndices.Add(candidate.VisibleIndex);
                continue;
            }

            if (!batchIndicesByModel.TryGetValue(candidate.ModelKey, out List<int>? visibleIndices))
            {
                visibleIndices = new List<int>();
                batchIndicesByModel.Add(candidate.ModelKey, visibleIndices);
            }

            visibleIndices.Add(candidate.VisibleIndex);
        }

        var batches = new List<WorldWmoOpaqueBatch>(batchIndicesByModel.Count);
        foreach ((string modelKey, List<int> visibleIndices) in batchIndicesByModel)
            batches.Add(new WorldWmoOpaqueBatch(modelKey, visibleIndices));

        return new WorldWmoOpaqueBatchPlan(batches, fallbackVisibleIndices);
    }

    public static int ExecuteVisibleWmoOpaque(WorldVisibilityFrame visibility, Action<WorldVisibleWmoEntry> renderVisibleWmo)
    {
        int renderedCount = 0;
        for (int i = 0; i < visibility.VisibleWmos.Count; i++)
        {
            renderVisibleWmo(visibility.VisibleWmos[i]);
            renderedCount++;
        }

        return renderedCount;
    }

    public static int ExecuteVisibleMdxAnimation(
        WorldObjectPassFrame passFrame,
        WorldVisibilityFrame visibility,
        Action<WorldVisibleMdxEntry> updateVisibleMdxAnimation)
    {
        passFrame.UpdatedMdxModelKeys.Clear();
        int updatedRendererCount = 0;

        for (int i = 0; i < visibility.VisibleMdx.Count; i++)
        {
            WorldVisibleMdxEntry visible = visibility.VisibleMdx[i];
            if (!passFrame.UpdatedMdxModelKeys.Add(visible.Instance.ModelKey))
                continue;

            updateVisibleMdxAnimation(visible);
            updatedRendererCount++;
        }

        return updatedRendererCount;
    }

    public static void PlanOpaqueMdxRoutes(
        WorldObjectPassFrame passFrame,
        WorldVisibilityFrame visibility,
        Func<WorldVisibleMdxEntry, bool> requiresUnbatchedRender,
        Func<WorldVisibleMdxEntry, bool>? includeOpaque = null)
    {
        passFrame.OpaqueVisibleMdxRoutes.Clear();
        passFrame.UnbatchedVisibleMdxIndices.Clear();
        passFrame.FirstOpaqueBatchedVisibleMdxIndex = -1;

        for (int i = 0; i < visibility.VisibleMdx.Count; i++)
        {
            WorldVisibleMdxEntry visible = visibility.VisibleMdx[i];
            if (includeOpaque != null && !includeOpaque(visible))
                continue;

            bool requiresUnbatched = requiresUnbatchedRender(visible);
            passFrame.OpaqueVisibleMdxRoutes.Add(new WorldVisibleMdxPassRoute(i, requiresUnbatched));

            if (requiresUnbatched)
                passFrame.UnbatchedVisibleMdxIndices.Add(i);

            if (!requiresUnbatched && passFrame.FirstOpaqueBatchedVisibleMdxIndex < 0)
                passFrame.FirstOpaqueBatchedVisibleMdxIndex = i;
        }
    }

    public static (int batchedCount, int unbatchedCount) ExecutePlannedOpaqueMdx(
        WorldObjectPassFrame passFrame,
        WorldVisibilityFrame visibility,
        Action<WorldVisibleMdxEntry> renderUnbatched,
        Action<WorldVisibleMdxEntry> renderBatched)
    {
        int batchedCount = 0;
        int unbatchedCount = 0;

        for (int i = 0; i < passFrame.OpaqueVisibleMdxRoutes.Count; i++)
        {
            WorldVisibleMdxPassRoute route = passFrame.OpaqueVisibleMdxRoutes[i];
            WorldVisibleMdxEntry visible = visibility.VisibleMdx[route.VisibleMdxIndex];
            if (route.RequiresUnbatchedRender)
            {
                renderUnbatched(visible);
                unbatchedCount++;
            }
            else
            {
                renderBatched(visible);
                batchedCount++;
            }
        }

        return (batchedCount, unbatchedCount);
    }

    public static void PlanTransparentMdxRoutes(
        WorldObjectPassFrame passFrame,
        WorldVisibilityFrame visibility,
        Func<WorldVisibleMdxEntry, bool>? includeTransparent = null)
    {
        passFrame.TransparentVisibleMdxRoutes.Clear();
        for (int i = 0; i < visibility.VisibleMdx.Count; i++)
        {
            WorldVisibleMdxEntry visible = visibility.VisibleMdx[i];
            if (includeTransparent != null && !includeTransparent(visible))
                continue;

            bool requiresUnbatched = passFrame.UnbatchedVisibleMdxIndices.Contains(i);

            passFrame.TransparentVisibleMdxRoutes.Add(new WorldVisibleMdxPassRoute(i, requiresUnbatched));
        }

        passFrame.TransparentVisibleMdxRoutes.Sort((left, right) =>
            visibility.VisibleMdx[right.VisibleMdxIndex].CenterDistanceSq.CompareTo(visibility.VisibleMdx[left.VisibleMdxIndex].CenterDistanceSq));
    }

    public static (int batchedCount, int unbatchedCount) ExecutePlannedTransparentMdx(
        WorldObjectPassFrame passFrame,
        WorldVisibilityFrame visibility,
        Action<WorldVisibleMdxEntry> renderUnbatched,
        Action<WorldVisibleMdxEntry> renderBatched)
    {
        int batchedCount = 0;
        int unbatchedCount = 0;

        for (int i = 0; i < passFrame.TransparentVisibleMdxRoutes.Count; i++)
        {
            WorldVisibleMdxPassRoute route = passFrame.TransparentVisibleMdxRoutes[i];
            WorldVisibleMdxEntry visible = visibility.VisibleMdx[route.VisibleMdxIndex];
            if (route.RequiresUnbatchedRender)
            {
                renderUnbatched(visible);
                unbatchedCount++;
            }
            else
            {
                renderBatched(visible);
                batchedCount++;
            }
        }

        return (batchedCount, unbatchedCount);
    }
}
