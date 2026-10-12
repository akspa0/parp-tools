using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Runtime.World.Passes;

public static class WorldObjectPassCoordinator
{
    /// <param name="ModelKey">Placement's model key; placements batch by model.</param>
    /// <param name="CanBatch">Renderer supports the GPU-instanced opaque shell.</param>
    /// <param name="VisibleIndex">Index into the frame's visible WMO list.</param>
    /// <param name="ReachedBySceneLight">
    /// A scene light's attenuation reaches this placement's world bounds. Such placements keep the
    /// per-placement path so they get their own light set (Epic 249 R-10b / archived Spec 242 FR-002).
    /// </param>
    /// <param name="EmitsSceneLights">The placement's own model emits scene lights (diagnostics only).</param>
    public readonly record struct WorldWmoOpaqueBatchCandidate(
        string ModelKey,
        bool CanBatch,
        int VisibleIndex,
        bool ReachedBySceneLight = false,
        bool EmitsSceneLights = false);

    public readonly record struct WorldWmoOpaqueBatch(string ModelKey, IReadOnlyList<int> VisibleIndices);

    /// <param name="LitFallbackCount">Placements that could batch but a scene light reaches.</param>
    /// <param name="SelfLitFallbackCount">Of those, placements whose own model emits lights.</param>
    public readonly record struct WorldWmoOpaqueBatchPlan(
        IReadOnlyList<WorldWmoOpaqueBatch> Batches,
        IReadOnlyList<int> FallbackVisibleIndices,
        int LitFallbackCount = 0,
        int SelfLitFallbackCount = 0)
    {
        public int BatchedPlacementCount
        {
            get
            {
                int count = 0;
                for (int i = 0; i < Batches.Count; i++)
                    count += Batches[i].VisibleIndices.Count;
                return count;
            }
        }
    }

    /// <summary>
    /// Partitions visible WMO placements into per-model instanced batches and a per-placement fallback.
    /// Deterministic: identical candidates always give the identical partition and fallback order.
    /// </summary>
    public static WorldWmoOpaqueBatchPlan PlanOpaqueWmoBatches(
        IReadOnlyList<WorldWmoOpaqueBatchCandidate> candidates)
    {
        var batchIndicesByModel = new Dictionary<string, List<int>>(StringComparer.OrdinalIgnoreCase);
        var fallbackVisibleIndices = new List<int>();
        int litFallbackCount = 0;
        int selfLitFallbackCount = 0;

        for (int i = 0; i < candidates.Count; i++)
        {
            WorldWmoOpaqueBatchCandidate candidate = candidates[i];
            if (!candidate.CanBatch || string.IsNullOrWhiteSpace(candidate.ModelKey))
            {
                fallbackVisibleIndices.Add(candidate.VisibleIndex);
                continue;
            }

            if (candidate.ReachedBySceneLight)
            {
                fallbackVisibleIndices.Add(candidate.VisibleIndex);
                litFallbackCount++;
                if (candidate.EmitsSceneLights)
                    selfLitFallbackCount++;
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

        return new WorldWmoOpaqueBatchPlan(batches, fallbackVisibleIndices, litFallbackCount, selfLitFallbackCount);
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
