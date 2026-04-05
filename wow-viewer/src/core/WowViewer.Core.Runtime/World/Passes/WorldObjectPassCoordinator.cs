using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Runtime.World.Passes;

public static class WorldObjectPassCoordinator
{
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
        Func<WorldVisibleMdxEntry, bool> requiresUnbatchedRender)
    {
        passFrame.OpaqueVisibleMdxRoutes.Clear();
        passFrame.UnbatchedVisibleMdxIndices.Clear();
        passFrame.FirstOpaqueBatchedVisibleMdxIndex = -1;

        for (int i = 0; i < visibility.VisibleMdx.Count; i++)
        {
            WorldVisibleMdxEntry visible = visibility.VisibleMdx[i];
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