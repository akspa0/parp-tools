namespace WowViewer.Core.Runtime.World;

public static class WorldRenderOptimizationAdvisor
{
    public static string BuildHint(WorldRenderFrameStats stats)
    {
        double mdxCost = stats.MdxAnimation.DurationMs
            + stats.MdxVisibility.DurationMs
            + stats.MdxOpaqueSubmission.DurationMs
            + stats.MdxTransparentSort.DurationMs
            + stats.MdxTransparentSubmission.DurationMs;
        double wmoCost = stats.WmoVisibility.DurationMs + stats.WmoSubmission.DurationMs;
        double terrainCost = stats.Terrain.DurationMs + stats.Wdl.DurationMs + stats.Liquid.DurationMs;
        double overlayCost = stats.Overlay.DurationMs;
        double assetCost = stats.DeferredAssetLoads.DurationMs;

        if (mdxCost <= 0.0 && wmoCost <= 0.0 && terrainCost <= 0.0 && overlayCost <= 0.0 && assetCost <= 0.0)
            return "Next win pending: render stats have not captured a world frame yet.";

        if (mdxCost >= wmoCost + 0.75 && mdxCost >= terrainCost && mdxCost >= overlayCost && mdxCost >= assetCost)
            return "Next win: MDX batching/state reduction is costing more than the other measured world layers.";

        if (wmoCost >= mdxCost + 0.75 && wmoCost >= terrainCost && wmoCost >= overlayCost && wmoCost >= assetCost)
            return "Next win: WMO scene-pass extraction is the larger measured object-side cost.";

        if (terrainCost >= mdxCost && terrainCost >= wmoCost && terrainCost >= overlayCost && terrainCost >= assetCost)
            return "Next win: terrain/WDL/liquid cost is dominating the captured frame.";

        if (overlayCost >= mdxCost && overlayCost >= wmoCost && overlayCost >= terrainCost && overlayCost >= assetCost)
            return "Next win: overlay/debug work should be pushed later and made cheaper when disabled.";

        if (assetCost >= mdxCost && assetCost >= wmoCost && assetCost >= terrainCost && assetCost >= overlayCost)
            return "Next win: deferred asset-load drain is spiking the frame more than draw submission.";

        return "Next win: costs are mixed; compare MDX and WMO layers on the live map before the next slice.";
    }
}