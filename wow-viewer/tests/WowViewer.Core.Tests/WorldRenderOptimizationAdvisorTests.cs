using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

public sealed class WorldRenderOptimizationAdvisorTests
{
    [Fact]
    public void BuildHint_EmptyStats_ReturnsPendingCaptureMessage()
    {
        string hint = WorldRenderOptimizationAdvisor.BuildHint(WorldRenderFrameStats.Empty);

        Assert.Equal("Next win pending: render stats have not captured a world frame yet.", hint);
    }

    [Fact]
    public void BuildHint_MdxDominates_ReturnsMdxGuidance()
    {
        WorldRenderFrameStats stats = CreateStats(
            mdxAnimationMs: 1.5,
            mdxVisibilityMs: 1.0,
            mdxOpaqueSubmissionMs: 1.2,
            mdxTransparentSortMs: 0.8,
            mdxTransparentSubmissionMs: 0.9,
            wmoVisibilityMs: 0.4,
            wmoSubmissionMs: 0.5,
            terrainMs: 0.6,
            wdlMs: 0.1,
            liquidMs: 0.2,
            overlayMs: 0.3,
            deferredAssetLoadMs: 0.4);

        string hint = WorldRenderOptimizationAdvisor.BuildHint(stats);

        Assert.Equal("Next win: MDX batching/state reduction is costing more than the other measured world layers.", hint);
    }

    [Fact]
    public void BuildHint_TerrainDominates_ReturnsTerrainGuidance()
    {
        WorldRenderFrameStats stats = CreateStats(
            mdxAnimationMs: 0.5,
            mdxVisibilityMs: 0.4,
            mdxOpaqueSubmissionMs: 0.3,
            mdxTransparentSortMs: 0.2,
            mdxTransparentSubmissionMs: 0.1,
            wmoVisibilityMs: 0.6,
            wmoSubmissionMs: 0.5,
            terrainMs: 2.1,
            wdlMs: 1.2,
            liquidMs: 0.9,
            overlayMs: 0.4,
            deferredAssetLoadMs: 0.2);

        string hint = WorldRenderOptimizationAdvisor.BuildHint(stats);

        Assert.Equal("Next win: terrain/WDL/liquid cost is dominating the captured frame.", hint);
    }

    private static WorldRenderFrameStats CreateStats(
        double mdxAnimationMs,
        double mdxVisibilityMs,
        double mdxOpaqueSubmissionMs,
        double mdxTransparentSortMs,
        double mdxTransparentSubmissionMs,
        double wmoVisibilityMs,
        double wmoSubmissionMs,
        double terrainMs,
        double wdlMs,
        double liquidMs,
        double overlayMs,
        double deferredAssetLoadMs)
    {
        return new WorldRenderFrameStats(
            TotalCpuMs: mdxAnimationMs + mdxVisibilityMs + mdxOpaqueSubmissionMs + mdxTransparentSortMs + mdxTransparentSubmissionMs + wmoVisibilityMs + wmoSubmissionMs + terrainMs + wdlMs + liquidMs + overlayMs + deferredAssetLoadMs,
            PendingAssetLoadCount: 0,
            TerrainChunksRendered: 0,
            WdlVisibleTileCount: 0,
            VisibleWmoCount: 0,
            VisibleMdxCount: 0,
            VisibleTaxiMdxCount: 0,
            OpaqueBatchedMdxCount: 0,
            OpaqueUnbatchedMdxCount: 0,
            TransparentBatchedMdxCount: 0,
            TransparentUnbatchedMdxCount: 0,
            DeferredAssetLoads: new WorldRenderStageStats(deferredAssetLoadMs),
            TaxiActorUpdate: new WorldRenderStageStats(0),
            Lighting: new WorldRenderStageStats(0),
            Sky: new WorldRenderStageStats(0),
            SkyboxBackdrop: new WorldRenderStageStats(0),
            Wdl: new WorldRenderStageStats(wdlMs),
            Terrain: new WorldRenderStageStats(terrainMs),
            WmoVisibility: new WorldRenderStageStats(wmoVisibilityMs),
            WmoSubmission: new WorldRenderStageStats(wmoSubmissionMs),
            MdxAnimation: new WorldRenderStageStats(mdxAnimationMs),
            MdxVisibility: new WorldRenderStageStats(mdxVisibilityMs),
            MdxOpaqueSubmission: new WorldRenderStageStats(mdxOpaqueSubmissionMs),
            Liquid: new WorldRenderStageStats(liquidMs),
            MdxTransparentSort: new WorldRenderStageStats(mdxTransparentSortMs),
            MdxTransparentSubmission: new WorldRenderStageStats(mdxTransparentSubmissionMs),
            Overlay: new WorldRenderStageStats(overlayMs));
    }
}