using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Tests.World;

public sealed class WorldScenePortalGraphTests
{
    [Fact]
    public void TraverseStopsAtDepthLimitAndReportsFallback()
    {
        WorldScenePortalGraphBuildResult build = WorldScenePortalGraph.Build(
            ["group/a", "group/b", "group/c"],
            [
                new WorldScenePortalLink("portal/ab", "group/a", "group/b"),
                new WorldScenePortalLink("portal/bc", "group/b", "group/c"),
            ]);

        WorldScenePortalTraversalResult result = build.Graph.Traverse("group/a", maximumDepth: 1);

        Assert.Equal(["group/a", "group/b"], result.VisibleNodeIds);
        Assert.Single(result.TraversedLinks);
        Assert.True(result.Diagnostics.FallbackRequired);
        Assert.Contains("maximum_depth_reached", result.Diagnostics.FallbackReason);
        Assert.Equal(1, result.Diagnostics.DepthLimitHitCount);
        Assert.Equal(1, result.Diagnostics.MaxDepthReached);
    }

    [Fact]
    public void CyclesAreVisitedOnceAndReportedWithoutInfiniteTraversal()
    {
        WorldScenePortalGraphBuildResult build = WorldScenePortalGraph.Build(
            ["group/a", "group/b"],
            [
                new WorldScenePortalLink("portal/ab", "group/a", "group/b"),
                new WorldScenePortalLink("portal/ba", "group/b", "group/a"),
            ]);

        WorldScenePortalTraversalResult result = build.Graph.Traverse("group/a", maximumDepth: 8);

        Assert.Equal(["group/a", "group/b"], result.VisibleNodeIds);
        Assert.Equal(1, result.Diagnostics.CycleCount);
        Assert.False(result.Diagnostics.FallbackRequired);
        Assert.Equal(2, result.Diagnostics.VisitedNodeCount);
    }

    [Fact]
    public void MalformedLinksAreRejectedAndRequireFallback()
    {
        WorldScenePortalGraphBuildResult build = WorldScenePortalGraph.Build(
            ["group/a", "group/b"],
            [
                new WorldScenePortalLink("portal/valid", "group/a", "group/b"),
                new WorldScenePortalLink("portal/missing-destination", "group/a", "group/missing"),
                new WorldScenePortalLink("portal/valid", "group/b", "group/a"),
            ]);

        WorldScenePortalTraversalResult result = build.Graph.Traverse("group/a", maximumDepth: 8);

        Assert.Equal(2, build.RejectedLinks.Count);
        Assert.Equal(1, build.AcceptedLinkCount);
        Assert.True(result.Diagnostics.FallbackRequired);
        Assert.Contains("malformed_portal_edge", result.Diagnostics.FallbackReason ?? "malformed_portal_edge");
    }

    [Fact]
    public void MissingPortalDataAndEntryAreExplicitlyFailOpen()
    {
        WorldScenePortalGraphBuildResult build = WorldScenePortalGraph.Build(["group/a"], []);

        WorldScenePortalTraversalResult missingEntry = build.Graph.Traverse("group/missing", maximumDepth: 4);
        WorldScenePortalTraversalResult presentEntry = build.Graph.Traverse("group/a", maximumDepth: 4);

        Assert.True(missingEntry.Diagnostics.FallbackRequired);
        Assert.Contains("portal_data_absent", missingEntry.Diagnostics.FallbackReason);
        Assert.Contains("entry_node_missing", missingEntry.Diagnostics.FallbackReason);
        Assert.Equal(["group/a"], presentEntry.VisibleNodeIds);
        Assert.True(presentEntry.Diagnostics.FallbackRequired);
        Assert.Equal("portal_data_absent", presentEntry.Diagnostics.FallbackReason);
    }
}
