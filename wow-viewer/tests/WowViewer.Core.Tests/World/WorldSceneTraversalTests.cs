using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Tests.World;

public sealed class WorldSceneTraversalTests
{
    [Fact]
    public void RejectedRegionSkipsEveryDescendantAndRecordsAttribution()
    {
        SyntheticWorldWorkloadBuildResult fixture = SyntheticWorldWorkloadBuilder.Build(new()
        {
            ResidentRegionCount = 2,
            ChunksPerRegion = 4,
            WmoPlacements = 1,
            WmoGroupsPerPlacement = 2,
            M2Placements = 4,
            RepeatedAssetCount = 2,
            Pm4OverlayCount = 1
        });
        HashSet<string> testedIds = new(StringComparer.Ordinal);

        WorldSceneTraversalResult result = WorldSceneTraversal.Traverse(
            fixture.Graph,
            node =>
            {
                testedIds.Add(node.Id);
                return !node.Id.Equals("synthetic/region/0001", StringComparison.Ordinal);
            });

        Assert.Contains(result.RejectedNodes, node => node.Id == "synthetic/region/0001");
        Assert.DoesNotContain(result.VisibleNodes, node => node.Id.StartsWith("synthetic/region/0001/", StringComparison.Ordinal));
        Assert.DoesNotContain(testedIds, id => id.StartsWith("synthetic/region/0001/", StringComparison.Ordinal));
        Assert.True(result.Diagnostics.RejectedNodeCount >= 1);
        Assert.True(result.Diagnostics.SkippedDescendantCount > 0);
        Assert.True(result.Diagnostics.IndividuallyTestedNodeCount < fixture.Graph.Count);
    }

    [Fact]
    public void DefaultTraversalReturnsOnlyRenderableNodes()
    {
        SyntheticWorldWorkloadBuildResult fixture = SyntheticWorldWorkloadBuilder.Build(new()
        {
            ResidentRegionCount = 1,
            ChunksPerRegion = 4,
            WmoPlacements = 1,
            WmoGroupsPerPlacement = 2,
            M2Placements = 4,
            RepeatedAssetCount = 2,
            Pm4OverlayCount = 1
        });

        WorldSceneTraversalResult result = WorldSceneTraversal.Traverse(fixture.Graph, static _ => true);

        Assert.NotEmpty(result.VisibleNodes);
        Assert.All(result.VisibleNodes, node => Assert.True(node.IsRenderable));
        Assert.Equal(result.VisibleNodes.Count, result.Diagnostics.VisibleRenderableNodeCount);
        Assert.Empty(result.RejectedNodes);
    }

    [Fact]
    public void NonRejectableNodesAreIncludedWithoutCallingVisibilityPredicate()
    {
        SyntheticWorldWorkloadBuildResult fixture = SyntheticWorldWorkloadBuilder.Build(new()
        {
            ResidentRegionCount = 1,
            ChunksPerRegion = 1,
            WmoPlacements = 0,
            M2Placements = 0,
            Pm4OverlayCount = 0
        });
        WorldSceneNode unknown = new(
            "synthetic/region/0000/unknown",
            WorldSceneNodeKind.M2Placement,
            System.Numerics.Matrix4x4.Identity,
            System.Numerics.Vector3.Zero,
            System.Numerics.Vector3.Zero,
            boundsKnown: false,
            isRenderable: true,
            isQueryable: true);
        fixture.Graph.Attach("synthetic/region/0000/chunk/0000", unknown);

        HashSet<string> testedIds = new(StringComparer.Ordinal);
        WorldSceneTraversalResult result = WorldSceneTraversal.Traverse(
            fixture.Graph,
            node =>
            {
                testedIds.Add(node.Id);
                return false;
            });

        Assert.Contains(result.VisibleNodes, node => node.Id == unknown.Id);
        Assert.DoesNotContain(unknown.Id, testedIds);
        Assert.True(result.Diagnostics.NonRejectableNodeCount >= 1);
    }
}
