using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Tests.World;

public sealed class SyntheticWorldWorkloadTests
{
    [Fact]
    public void SameDefinitionAndSeedProduceTheSameManifestAndSnapshot()
    {
        SyntheticWorldWorkloadDefinition definition = new()
        {
            FixtureName = "deterministic-test",
            Seed = 77,
            ResidentRegionCount = 2,
            ChunksPerRegion = 9,
            WmoPlacements = 2,
            WmoGroupsPerPlacement = 3,
            M2Placements = 12,
            RepeatedAssetCount = 3,
            Pm4OverlayCount = 2,
            PortalLinkCount = 4
        };

        SyntheticWorldWorkloadBuildResult first = SyntheticWorldWorkloadBuilder.Build(definition);
        SyntheticWorldWorkloadBuildResult second = SyntheticWorldWorkloadBuilder.Build(definition);

        Assert.Equal(first.Manifest.ManifestSha256, second.Manifest.ManifestSha256);
        Assert.Equal(first.Manifest.ToJson(false), second.Manifest.ToJson(false));
        Assert.Equal(first.Graph.CreateSnapshot().NodeIds, second.Graph.CreateSnapshot().NodeIds);
        Assert.Equal(first.Graph.CreateSnapshot().NodeCount, second.Graph.CreateSnapshot().NodeCount);
    }

    [Fact]
    public void ManifestRoundTripsWithHashAndNestedContent()
    {
        SyntheticWorldWorkloadBuildResult result = SyntheticWorldWorkloadBuilder.Build(new()
        {
            FixtureName = "round-trip-test",
            ResidentRegionCount = 2,
            ChunksPerRegion = 4,
            WmoPlacements = 1,
            WmoGroupsPerPlacement = 4,
            M2Placements = 8,
            RepeatedAssetCount = 2,
            Pm4OverlayCount = 1,
            PortalLinkCount = 3
        });

        SyntheticWorldWorkload restored = SyntheticWorldWorkload.FromJson(result.Manifest.ToJson(false));

        Assert.Equal(result.Manifest.ManifestSha256, restored.ManifestSha256);
        Assert.Equal(result.Manifest.Nodes.Count, restored.Nodes.Count);
        Assert.Contains(restored.Nodes, node => node.Kind == WorldSceneNodeKind.WmoGroup);
        Assert.Contains(restored.Nodes, node => node.Kind == WorldSceneNodeKind.M2Placement);
        Assert.Contains(restored.Nodes, node => node.Kind == WorldSceneNodeKind.Pm4Structure);
        Assert.NotEmpty(restored.PortalLinks);
    }

    [Fact]
    public void WorkloadSnapshotReportsNestedKindsPassesAndUpdates()
    {
        SyntheticWorldWorkloadBuildResult result = SyntheticWorldWorkloadBuilder.Build(new()
        {
            ResidentRegionCount = 1,
            ChunksPerRegion = 4,
            WmoPlacements = 1,
            WmoGroupsPerPlacement = 4,
            M2Placements = 6,
            RepeatedAssetCount = 2,
            Pm4OverlayCount = 1
        });

        WorldSceneGraphSnapshot snapshot = result.Graph.CreateSnapshot();

        Assert.True(snapshot.NodeKindCounts[WorldSceneNodeKind.Map] == 1);
        Assert.True(snapshot.NodeKindCounts[WorldSceneNodeKind.Tile] == 1);
        Assert.Equal(4, snapshot.NodeKindCounts[WorldSceneNodeKind.Chunk]);
        Assert.Equal(4, snapshot.NodeKindCounts[WorldSceneNodeKind.WmoGroup]);
        Assert.Equal(6, snapshot.NodeKindCounts[WorldSceneNodeKind.M2Placement]);
        Assert.Equal(1, snapshot.NodeKindCounts[WorldSceneNodeKind.Pm4Structure]);
        Assert.True(snapshot.RenderableCount > 0);
        Assert.True(snapshot.UpdateRequiredCount > 0);
        Assert.True(snapshot.MaxDepth >= 3);
        Assert.Equal(snapshot.NodeCount, result.Manifest.Nodes.Count);
    }

    [Fact]
    public void ManifestRejectsImageOnlyWorkloadClassAndTamperedHash()
    {
        SyntheticWorldWorkloadBuildResult result = SyntheticWorldWorkloadBuilder.Build(new());
        string json = result.Manifest.ToJson(false);

        string imageOnlyJson = json.Replace(
            "\"workload_class\":\"synthetic_world_scene\"",
            "\"workload_class\":\"synthetic_minimap_asset\"",
            StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => SyntheticWorldWorkload.FromJson(imageOnlyJson));

        string tamperedJson = json.Replace("synthetic-world-v1", "tampered-fixture", StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => SyntheticWorldWorkload.FromJson(tamperedJson));
    }
}
