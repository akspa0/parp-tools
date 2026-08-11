using System.Numerics;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Tests.World;

public sealed class WorldSceneGraphObjectAdapterTests
{
    [Fact]
    public void BuildGroupsPlacementsByTileAndPreservesStablePlacementMetadata()
    {
        WorldSceneGraphBuildResult result = WorldSceneGraphObjectAdapter.Build(
        [
            Placement("wmo/002", WorldSceneNodeKind.WmoPlacement, 4, 8, new Vector3(30f, 0f, 0f)),
            Placement("m2/001", WorldSceneNodeKind.M2Placement, 4, 8, new Vector3(5f, 0f, 0f), requiresUpdate: true),
            Placement("m2/003", WorldSceneNodeKind.M2Placement, 9, 2, new Vector3(100f, 0f, 0f)),
            Placement("external/m2/001", WorldSceneNodeKind.M2Placement, null, null, new Vector3(-25f, 0f, 0f), external: true),
        ]);

        WorldSceneGraphSnapshot snapshot = result.Graph.CreateSnapshot();

        Assert.Equal(8, snapshot.NodeCount);
        Assert.Equal(3, snapshot.NodeKindCounts[WorldSceneNodeKind.M2Placement]);
        Assert.Equal(1, snapshot.NodeKindCounts[WorldSceneNodeKind.WmoPlacement]);
        Assert.Equal(2, snapshot.NodeKindCounts[WorldSceneNodeKind.Tile]);
        Assert.Equal(1, snapshot.NodeKindCounts[WorldSceneNodeKind.SyntheticProxy]);
        Assert.Equal(1, snapshot.UpdateRequiredCount);
        Assert.True(result.Graph.TryGetNode("world/tile/04/08", out WorldSceneNode? tile));
        Assert.NotNull(tile);
        Assert.Equal(2, tile!.Children.Count);
        Assert.Equal("m2/001", tile.Children[0].Id);
        Assert.Equal("wmo/002", tile.Children[1].Id);
        Assert.Equal("m2/001", result.PlacementsByNodeId["m2/001"].Id);
        Assert.Equal("m2/001", tile.Children[0].Id);
    }

    [Fact]
    public void UnknownPlacementBoundsKeepTileAndMapFailOpen()
    {
        WorldSceneGraphObjectPlacement unknown = new(
            "m2/unknown",
            WorldSceneNodeKind.M2Placement,
            new WorldObjectInstance
            {
                ModelKey = "world/unknown.m2",
                Transform = Matrix4x4.CreateTranslation(10f, 0f, 0f),
                HasTileCoordinate = true,
                TileX = 1,
                TileY = 2,
                BoundsResolved = false,
            });

        WorldSceneGraphBuildResult result = WorldSceneGraphObjectAdapter.Build([unknown]);

        Assert.False(result.Graph.Root.CanRejectSubtree);
        Assert.True(result.Graph.TryGetNode("world/tile/01/02", out WorldSceneNode? tile));
        Assert.NotNull(tile);
        Assert.False(tile!.CanRejectSubtree);
        Assert.False(result.Graph.TryGetNode("m2/unknown", out WorldSceneNode? node) && node!.CanRejectSubtree);
    }

    [Fact]
    public void RebuildingTheSamePlacementsProducesTheSameNodeInventory()
    {
        WorldSceneGraphObjectPlacement[] placements =
        [
            Placement("wmo/a", WorldSceneNodeKind.WmoPlacement, 2, 3, new Vector3(8f, 2f, 0f)),
            Placement("m2/a", WorldSceneNodeKind.M2Placement, 2, 3, new Vector3(4f, 3f, 0f)),
        ];

        WorldSceneGraphBuildResult first = WorldSceneGraphObjectAdapter.Build(placements);
        WorldSceneGraphBuildResult second = WorldSceneGraphObjectAdapter.Build(placements);

        Assert.Equal(first.Graph.CreateSnapshot().NodeIds, second.Graph.CreateSnapshot().NodeIds);
        Assert.Equal(first.Graph.CreateSnapshot().RootBoundsMin, second.Graph.CreateSnapshot().RootBoundsMin);
        Assert.Equal(first.Graph.CreateSnapshot().RootBoundsMax, second.Graph.CreateSnapshot().RootBoundsMax);
    }

    [Fact]
    public void PlacementChildrenBecomeNestedNodesWithoutDuplicatingPlacementState()
    {
        WorldSceneGraphObjectPlacement placement = Placement(
            "wmo/interior",
            WorldSceneNodeKind.WmoPlacement,
            1,
            1,
            Vector3.Zero) with
        {
            Children =
            [
                new WorldSceneGraphChildNode(
                    "wmo/interior/group/0000",
                    WorldSceneNodeKind.WmoGroup,
                    Matrix4x4.Identity,
                    Vector3.Zero,
                    new Vector3(1f, 1f, 1f),
                    PortalGroup: 0),
                new WorldSceneGraphChildNode(
                    "wmo/interior/group/0001",
                    WorldSceneNodeKind.WmoGroup,
                    Matrix4x4.CreateTranslation(0.5f, 0f, 0f),
                    new Vector3(-0.25f),
                    new Vector3(0.25f),
                    PortalGroup: 1),
            ]
        };

        WorldSceneGraphBuildResult result = WorldSceneGraphObjectAdapter.Build([placement]);

        Assert.True(result.Graph.TryGetNode("wmo/interior", out WorldSceneNode? placementNode));
        Assert.NotNull(placementNode);
        Assert.Equal(2, placementNode!.Children.Count);
        Assert.Equal(WorldSceneNodeKind.WmoGroup, placementNode.Children[0].Kind);
        Assert.Equal(1, placementNode.Children[1].PortalGroup);
        Assert.Equal(5, result.Graph.CreateSnapshot().NodeCount);
        Assert.Equal("world/wmo/interior.asset", placementNode.AssetKey);
    }

    [Fact]
    public void SpatialBucketsNestResidentDoodadsUnderTerrainChunks()
    {
        WorldSceneGraphBuildResult result = WorldSceneGraphObjectAdapter.Build(
        [
            Placement("m2/a", WorldSceneNodeKind.M2Placement, 3, 4, new Vector3(10f, 0f, 0f)) with
            {
                SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "02/03")
            },
            Placement("m2/b", WorldSceneNodeKind.M2Placement, 3, 4, new Vector3(20f, 0f, 0f)) with
            {
                SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "02/03")
            },
            Placement("m2/c", WorldSceneNodeKind.M2Placement, 3, 4, new Vector3(40f, 0f, 0f)) with
            {
                SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "02/04")
            },
        ]);

        Assert.True(result.Graph.TryGetNode("world/tile/03/04", out WorldSceneNode? tile));
        Assert.NotNull(tile);
        Assert.Equal(2, tile!.Children.Count);
        Assert.All(tile.Children, child => Assert.Equal(WorldSceneNodeKind.Chunk, child.Kind));
        Assert.Equal(
            "world/tile/03/04/chunk/02/03",
            tile.Children[0].Id);
        Assert.Equal(2, tile.Children[0].Children.Count);
        Assert.All(tile.Children[0].Children, child => Assert.Equal(WorldSceneNodeKind.M2Placement, child.Kind));
        Assert.Equal(3, result.Graph.CreateSnapshot().MaxDepth);
        Assert.Equal(7, result.Graph.CreateSnapshot().NodeCount);
    }

    [Fact]
    public void UnknownSpatialBucketBoundsKeepChunkAndAncestorsFailOpen()
    {
        WorldSceneGraphObjectPlacement placement = Placement(
            "m2/unknown",
            WorldSceneNodeKind.M2Placement,
            1,
            2,
            Vector3.Zero) with
        {
            Instance = Placement(
                "m2/unknown",
                WorldSceneNodeKind.M2Placement,
                1,
                2,
                Vector3.Zero).Instance with
            {
                BoundsResolved = false,
                BoundsMin = Vector3.Zero,
                BoundsMax = Vector3.Zero,
            },
            SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "01/01")
        };

        WorldSceneGraphBuildResult result = WorldSceneGraphObjectAdapter.Build([placement]);

        Assert.True(result.Graph.TryGetNode("world/tile/01/02/chunk/01/01", out WorldSceneNode? chunk));
        Assert.NotNull(chunk);
        Assert.False(chunk!.CanRejectSubtree);
        Assert.False(result.Graph.Root.CanRejectSubtree);
    }

    [Fact]
    public void RejectedTerrainChunkSkipsItsDoodadDescendants()
    {
        WorldSceneGraphBuildResult result = WorldSceneGraphObjectAdapter.Build(
        [
            Placement("m2/a", WorldSceneNodeKind.M2Placement, 3, 4, new Vector3(10f, 0f, 0f)) with
            {
                SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "02/03")
            },
            Placement("m2/b", WorldSceneNodeKind.M2Placement, 3, 4, new Vector3(20f, 0f, 0f)) with
            {
                SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "02/03")
            },
        ]);
        HashSet<string> testedIds = new(StringComparer.Ordinal);

        WorldSceneTraversalResult traversal = WorldSceneTraversal.Traverse(
            result.Graph,
            node =>
            {
                testedIds.Add(node.Id);
                return !node.Id.Equals("world/tile/03/04/chunk/02/03", StringComparison.Ordinal);
            });

        Assert.DoesNotContain("m2/a", testedIds);
        Assert.DoesNotContain("m2/b", testedIds);
        Assert.Contains(
            traversal.RejectedNodes,
            node => node.Id == "world/tile/03/04/chunk/02/03");
        Assert.Equal(2, traversal.Diagnostics.SkippedDescendantCount);
        Assert.Equal(1, traversal.Diagnostics.RejectedNodeCountsByKind[WorldSceneNodeKind.Chunk]);
        Assert.Equal(2, traversal.Diagnostics.SkippedDescendantCountsByKind[WorldSceneNodeKind.M2Placement]);
        Assert.Equal(1, traversal.Diagnostics.IndividuallyTestedNodeCountsByKind[WorldSceneNodeKind.Chunk]);
    }

    [Fact]
    public void DeferredM2VisibilityLeavesChunkCullingToTheGraph()
    {
        WorldSceneGraphBuildResult result = WorldSceneGraphObjectAdapter.Build(
        [
            Placement("m2/a", WorldSceneNodeKind.M2Placement, 3, 4, new Vector3(10f, 0f, 0f)) with
            {
                SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "02/03")
            },
            Placement("m2/b", WorldSceneNodeKind.M2Placement, 3, 4, new Vector3(20f, 0f, 0f)) with
            {
                SpatialBucket = new WorldSceneGraphSpatialBucket(WorldSceneNodeKind.Chunk, "02/03")
            },
        ]);
        HashSet<string> testedIds = new(StringComparer.Ordinal);

        WorldSceneTraversalResult traversal = WorldSceneTraversal.Traverse(
            result.Graph,
            node =>
            {
                testedIds.Add(node.Id);
                return true;
            },
            shouldEvaluateVisibility: static node => node.Kind != WorldSceneNodeKind.M2Placement);

        Assert.Contains("world/tile/03/04/chunk/02/03", testedIds);
        Assert.DoesNotContain("m2/a", testedIds);
        Assert.DoesNotContain("m2/b", testedIds);
        Assert.Equal(2, traversal.Diagnostics.DeferredVisibilityTestCount);
        Assert.Equal(2, traversal.Diagnostics.DeferredVisibilityTestCountsByKind[WorldSceneNodeKind.M2Placement]);
        Assert.Contains(traversal.VisibleNodes, node => node.Id == "m2/a");
        Assert.Contains(traversal.VisibleNodes, node => node.Id == "m2/b");
    }

    private static WorldSceneGraphObjectPlacement Placement(
        string id,
        WorldSceneNodeKind kind,
        int? tileX,
        int? tileY,
        Vector3 position,
        bool external = false,
        bool requiresUpdate = false)
    {
        return new WorldSceneGraphObjectPlacement(
            id,
            kind,
            new WorldObjectInstance
            {
                ModelKey = $"world/{id}.asset",
                Transform = Matrix4x4.CreateTranslation(position),
                LocalBoundsMin = new Vector3(-1f, -1f, -1f),
                LocalBoundsMax = new Vector3(1f, 1f, 1f),
                BoundsMin = position - Vector3.One,
                BoundsMax = position + Vector3.One,
                BoundsResolved = true,
                HasTileCoordinate = tileX.HasValue && tileY.HasValue,
                TileX = tileX ?? 0,
                TileY = tileY ?? 0,
            },
            external,
            WorldSceneRenderPass.Opaque,
            IsQueryable: true,
            RequiresUpdate: requiresUpdate);
    }
}
