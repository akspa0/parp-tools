using System.Numerics;
using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Tests.World;

public sealed class WorldSceneGraphTests
{
    [Fact]
    public void AttachPropagatesWorldTransformAndCountsNestedNodes()
    {
        WorldSceneNode root = Node("root", WorldSceneNodeKind.Map, Vector3.Zero, new Vector3(100f, 100f, 100f));
        WorldSceneGraph graph = new(root);
        WorldSceneNode tile = Node("tile", WorldSceneNodeKind.Tile, new Vector3(10f, 20f, 0f), new Vector3(40f, 40f, 40f));
        WorldSceneNode chunk = Node("chunk", WorldSceneNodeKind.Chunk, new Vector3(2f, 3f, 4f), new Vector3(5f, 5f, 5f), renderable: true);

        graph.Attach(root.Id, tile);
        graph.Attach(tile.Id, chunk);

        Assert.Equal(new Vector3(12f, 23f, 4f), chunk.WorldTransform.Translation);
        Assert.Equal(3, graph.Count);
        Assert.Equal(3, graph.CreateSnapshot().NodeCount);
        Assert.Equal(1, graph.CreateSnapshot().NodeKindCounts[WorldSceneNodeKind.Chunk]);
        Assert.True(root.CanRejectSubtree);
    }

    [Fact]
    public void DetachRemovesTheCompleteSubtreeFromTheGraphIndex()
    {
        WorldSceneNode root = Node("root", WorldSceneNodeKind.Map, Vector3.Zero, new Vector3(100f, 100f, 100f));
        WorldSceneGraph graph = new(root);
        WorldSceneNode tile = Node("tile", WorldSceneNodeKind.Tile, Vector3.Zero, new Vector3(40f, 40f, 40f));
        WorldSceneNode chunk = Node("chunk", WorldSceneNodeKind.Chunk, Vector3.Zero, new Vector3(10f, 10f, 10f));
        WorldSceneNode objectNode = Node("object", WorldSceneNodeKind.M2Placement, Vector3.Zero, new Vector3(1f, 1f, 1f), renderable: true);

        graph.Attach(root.Id, tile);
        graph.Attach(tile.Id, chunk);
        graph.Attach(chunk.Id, objectNode);

        Assert.True(graph.Detach(tile.Id, out WorldSceneNode? detached));
        Assert.Same(tile, detached);
        Assert.Equal(1, graph.Count);
        Assert.False(graph.TryGetNode("tile", out _));
        Assert.False(graph.TryGetNode("chunk", out _));
        Assert.False(graph.TryGetNode("object", out _));
        Assert.Empty(root.Children);
    }

    [Fact]
    public void AttachmentRejectsDuplicateIdsAndSecondParents()
    {
        WorldSceneNode root = Node("root", WorldSceneNodeKind.Map, Vector3.Zero, new Vector3(100f, 100f, 100f));
        WorldSceneGraph graph = new(root);
        WorldSceneNode firstParent = Node("first", WorldSceneNodeKind.Tile, Vector3.Zero, new Vector3(40f, 40f, 40f));
        WorldSceneNode secondParent = Node("second", WorldSceneNodeKind.Tile, new Vector3(50f, 0f, 0f), new Vector3(40f, 40f, 40f));
        WorldSceneNode child = Node("child", WorldSceneNodeKind.Chunk, Vector3.Zero, new Vector3(5f, 5f, 5f));

        graph.Attach(root.Id, firstParent);
        graph.Attach(root.Id, secondParent);
        graph.Attach(firstParent.Id, child);

        Assert.Throws<InvalidOperationException>(() => graph.Attach(secondParent.Id, child));
        Assert.Throws<InvalidOperationException>(() => graph.Attach(root.Id, Node("child", WorldSceneNodeKind.Chunk, Vector3.Zero, new Vector3(5f, 5f, 5f))));
    }

    [Fact]
    public void UnknownChildBoundsMakeAncestorNonRejectable()
    {
        WorldSceneNode root = Node("root", WorldSceneNodeKind.Map, Vector3.Zero, new Vector3(100f, 100f, 100f));
        WorldSceneGraph graph = new(root);
        WorldSceneNode unknown = new(
            "unknown",
            WorldSceneNodeKind.M2Placement,
            Matrix4x4.Identity,
            Vector3.Zero,
            Vector3.Zero,
            boundsKnown: false,
            isRenderable: true,
            isQueryable: true);

        graph.Attach(root.Id, unknown);

        Assert.False(unknown.CanRejectSubtree);
        Assert.False(root.CanRejectSubtree);
        graph.ValidateInvariants();
    }

    private static WorldSceneNode Node(
        string id,
        WorldSceneNodeKind kind,
        Vector3 translation,
        Vector3 boundsMax,
        bool renderable = false)
    {
        return new WorldSceneNode(
            id,
            kind,
            Matrix4x4.CreateTranslation(translation),
            Vector3.Zero,
            boundsMax,
            isRenderable: renderable,
            isQueryable: renderable,
            renderPassMask: renderable ? WorldSceneRenderPass.Opaque : WorldSceneRenderPass.None);
    }
}
