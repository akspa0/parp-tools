using System.Numerics;
using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Tests.World;

public sealed class WorldScenePortalVisibilityEvaluatorTests
{
    [Fact]
    public void EvaluatesReachableGroupsThroughNestedPortalVolumes()
    {
        WorldScenePortalAdapterResult adapter = WorldScenePortalAdapter.Build(
            [new WorldSceneWmoPortalGroupReadModel(0), new WorldSceneWmoPortalGroupReadModel(1)],
            [new WorldSceneWmoPortalReadModel(
                3,
                [
                    new Vector3(5f, -1f, -1f),
                    new Vector3(5f, 1f, -1f),
                    new Vector3(5f, 1f, 1f),
                    new Vector3(5f, -1f, 1f),
                ],
                Vector3.UnitX,
                -5f,
                [
                    new WorldSceneWmoPortalReferenceReadModel(0, 3, 0, -1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 3, 1, 1),
                ])],
            "wmo");
        WorldSceneNode placement = PlacementNode();

        WorldScenePortalVisibilityResult result = WorldScenePortalVisibilityEvaluator.Evaluate(
            adapter,
            placement,
            Vector3.Zero,
            maximumDepth: 4);

        Assert.False(result.Diagnostics.FallbackRequired);
        Assert.Equal(0, result.Diagnostics.SourceGroupIndex);
        Assert.Equal(["wmo/group/0000", "wmo/group/0001"], result.VisibleNodeIds);
        Assert.Equal(1, result.Diagnostics.TestedPortalCount);
    }

    [Fact]
    public void CameraOutsideAllGroupsFailsOpenToAllGraphNodes()
    {
        WorldScenePortalAdapterResult adapter = WorldScenePortalAdapter.Build(
            [new WorldSceneWmoPortalGroupReadModel(0), new WorldSceneWmoPortalGroupReadModel(1)],
            [new WorldSceneWmoPortalReadModel(
                3,
                [
                    new Vector3(5f, -1f, -1f),
                    new Vector3(5f, 1f, -1f),
                    new Vector3(5f, 1f, 1f),
                ],
                Vector3.UnitX,
                -5f,
                [
                    new WorldSceneWmoPortalReferenceReadModel(0, 3, 0, -1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 3, 1, 1),
                ])],
            "wmo");

        WorldScenePortalVisibilityResult result = WorldScenePortalVisibilityEvaluator.Evaluate(
            adapter,
            PlacementNode(),
            new Vector3(0f, 100f, 0f));

        Assert.True(result.Diagnostics.FallbackRequired);
        Assert.Equal("camera_group_unknown", result.Diagnostics.FallbackReason);
        Assert.Equal(adapter.Graph.NodeIds, result.VisibleNodeIds);
    }

    [Fact]
    public void OffDoorwayDestinationIsNotReportedVisible()
    {
        WorldScenePortalAdapterResult adapter = WorldScenePortalAdapter.Build(
            [new WorldSceneWmoPortalGroupReadModel(0), new WorldSceneWmoPortalGroupReadModel(1)],
            [new WorldSceneWmoPortalReadModel(
                3,
                [
                    new Vector3(5f, -1f, -1f),
                    new Vector3(5f, 1f, -1f),
                    new Vector3(5f, 1f, 1f),
                    new Vector3(5f, -1f, 1f),
                ],
                Vector3.UnitX,
                -5f,
                [
                    new WorldSceneWmoPortalReferenceReadModel(0, 3, 0, -1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 3, 1, 1),
                ])],
            "wmo");
        WorldSceneNode placement = PlacementNode(groupOneMin: new Vector3(6f, 4f, -1f), groupOneMax: new Vector3(10f, 5f, 1f));

        WorldScenePortalVisibilityResult result = WorldScenePortalVisibilityEvaluator.Evaluate(
            adapter,
            placement,
            Vector3.Zero);

        Assert.False(result.Diagnostics.FallbackRequired);
        Assert.Equal(["wmo/group/0000"], result.VisibleNodeIds);
    }

    [Fact]
    public void SingularPlacementTransformFailsOpen()
    {
        WorldScenePortalAdapterResult adapter = ValidAdapter();
        WorldScenePortalVisibilityResult result = WorldScenePortalVisibilityEvaluator.Evaluate(
            adapter,
            PlacementNode(transform: Matrix4x4.CreateScale(0f)),
            Vector3.Zero);

        Assert.True(result.Diagnostics.FallbackRequired);
        Assert.Equal("placement_transform_invalid", result.Diagnostics.FallbackReason);
        Assert.Equal(adapter.Graph.NodeIds, result.VisibleNodeIds);
    }

    [Fact]
    public void MissingPortalGeometryFailsOpen()
    {
        WorldScenePortalAdapterResult adapter = WorldScenePortalAdapter.Build(
            [new WorldSceneWmoPortalGroupReadModel(0), new WorldSceneWmoPortalGroupReadModel(1)],
            [new WorldSceneWmoPortalReadModel(
                3,
                Vertices: null,
                Normal: Vector3.UnitX,
                PlaneDistance: -5f,
                References: [
                    new WorldSceneWmoPortalReferenceReadModel(0, 3, 0, -1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 3, 1, 1),
                ])],
            "wmo");

        WorldScenePortalVisibilityResult result = WorldScenePortalVisibilityEvaluator.Evaluate(
            adapter,
            PlacementNode(),
            Vector3.Zero);

        Assert.True(result.Diagnostics.FallbackRequired);
        Assert.Equal("malformed_portal_edge", result.Diagnostics.FallbackReason);
        Assert.Equal(adapter.Graph.NodeIds, result.VisibleNodeIds);
    }

    [Fact]
    public void MalformedPortalEdgeFailsOpen()
    {
        WorldScenePortalAdapterResult adapter = WorldScenePortalAdapter.Build(
            [new WorldSceneWmoPortalGroupReadModel(0), new WorldSceneWmoPortalGroupReadModel(1)],
            [new WorldSceneWmoPortalReadModel(
                3,
                [
                    new Vector3(5f, -1f, -1f),
                    new Vector3(5f, 1f, -1f),
                    new Vector3(5f, 1f, 1f),
                ],
                Vector3.UnitX,
                -5f,
                [
                    new WorldSceneWmoPortalReferenceReadModel(0, 3, 0, -1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 3, 2, 1),
                ])],
            "wmo");

        WorldScenePortalVisibilityResult result = WorldScenePortalVisibilityEvaluator.Evaluate(
            adapter,
            PlacementNode(),
            Vector3.Zero);

        Assert.True(result.Diagnostics.FallbackRequired);
        Assert.Equal("malformed_portal_edge", result.Diagnostics.FallbackReason);
        Assert.Equal(adapter.Graph.NodeIds, result.VisibleNodeIds);
    }

    private static WorldScenePortalAdapterResult ValidAdapter()
        => WorldScenePortalAdapter.Build(
            [new WorldSceneWmoPortalGroupReadModel(0), new WorldSceneWmoPortalGroupReadModel(1)],
            [new WorldSceneWmoPortalReadModel(
                3,
                [
                    new Vector3(5f, -1f, -1f),
                    new Vector3(5f, 1f, -1f),
                    new Vector3(5f, 1f, 1f),
                    new Vector3(5f, -1f, 1f),
                ],
                Vector3.UnitX,
                -5f,
                [
                    new WorldSceneWmoPortalReferenceReadModel(0, 3, 0, -1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 3, 1, 1),
                ])],
            "wmo");

    private static WorldSceneNode PlacementNode(
        Matrix4x4? transform = null,
        Vector3? groupOneMin = null,
        Vector3? groupOneMax = null)
    {
        WorldSceneNode placement = new(
            "wmo",
            WorldSceneNodeKind.WmoPlacement,
            transform ?? Matrix4x4.Identity,
            new Vector3(-1f, -2f, -2f),
            new Vector3(11f, 5f, 2f),
            boundsKnown: true);
        WorldSceneGraph graph = new(placement);
        graph.Attach("wmo", new WorldSceneNode(
            "wmo/group/0000",
            WorldSceneNodeKind.WmoGroup,
            Matrix4x4.Identity,
            new Vector3(-1f, -1f, -1f),
            new Vector3(4f, 1f, 1f),
            boundsKnown: true,
            portalGroup: 0));
        graph.Attach("wmo", new WorldSceneNode(
            "wmo/group/0001",
            WorldSceneNodeKind.WmoGroup,
            Matrix4x4.Identity,
            groupOneMin ?? new Vector3(6f, -1f, -1f),
            groupOneMax ?? new Vector3(10f, 1f, 1f),
            boundsKnown: true,
            portalGroup: 1));
        return placement;
    }
}
