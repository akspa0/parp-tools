using System.Numerics;
using WowViewer.Core.Runtime.World.SceneGraph;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests.World;

public sealed class WorldScenePortalAdapterTests
{
    [Fact]
    public void BuildsBidirectionalGroupAdjacencyAndPreservesPortalGeometry()
    {
        WorldScenePortalAdapterResult result = WorldScenePortalAdapter.Build(
            [
                new WorldSceneWmoPortalGroupReadModel(0),
                new WorldSceneWmoPortalGroupReadModel(1),
            ],
            [
                new WorldSceneWmoPortalReadModel(
                    3,
                    [
                        new Vector3(0f, -1f, -1f),
                        new Vector3(0f, 1f, -1f),
                        new Vector3(0f, 1f, 1f),
                        new Vector3(0f, -1f, 1f),
                    ],
                    Vector3.UnitX,
                    0f,
                    [
                        new WorldSceneWmoPortalReferenceReadModel(0, 3, 1, 1),
                        new WorldSceneWmoPortalReferenceReadModel(1, 3, 0, -1),
                    ]),
            ]);

        Assert.Equal(2, result.Graph.NodeCount);
        Assert.Equal(2, result.Graph.LinkCount);
        Assert.Equal(2, result.AcceptedLinkCount);
        Assert.Single(result.Geometries);
        Assert.Equal(3, result.Geometries[0].PortalIndex);
        Assert.Equal([0, 1], result.Geometries[0].GroupIndices);

        WorldScenePortalTraversalResult traversal = result.Graph.Traverse("wmo/group/0000", 2);
        Assert.Equal(["wmo/group/0000", "wmo/group/0001"], traversal.VisibleNodeIds);
        Assert.False(traversal.Diagnostics.FallbackRequired);
    }

    [Fact]
    public void UnknownGroupReferenceIsRetainedAsMalformedGraphData()
    {
        WorldScenePortalAdapterResult result = WorldScenePortalAdapter.Build(
            [new WorldSceneWmoPortalGroupReadModel(0)],
            [Portal(
                1,
                [
                    new WorldSceneWmoPortalReferenceReadModel(0, 1, 0, 1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 1, 9, -1),
                ])]);

        Assert.Single(result.Geometries);
        Assert.Equal(0, result.AcceptedLinkCount);
        Assert.Equal(2, result.RejectedLinks.Count);
        Assert.True(result.Graph.MalformedPortalData);
        Assert.True(result.Graph.Traverse("wmo/group/0000", 1).Diagnostics.FallbackRequired);
    }

    [Fact]
    public void InvalidPortalGeometryProducesExplicitFallbackInsteadOfConnectivity()
    {
        WorldScenePortalAdapterResult result = WorldScenePortalAdapter.Build(
            [
                new WorldSceneWmoPortalGroupReadModel(0),
                new WorldSceneWmoPortalGroupReadModel(1),
            ],
            [Portal(
                7,
                [
                    new WorldSceneWmoPortalReferenceReadModel(0, 7, 0, 1),
                    new WorldSceneWmoPortalReferenceReadModel(1, 7, 1, -1),
                ],
                vertices: [new Vector3(float.NaN, 0f, 0f), Vector3.UnitY, Vector3.UnitZ])]);

        Assert.Empty(result.Geometries);
        Assert.Equal([7], result.RejectedPortalIndices);
        Assert.Empty(result.Graph.Traverse("wmo/group/0000", 1).TraversedLinks);
        Assert.True(result.Graph.MalformedPortalData);
    }

    [Fact]
    public void ExistingWmoRenderDocumentReadModelsAreMappedWithoutReadingFiles()
    {
        WorldScenePortalAdapterResult result = WorldScenePortalAdapter.Build(DocumentWithPortal(), "client/wmo");

        Assert.Equal(2, result.DeclaredGroupCount);
        Assert.Equal(1, result.DeclaredPortalCount);
        Assert.Equal(2, result.Graph.LinkCount);
        Assert.Single(result.Geometries);
        Assert.Equal(["client/wmo/group/0000", "client/wmo/group/0001"],
            result.Graph.Traverse("client/wmo/group/0000", 2).VisibleNodeIds);
    }

    private static WorldSceneWmoPortalReadModel Portal(
        int portalIndex,
        IReadOnlyList<WorldSceneWmoPortalReferenceReadModel> references,
        IReadOnlyList<Vector3>? vertices = null)
        => new(
            portalIndex,
            vertices ??
            [
                new Vector3(0f, -1f, -1f),
                new Vector3(0f, 1f, -1f),
                new Vector3(0f, 0f, 1f),
            ],
            Vector3.UnitX,
            0f,
            references);

    private static WmoRenderDocument DocumentWithPortal()
    {
        return new WmoRenderDocument(
            "fixture.wmo",
            17,
            new WmoSummary(
                "fixture.wmo", 17, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                false, 0, Vector3.Zero, Vector3.One),
            Array.Empty<WmoMaterialDetail>(),
            [EmbeddedGroup(0), EmbeddedGroup(1)],
            [
                new WmoPortalVertexDetail(0, new Vector3(0f, -1f, -1f)),
                new WmoPortalVertexDetail(1, new Vector3(0f, 1f, -1f)),
                new WmoPortalVertexDetail(2, new Vector3(0f, 1f, 1f)),
                new WmoPortalVertexDetail(3, new Vector3(0f, -1f, 1f)),
            ],
            [new WmoPortalDetail(
                3,
                0,
                4,
                [
                    new Vector3(0f, -1f, -1f),
                    new Vector3(0f, 1f, -1f),
                    new Vector3(0f, 1f, 1f),
                    new Vector3(0f, -1f, 1f),
                ],
                Vector3.UnitX,
                0f)],
            [
                new WmoPortalReferenceDetail(0, 3, 0, -1),
                new WmoPortalReferenceDetail(1, 3, 1, 1),
            ],
            Array.Empty<WmoDoodadSetDetail>(),
            Array.Empty<WmoDoodadPlacementDetail>());
    }

    private static WmoEmbeddedGroupMeshDetail EmbeddedGroup(int groupIndex)
    {
        WmoGroupSummary summary = new(
            "fixture.wmo", 17, 0, 0, 0, 0, Vector3.Zero, Vector3.One,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false);
        WmoGroupMeshDetail mesh = new(
            "fixture.wmo", 17, 0, null,
            Array.Empty<Vector3>(),
            Array.Empty<Vector3>(),
            Array.Empty<ushort>(),
            Array.Empty<Vector2>(),
            Array.Empty<IReadOnlyList<Vector2>>(),
            Array.Empty<uint>(),
            Array.Empty<IReadOnlyList<uint>>(),
            Array.Empty<WmoGroupFaceMaterialDetail>(),
            Array.Empty<WmoGroupBatchDetail>());

        return new WmoEmbeddedGroupMeshDetail(
            groupIndex,
            0,
            summary,
            mesh,
            null,
            Array.Empty<ushort>(),
            Array.Empty<ushort>());
    }
}
