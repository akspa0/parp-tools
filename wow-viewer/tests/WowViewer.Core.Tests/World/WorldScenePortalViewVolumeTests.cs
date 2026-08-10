using System.Numerics;
using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Tests.World;

public sealed class WorldScenePortalViewVolumeTests
{
    [Fact]
    public void ChildVolumeKeepsParentPlanesAndRestrictsPointsToDoorwayCone()
    {
        WorldScenePortalViewVolume root = WorldScenePortalViewVolume.CreateRoot(
            [new WorldScenePortalPlane(Vector3.UnitX, 0f)]);
        WorldScenePortalGeometry portal = Portal();

        WorldScenePortalViewVolumeBuildResult result = WorldScenePortalViewVolumeBuilder.BuildChild(
            root,
            portal,
            "wmo/group/0000",
            "wmo/group/0001",
            Vector3.Zero,
            maximumDepth: 4);

        Assert.False(result.FallbackRequired);
        Assert.NotNull(result.Volume);
        Assert.Equal(6, result.Volume!.Planes.Count);
        Assert.True(result.Volume.ContainsPoint(new Vector3(10f, 0f, 0f)));
        Assert.False(result.Volume.ContainsPoint(new Vector3(10f, 4f, 0f)));
        Assert.True(result.Volume.IntersectsBounds(new Vector3(9f, -0.5f, -0.5f), new Vector3(11f, 0.5f, 0.5f)));
        Assert.False(result.Volume.IntersectsBounds(new Vector3(9f, 4f, -0.5f), new Vector3(11f, 5f, 0.5f)));
    }

    [Fact]
    public void DepthLimitAndUnknownSideFailOpenToParentVolume()
    {
        WorldScenePortalViewVolume root = WorldScenePortalViewVolume.CreateRoot();
        WorldScenePortalGeometry portal = Portal(destinationSide: 0);

        WorldScenePortalViewVolumeBuildResult depthResult = WorldScenePortalViewVolumeBuilder.BuildChild(
            root,
            portal,
            "wmo/group/0000",
            "wmo/group/0001",
            Vector3.Zero,
            maximumDepth: 0);
        WorldScenePortalViewVolumeBuildResult sideResult = WorldScenePortalViewVolumeBuilder.BuildChild(
            root,
            portal,
            "wmo/group/0000",
            "wmo/group/0001",
            Vector3.Zero,
            maximumDepth: 4);

        Assert.True(depthResult.FallbackRequired);
        Assert.Equal("maximum_depth_reached", depthResult.FallbackReason);
        Assert.Same(root, depthResult.Volume);
        Assert.True(sideResult.FallbackRequired);
        Assert.Equal("portal_side_unknown", sideResult.FallbackReason);
    }

    [Fact]
    public void CameraOnPortalPlaneAndDegenerateEdgesRequireFallback()
    {
        WorldScenePortalViewVolume root = WorldScenePortalViewVolume.CreateRoot();
        WorldScenePortalViewVolumeBuildResult onPlane = WorldScenePortalViewVolumeBuilder.BuildChild(
            root,
            Portal(),
            "wmo/group/0000",
            "wmo/group/0001",
            new Vector3(5f, 0f, 0f),
            maximumDepth: 4);
        WorldScenePortalViewVolumeBuildResult degenerate = WorldScenePortalViewVolumeBuilder.BuildChild(
            root,
            Portal(vertices: [new Vector3(5f, 0f, 0f), new Vector3(5f, 0f, 0f), new Vector3(5f, 0f, 0f)]),
            "wmo/group/0000",
            "wmo/group/0001",
            Vector3.Zero,
            maximumDepth: 4);

        Assert.Equal("camera_on_portal_plane", onPlane.FallbackReason);
        Assert.Equal("portal_edge_degenerate", degenerate.FallbackReason);
    }

    private static WorldScenePortalGeometry Portal(
        short destinationSide = 1,
        IReadOnlyList<Vector3>? vertices = null)
        => new(
            3,
            vertices ??
            [
                new Vector3(5f, -1f, -1f),
                new Vector3(5f, 1f, -1f),
                new Vector3(5f, 1f, 1f),
                new Vector3(5f, -1f, 1f),
            ],
            Vector3.UnitX,
            -5f,
            [0, 1],
            [
                new WorldScenePortalGroupSide(0, -1),
                new WorldScenePortalGroupSide(1, destinationSide),
            ]);
}
