using WowViewer.Core.Runtime.World.Sky;

namespace WowViewer.Core.Tests;

public sealed class SkyDomeVertexBuilderTests
{
    [Fact]
    public void Build_UsesTerrainZAsVerticalGradientAxis()
    {
        SkyDomeMeshData mesh = SkyDomeVertexBuilder.Build(segments: 4, rings: 2, radius: 10f);
        int horizon = 0;
        int zenith = (2 * (4 + 1)) * 4;

        Assert.Equal(10f, mesh.Vertices[horizon + 0], 5);
        Assert.Equal(0f, mesh.Vertices[horizon + 1], 5);
        Assert.Equal(0f, mesh.Vertices[horizon + 2], 5);
        Assert.Equal(0f, mesh.Vertices[horizon + 3], 5);

        Assert.Equal(0f, mesh.Vertices[zenith + 0], 4);
        Assert.Equal(0f, mesh.Vertices[zenith + 1], 4);
        Assert.Equal(10f, mesh.Vertices[zenith + 2], 5);
        Assert.Equal(1f, mesh.Vertices[zenith + 3], 5);
        Assert.Equal(4 * 2 * 6, mesh.Indices.Length);
    }
}
