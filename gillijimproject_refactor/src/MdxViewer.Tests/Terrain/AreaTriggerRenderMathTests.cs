using System.Linq;
using System.Numerics;
using MdxViewer.Rendering;
using MdxViewer.Terrain;
using Xunit;

namespace MdxViewer.Tests.Terrain;

public sealed class AreaTriggerRenderMathTests
{
    [Fact]
    public void ToScenePosition_UsesTerrainRendererConversion_ForTerrainMaps()
    {
        var position = AreaTriggerRenderMath.ToScenePosition(new Vector3(1200f, 3400f, 55f), useRawWorldCoordinates: false);

        Assert.Equal(WoWConstants.MapOrigin - 3400f, position.X);
        Assert.Equal(WoWConstants.MapOrigin - 1200f, position.Y);
        Assert.Equal(55f, position.Z);
    }

    [Fact]
    public void ToScenePosition_PreservesRawWorldCoordinates_ForWmoBasedMaps()
    {
        var wowPosition = new Vector3(1200f, 3400f, 55f);

        var position = AreaTriggerRenderMath.ToScenePosition(wowPosition, useRawWorldCoordinates: true);

        Assert.Equal(wowPosition, position);
    }

    [Fact]
    public void BuildBoxCorners_CentersHeightAndAppliesYaw()
    {
        var corners = AreaTriggerRenderMath.BuildBoxCorners(Vector3.Zero, length: 4f, width: 2f, height: 6f, yawRadians: MathF.PI * 0.5f);

        Assert.Equal(-3f, corners.Min(corner => corner.Z));
        Assert.Equal(3f, corners.Max(corner => corner.Z));
        Assert.Contains(corners, corner => Vector3.Distance(corner, new Vector3(1f, -2f, -3f)) < 0.0001f);
        Assert.Contains(corners, corner => Vector3.Distance(corner, new Vector3(-1f, 2f, 3f)) < 0.0001f);
    }
}