using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests.World;

public sealed class WorldDistanceMathTests
{
    [Fact]
    public void TileAtCameraEdgeHasZeroDistanceEvenWhenItsCenterIsFarAway()
    {
        Vector3 camera = new(0f, 500f, 0f);
        Vector3 min = new(0f, 0f, -20f);
        Vector3 max = new(1000f, 1000f, 20f);

        Assert.Equal(0f, WorldDistanceMath.DistanceSquaredPointToAabb(camera, min, max));
        Assert.True(Vector3.DistanceSquared(camera, (min + max) * 0.5f) > 100f * 100f);
    }

    [Fact]
    public void PointOutsideBoundsUsesNearestBoundsPoint()
    {
        float distanceSquared = WorldDistanceMath.DistanceSquaredPointToAabb(
            new Vector3(12f, -4f, 8f),
            new Vector3(0f, 0f, 0f),
            new Vector3(10f, 10f, 10f));

        Assert.Equal(4f + 16f, distanceSquared);
    }

    [Fact]
    public void ReversedBoundsAreHandledAsAnAabb()
    {
        float distanceSquared = WorldDistanceMath.DistanceSquaredPointToAabb(
            new Vector3(12f, -4f, 8f),
            new Vector3(10f, 10f, 10f),
            new Vector3(0f, 0f, 0f));

        Assert.Equal(4f + 16f, distanceSquared);
    }
}
