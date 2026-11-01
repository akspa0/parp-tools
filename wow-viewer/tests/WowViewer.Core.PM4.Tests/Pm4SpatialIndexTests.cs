using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WowViewer.Core.PM4.Services;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4SpatialIndexTests
{
    [Fact]
    public void RayAABBIntersect_HitsBoxDirectly()
    {
        Vector3 origin = new(0f, 0f, -10f);
        Vector3 dir = new(0f, 0f, 1f);
        Vector3 bmin = new(-1f, -1f, -1f);
        Vector3 bmax = new(1f, 1f, 1f);

        float hitDist = Pm4SpatialMath.RayAABBIntersect(origin, dir, bmin, bmax);

        Assert.True(hitDist > 0f);
        Assert.Equal(9.0f, hitDist, 0.01f);
    }

    [Fact]
    public void RayAABBIntersect_MissesBox()
    {
        Vector3 origin = new(0f, 0f, -10f);
        Vector3 dir = new(0f, 1f, 0f); // Parallel pointing away
        Vector3 bmin = new(-1f, -1f, -1f);
        Vector3 bmax = new(1f, 1f, 1f);

        float hitDist = Pm4SpatialMath.RayAABBIntersect(origin, dir, bmin, bmax);

        Assert.Equal(-1.0f, hitDist);
    }

    [Fact]
    public void RayTriangleIntersect_HitsInterior()
    {
        Vector3 origin = new(2f, 2f, -10f);
        Vector3 dir = new(0f, 0f, 1f);
        Vector3 v0 = new(0f, 0f, 0f);
        Vector3 v1 = new(10f, 0f, 0f);
        Vector3 v2 = new(0f, 10f, 0f);

        float hitDist = Pm4SpatialMath.RayTriangleIntersect(origin, dir, v0, v1, v2);

        Assert.True(hitDist > 0f);
        Assert.Equal(10.0f, hitDist, 0.01f);
    }

    [Fact]
    public void Get3x3Neighborhood_Returns9TilesInBounds()
    {
        var tiles = Pm4SpatialMath.Get3x3Neighborhood(35, 38).ToList();

        Assert.Equal(9, tiles.Count);
        Assert.Contains((34, 37), tiles);
        Assert.Contains((35, 38), tiles);
        Assert.Contains((36, 39), tiles);
    }

    [Fact]
    public void Get3x3Neighborhood_ClampsToGrid()
    {
        var cornerTiles = Pm4SpatialMath.Get3x3Neighborhood(0, 0).ToList();

        Assert.Equal(4, cornerTiles.Count);
        Assert.Contains((0, 0), cornerTiles);
        Assert.Contains((1, 0), cornerTiles);
        Assert.Contains((0, 1), cornerTiles);
        Assert.Contains((1, 1), cornerTiles);
    }
}
