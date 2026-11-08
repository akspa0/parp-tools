using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Services;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4SpatialClusteringTests
{
    private sealed class TestHull
    {
        public int Id { get; init; }
        public Vector3 Min { get; init; }
        public Vector3 Max { get; init; }
        public uint LinkGroupId { get; init; }
    }

    [Fact]
    public void OverlapsWithTolerance_WhenOverlappingOrWithinTolerance_ReturnsTrue()
    {
        var minA = new Vector3(0, 0, 0);
        var maxA = new Vector3(10, 10, 10);

        // Directly overlapping
        var minB = new Vector3(5, 5, 5);
        var maxB = new Vector3(15, 15, 15);
        Assert.True(Pm4SpatialClustering.OverlapsWithTolerance(minA, maxA, minB, maxB, tolerance: 2.5f));

        // Separated by 1.5yd (< 2.5yd tolerance)
        var minC = new Vector3(11.5f, 0, 0);
        var maxC = new Vector3(20, 10, 10);
        Assert.True(Pm4SpatialClustering.OverlapsWithTolerance(minA, maxA, minC, maxC, tolerance: 2.5f));

        // Separated by 4.0yd (> 2.5yd tolerance)
        var minD = new Vector3(14.0f, 0, 0);
        var maxD = new Vector3(25, 10, 10);
        Assert.False(Pm4SpatialClustering.OverlapsWithTolerance(minA, maxA, minD, maxD, tolerance: 2.5f));
    }

    [Fact]
    public void DisjointSet_UnionAndFind_MergesCorrectly()
    {
        var dsu = new Pm4SpatialClustering.DisjointSet(5);
        dsu.Union(0, 1);
        dsu.Union(1, 2);
        dsu.Union(3, 4);

        Assert.Equal(dsu.Find(0), dsu.Find(2));
        Assert.Equal(dsu.Find(3), dsu.Find(4));
        Assert.NotEqual(dsu.Find(0), dsu.Find(3));
    }

    [Fact]
    public void ClusterItems_SpatialProximity_ClustersAdjacentHulls()
    {
        // 3 hulls representing ship keel, lower deck, upper deck (adjacent in Z)
        var hulls = new List<TestHull>
        {
            new() { Id = 0, Min = new Vector3(0, 0, 0), Max = new Vector3(30, 10, 5), LinkGroupId = 0 },
            new() { Id = 1, Min = new Vector3(0, 0, 4.5f), Max = new Vector3(30, 10, 10), LinkGroupId = 0 },
            new() { Id = 2, Min = new Vector3(0, 0, 9.5f), Max = new Vector3(30, 10, 15), LinkGroupId = 0 },
            // A separate dock 50 yards away
            new() { Id = 3, Min = new Vector3(100, 100, 0), Max = new Vector3(120, 110, 2), LinkGroupId = 0 }
        };

        var clusters = Pm4SpatialClustering.ClusterItems(
            hulls,
            (h, _) => (h.Min, h.Max),
            (h, _) => h.LinkGroupId,
            tolerance: 2.5f);

        Assert.Equal(2, clusters.Count);

        // Ship cluster should contain hulls 0, 1, 2
        var shipCluster = clusters.Find(c => c.Contains(0));
        Assert.NotNull(shipCluster);
        Assert.Equal(3, shipCluster.Count);
        Assert.Contains(1, shipCluster);
        Assert.Contains(2, shipCluster);

        // Dock cluster should contain only hull 3
        var dockCluster = clusters.Find(c => c.Contains(3));
        Assert.NotNull(dockCluster);
        Assert.Single(dockCluster);
    }

    [Fact]
    public void ClusterItems_SharedLinkGroupId_BridgesDisjointElevations()
    {
        // Two hulls physically separated by 10 yards vertically, but sharing MSLK link group ID
        var hulls = new List<TestHull>
        {
            new() { Id = 0, Min = new Vector3(0, 0, 0), Max = new Vector3(10, 10, 2), LinkGroupId = 0x1234 },
            new() { Id = 1, Min = new Vector3(0, 0, 15), Max = new Vector3(10, 10, 17), LinkGroupId = 0x1234 },
            // An unrelated hull
            new() { Id = 2, Min = new Vector3(50, 50, 0), Max = new Vector3(60, 60, 5), LinkGroupId = 0x9999 }
        };

        var clusters = Pm4SpatialClustering.ClusterItems(
            hulls,
            (h, _) => (h.Min, h.Max),
            (h, _) => h.LinkGroupId,
            tolerance: 2.5f);

        Assert.Equal(2, clusters.Count);
        var linkedCluster = clusters.Find(c => c.Contains(0));
        Assert.NotNull(linkedCluster);
        Assert.Equal(2, linkedCluster.Count);
        Assert.Contains(1, linkedCluster);
    }

    [Fact]
    public void AggregateBounds_CalculatesTotalExtentsCentroidAndDimensions()
    {
        var bounds = new List<(Vector3 Min, Vector3 Max)>
        {
            (new Vector3(0, 0, 0), new Vector3(10, 10, 5)),
            (new Vector3(5, 5, 5), new Vector3(20, 15, 12))
        };

        Pm4SpatialClustering.AggregateBounds(bounds, out Vector3 totalMin, out Vector3 totalMax, out Vector3 centroid, out Vector3 dimensions);

        Assert.Equal(new Vector3(0, 0, 0), totalMin);
        Assert.Equal(new Vector3(20, 15, 12), totalMax);
        Assert.Equal(new Vector3(20, 15, 12), dimensions);

        // Centers are (5, 5, 2.5) and (12.5, 10, 8.5) -> avg = (8.75, 7.5, 5.5)
        Assert.Equal(8.75f, centroid.X, precision: 3);
        Assert.Equal(7.50f, centroid.Y, precision: 3);
        Assert.Equal(5.50f, centroid.Z, precision: 3);
    }
}
