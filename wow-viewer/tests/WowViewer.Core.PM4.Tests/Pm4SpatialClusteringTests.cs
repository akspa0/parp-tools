using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Models;
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

    [Fact]
    public void ClusterCrossTileCompounds_WithCoincidentMscnPoints_MergesAcrossTileBoundary()
    {
        // Compound A on tile (13, 51) touching boundary at X = 27200.0
        var compA = new Pm4CrossTileCompoundDescriptor(
            CompoundId: 0,
            TileX: 13,
            TileY: 51,
            BoundsMin: new Vector3(27200.0f, 7300.0f, 10.0f),
            BoundsMax: new Vector3(27208.8f, 7326.2f, 15.1f),
            MscnWorldPoints: new[]
            {
                new Vector3(27200.002f, 7302.459f, 13.351f), // Seam point (matches compB within 0.003yd)
                new Vector3(27204.230f, 7303.969f, 12.152f)
            });

        // Compound B on tile (13, 50) across boundary at X = 27200.0
        var compB = new Pm4CrossTileCompoundDescriptor(
            CompoundId: 1,
            TileX: 13,
            TileY: 50,
            BoundsMin: new Vector3(27160.0f, 7290.0f, 8.0f),
            BoundsMax: new Vector3(27200.0f, 7340.0f, 25.0f),
            MscnWorldPoints: new[]
            {
                new Vector3(27199.998f, 7302.458f, 13.351f), // Matching seam point
                new Vector3(27180.000f, 7310.000f, 18.000f)
            });

        // Compound C on an unrelated tile far away
        var compC = new Pm4CrossTileCompoundDescriptor(
            CompoundId: 2,
            TileX: 13,
            TileY: 50,
            BoundsMin: new Vector3(26500.0f, 7000.0f, 0.0f),
            BoundsMax: new Vector3(26520.0f, 7020.0f, 10.0f),
            MscnWorldPoints: Array.Empty<Vector3>());

        var clusters = Pm4SpatialClustering.ClusterCrossTileCompounds(
            new[] { compA, compB, compC },
            mscnCoincidenceTolerance: 0.5f,
            boundaryAabbTolerance: 2.5f);

        Assert.Equal(2, clusters.Count);

        var mergedCluster = clusters.Find(c => c.Contains(0));
        Assert.NotNull(mergedCluster);
        Assert.Equal(2, mergedCluster.Count);
        Assert.Contains(1, mergedCluster);

        var cCluster = clusters.Find(c => c.Contains(2));
        Assert.NotNull(cCluster);
        Assert.Single(cCluster);
    }

    [Fact]
    public void ClusterCrossTileCompounds_WithSharedConnectorKeys_MergesAcrossTileBoundary()
    {
        var sharedKey = new Pm4ConnectorKey(13600, 3651, 6);

        var compA = new Pm4CrossTileCompoundDescriptor(
            CompoundId: 0,
            TileX: 13,
            TileY: 51,
            BoundsMin: new Vector3(27198.0f, 7300.0f, 10.0f),
            BoundsMax: new Vector3(27205.0f, 7310.0f, 15.0f),
            MscnWorldPoints: Array.Empty<Vector3>(),
            ConnectorKeys: new HashSet<Pm4ConnectorKey> { sharedKey });

        var compB = new Pm4CrossTileCompoundDescriptor(
            CompoundId: 1,
            TileX: 13,
            TileY: 50,
            BoundsMin: new Vector3(27195.0f, 7300.0f, 10.0f),
            BoundsMax: new Vector3(27202.0f, 7310.0f, 15.0f),
            MscnWorldPoints: Array.Empty<Vector3>(),
            ConnectorKeys: new HashSet<Pm4ConnectorKey> { sharedKey });

        var clusters = Pm4SpatialClustering.ClusterCrossTileCompounds(
            new[] { compA, compB });

        Assert.Single(clusters);
        Assert.Contains(0, clusters[0]);
        Assert.Contains(1, clusters[0]);
    }
}
