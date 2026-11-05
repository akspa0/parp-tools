using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Footprint;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4FootprintExtrusionTests
{
    [Fact]
    public void ComputeConvexHull2D_ExtractsOuterPerimeterPointsInOrder()
    {
        // 4 corners of a 10x10 square plus interior points
        var points = new List<Vector2>
        {
            new(0f, 0f),
            new(10f, 0f),
            new(10f, 10f),
            new(0f, 10f),
            new(5f, 5f), // interior
            new(2f, 3f), // interior
            new(8f, 7f), // interior
        };

        List<Vector2> hull = Pm4FootprintExtruder.ComputeConvexHull2D(points);

        Assert.Equal(4, hull.Count);
        // All 4 outer corners must be in the hull
        Assert.Contains(new Vector2(0f, 0f), hull);
        Assert.Contains(new Vector2(10f, 0f), hull);
        Assert.Contains(new Vector2(10f, 10f), hull);
        Assert.Contains(new Vector2(0f, 10f), hull);
    }

    [Fact]
    public void ComputePrincipalOrientation_CenterIsOrientedBoundingBoxCenterNotVertexMean()
    {
        // L-shaped footprint: the vertex mean is (4, 4), pulled toward the dense corner
        // cluster, but the hull's minimal oriented bounding box (PCA frame, rotated -45
        // degrees for this shape) is centered at (3, 3) — which also matches the polygon
        // area centroid (~3.2, 3.2) far better than the mean. The returned center must be
        // the OBB center, not the vertex mean: placement paths solve
        // position = prism.Centroid - rotCenter, so a biased centroid shifts every aligned
        // model off the hull on X/Y.
        var lShape = new List<Vector2>
        {
            new(0f, 0f),
            new(10f, 0f),
            new(10f, 2f),
            new(2f, 2f),
            new(2f, 10f),
            new(0f, 10f),
        };

        Pm4FootprintExtruder.ComputePrincipalOrientation(
            lShape,
            out _,
            out float length,
            out float width,
            out Vector2 center);

        // OBB center of the L-shape is (3, 3); the vertex mean would be (4, 4).
        Assert.InRange(center.X, 2.9f, 3.1f);
        Assert.InRange(center.Y, 2.9f, 3.1f);

        // Full extents are still 10 x 10 (the OBB spans both arms of the L).
        Assert.InRange(length, 13.9f, 14.4f);
        Assert.InRange(width, 8.4f, 8.6f);
    }

    [Fact]
    public void ComputePrincipalOrientation_RecoversUnrotatedDimensionsAndYaw()
    {
        // Construct a 20x10 rectangle centered at (100, 200) rotated by 45 degrees
        float angleRad = 45f * (MathF.PI / 180f);
        float cos = MathF.Cos(angleRad);
        float sin = MathF.Sin(angleRad);
        Vector2 center = new(100f, 200f);

        Vector2[] localCorners =
        [
            new(-10f, -5f),
            new(10f, -5f),
            new(10f, 5f),
            new(-10f, 5f),
        ];

        var worldPoints = new List<Vector2>();
        foreach (Vector2 p in localCorners)
        {
            Vector2 rot = new(p.X * cos - p.Y * sin, p.X * sin + p.Y * cos);
            worldPoints.Add(center + rot);
        }

        Pm4FootprintExtruder.ComputePrincipalOrientation(
            worldPoints,
            out float yawDegrees,
            out float length,
            out float width,
            out Vector2 computedCenter);

        // Yaw should be ~45 degrees (or -135)
        float normalizedYaw = ((yawDegrees % 180f) + 180f) % 180f;
        Assert.InRange(normalizedYaw, 44.5f, 45.5f);

        // Dimensions should match unrotated length (20) and width (10)
        Assert.InRange(length, 19.5f, 20.5f);
        Assert.InRange(width, 9.5f, 10.5f);

        // Center should match (100, 200)
        Assert.InRange(computedCenter.X, 99.9f, 100.1f);
        Assert.InRange(computedCenter.Y, 199.9f, 200.1f);
    }

    [Fact]
    public void ResolveBaseElevation_PrefersPlacementZForWmoAndMprlForDoodad()
    {
        float topZ = 50f;

        // 1. When explicit placement-Z is present (WMO), use it
        float wmoBase = Pm4FootprintExtruder.ResolveBaseElevation(
            topZ,
            mprlContacts: null,
            placementZ: 35f,
            fallbackBoundsMinZ: 30f);
        Assert.Equal(35f, wmoBase);

        // 2. When doodad has MPRL terrain contacts, use mean of contacts
        var mprl = new List<Vector3>
        {
            new(10f, 10f, 42f),
            new(12f, 12f, 44f),
        };
        float doodadBase = Pm4FootprintExtruder.ResolveBaseElevation(
            topZ,
            mprlContacts: mprl,
            placementZ: 0f, // raw doodads have 0 placement Z
            fallbackBoundsMinZ: 30f);
        Assert.Equal(43f, doodadBase);

        // 3. When no MPRL or placement-Z, fallback to bounds min
        float fallbackBase = Pm4FootprintExtruder.ResolveBaseElevation(
            topZ,
            mprlContacts: null,
            placementZ: null,
            fallbackBoundsMinZ: 38f);
        Assert.Equal(38f, fallbackBase);
    }

    [Fact]
    public void ExtrudeFootprint_GeneratesClosed3DPrismAndSyntheticWallQuads()
    {
        // 4 surface vertices forming a 10x10 roof at Z = 50
        var surfaceVertices = new List<Vector3>
        {
            new(100f, 200f, 50f),
            new(110f, 200f, 50f),
            new(110f, 210f, 50f),
            new(100f, 210f, 50f),
        };

        var mprl = new List<Vector3>
        {
            new(105f, 205f, 40f), // ground level at 40
        };

        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(
            surfaceVertices,
            mprlContacts: mprl,
            placementZ: null,
            fallbackBoundsMinZ: 30f);

        Assert.Equal(50f, prism.TopZ);
        Assert.Equal(40f, prism.BaseZ);
        Assert.Equal(10f, prism.Height);

        // Centroid should be in center of prism (105, 205, 45)
        Assert.InRange(prism.Centroid.X, 104.9f, 105.1f);
        Assert.InRange(prism.Centroid.Y, 204.9f, 205.1f);
        Assert.InRange(prism.Centroid.Z, 44.9f, 45.1f);

        // 4 corners in 2D convex hull
        Assert.Equal(4, prism.ConvexHull2D.Count);

        // 4 wall quads
        Assert.Equal(4, prism.Walls.Count);

        // 4 quads * 6 vertices per quad = 24 wall triangle vertices
        Assert.Equal(24, prism.WallTriangles.Count);

        // Top edge of each wall must be at TopZ (50), base at BaseZ (40)
        foreach (Pm4ExtrudedWall wall in prism.Walls)
        {
            Assert.Equal(50f, wall.TopV0.Z);
            Assert.Equal(50f, wall.TopV1.Z);
            Assert.Equal(40f, wall.BaseV0.Z);
            Assert.Equal(40f, wall.BaseV1.Z);
        }
    }
}

