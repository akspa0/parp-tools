using System.Numerics;
using WowViewer.Core.PM4.Matching;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4Geometry3DShapeSignatureTests
{
    [Fact]
    public void Compare3D_IdenticalGeometry_YieldsNearPerfectScore()
    {
        // Construct a simple 3D boat hull mesh: flat bottom, angled sides, tapered bow
        var triangles = new List<(Vector3 A, Vector3 B, Vector3 C)>
        {
            // Floor
            (new Vector3(-5f, -2f, 0f), new Vector3(5f, -2f, 0f), new Vector3(5f, 2f, 0f)),
            (new Vector3(-5f, -2f, 0f), new Vector3(5f, 2f, 0f), new Vector3(-5f, 2f, 0f)),
            // Left angled side
            (new Vector3(-5f, -2f, 0f), new Vector3(5f, -2f, 0f), new Vector3(5f, -3f, 2f)),
            (new Vector3(-5f, -2f, 0f), new Vector3(5f, -3f, 2f), new Vector3(-5f, -3f, 2f)),
            // Right angled side
            (new Vector3(-5f, 2f, 0f), new Vector3(5f, 2f, 0f), new Vector3(5f, 3f, 2f)),
            (new Vector3(-5f, 2f, 0f), new Vector3(5f, 3f, 2f), new Vector3(-5f, 3f, 2f)),
            // Bow
            (new Vector3(5f, -2f, 0f), new Vector3(8f, 0f, 1f), new Vector3(5f, 2f, 0f)),
        };

        var sigA = Pm4Geometry3DShapeSignature.FromGeometry(triangles);
        var sigB = Pm4Geometry3DShapeSignature.FromGeometry(triangles);

        float score = Pm4Geometry3DShapeSignature.Compare3D(sigA, sigB, out float bestYaw);

        Assert.InRange(score, 0.95f, 1.0f);
        Assert.Equal(0f, bestYaw);
    }

    [Fact]
    public void Compare3D_RotatedGeometry_RecoversYawOffset()
    {
        var trianglesA = new List<(Vector3 A, Vector3 B, Vector3 C)>
        {
            (new Vector3(-10f, -2f, 0f), new Vector3(10f, -2f, 0f), new Vector3(10f, 2f, 0f)),
            (new Vector3(-10f, -2f, 0f), new Vector3(10f, 2f, 0f), new Vector3(-10f, 2f, 0f)),
            (new Vector3(-10f, -2f, 0f), new Vector3(10f, -2f, 0f), new Vector3(10f, -2f, 4f)),
        };

        // Rotate 90 degrees around Z: (x, y, z) -> (-y, x, z)
        var trianglesB = new List<(Vector3 A, Vector3 B, Vector3 C)>();
        foreach (var (a, b, c) in trianglesA)
        {
            trianglesB.Add((
                new Vector3(-a.Y, a.X, a.Z),
                new Vector3(-b.Y, b.X, b.Z),
                new Vector3(-c.Y, c.X, c.Z)));
        }

        var sigA = Pm4Geometry3DShapeSignature.FromGeometry(trianglesA);
        var sigB = Pm4Geometry3DShapeSignature.FromGeometry(trianglesB);

        float score = Pm4Geometry3DShapeSignature.Compare3D(sigA, sigB, out float bestYaw);

        Assert.InRange(score, 0.85f, 1.0f);
        // The rotation difference should match 90 or 270 degrees
        Assert.True(bestYaw == 90f || bestYaw == 270f, $"Expected 90 or 270 degrees, got {bestYaw}");
    }

    [Fact]
    public void Compare3D_DifferentiatesDisparate3DShapes()
    {
        // Shape 1: Boat hull (sloped sides, high length-to-width ratio)
        var boatTriangles = new List<(Vector3 A, Vector3 B, Vector3 C)>
        {
            (new Vector3(-10f, -2f, 0f), new Vector3(10f, -2f, 0f), new Vector3(10f, 2f, 0f)),
            (new Vector3(-10f, -2f, 0f), new Vector3(10f, 2f, 0f), new Vector3(-10f, 2f, 0f)),
            (new Vector3(-10f, -2f, 0f), new Vector3(10f, -2f, 0f), new Vector3(10f, -4f, 3f)),
            (new Vector3(-10f, 2f, 0f), new Vector3(10f, 2f, 0f), new Vector3(10f, 4f, 3f)),
        };

        // Shape 2: Flat tabletop on vertical legs (different vertical profile and normal distribution)
        var tableTriangles = new List<(Vector3 A, Vector3 B, Vector3 C)>
        {
            // Thin horizontal slab at top
            (new Vector3(-5f, -5f, 10f), new Vector3(5f, -5f, 10f), new Vector3(5f, 5f, 10f)),
            (new Vector3(-5f, -5f, 10f), new Vector3(5f, 5f, 10f), new Vector3(-5f, 5f, 10f)),
            // Vertical leg
            (new Vector3(-4f, -4f, 0f), new Vector3(-4f, -4f, 10f), new Vector3(-3f, -4f, 10f)),
            (new Vector3(4f, 4f, 0f), new Vector3(4f, 4f, 10f), new Vector3(3f, 4f, 10f)),
        };

        var boatSig = Pm4Geometry3DShapeSignature.FromGeometry(boatTriangles);
        var tableSig = Pm4Geometry3DShapeSignature.FromGeometry(tableTriangles);

        float score = Pm4Geometry3DShapeSignature.Compare3D(boatSig, tableSig, out _);

        // Disparate 3D shapes should have low similarity score
        Assert.True(score < 0.65f, $"Expected disparate shapes to score < 0.65, got {score:F2}");
    }
}

