using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace WowViewer.Core.PM4.Footprint;

/// <summary>
/// A single synthetic vertical wall quad connecting an edge of the top walkable surface
/// down to the resolved ground/base elevation.
/// </summary>
public sealed record Pm4ExtrudedWall(
    Vector3 TopV0,
    Vector3 TopV1,
    Vector3 BaseV0,
    Vector3 BaseV1);

/// <summary>
/// A reified 3D envelope prism computed from PM4 top surface geometry, MPRL terrain contacts,
/// and Placement-Z markers.
/// </summary>
public sealed record Pm4FootprintPrism(
    IReadOnlyList<Vector2> ConvexHull2D,
    float TopZ,
    float BaseZ,
    float Height,
    Vector3 Centroid,
    float YawDegrees,
    float UnrotatedLength,
    float UnrotatedWidth,
    IReadOnlyList<Pm4ExtrudedWall> Walls,
    IReadOnlyList<Vector3> WallTriangles);

/// <summary>
/// Pure algorithmic service that reconstructs 3D footprint prisms and synthetic vertical walls
/// from PM4 top walkable surfaces, MPRL terrain contacts, and Placement-Z markers.
/// </summary>
public static class Pm4FootprintExtruder
{
    /// <summary>
    /// Computes the 2D convex hull of a collection of 2D points using Andrew's monotone chain algorithm (O(n log n)).
    /// </summary>
    public static List<Vector2> ComputeConvexHull2D(IReadOnlyList<Vector2> points)
    {
        if (points == null || points.Count < 3)
            return points != null ? new List<Vector2>(points) : [];

        // Sort points lexicographically by X, then Y
        var sorted = points.OrderBy(static p => p.X).ThenBy(static p => p.Y).ToList();

        // Deduplicate consecutive identical points
        var unique = new List<Vector2>(sorted.Count) { sorted[0] };
        for (int i = 1; i < sorted.Count; i++)
        {
            if (Vector2.DistanceSquared(sorted[i], unique[^1]) > 1e-6f)
                unique.Add(sorted[i]);
        }

        if (unique.Count < 3)
            return unique;

        int n = unique.Count;
        var hull = new List<Vector2>(2 * n);

        // Lower hull
        for (int i = 0; i < n; i++)
        {
            while (hull.Count >= 2 && CrossProduct(hull[^2], hull[^1], unique[i]) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(unique[i]);
        }

        // Upper hull
        for (int i = n - 2, t = hull.Count + 1; i >= 0; i--)
        {
            while (hull.Count >= t && CrossProduct(hull[^2], hull[^1], unique[i]) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(unique[i]);
        }

        // Remove the last redundant point (same as first)
        if (hull.Count > 1)
            hull.RemoveAt(hull.Count - 1);

        return hull;
    }

    /// <summary>
    /// Computes the 2D cross-product of vectors (b - a) and (c - a).
    /// </summary>
    private static float CrossProduct(Vector2 a, Vector2 b, Vector2 c)
    {
        return (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
    }

    /// <summary>
    /// Resolves the ground/base elevation ($Z_{\text{base}}$) using MPRL terrain contacts, Placement-Z,
    /// or fallback bounding minimum.
    /// </summary>
    public static float ResolveBaseElevation(
        float topZ,
        IReadOnlyList<Vector3>? mprlContacts,
        float? placementZ,
        float fallbackBoundsMinZ)
    {
        // 1. If explicit Placement-Z is valid (WMO root), respect it
        if (placementZ.HasValue && MathF.Abs(placementZ.Value) > 0.01f && placementZ.Value < topZ)
        {
            return placementZ.Value;
        }

        // 2. If MPRL terrain contacts are present (M2 doodads / terrain intersections), use mean terrain elevation
        if (mprlContacts is { Count: > 0 })
        {
            float mprlSum = 0f;
            int count = 0;
            for (int i = 0; i < mprlContacts.Count; i++)
            {
                float z = mprlContacts[i].Z;
                if (!float.IsNaN(z) && !float.IsInfinity(z) && z < topZ + 0.1f)
                {
                    mprlSum += z;
                    count++;
                }
            }
            if (count > 0)
                return mprlSum / count;
        }

        // 3. Fallback to object bounding box minimum
        if (!float.IsNaN(fallbackBoundsMinZ) && !float.IsInfinity(fallbackBoundsMinZ) && fallbackBoundsMinZ < topZ - 0.05f)
        {
            return fallbackBoundsMinZ;
        }

        // 4. Default to standard 3-yard doodad height if completely unanchored
        return topZ - 3.0f;
    }

    /// <summary>
    /// Computes the 2D principal orientation (Yaw angle) and unrotated dimensions (Length, Width)
    /// of a 2D point cloud via 2D Principal Component Analysis (covariance matrix).
    /// </summary>
    public static void ComputePrincipalOrientation(
        IReadOnlyList<Vector2> points,
        out float yawDegrees,
        out float length,
        out float width,
        out Vector2 center)
    {
        yawDegrees = 0f;
        length = 0f;
        width = 0f;
        center = Vector2.Zero;

        if (points == null || points.Count == 0)
            return;

        // 1. Compute centroid
        float sumX = 0f;
        float sumY = 0f;
        for (int i = 0; i < points.Count; i++)
        {
            sumX += points[i].X;
            sumY += points[i].Y;
        }
        center = new Vector2(sumX / points.Count, sumY / points.Count);

        if (points.Count == 1)
            return;

        // 2. Covariance matrix elements
        float cxx = 0f;
        float cyy = 0f;
        float cxy = 0f;
        for (int i = 0; i < points.Count; i++)
        {
            float dx = points[i].X - center.X;
            float dy = points[i].Y - center.Y;
            cxx += dx * dx;
            cyy += dy * dy;
            cxy += dx * dy;
        }

        // 3. Principal angle
        if (MathF.Abs(cxx - cyy) > 1e-5f || MathF.Abs(cxy) > 1e-5f)
        {
            float angleRad = 0.5f * MathF.Atan2(2f * cxy, cxx - cyy);
            yawDegrees = angleRad * (180f / MathF.PI);
            while (yawDegrees > 180f) yawDegrees -= 360f;
            while (yawDegrees < -180f) yawDegrees += 360f;
        }

        // 4. Project points onto principal and secondary axes to find true unrotated extents
        float rad = yawDegrees * (MathF.PI / 180f);
        Vector2 axisL = new(MathF.Cos(rad), MathF.Sin(rad));
        Vector2 axisW = new(-MathF.Sin(rad), MathF.Cos(rad));

        float minL = float.MaxValue, maxL = float.MinValue;
        float minW = float.MaxValue, maxW = float.MinValue;

        for (int i = 0; i < points.Count; i++)
        {
            Vector2 rel = points[i] - center;
            float projL = Vector2.Dot(rel, axisL);
            float projW = Vector2.Dot(rel, axisW);

            if (projL < minL) minL = projL;
            if (projL > maxL) maxL = projL;
            if (projW < minW) minW = projW;
            if (projW > maxW) maxW = projW;
        }

        length = MathF.Max(0.01f, maxL - minL);
        width = MathF.Max(0.01f, maxW - minW);

        // The vertex mean is biased toward vertex clusters on irregular footprints (L-shapes,
        // wings, bridges), which shifted every placement solved against prism.Centroid off the
        // hull's visual center on X/Y. Use the oriented bounding box center instead: the
        // midpoint of the principal-axis projections. Identical to the mean for symmetric
        // shapes, unbiased for everything else.
        center += axisL * ((minL + maxL) * 0.5f) + axisW * ((minW + maxW) * 0.5f);

        // Ensure length >= width (canonicalize primary axis as the longer dimension)
        if (width > length)
        {
            (length, width) = (width, length);
            yawDegrees += 90f;
            while (yawDegrees > 180f) yawDegrees -= 360f;
            while (yawDegrees < -180f) yawDegrees += 360f;
        }
    }

    /// <summary>
    /// Extrudes the 2D perimeter of the top surface down to the base elevation, generating
    /// a closed 3D prism with synthetic vertical walls.
    /// </summary>
    public static Pm4FootprintPrism ExtrudeFootprint(
        IReadOnlyList<Vector3> surfaceVertices,
        IReadOnlyList<Vector3>? mprlContacts,
        float? placementZ,
        float fallbackBoundsMinZ)
    {
        if (surfaceVertices == null || surfaceVertices.Count == 0)
        {
            return new Pm4FootprintPrism(
                [], 0f, 0f, 0f, Vector3.Zero, 0f, 0f, 0f, [], []);
        }

        // 1. Calculate top surface elevation
        float topZ = float.MinValue;
        var points2D = new List<Vector2>(surfaceVertices.Count);
        for (int i = 0; i < surfaceVertices.Count; i++)
        {
            Vector3 v = surfaceVertices[i];
            if (v.Z > topZ)
                topZ = v.Z;
            points2D.Add(new Vector2(v.X, v.Y));
        }

        // 2. Compute 2D Convex Hull of footprint
        List<Vector2> hull2D = ComputeConvexHull2D(points2D);

        // 3. Resolve base elevation
        float baseZ = ResolveBaseElevation(topZ, mprlContacts, placementZ, fallbackBoundsMinZ);
        float height = MathF.Max(0.01f, topZ - baseZ);

        // 4. Compute principal orientation and unrotated dimensions
        ComputePrincipalOrientation(hull2D, out float yawDegrees, out float unrotatedLength, out float unrotatedWidth, out Vector2 center2D);
        Vector3 centroid = new(center2D.X, center2D.Y, (topZ + baseZ) * 0.5f);

        // 5. Generate vertical wall quads and triangles
        var walls = new List<Pm4ExtrudedWall>(hull2D.Count);
        var wallTriangles = new List<Vector3>(hull2D.Count * 6);

        if (hull2D.Count >= 3)
        {
            for (int i = 0; i < hull2D.Count; i++)
            {
                Vector2 p0 = hull2D[i];
                Vector2 p1 = hull2D[(i + 1) % hull2D.Count];

                Vector3 topV0 = new(p0.X, p0.Y, topZ);
                Vector3 topV1 = new(p1.X, p1.Y, topZ);
                Vector3 baseV0 = new(p0.X, p0.Y, baseZ);
                Vector3 baseV1 = new(p1.X, p1.Y, baseZ);

                walls.Add(new Pm4ExtrudedWall(topV0, topV1, baseV0, baseV1));

                // Quad as two CCW triangles: (topV0, baseV0, baseV1) and (topV0, baseV1, topV1)
                wallTriangles.Add(topV0);
                wallTriangles.Add(baseV0);
                wallTriangles.Add(baseV1);

                wallTriangles.Add(topV0);
                wallTriangles.Add(baseV1);
                wallTriangles.Add(topV1);
            }
        }

        return new Pm4FootprintPrism(
            hull2D,
            topZ,
            baseZ,
            height,
            centroid,
            yawDegrees,
            unrotatedLength,
            unrotatedWidth,
            walls,
            wallTriangles);
    }
}

