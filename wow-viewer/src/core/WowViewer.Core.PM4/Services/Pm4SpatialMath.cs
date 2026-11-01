using System;
using System.Collections.Generic;
using System.Numerics;

namespace WowViewer.Core.PM4.Services;

/// <summary>
/// Mathematical spatial operations, bounding volumes, and ray intersections for PM4 collision structures (Spec 288).
/// </summary>
public static class Pm4SpatialMath
{
    /// <summary>
    /// Computes world AABB for a 533.33yd map tile at (tileX, tileY).
    /// </summary>
    public static void ComputeTileWorldBounds(
        int tileX,
        int tileY,
        float minZ,
        float maxZ,
        out Vector3 tileMin,
        out Vector3 tileMax)
    {
        float tileMinX = 17066.666f - (tileX + 1) * 533.3333f;
        float tileMaxX = 17066.666f - tileX * 533.3333f;
        float tileMinY = 17066.666f - (tileY + 1) * 533.3333f;
        float tileMaxY = 17066.666f - tileY * 533.3333f;
        tileMin = new Vector3(tileMinX - 10f, tileMinY - 10f, minZ - 100f);
        tileMax = new Vector3(tileMaxX + 10f, tileMaxY + 10f, maxZ + 100f);
    }

    /// <summary>
    /// Tests ray against axis-aligned bounding box (AABB) using slabs method.
    /// Returns positive distance to entry point if hit, or -1f if missed.
    /// </summary>
    public static float RayAABBIntersect(Vector3 origin, Vector3 dir, Vector3 bmin, Vector3 bmax)
    {
        float tmin = 0f;
        float tmax = float.MaxValue;

        // X axis
        if (MathF.Abs(dir.X) < 1e-8f)
        {
            if (origin.X < bmin.X || origin.X > bmax.X) return -1f;
        }
        else
        {
            float inv = 1f / dir.X;
            float t1 = (bmin.X - origin.X) * inv;
            float t2 = (bmax.X - origin.X) * inv;
            if (t1 > t2) (t1, t2) = (t2, t1);
            tmin = MathF.Max(tmin, t1);
            tmax = MathF.Min(tmax, t2);
            if (tmin > tmax) return -1f;
        }

        // Y axis
        if (MathF.Abs(dir.Y) < 1e-8f)
        {
            if (origin.Y < bmin.Y || origin.Y > bmax.Y) return -1f;
        }
        else
        {
            float inv = 1f / dir.Y;
            float t1 = (bmin.Y - origin.Y) * inv;
            float t2 = (bmax.Y - origin.Y) * inv;
            if (t1 > t2) (t1, t2) = (t2, t1);
            tmin = MathF.Max(tmin, t1);
            tmax = MathF.Min(tmax, t2);
            if (tmin > tmax) return -1f;
        }

        // Z axis
        if (MathF.Abs(dir.Z) < 1e-8f)
        {
            if (origin.Z < bmin.Z || origin.Z > bmax.Z) return -1f;
        }
        else
        {
            float inv = 1f / dir.Z;
            float t1 = (bmin.Z - origin.Z) * inv;
            float t2 = (bmax.Z - origin.Z) * inv;
            if (t1 > t2) (t1, t2) = (t2, t1);
            tmin = MathF.Max(tmin, t1);
            tmax = MathF.Min(tmax, t2);
            if (tmin > tmax) return -1f;
        }

        return tmin;
    }

    /// <summary>
    /// Möller–Trumbore ray-triangle intersection.
    /// </summary>
    public static float RayTriangleIntersect(
        Vector3 origin,
        Vector3 dir,
        Vector3 v0,
        Vector3 v1,
        Vector3 v2)
    {
        const float epsilon = 1e-7f;
        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        Vector3 h = Vector3.Cross(dir, edge2);
        float a = Vector3.Dot(edge1, h);

        if (a > -epsilon && a < epsilon)
            return -1f;

        float f = 1.0f / a;
        Vector3 s = origin - v0;
        float u = f * Vector3.Dot(s, h);

        if (u < 0.0f || u > 1.0f)
            return -1f;

        Vector3 q = Vector3.Cross(s, edge1);
        float v = f * Vector3.Dot(dir, q);

        if (v < 0.0f || u + v > 1.0f)
            return -1f;

        float t = f * Vector3.Dot(edge2, q);
        return t > epsilon ? t : -1f;
    }

    /// <summary>
    /// Returns 3x3 tile grid coordinates surrounding (centerTileX, centerTileY) clamped to [0, 63].
    /// </summary>
    public static IEnumerable<(int tileX, int tileY)> Get3x3Neighborhood(int centerTileX, int centerTileY)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                int tx = centerTileX + dx;
                int ty = centerTileY + dy;
                if (tx >= 0 && tx < 64 && ty >= 0 && ty < 64)
                    yield return (tx, ty);
            }
        }
    }
}
