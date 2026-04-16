using System.Numerics;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Models;

public static class ModelFootprintReader
{
    private const int MaxFootprintSamplesPerSource = 2048;

    public static Vector2[][]? TryRead(byte[] data, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (data.Length < 16)
            return null;

        string extension = Path.GetExtension(sourcePath);
        if (extension.Equals(".wmo", StringComparison.OrdinalIgnoreCase))
            return null;

        if (extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
            return TryReadMdx(data, sourcePath) ?? TryReadM2(data, sourcePath);

        return TryReadM2(data, sourcePath) ?? TryReadMdx(data, sourcePath);
    }

    private static Vector2[][]? TryReadM2(byte[] data, string sourcePath)
    {
        try
        {
            using MemoryStream stream = new(data, writable: false);
            M2GeometryDocument geometry = M2GeometryReader.Read(stream, sourcePath);
            Vector2[]? hull = BuildFootprintHull(
                geometry.Vertices.Count,
                index => new Vector2(geometry.Vertices[index].Position.X, geometry.Vertices[index].Position.Z));
            return hull is { Length: >= 3 } ? [hull] : null;
        }
        catch
        {
            return null;
        }
    }

    private static Vector2[][]? TryReadMdx(byte[] data, string sourcePath)
    {
        try
        {
            using MemoryStream stream = new(data, writable: false);
            MdxGeometryFile geometry = MdxGeometryReader.Read(stream, sourcePath);
            List<Vector2[]> polygons = [];
            foreach (MdxGeosetGeometry geoset in geometry.Geosets)
            {
                Vector2[]? hull = BuildFootprintHull(
                    geoset.Vertices.Count,
                    index => new Vector2(geoset.Vertices[index].X, geoset.Vertices[index].Z));
                if (hull is { Length: >= 3 })
                    polygons.Add(hull);
            }

            return polygons.Count > 0 ? [.. polygons] : null;
        }
        catch
        {
            return null;
        }
    }

    private static Vector2[]? BuildFootprintHull(int pointCount, Func<int, Vector2> pointAccessor)
    {
        if (pointCount < 3)
            return null;

        int step = Math.Max(1, (int)Math.Ceiling(pointCount / (double)MaxFootprintSamplesPerSource));
        List<Vector2> points = new(Math.Min(pointCount, MaxFootprintSamplesPerSource) + 1);
        for (int index = 0; index < pointCount; index += step)
        {
            Vector2 point = pointAccessor(index);
            if (float.IsFinite(point.X) && float.IsFinite(point.Y))
                points.Add(point);
        }

        int lastIndex = pointCount - 1;
        if ((lastIndex % step) != 0)
        {
            Vector2 point = pointAccessor(lastIndex);
            if (float.IsFinite(point.X) && float.IsFinite(point.Y))
                points.Add(point);
        }

        return BuildConvexHull(points);
    }

    private static Vector2[]? BuildConvexHull(List<Vector2> points)
    {
        if (points.Count < 3)
            return null;

        points.Sort(static (left, right) =>
        {
            int compareX = left.X.CompareTo(right.X);
            return compareX != 0 ? compareX : left.Y.CompareTo(right.Y);
        });

        List<Vector2> uniquePoints = new(points.Count);
        foreach (Vector2 point in points)
        {
            if (uniquePoints.Count > 0 && Vector2.DistanceSquared(uniquePoints[^1], point) < 0.000001f)
                continue;

            uniquePoints.Add(point);
        }

        if (uniquePoints.Count < 3)
            return null;

        List<Vector2> hull = new(uniquePoints.Count * 2);
        foreach (Vector2 point in uniquePoints)
        {
            while (hull.Count >= 2 && Cross(hull[^2], hull[^1], point) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(point);
        }

        int lowerCount = hull.Count;
        for (int index = uniquePoints.Count - 2; index >= 0; index--)
        {
            Vector2 point = uniquePoints[index];
            while (hull.Count > lowerCount && Cross(hull[^2], hull[^1], point) <= 0f)
                hull.RemoveAt(hull.Count - 1);
            hull.Add(point);
        }

        if (hull.Count <= 3)
            return null;

        hull.RemoveAt(hull.Count - 1);
        return hull.ToArray();
    }

    private static float Cross(Vector2 origin, Vector2 left, Vector2 right)
    {
        Vector2 a = left - origin;
        Vector2 b = right - origin;
        return (a.X * b.Y) - (a.Y * b.X);
    }
}