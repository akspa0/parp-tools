using System.Numerics;
using WmoBspConverter.Bsp;

namespace WmoBspConverter.Wmo;

internal readonly record struct GeometryBounds(Vector3 Min, Vector3 Max)
{
    public Vector3 Center => (Min + Max) * 0.5f;
}

internal readonly record struct GeometryContext(GeometryBounds Bounds, Vector3 Offset, GeometryBounds PaddedBounds);

internal static class WmoGeometryHelper
{
    public static Vector3 ToQuake3(Vector3 value) => new(value.X, -value.Z, value.Y);

    public static GeometryBounds ComputeBounds(BspFile bspFile)
    {
        var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        foreach (var vertex in bspFile.Vertices)
        {
            var transformed = ToQuake3(vertex.Position);
            min = Vector3.Min(min, transformed);
            max = Vector3.Max(max, transformed);
        }

        if (bspFile.Vertices.Count == 0)
        {
            min = Vector3.Zero;
            max = Vector3.Zero;
        }

        return new GeometryBounds(min, max);
    }

    public static GeometryContext Prepare(BspFile bspFile)
    {
        var bounds = ComputeBounds(bspFile);
        var offset = bounds.Center;
        var padding = new Vector3(128f, 128f, 128f);
        var padded = new GeometryBounds(bounds.Min - padding - offset, bounds.Max + padding - offset);
        return new GeometryContext(bounds, offset, padded);
    }
}
