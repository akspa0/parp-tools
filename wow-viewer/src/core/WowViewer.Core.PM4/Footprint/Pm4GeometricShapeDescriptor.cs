using System;
using System.Collections.Generic;
using System.Numerics;

namespace WowViewer.Core.PM4.Footprint;

/// <summary>
/// Scale-invariant geometric signature describing the 3D proportions and pivot offset of a model
/// or a PM4 footprint prism.
/// </summary>
public sealed record Pm4GeometricShapeDescriptor(
    float Length,
    float Width,
    float Height,
    float AspectXY,
    float AspectZH,
    float FootprintArea,
    Vector3 LocalOriginOffset)
{
    public Vector3 BoundingBoxDimensions
    {
        get => new(Length, Width, Height);
        init
        {
            Length = MathF.Max(value.X, value.Y);
            Width = MathF.Min(value.X, value.Y);
            Height = MathF.Max(0.01f, value.Z);
        }
    }

    public Vector3 PrismCentroid { get; init; }
    public float PrimaryAxisAngleRadians { get; init; }
    public IReadOnlyList<Vector2>? FootprintHullVertices { get; init; }

    /// <summary>
    /// Builds a shape descriptor from model-space bounding extents (LocalBoundsMin, LocalBoundsMax).
    /// Captures the WC3-heritage asymmetric pivot offset (where (0,0,0) is at the door/stairs, not center).
    /// </summary>
    public static Pm4GeometricShapeDescriptor FromBounds(Vector3 localBoundsMin, Vector3 localBoundsMax)
    {
        float dx = MathF.Abs(localBoundsMax.X - localBoundsMin.X);
        float dy = MathF.Abs(localBoundsMax.Y - localBoundsMin.Y);
        float dz = MathF.Abs(localBoundsMax.Z - localBoundsMin.Z);

        float length = MathF.Max(dx, dy);
        float width = MathF.Min(dx, dy);
        float height = MathF.Max(0.01f, dz);

        float aspectXY = length / MathF.Max(0.01f, width);
        float aspectZH = height / MathF.Max(0.01f, length);
        float footprintArea = length * width;

        // Offset from model local center (min+max)/2 to model origin (0,0,0)
        Vector3 center = (localBoundsMin + localBoundsMax) * 0.5f;
        Vector3 originOffset = -center;

        return new Pm4GeometricShapeDescriptor(
            length,
            width,
            height,
            aspectXY,
            aspectZH,
            footprintArea,
            originOffset);
    }

    /// <summary>
    /// Builds a shape descriptor from a reified PM4 3D footprint prism.
    /// </summary>
    public static Pm4GeometricShapeDescriptor FromPrism(Pm4FootprintPrism prism)
    {
        ArgumentNullException.ThrowIfNull(prism);

        float length = MathF.Max(0.01f, prism.UnrotatedLength);
        float width = MathF.Max(0.01f, prism.UnrotatedWidth);
        float height = MathF.Max(0.01f, prism.Height);

        float aspectXY = length / width;
        float aspectZH = height / length;

        float area = ComputePolygonArea2D(prism.ConvexHull2D);
        if (area <= 0.01f)
            area = length * width;

        return new Pm4GeometricShapeDescriptor(
            length,
            width,
            height,
            aspectXY,
            aspectZH,
            area,
            Vector3.Zero)
        {
            PrismCentroid = prism.Centroid,
            PrimaryAxisAngleRadians = prism.YawDegrees * (MathF.PI / 180f),
            FootprintHullVertices = prism.ConvexHull2D
        };
    }

    /// <summary>
    /// Computes the 2D planar polygon area using the Shoelace formula.
    /// </summary>
    public static float ComputePolygonArea2D(IReadOnlyList<Vector2> polygon)
    {
        if (polygon == null || polygon.Count < 3)
            return 0f;

        float area = 0f;
        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 p0 = polygon[i];
            Vector2 p1 = polygon[(i + 1) % polygon.Count];
            area += (p0.X * p1.Y) - (p1.X * p0.Y);
        }

        return MathF.Abs(area) * 0.5f;
    }
}

