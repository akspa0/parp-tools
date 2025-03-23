using System;

namespace ArcaneFileParser.Core.Common.Types;

/// <summary>
/// Represents an axis-aligned bounding box defined by minimum and maximum points.
/// </summary>
public readonly struct BoundingBox : IEquatable<BoundingBox>
{
    public readonly Vector3F Min;
    public readonly Vector3F Max;

    public BoundingBox(Vector3F min, Vector3F max)
    {
        Min = min;
        Max = max;
    }

    public Vector3F Center => (Min + Max) * 0.5f;
    public Vector3F Extents => (Max - Min) * 0.5f;
    public Vector3F Size => Max - Min;

    public float Width => Max.X - Min.X;
    public float Height => Max.Y - Min.Y;
    public float Depth => Max.Z - Min.Z;

    public float Volume => Width * Height * Depth;

    public bool Contains(Vector3F point) =>
        point.X >= Min.X && point.X <= Max.X &&
        point.Y >= Min.Y && point.Y <= Max.Y &&
        point.Z >= Min.Z && point.Z <= Max.Z;

    public bool Intersects(BoundingBox other) =>
        Min.X <= other.Max.X && Max.X >= other.Min.X &&
        Min.Y <= other.Max.Y && Max.Y >= other.Min.Y &&
        Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;

    public static BoundingBox CreateFromPoints(ReadOnlySpan<Vector3F> points)
    {
        if (points.IsEmpty)
            return new BoundingBox(Vector3F.Zero, Vector3F.Zero);

        var min = new Vector3F(float.MaxValue, float.MaxValue, float.MaxValue);
        var max = new Vector3F(float.MinValue, float.MinValue, float.MinValue);

        foreach (var point in points)
        {
            min = new Vector3F(
                MathF.Min(min.X, point.X),
                MathF.Min(min.Y, point.Y),
                MathF.Min(min.Z, point.Z)
            );
            max = new Vector3F(
                MathF.Max(max.X, point.X),
                MathF.Max(max.Y, point.Y),
                MathF.Max(max.Z, point.Z)
            );
        }

        return new BoundingBox(min, max);
    }

    public static BoundingBox Combine(BoundingBox a, BoundingBox b) => new(
        new Vector3F(
            MathF.Min(a.Min.X, b.Min.X),
            MathF.Min(a.Min.Y, b.Min.Y),
            MathF.Min(a.Min.Z, b.Min.Z)
        ),
        new Vector3F(
            MathF.Max(a.Max.X, b.Max.X),
            MathF.Max(a.Max.Y, b.Max.Y),
            MathF.Max(a.Max.Z, b.Max.Z)
        )
    );

    public bool Equals(BoundingBox other) => Min == other.Min && Max == other.Max;
    public override bool Equals(object? obj) => obj is BoundingBox other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Min, Max);

    public static bool operator ==(BoundingBox left, BoundingBox right) => left.Equals(right);
    public static bool operator !=(BoundingBox left, BoundingBox right) => !left.Equals(right);

    public override string ToString() => $"Min: {Min}, Max: {Max}";
} 