using System;
using System.Numerics;

namespace ArcaneFileParser.Core.Common.Types;

/// <summary>
/// Represents a 2D vector with single-precision floating-point components.
/// Used for texture coordinates and 2D positions.
/// </summary>
public readonly struct Vector2F : IEquatable<Vector2F>
{
    public readonly float X;
    public readonly float Y;

    public Vector2F(float x, float y)
    {
        X = x;
        Y = y;
    }

    public static Vector2F Zero => new(0, 0);
    public static Vector2F One => new(1, 1);

    public float Length => MathF.Sqrt(X * X + Y * Y);
    public float LengthSquared => X * X + Y * Y;

    public Vector2F Normalized
    {
        get
        {
            float length = Length;
            if (length < float.Epsilon)
                return Zero;
            return new Vector2F(X / length, Y / length);
        }
    }

    public static Vector2F operator +(Vector2F a, Vector2F b) => new(a.X + b.X, a.Y + b.Y);
    public static Vector2F operator -(Vector2F a, Vector2F b) => new(a.X - b.X, a.Y - b.Y);
    public static Vector2F operator *(Vector2F a, float scalar) => new(a.X * scalar, a.Y * scalar);
    public static Vector2F operator /(Vector2F a, float scalar) => new(a.X / scalar, a.Y / scalar);
    public static Vector2F operator -(Vector2F v) => new(-v.X, -v.Y);

    public static float Dot(Vector2F a, Vector2F b) => a.X * b.X + a.Y * b.Y;

    public bool Equals(Vector2F other) => X == other.X && Y == other.Y;
    public override bool Equals(object? obj) => obj is Vector2F other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(X, Y);

    public static bool operator ==(Vector2F left, Vector2F right) => left.Equals(right);
    public static bool operator !=(Vector2F left, Vector2F right) => !left.Equals(right);

    public override string ToString() => $"({X}, {Y})";

    // Conversion operators
    public static implicit operator Vector2(Vector2F v) => new(v.X, v.Y);
    public static implicit operator Vector2F(Vector2 v) => new(v.X, v.Y);
} 