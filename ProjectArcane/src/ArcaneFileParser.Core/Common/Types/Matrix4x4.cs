using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ArcaneFileParser.Core.Common.Types;

/// <summary>
/// Represents a 4x4 matrix optimized for 3D transformations.
/// Uses row-major order to match the WoW file format conventions.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Matrix4x4F : IEquatable<Matrix4x4F>
{
    public readonly float M11, M12, M13, M14;
    public readonly float M21, M22, M23, M24;
    public readonly float M31, M32, M33, M34;
    public readonly float M41, M42, M43, M44;

    public Matrix4x4F(
        float m11, float m12, float m13, float m14,
        float m21, float m22, float m23, float m24,
        float m31, float m32, float m33, float m34,
        float m41, float m42, float m43, float m44)
    {
        M11 = m11; M12 = m12; M13 = m13; M14 = m14;
        M21 = m21; M22 = m22; M23 = m23; M24 = m24;
        M31 = m31; M32 = m32; M33 = m33; M34 = m34;
        M41 = m41; M42 = m42; M43 = m43; M44 = m44;
    }

    public static Matrix4x4F Identity => new(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );

    public bool IsIdentity =>
        M11 == 1f && M12 == 0f && M13 == 0f && M14 == 0f &&
        M21 == 0f && M22 == 1f && M23 == 0f && M24 == 0f &&
        M31 == 0f && M32 == 0f && M33 == 1f && M34 == 0f &&
        M41 == 0f && M42 == 0f && M43 == 0f && M44 == 1f;

    public Vector3F Translation => new(M41, M42, M43);

    public static Matrix4x4F CreateTranslation(Vector3F position) => new(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        position.X, position.Y, position.Z, 1
    );

    public static Matrix4x4F CreateScale(Vector3F scale) => new(
        scale.X, 0, 0, 0,
        0, scale.Y, 0, 0,
        0, 0, scale.Z, 0,
        0, 0, 0, 1
    );

    public static Matrix4x4F CreateRotationX(float radians)
    {
        float cos = MathF.Cos(radians);
        float sin = MathF.Sin(radians);

        return new Matrix4x4F(
            1, 0, 0, 0,
            0, cos, -sin, 0,
            0, sin, cos, 0,
            0, 0, 0, 1
        );
    }

    public static Matrix4x4F CreateRotationY(float radians)
    {
        float cos = MathF.Cos(radians);
        float sin = MathF.Sin(radians);

        return new Matrix4x4F(
            cos, 0, sin, 0,
            0, 1, 0, 0,
            -sin, 0, cos, 0,
            0, 0, 0, 1
        );
    }

    public static Matrix4x4F CreateRotationZ(float radians)
    {
        float cos = MathF.Cos(radians);
        float sin = MathF.Sin(radians);

        return new Matrix4x4F(
            cos, -sin, 0, 0,
            sin, cos, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
    }

    public static Matrix4x4F operator *(Matrix4x4F value1, Matrix4x4F value2)
    {
        return new Matrix4x4F(
            // Row 1
            value1.M11 * value2.M11 + value1.M12 * value2.M21 + value1.M13 * value2.M31 + value1.M14 * value2.M41,
            value1.M11 * value2.M12 + value1.M12 * value2.M22 + value1.M13 * value2.M32 + value1.M14 * value2.M42,
            value1.M11 * value2.M13 + value1.M12 * value2.M23 + value1.M13 * value2.M33 + value1.M14 * value2.M43,
            value1.M11 * value2.M14 + value1.M12 * value2.M24 + value1.M13 * value2.M34 + value1.M14 * value2.M44,
            // Row 2
            value1.M21 * value2.M11 + value1.M22 * value2.M21 + value1.M23 * value2.M31 + value1.M24 * value2.M41,
            value1.M21 * value2.M12 + value1.M22 * value2.M22 + value1.M23 * value2.M32 + value1.M24 * value2.M42,
            value1.M21 * value2.M13 + value1.M22 * value2.M23 + value1.M23 * value2.M33 + value1.M24 * value2.M43,
            value1.M21 * value2.M14 + value1.M22 * value2.M24 + value1.M23 * value2.M34 + value1.M24 * value2.M44,
            // Row 3
            value1.M31 * value2.M11 + value1.M32 * value2.M21 + value1.M33 * value2.M31 + value1.M34 * value2.M41,
            value1.M31 * value2.M12 + value1.M32 * value2.M22 + value1.M33 * value2.M32 + value1.M34 * value2.M42,
            value1.M31 * value2.M13 + value1.M32 * value2.M23 + value1.M33 * value2.M33 + value1.M34 * value2.M43,
            value1.M31 * value2.M14 + value1.M32 * value2.M24 + value1.M33 * value2.M34 + value1.M34 * value2.M44,
            // Row 4
            value1.M41 * value2.M11 + value1.M42 * value2.M21 + value1.M43 * value2.M31 + value1.M44 * value2.M41,
            value1.M41 * value2.M12 + value1.M42 * value2.M22 + value1.M43 * value2.M32 + value1.M44 * value2.M42,
            value1.M41 * value2.M13 + value1.M42 * value2.M23 + value1.M43 * value2.M33 + value1.M44 * value2.M43,
            value1.M41 * value2.M14 + value1.M42 * value2.M24 + value1.M43 * value2.M34 + value1.M44 * value2.M44
        );
    }

    public Vector3F TransformPoint(Vector3F point)
    {
        return new Vector3F(
            point.X * M11 + point.Y * M21 + point.Z * M31 + M41,
            point.X * M12 + point.Y * M22 + point.Z * M32 + M42,
            point.X * M13 + point.Y * M23 + point.Z * M33 + M43
        );
    }

    public Vector3F TransformVector(Vector3F vector)
    {
        return new Vector3F(
            vector.X * M11 + vector.Y * M21 + vector.Z * M31,
            vector.X * M12 + vector.Y * M22 + vector.Z * M32,
            vector.X * M13 + vector.Y * M23 + vector.Z * M33
        );
    }

    public bool Equals(Matrix4x4F other)
    {
        return M11 == other.M11 && M12 == other.M12 && M13 == other.M13 && M14 == other.M14 &&
               M21 == other.M21 && M22 == other.M22 && M23 == other.M23 && M24 == other.M24 &&
               M31 == other.M31 && M32 == other.M32 && M33 == other.M33 && M34 == other.M34 &&
               M41 == other.M41 && M42 == other.M42 && M43 == other.M43 && M44 == other.M44;
    }

    public override bool Equals(object? obj) => obj is Matrix4x4F other && Equals(other);

    public override int GetHashCode()
    {
        return HashCode.Combine(
            HashCode.Combine(M11, M12, M13, M14),
            HashCode.Combine(M21, M22, M23, M24),
            HashCode.Combine(M31, M32, M33, M34),
            HashCode.Combine(M41, M42, M43, M44)
        );
    }

    public static bool operator ==(Matrix4x4F left, Matrix4x4F right) => left.Equals(right);
    public static bool operator !=(Matrix4x4F left, Matrix4x4F right) => !left.Equals(right);

    // Conversion operators
    public static implicit operator Matrix4x4(Matrix4x4F m) => new(
        m.M11, m.M12, m.M13, m.M14,
        m.M21, m.M22, m.M23, m.M24,
        m.M31, m.M32, m.M33, m.M34,
        m.M41, m.M42, m.M43, m.M44
    );

    public static implicit operator Matrix4x4F(Matrix4x4 m) => new(
        m.M11, m.M12, m.M13, m.M14,
        m.M21, m.M22, m.M23, m.M24,
        m.M31, m.M32, m.M33, m.M34,
        m.M41, m.M42, m.M43, m.M44
    );
} 