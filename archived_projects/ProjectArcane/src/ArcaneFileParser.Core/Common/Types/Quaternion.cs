using System;
using System.Numerics;

namespace ArcaneFileParser.Core.Common.Types;

/// <summary>
/// Represents a quaternion for 3D rotations.
/// Uses XYZW layout to match WoW file formats.
/// </summary>
public readonly struct QuaternionF : IEquatable<QuaternionF>
{
    public readonly float X;
    public readonly float Y;
    public readonly float Z;
    public readonly float W;

    public QuaternionF(float x, float y, float z, float w)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
    }

    public static QuaternionF Identity => new(0, 0, 0, 1);

    public float Length => MathF.Sqrt(X * X + Y * Y + Z * Z + W * W);
    public float LengthSquared => X * X + Y * Y + Z * Z + W * W;

    public QuaternionF Normalized
    {
        get
        {
            float length = Length;
            if (length < float.Epsilon)
                return Identity;
            return new QuaternionF(X / length, Y / length, Z / length, W / length);
        }
    }

    public QuaternionF Conjugate => new(-X, -Y, -Z, W);

    public static QuaternionF CreateFromAxisAngle(Vector3F axis, float angle)
    {
        float halfAngle = angle * 0.5f;
        float sin = MathF.Sin(halfAngle);
        float cos = MathF.Cos(halfAngle);
        
        return new QuaternionF(
            axis.X * sin,
            axis.Y * sin,
            axis.Z * sin,
            cos
        );
    }

    public static QuaternionF CreateFromYawPitchRoll(float yaw, float pitch, float roll)
    {
        float halfRoll = roll * 0.5f;
        float halfPitch = pitch * 0.5f;
        float halfYaw = yaw * 0.5f;

        float sinRoll = MathF.Sin(halfRoll);
        float cosRoll = MathF.Cos(halfRoll);
        float sinPitch = MathF.Sin(halfPitch);
        float cosPitch = MathF.Cos(halfPitch);
        float sinYaw = MathF.Sin(halfYaw);
        float cosYaw = MathF.Cos(halfYaw);

        return new QuaternionF(
            cosYaw * sinPitch * cosRoll + sinYaw * cosPitch * sinRoll,
            sinYaw * cosPitch * cosRoll - cosYaw * sinPitch * sinRoll,
            cosYaw * cosPitch * sinRoll - sinYaw * sinPitch * cosRoll,
            cosYaw * cosPitch * cosRoll + sinYaw * sinPitch * sinRoll
        );
    }

    public static QuaternionF CreateFromRotationMatrix(Matrix4x4F matrix)
    {
        float trace = matrix.M11 + matrix.M22 + matrix.M33;

        if (trace > 0f)
        {
            float s = MathF.Sqrt(trace + 1.0f);
            float w = s * 0.5f;
            s = 0.5f / s;
            return new QuaternionF(
                (matrix.M32 - matrix.M23) * s,
                (matrix.M13 - matrix.M31) * s,
                (matrix.M21 - matrix.M12) * s,
                w
            );
        }

        if (matrix.M11 >= matrix.M22 && matrix.M11 >= matrix.M33)
        {
            float s = MathF.Sqrt(1.0f + matrix.M11 - matrix.M22 - matrix.M33);
            float invS = 0.5f / s;
            return new QuaternionF(
                0.5f * s,
                (matrix.M12 + matrix.M21) * invS,
                (matrix.M13 + matrix.M31) * invS,
                (matrix.M32 - matrix.M23) * invS
            );
        }

        if (matrix.M22 > matrix.M33)
        {
            float s = MathF.Sqrt(1.0f + matrix.M22 - matrix.M11 - matrix.M33);
            float invS = 0.5f / s;
            return new QuaternionF(
                (matrix.M21 + matrix.M12) * invS,
                0.5f * s,
                (matrix.M32 + matrix.M23) * invS,
                (matrix.M13 - matrix.M31) * invS
            );
        }

        {
            float s = MathF.Sqrt(1.0f + matrix.M33 - matrix.M11 - matrix.M22);
            float invS = 0.5f / s;
            return new QuaternionF(
                (matrix.M31 + matrix.M13) * invS,
                (matrix.M32 + matrix.M23) * invS,
                0.5f * s,
                (matrix.M21 - matrix.M12) * invS
            );
        }
    }

    public Matrix4x4F ToMatrix()
    {
        float xx = X * X;
        float yy = Y * Y;
        float zz = Z * Z;
        float xy = X * Y;
        float xz = X * Z;
        float yz = Y * Z;
        float wx = W * X;
        float wy = W * Y;
        float wz = W * Z;

        return new Matrix4x4F(
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0,
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0,
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0,
            0, 0, 0, 1
        );
    }

    public static QuaternionF operator *(QuaternionF a, QuaternionF b)
    {
        return new QuaternionF(
            a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
            a.W * b.Y - a.X * b.Z + a.Y * b.W + a.Z * b.X,
            a.W * b.Z + a.X * b.Y - a.Y * b.X + a.Z * b.W,
            a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z
        );
    }

    public Vector3F Rotate(Vector3F vector)
    {
        QuaternionF vectorQuat = new(vector.X, vector.Y, vector.Z, 0);
        QuaternionF result = this * vectorQuat * Conjugate;
        return new Vector3F(result.X, result.Y, result.Z);
    }

    public static QuaternionF Slerp(QuaternionF q1, QuaternionF q2, float t)
    {
        float dot = q1.X * q2.X + q1.Y * q2.Y + q1.Z * q2.Z + q1.W * q2.W;
        
        if (dot < 0)
        {
            q2 = new QuaternionF(-q2.X, -q2.Y, -q2.Z, -q2.W);
            dot = -dot;
        }

        if (dot > 0.9995f)
        {
            return new QuaternionF(
                q1.X + t * (q2.X - q1.X),
                q1.Y + t * (q2.Y - q1.Y),
                q1.Z + t * (q2.Z - q1.Z),
                q1.W + t * (q2.W - q1.W)
            ).Normalized;
        }

        float angle = MathF.Acos(dot);
        float sinAngle = MathF.Sin(angle);
        float invSinAngle = 1.0f / sinAngle;
        float t1 = MathF.Sin((1.0f - t) * angle) * invSinAngle;
        float t2 = MathF.Sin(t * angle) * invSinAngle;

        return new QuaternionF(
            q1.X * t1 + q2.X * t2,
            q1.Y * t1 + q2.Y * t2,
            q1.Z * t1 + q2.Z * t2,
            q1.W * t1 + q2.W * t2
        );
    }

    public bool Equals(QuaternionF other) =>
        X == other.X && Y == other.Y && Z == other.Z && W == other.W;

    public override bool Equals(object? obj) =>
        obj is QuaternionF other && Equals(other);

    public override int GetHashCode() =>
        HashCode.Combine(X, Y, Z, W);

    public static bool operator ==(QuaternionF left, QuaternionF right) =>
        left.Equals(right);

    public static bool operator !=(QuaternionF left, QuaternionF right) =>
        !left.Equals(right);

    public override string ToString() =>
        $"({X}, {Y}, {Z}, {W})";

    // Conversion operators
    public static implicit operator Quaternion(QuaternionF q) =>
        new(q.X, q.Y, q.Z, q.W);

    public static implicit operator QuaternionF(Quaternion q) =>
        new(q.X, q.Y, q.Z, q.W);
} 