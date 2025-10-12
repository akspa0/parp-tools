using System;
using System.Runtime.InteropServices;

namespace ArcaneFileParser.Core.Common.Types;

/// <summary>
/// Represents a 32-bit RGBA color with 8 bits per channel.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct ColorRGBA : IEquatable<ColorRGBA>
{
    public readonly byte R;
    public readonly byte G;
    public readonly byte B;
    public readonly byte A;

    public ColorRGBA(byte r, byte g, byte b, byte a)
    {
        R = r;
        G = g;
        B = b;
        A = a;
    }

    public static ColorRGBA FromUInt32(uint value) => new(
        (byte)((value >> 24) & 0xFF),
        (byte)((value >> 16) & 0xFF),
        (byte)((value >> 8) & 0xFF),
        (byte)(value & 0xFF)
    );

    public uint ToUInt32() => ((uint)R << 24) | ((uint)G << 16) | ((uint)B << 8) | A;

    public static ColorRGBA White => new(255, 255, 255, 255);
    public static ColorRGBA Black => new(0, 0, 0, 255);
    public static ColorRGBA Transparent => new(0, 0, 0, 0);

    public bool Equals(ColorRGBA other) => R == other.R && G == other.G && B == other.B && A == other.A;
    public override bool Equals(object? obj) => obj is ColorRGBA other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(R, G, B, A);

    public static bool operator ==(ColorRGBA left, ColorRGBA right) => left.Equals(right);
    public static bool operator !=(ColorRGBA left, ColorRGBA right) => !left.Equals(right);

    public override string ToString() => $"RGBA({R}, {G}, {B}, {A})";
}

/// <summary>
/// Represents a 32-bit BGRA color with 8 bits per channel.
/// This format is commonly used in WoW file formats.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct ColorBGRA : IEquatable<ColorBGRA>
{
    public readonly byte B;
    public readonly byte G;
    public readonly byte R;
    public readonly byte A;

    public ColorBGRA(byte b, byte g, byte r, byte a)
    {
        B = b;
        G = g;
        R = r;
        A = a;
    }

    public static ColorBGRA FromUInt32(uint value) => new(
        (byte)((value >> 24) & 0xFF),
        (byte)((value >> 16) & 0xFF),
        (byte)((value >> 8) & 0xFF),
        (byte)(value & 0xFF)
    );

    public uint ToUInt32() => ((uint)B << 24) | ((uint)G << 16) | ((uint)R << 8) | A;

    public static ColorBGRA White => new(255, 255, 255, 255);
    public static ColorBGRA Black => new(0, 0, 0, 255);
    public static ColorBGRA Transparent => new(0, 0, 0, 0);

    public ColorRGBA ToRGBA() => new(R, G, B, A);
    public static ColorBGRA FromRGBA(ColorRGBA rgba) => new(rgba.B, rgba.G, rgba.R, rgba.A);

    public bool Equals(ColorBGRA other) => B == other.B && G == other.G && R == other.R && A == other.A;
    public override bool Equals(object? obj) => obj is ColorBGRA other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(B, G, R, A);

    public static bool operator ==(ColorBGRA left, ColorBGRA right) => left.Equals(right);
    public static bool operator !=(ColorBGRA left, ColorBGRA right) => !left.Equals(right);

    public override string ToString() => $"BGRA({B}, {G}, {R}, {A})";
} 