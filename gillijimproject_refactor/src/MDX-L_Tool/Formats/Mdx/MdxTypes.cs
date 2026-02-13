using System.Runtime.InteropServices;

namespace MdxLTool.Formats.Mdx;

/// <summary>
/// Core MDX data types based on Ghidra analysis of WoW 0.5.3 client.
/// Maps to NTempest library types used in the original engine.
/// </summary>

/// <summary>NTempest::C3Vector - 3D position/normal</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct C3Vector
{
    public float X, Y, Z;

    public C3Vector(float x, float y, float z) { X = x; Y = y; Z = z; }
    public override string ToString() => $"({X:F4}, {Y:F4}, {Z:F4})";
}

/// <summary>NTempest::C2Vector - 2D texture coordinate</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct C2Vector
{
    public float U, V;

    public C2Vector(float u, float v) { U = u; V = v; }
    public override string ToString() => $"({U:F4}, {V:F4})";
}

/// <summary>NTempest::C4Quaternion - rotation</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct C4Quaternion
{
    public float X, Y, Z, W;

    public C4Quaternion(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }
}

/// <summary>
/// NTempest::C4QuaternionCompressed - 64-bit packed quaternion used in KGRT tracks.
/// Ghidra-verified decompression from 0x0074d690 / 0x0075ba30 / 0x0075bad0.
/// Three signed components packed as 21-bit integers; W reconstructed from unit norm.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct C4QuaternionCompressed
{
    public uint Data0; // low 32 bits
    public uint Data1; // high 32 bits

    /// <summary>Decompress to full C4Quaternion (X,Y,Z,W floats)</summary>
    public C4Quaternion Decompress()
    {
        // Arithmetic right shifts on signed reinterpretation
        int xq = ((int)Data1) >> 10;
        int yq = ((int)((Data1 << 22) | (Data0 >> 10))) >> 11;
        int zq = ((int)(Data0 << 11)) >> 11;

        // Scale: x uses 2^-21, y and z use 2^-20
        const float scaleX = 1.0f / (1 << 21); // 2^-21
        const float scaleYZ = 1.0f / (1 << 20); // 2^-20

        float x = xq * scaleX;
        float y = yq * scaleYZ;
        float z = zq * scaleYZ;

        // Reconstruct W from unit quaternion constraint
        float s = x * x + y * y + z * z;
        float w = (MathF.Abs(s - 1.0f) < scaleYZ) ? 0.0f : MathF.Sqrt(1.0f - s);

        return new C4Quaternion(x, y, z, w);
    }
}

/// <summary>CAaBox - axis-aligned bounding box</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct CAaBox
{
    public C3Vector Min;
    public C3Vector Max;
}

/// <summary>CMdlBounds - bounding volume (from offset 0xE0 in MDLGEOSET)</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct CMdlBounds
{
    public CAaBox Extent;
    public float Radius;
}

/// <summary>CiRange - integer range for animation times</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct CiRange
{
    public int Start;
    public int End;
}

/// <summary>Keyframe interpolation type</summary>
public enum MdlTrackType : uint
{
    NoInterp = 0,
    Linear = 1,
    Hermite = 2,
    Bezier = 3
}

/// <summary>Texture blend mode</summary>
public enum MdlTexOp : uint
{
    Load = 0,
    Transparent = 1,
    Blend = 2,
    Add = 3,
    AddAlpha = 4,
    Modulate = 5,
    Modulate2X = 6
}

/// <summary>Material geometry flags</summary>
[Flags]
public enum MdlGeoFlags : uint
{
    None = 0,
    Unshaded = 0x1,
    SphereEnvMap = 0x2,
    WrapWidth = 0x4,
    WrapHeight = 0x8,
    TwoSided = 0x10,
    Unfogged = 0x20,
    NoDepthTest = 0x40,
    NoDepthSet = 0x80,
    NoFallback = 0x100
}
