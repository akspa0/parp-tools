using System.Numerics;

namespace WoWToolbox.Core.v2.Foundation.PM4
{
    /// <summary>
    /// Minimal placeholder structs mirroring legacy PM4 parsing types used by some services.
    /// They wrap a <see cref="Vector3"/> position so that coordinate utilities can continue to compile.
    /// Replace with full implementations once the real chunk-parsing structs are ported.
    /// </summary>
    public readonly struct Vector3_Short
    {
        public readonly short X;
        public readonly short Y;
        public readonly short Z;

        public Vector3_Short(short x, short y, short z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public Vector3 ToVector3() => new(X, Y, Z);
    }

    public readonly struct MSVT_Vertex
    {
        public Vector3 Position { get; init; }
    }

    public readonly struct MSCN_Vertex
    {
        public Vector3 Position { get; init; }
    }

    public readonly struct MSPV_Vertex
    {
        public Vector3 Position { get; init; }
    }

    public readonly struct MPRL_Entry
    {
        public Vector3 Position { get; init; }
    }
}
