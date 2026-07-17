using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>
/// Unified model for WL* files (WLW, WLM, WLQ, WLL).
/// These are loose "Water Level" files containing liquid heightmaps.
/// Each block is 360 bytes with a 4x4 vertex grid.
/// </summary>
public sealed class WlFile
{
    public WlHeader Header { get; init; } = new();
    public IReadOnlyList<WlBlock> Blocks { get; init; } = Array.Empty<WlBlock>();
}

public enum WlFileType
{
    WLW,  // Water Level Water
    WLM,  // Water Level Magma (always magma)
    WLQ,  // Water Level (alternate format, WMO-style types)
    // Water Level Lava. The canonical terrain liquid palette represents lava as Magma.
    WLL
}

public enum WlLiquidType
{
    StillWater = 0,
    Ocean = 1,
    River = 2,
    Magma = 3,
    Slime = 4,
    FastWater = 5
}

public sealed class WlHeader
{
    public ReadOnlyMemory<byte> Magic { get; init; } = ReadOnlyMemory<byte>.Empty;
    public WlFileType FileType { get; init; }
    public ushort Version { get; init; }
    public ushort Unk06 { get; init; }
    public ushort RawLiquidType { get; init; }
    public ushort Padding { get; init; }
    public uint BlockCount { get; init; }
    public WlLiquidType LiquidType { get; init; }
}

public sealed class WlBlock
{
    /// <summary>16 vertices in 4x4 grid (z-up). Layout starts at lower-right corner.</summary>
    public Vector3[] Vertices { get; init; } = new Vector3[16];

    /// <summary>Internal grid X coordinate.</summary>
    public float CoordX { get; init; }

    /// <summary>Internal grid Y coordinate.</summary>
    public float CoordY { get; init; }

    /// <summary>Unknown data (80 ushorts).</summary>
    public ushort[] Data { get; init; } = new ushort[80];

    /// <summary>Gets heights in standard row-major order (reversed from file layout).</summary>
    public float[] GetHeights4x4()
    {
        var heights = new float[16];
        for (int i = 0; i < 16; i++)
            heights[15 - i] = Vertices[i].Z;
        return heights;
    }

    /// <summary>Gets the world position from the first vertex.</summary>
    public Vector3 WorldPosition => Vertices[0];
}
