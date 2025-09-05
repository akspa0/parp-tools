using System;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// MCLQ data for a single MCNK: 9x9 vertex grids (height/depth) and 8x8 per-tile type/flags.
/// </summary>
public sealed class MclqData
{
    public const int VertexGrid = 9;  // 9x9
    public const int TileGrid = 8;    // 8x8

    /// <summary>9x9 heights array (row-major). For depth-only cases, heights may be zeros.</summary>
    public float[] Heights { get; }

    /// <summary>9x9 depth array (row-major). For magma, values may be unused but are preserved.</summary>
    public byte[] Depth { get; }

    /// <summary>8x8 liquid types (lower 4-bit values interpreted to <see cref="MclqLiquidType"/>).</summary>
    public MclqLiquidType[,] Types { get; }

    /// <summary>8x8 flags (upper bits). See <see cref="MclqTileFlags"/>.</summary>
    public MclqTileFlags[,] Flags { get; }

    public MclqData(float[] heights, byte[] depth, MclqLiquidType[,] types, MclqTileFlags[,] flags)
    {
        Heights = heights ?? throw new ArgumentNullException(nameof(heights));
        Depth = depth ?? throw new ArgumentNullException(nameof(depth));
        Types = types ?? throw new ArgumentNullException(nameof(types));
        Flags = flags ?? throw new ArgumentNullException(nameof(flags));
        ValidateDimensions();
    }

    private void ValidateDimensions()
    {
        if (Heights.Length != VertexGrid * VertexGrid)
            throw new ArgumentException("Heights must be 9x9 = 81 elements", nameof(Heights));
        if (Depth.Length != VertexGrid * VertexGrid)
            throw new ArgumentException("Depth must be 9x9 = 81 elements", nameof(Depth));
        if (Types.GetLength(0) != TileGrid || Types.GetLength(1) != TileGrid)
            throw new ArgumentException("Types must be 8x8", nameof(Types));
        if (Flags.GetLength(0) != TileGrid || Flags.GetLength(1) != TileGrid)
            throw new ArgumentException("Flags must be 8x8", nameof(Flags));
    }
}
