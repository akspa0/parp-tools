using System.Collections.Generic;
using WoWRollback.LkToAlphaModule.Liquids;

namespace WoWRollback.LkToAlphaModule.Models;

/// <summary>
/// Aggregates all data required to construct a Lich King ADT.
/// </summary>
public sealed class LkAdtSource
{
    public required string MapName { get; init; }
    public required int TileX { get; init; }
    public required int TileY { get; init; }

    /// <summary>
    /// Texture filename tables (MMDX/MWMO) and index indirection tables (MMID/MWID).
    /// </summary>
    public List<string> MmdxFilenames { get; } = new();
    public List<int> MmidOffsets { get; } = new();
    public List<string> MwmoFilenames { get; } = new();
    public List<int> MwidOffsets { get; } = new();

    /// <summary>
    /// Doodad (MDDF) and map object (MODF) placements associated with the tile.
    /// </summary>
    public List<LkMddfPlacement> MddfPlacements { get; } = new();
    public List<LkModfPlacement> ModfPlacements { get; } = new();

    /// <summary>
    /// Liquid data per MCNK (MH2O instances).
    /// </summary>
    public Mh2oChunk?[] Mh2oByChunk { get; } = new Mh2oChunk?[256];

    /// <summary>
    /// Optional MFBO/MTXF payloads.
    /// </summary>
    public byte[] MfboRaw { get; set; } = System.Array.Empty<byte>();
    public byte[] MtxfRaw { get; set; } = System.Array.Empty<byte>();

    /// <summary>
    /// MCNK sources in row-major 16x16 order.
    /// </summary>
    public List<LkMcnkSource> Mcnks { get; } = new();
}

public sealed record LkMddfPlacement(
    int NameIndex,
    int UniqueId,
    float PositionX,
    float PositionY,
    float PositionZ,
    float RotationX,
    float RotationY,
    float RotationZ,
    float Scale,
    ushort Flags);

public sealed record LkModfPlacement(
    int NameIndex,
    int UniqueId,
    float PositionX,
    float PositionY,
    float PositionZ,
    float RotationX,
    float RotationY,
    float RotationZ,
    float ExtentsX,
    float ExtentsY,
    float ExtentsZ,
    ushort Flags,
    ushort DoodadSet,
    ushort NameSet,
    ushort Scale);
