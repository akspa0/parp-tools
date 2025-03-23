using System;
using System.IO;
using System.Runtime.InteropServices;

namespace ArcaneFileParser.Core.Chunks.Wdt;

/// <summary>
/// Map header chunk containing global map information.
/// </summary>
public class MapHeaderChunk : VersionedChunkBase
{
    /// <summary>
    /// The expected signature for MPHD chunks.
    /// </summary>
    public const uint ExpectedSignature = 0x4448504D; // "MPHD"

    /// <summary>
    /// The expected version for MPHD chunks.
    /// </summary>
    protected override uint ExpectedVersion => 18;

    /// <summary>
    /// Flags indicating global map properties.
    /// </summary>
    [Flags]
    public enum MapFlags : uint
    {
        None = 0x0,
        UsesGlobalMapObj = 0x1,          // Uses global map objects
        HasVertexShading = 0x2,          // Has vertex shading (MCNK.MCCV)
        UsesLiquid_v2 = 0x4,             // Uses the MH2O liquid system
        UsesLiquid_v1 = 0x8,             // Uses the MCLQ liquid system
        UsesVertexLighting = 0x10,       // Uses vertex lighting
        HasPvpObjective = 0x20,          // Map has PvP objective
        UsesLiquid_v3 = 0x40,            // Uses the new liquid system (Legion+)
        IsBattleground = 0x80,           // Map is a battleground
        NoTerrainShading = 0x100,        // Disables terrain shading
        IsRaid = 0x200,                  // Map is a raid instance
        ShowMinimapInExterior = 0x400,   // Shows minimap in exterior
        ShowMinimapInInterior = 0x800,   // Shows minimap in interior
        UsesHeightTexturing = 0x1000,    // Uses height-based texturing
        HasFlightBoundary = 0x2000,      // Has flight boundaries (MFBO)
        UsesVertexColoring = 0x4000,     // Uses vertex coloring
        IsInterior = 0x8000,             // Map is an interior
        DontFixAlpha = 0x10000,          // Don't fix alpha values
        IsOcean = 0x20000,               // Map is an ocean
        IsMountAllowed = 0x40000,        // Mounting is allowed
        IsFlexibleLiquid = 0x80000,      // Uses flexible liquid system
    }

    /// <summary>
    /// Gets the flags indicating various map properties.
    /// </summary>
    public MapFlags Flags { get; }

    /// <summary>
    /// Gets the minimum terrain height for the map.
    /// </summary>
    public float MinTerrainHeight { get; }

    /// <summary>
    /// Gets the maximum terrain height for the map.
    /// </summary>
    public float MaxTerrainHeight { get; }

    /// <summary>
    /// Gets the number of doodad sets in the map.
    /// </summary>
    public uint DoodadSetCount { get; }

    /// <summary>
    /// Gets the number of map tile cells per side (usually 64).
    /// </summary>
    public uint TilesPerSide { get; }

    /// <summary>
    /// Creates a new instance of the MPHD chunk.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public MapHeaderChunk(BinaryReader reader) : base(reader)
    {
        if (!ValidateSignature(ExpectedSignature))
        {
            return;
        }

        // Read the map header data
        Flags = (MapFlags)reader.ReadUInt32();
        MinTerrainHeight = reader.ReadSingle();
        MaxTerrainHeight = reader.ReadSingle();
        DoodadSetCount = reader.ReadUInt32();
        TilesPerSide = reader.ReadUInt32();

        EnsureAtEnd(reader);
    }

    /// <summary>
    /// Creates a string representation of the chunk for debugging.
    /// </summary>
    public override string ToString() =>
        $"MPHD [Version: {Version}, Flags: {Flags}, Height: {MinTerrainHeight:F2} to {MaxTerrainHeight:F2}, " +
        $"DoodadSets: {DoodadSetCount}, TilesPerSide: {TilesPerSide}, Valid: {IsValid}]";
} 