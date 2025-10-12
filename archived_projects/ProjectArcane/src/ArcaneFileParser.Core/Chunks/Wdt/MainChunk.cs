using System;
using System.IO;
using System.Runtime.InteropServices;

namespace ArcaneFileParser.Core.Chunks.Wdt;

/// <summary>
/// Main data chunk for WDT files containing map tile information.
/// </summary>
public class MainChunk : ChunkBase
{
    /// <summary>
    /// The expected signature for MAIN chunks.
    /// </summary>
    public const uint ExpectedSignature = 0x4E49414D; // "MAIN"

    /// <summary>
    /// The expected size of the MAIN chunk (4096 tiles * 8 bytes per tile).
    /// </summary>
    private const uint ExpectedSize = 4096 * 8;

    /// <summary>
    /// Flags indicating the state of each map tile.
    /// </summary>
    [Flags]
    public enum MapTileFlags : uint
    {
        None = 0,
        HasADT = 0x1,          // ADT file exists
        AllWater = 0x2,        // Tile is completely filled with water
        LoadedLowRes = 0x4,    // Low resolution terrain loaded (WoD+)
        LoadedHighRes = 0x8,   // High resolution terrain loaded (WoD+)
        HasMCCV = 0x10,        // Contains vertex colors (WotLK+)
        HasMH2O = 0x20,        // Contains water data (WotLK+)
        HasMCSE = 0x40,        // Contains sound emitters
        HasMCLQ = 0x80,        // Contains legacy liquid data
    }

    /// <summary>
    /// Structure representing a single map tile entry.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MapTileEntry
    {
        public MapTileFlags Flags;
        public uint AsyncId;
    }

    /// <summary>
    /// Gets the array of map tile entries (64x64 grid).
    /// </summary>
    public MapTileEntry[] Tiles { get; }

    /// <summary>
    /// Creates a new instance of the MAIN chunk.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public MainChunk(BinaryReader reader) : base(reader)
    {
        if (!ValidateSignature(ExpectedSignature))
        {
            return;
        }

        // MAIN chunks must be exactly 4096 * 8 bytes (64x64 grid of 8-byte entries)
        if (Size != ExpectedSize)
        {
            IsValid = false;
            EnsureAtEnd(reader);
            return;
        }

        // Read all tile entries
        Tiles = new MapTileEntry[4096];
        for (int i = 0; i < 4096; i++)
        {
            Tiles[i] = new MapTileEntry
            {
                Flags = (MapTileFlags)reader.ReadUInt32(),
                AsyncId = reader.ReadUInt32()
            };
        }

        EnsureAtEnd(reader);
    }

    /// <summary>
    /// Gets whether a specific tile exists at the given coordinates.
    /// </summary>
    /// <param name="x">The X coordinate (0-63).</param>
    /// <param name="y">The Y coordinate (0-63).</param>
    /// <returns>True if the tile exists, false otherwise.</returns>
    public bool HasTile(int x, int y)
    {
        if (x < 0 || x > 63 || y < 0 || y > 63)
            return false;

        int index = y * 64 + x;
        return (Tiles[index].Flags & MapTileFlags.HasADT) != 0;
    }

    /// <summary>
    /// Gets the tile entry at the specified coordinates.
    /// </summary>
    /// <param name="x">The X coordinate (0-63).</param>
    /// <param name="y">The Y coordinate (0-63).</param>
    /// <returns>The tile entry at the specified coordinates.</returns>
    public MapTileEntry GetTile(int x, int y)
    {
        if (x < 0 || x > 63 || y < 0 || y > 63)
            throw new ArgumentOutOfRangeException($"Coordinates must be between 0 and 63 (got {x},{y})");

        int index = y * 64 + x;
        return Tiles[index];
    }

    /// <summary>
    /// Creates a string representation of the chunk for debugging.
    /// </summary>
    public override string ToString()
    {
        int tileCount = 0;
        for (int i = 0; i < 4096; i++)
        {
            if ((Tiles[i].Flags & MapTileFlags.HasADT) != 0)
                tileCount++;
        }

        return $"MAIN [Size: {Size}, Valid: {IsValid}, Tiles: {tileCount}/4096]";
    }
} 