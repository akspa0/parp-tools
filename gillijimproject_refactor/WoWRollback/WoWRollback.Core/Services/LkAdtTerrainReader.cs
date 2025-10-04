using System;
using System.Collections.Generic;
using System.IO;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

/// <summary>
/// Reads MCNK terrain metadata directly from cached LK ADT files.
/// </summary>
public static class LkAdtTerrainReader
{
    /// <summary>
    /// Loads terrain data for the specified map using cached LK ADT files.
    /// </summary>
    public static List<McnkTerrainEntry> ReadFromLkAdts(
        string cachedMapsDir,
        string version,
        string mapName)
    {
        var entries = new List<McnkTerrainEntry>();

        if (string.IsNullOrWhiteSpace(cachedMapsDir) ||
            string.IsNullOrWhiteSpace(version) ||
            string.IsNullOrWhiteSpace(mapName))
        {
            return entries;
        }

        var mapDir = Path.Combine(cachedMapsDir, version, mapName);
        if (!Directory.Exists(mapDir))
        {
            Console.WriteLine($"[terrain] Cached map directory not found: {mapDir}");
            return entries;
        }

        var adtFiles = Directory.GetFiles(mapDir, "*.adt", SearchOption.TopDirectoryOnly);
        foreach (var adtPath in adtFiles)
        {
            if (!TryParseTileCoordinates(Path.GetFileNameWithoutExtension(adtPath), out var tileRow, out var tileCol))
            {
                continue;
            }

            var chunks = LkAdtReader.ReadMcnkChunks(adtPath);
            foreach (var chunk in chunks)
            {
                entries.Add(new McnkTerrainEntry(
                    Map: mapName,
                    TileRow: tileRow,
                    TileCol: tileCol,
                    ChunkRow: chunk.ChunkY,
                    ChunkCol: chunk.ChunkX,
                    FlagsRaw: chunk.Flags,
                    HasMcsh: chunk.HasMcsh,
                    Impassible: (chunk.Flags & 0x01) != 0,
                    LqRiver: (chunk.Flags & 0x04) != 0,
                    LqOcean: (chunk.Flags & 0x08) != 0,
                    LqMagma: (chunk.Flags & 0x10) != 0,
                    LqSlime: (chunk.Flags & 0x20) != 0,
                    HasMccv: chunk.HasMccv,
                    HighResHoles: (chunk.Flags & 0x10000) != 0,
                    AreaId: chunk.AreaId,
                    NumLayers: 0,
                    HasHoles: false,
                    HoleType: "none",
                    HoleBitmapHex: "0x0000",
                    HoleCount: 0,
                    PositionX: chunk.PositionX,
                    PositionY: chunk.PositionY,
                    PositionZ: chunk.PositionZ
                ));
            }
        }

        Console.WriteLine($"[terrain] Loaded {entries.Count} MCNK chunks from {adtFiles.Length} LK ADTs ({mapName})");
        return entries;
    }

    private static bool TryParseTileCoordinates(string? fileName, out int tileRow, out int tileCol)
    {
        tileRow = 0;
        tileCol = 0;

        if (string.IsNullOrWhiteSpace(fileName))
        {
            return false;
        }

        var parts = fileName.Split('_');
        if (parts.Length < 3)
        {
            return false;
        }

        return int.TryParse(parts[^2], out tileRow) &&
               int.TryParse(parts[^1], out tileCol);
    }
}
