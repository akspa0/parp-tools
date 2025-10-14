using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using WoWRollback.Core.Services;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts MCNK terrain data from ADT files inside MPQs using Warcraft.NET parser.
/// </summary>
public sealed class AdtMpqTerrainExtractor
{
    public TerrainExtractionResult ExtractFromArchive(IArchiveSource source, string mapName, string outputCsvPath)
    {
        var records = new List<McnkTerrainRecord>();
        int tilesProcessed = 0;
        int tilesSkipped = 0;

        Console.WriteLine($"[AdtMpqTerrainExtractor] Extracting terrain for map: {mapName}");

        // Enumerate all ADT tiles for this map
        // ADT files are at: world/maps/{mapName}/{mapName}_{x}_{y}.adt
        var adtTiles = EnumerateAdtTiles(source, mapName);
        
        if (adtTiles.Count == 0)
        {
            Console.WriteLine($"[AdtMpqTerrainExtractor] No ADT tiles found for map: {mapName}");
            ExportToCsv(records, mapName, outputCsvPath);
            return new TerrainExtractionResult(
                Success: true,
                ChunksExtracted: 0,
                TilesProcessed: 0,
                CsvPath: outputCsvPath);
        }

        Console.WriteLine($"[AdtMpqTerrainExtractor] Found {adtTiles.Count} ADT tiles");

        foreach (var (tileX, tileY) in adtTiles)
        {
            try
            {
                var tileRecords = ExtractTileTerrainData(source, mapName, tileX, tileY);
                records.AddRange(tileRecords);
                tilesProcessed++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AdtMpqTerrainExtractor] Warning: Failed to process tile {tileX}_{tileY}: {ex.Message}");
                tilesSkipped++;
            }
        }

        Console.WriteLine($"[AdtMpqTerrainExtractor] Processed {tilesProcessed} tiles, skipped {tilesSkipped}");

        // Export to CSV
        ExportToCsv(records, mapName, outputCsvPath);

        return new TerrainExtractionResult(
            Success: true,
            ChunksExtracted: records.Count,
            TilesProcessed: tilesProcessed,
            CsvPath: outputCsvPath);
    }

    private List<(int X, int Y)> EnumerateAdtTiles(IArchiveSource source, string mapName)
    {
        var tiles = new List<(int X, int Y)>();

        // Check for ADT files in the map directory
        // Format: world/maps/{mapName}/{mapName}_{x}_{y}.adt
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var adtPath = $"world/maps/{mapName}/{mapName}_{x}_{y}.adt";
                if (source.FileExists(adtPath))
                {
                    tiles.Add((x, y));
                }
            }
        }

        return tiles;
    }

    private List<McnkTerrainRecord> ExtractTileTerrainData(
        IArchiveSource source,
        string mapName,
        int tileX,
        int tileY)
    {
        var records = new List<McnkTerrainRecord>();

        // Read ADT file from MPQ
        var adtPath = $"world/maps/{mapName}/{mapName}_{tileX}_{tileY}.adt";
        byte[] adtData;
        
        using (var stream = source.OpenFile(adtPath))
        using (var ms = new MemoryStream())
        {
            stream.CopyTo(ms);
            adtData = ms.ToArray();
        }

        if (adtData == null || adtData.Length == 0)
        {
            return records;
        }

        try
        {
            // Parse ADT using Warcraft.NET
            // NOTE: Warcraft.NET's Terrain parser is designed for WotLK+ format
            // Alpha 0.6.0 uses preliminary ADT v18 which may not parse correctly
            var terrain = new Terrain(adtData);

            // Extract MCNK chunks (16x16 grid per tile - 256 chunks expected)
            if (terrain.Chunks == null || terrain.Chunks.Length == 0)
            {
                Console.WriteLine($"[AdtMpqTerrainExtractor] WARNING: tile {tileX}_{tileY} has no MCNK chunks (may be unsupported format)");
                return records;
            }

            if (tileX == 0 && tileY == 0)
            {
                // Log first tile for debugging
                Console.WriteLine($"[AdtMpqTerrainExtractor] Tile {tileX}_{tileY}: Found {terrain.Chunks.Length} MCNK chunks");
            }

            foreach (var mcnk in terrain.Chunks)
            {
                if (mcnk?.Header == null)
                    continue;

                records.Add(new McnkTerrainRecord
                {
                    MapName = mapName,
                    TileX = tileX,
                    TileY = tileY,
                    ChunkX = (int)mcnk.Header.MapIndexX,
                    ChunkY = (int)mcnk.Header.MapIndexY,
                    AreaId = (int)mcnk.Header.AreaID,
                    Flags = (uint)mcnk.Header.Flags,
                    TextureLayers = mcnk.TextureLayers?.Layers?.Count ?? 0,
                    HasLiquids = mcnk.Header.LiquidSize > 8,  // LiquidSize > 8 means MCLQ data present
                    HasHoles = mcnk.Header.LowResHoles != 0 || mcnk.Header.HighResHoles != 0,
                    IsImpassible = ((uint)mcnk.Header.Flags & 0x00000001) != 0  // MCNK_IMPASS flag
                });
            }
        }
        catch (Exception ex)
        {
            // Alpha 0.6.0 ADTs use preliminary format that Warcraft.NET may not support
            Console.WriteLine($"[AdtMpqTerrainExtractor] Error parsing {adtPath}: {ex.Message}");
            Console.WriteLine($"[AdtMpqTerrainExtractor] Note: Alpha 0.6.0 uses preliminary ADT v18 format - parser may not support it");
        }

        return records;
    }

    private void ExportToCsv(List<McnkTerrainRecord> records, string mapName, string csvPath)
    {
        var csv = new System.Text.StringBuilder();
        csv.AppendLine("MapName,TileX,TileY,ChunkX,ChunkY,AreaId,Flags,TextureLayers,HasLiquids,HasHoles,IsImpassible");

        foreach (var record in records.OrderBy(r => r.TileY).ThenBy(r => r.TileX).ThenBy(r => r.ChunkY).ThenBy(r => r.ChunkX))
        {
            csv.AppendLine($"{record.MapName},{record.TileX},{record.TileY}," +
                $"{record.ChunkX},{record.ChunkY},{record.AreaId},{record.Flags}," +
                $"{record.TextureLayers},{record.HasLiquids},{record.HasHoles},{record.IsImpassible}");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(csvPath)!);
        File.WriteAllText(csvPath, csv.ToString());
        Console.WriteLine($"[AdtMpqTerrainExtractor] Terrain CSV: {csvPath}");
    }
}
