using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts MCNK terrain data from ADT files for terrain overlay generation.
/// Supports both pre-Cataclysm monolithic ADTs and Cataclysm+ split files.
/// </summary>
public sealed class AdtTerrainExtractor
{
    /// <summary>
    /// Extracts terrain data for all tiles in a map.
    /// </summary>
    public TerrainExtractionResult ExtractTerrainForMap(
        string mapDirectory,
        string mapName,
        string outputDir)
    {
        var tiles = AdtFormatDetector.EnumerateMapTiles(mapDirectory, mapName);
        var allRecords = new List<McnkTerrainRecord>();
        int tilesProcessed = 0;
        int tilesSkipped = 0;

        Console.WriteLine($"[AdtTerrainExtractor] Found {tiles.Count} ADT tiles in {mapDirectory}");
        
        if (tiles.Count == 0)
        {
            Console.WriteLine($"[AdtTerrainExtractor] No ADT tiles found - check mapDirectory path");
            var emptyCsvPath = Path.Combine(outputDir, $"{mapName}_terrain.csv");
            ExportToCsv(allRecords, mapName, emptyCsvPath); // Write empty CSV
            return new TerrainExtractionResult(Success: true, ChunksExtracted: 0, TilesProcessed: 0, CsvPath: emptyCsvPath);
        }
        
        Console.WriteLine($"[AdtTerrainExtractor] Extracting terrain from {tiles.Count} tiles...");

        foreach (var (tileX, tileY, format) in tiles)
        {
            try
            {
                var records = ExtractTileTerrainData(mapDirectory, mapName, tileX, tileY, format);
                allRecords.AddRange(records);
                tilesProcessed++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AdtTerrainExtractor] Warning: Failed to process tile {tileX}_{tileY}: {ex.Message}");
                tilesSkipped++;
            }
        }

        Console.WriteLine($"[AdtTerrainExtractor] Processed {tilesProcessed} tiles, skipped {tilesSkipped}");

        // Export to CSV
        var csvPath = Path.Combine(outputDir, $"{mapName}_terrain.csv");
        ExportToCsv(allRecords, mapName, csvPath);

        return new TerrainExtractionResult(
            Success: true,
            ChunksExtracted: allRecords.Count,
            TilesProcessed: tilesProcessed,
            CsvPath: csvPath);
    }

    private List<McnkTerrainRecord> ExtractTileTerrainData(
        string mapDirectory,
        string mapName,
        int tileX,
        int tileY,
        AdtFormatDetector.AdtFormat format)
    {
        var records = new List<McnkTerrainRecord>();

        // Determine ADT file path based on format
        string adtPath;
        if (format == AdtFormatDetector.AdtFormat.Cataclysm)
        {
            // Cataclysm+: use _tex0.adt for terrain data
            adtPath = Path.Combine(mapDirectory, $"{mapName}_{tileX}_{tileY}_tex0.adt");
        }
        else
        {
            // Pre-Cataclysm: monolithic ADT
            adtPath = Path.Combine(mapDirectory, $"{mapName}_{tileX}_{tileY}.adt");
        }

        if (!File.Exists(adtPath))
        {
            return records;
        }

        try
        {
            // Read ADT file
            var adtData = File.ReadAllBytes(adtPath);
            var terrain = new Terrain(adtData);

            // Extract MCNK chunks (16x16 grid per tile - 256 chunks expected)
            if (terrain.Chunks == null || terrain.Chunks.Length == 0)
            {
                Console.WriteLine($"[AdtTerrainExtractor] WARNING: tile {tileX}_{tileY} has no MCNK chunks! (Chunks={terrain.Chunks?.Length ?? 0})");
                return records;
            }
            
            Console.WriteLine($"[AdtTerrainExtractor] Tile {tileX}_{tileY}: Found {terrain.Chunks.Length} MCNK chunks");

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
            Console.WriteLine($"[AdtTerrainExtractor] Error reading {adtPath}: {ex.Message}");
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
        Console.WriteLine($"[AdtTerrainExtractor] Terrain CSV: {csvPath}");
    }
}

/// <summary>
/// Result of terrain extraction operation.
/// </summary>
public sealed record TerrainExtractionResult(
    bool Success,
    int ChunksExtracted,
    int TilesProcessed,
    string CsvPath);
