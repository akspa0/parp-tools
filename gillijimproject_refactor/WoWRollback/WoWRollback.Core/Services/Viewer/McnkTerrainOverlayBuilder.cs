using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds complete MCNK terrain overlays from CSV data
/// Generates JSON files for viewer consumption
/// </summary>
public static class McnkTerrainOverlayBuilder
{
    public static void BuildOverlaysForMap(
        string mapName,
        string csvDir,
        string outputDir,
        string version,
        AreaTableLookup? areaLookup = null,
        string? cachedMapsDir = null)
    {
        List<McnkTerrainEntry> allChunks;

        if (!string.IsNullOrWhiteSpace(cachedMapsDir) && Directory.Exists(cachedMapsDir))
        {
            allChunks = LkAdtTerrainReader.ReadFromLkAdts(cachedMapsDir, version, mapName);

            if (allChunks.Count == 0)
            {
                Console.WriteLine($"[terrain] No LK ADT data for {mapName}, falling back to CSV");
                allChunks = ReadCsvFallback(csvDir, mapName);
            }
        }
        else
        {
            allChunks = ReadCsvFallback(csvDir, mapName);
        }

        if (allChunks.Count == 0)
        {
            Console.WriteLine($"No terrain data for {mapName}, skipping");
            return;
        }

        // Group by tile
        var byTile = allChunks.GroupBy(c => (c.TileRow, c.TileCol));

        int tileCount = 0;
        foreach (var tileGroup in byTile)
        {
            var (tileRow, tileCol) = tileGroup.Key;
            var chunks = tileGroup.ToList();

            // Build all overlay types
            var terrainProps = TerrainPropertiesOverlayBuilder.Build(chunks, version);
            var liquids = LiquidsOverlayBuilder.Build(chunks, version);
            var holes = HolesOverlayBuilder.Build(chunks, version);

            // AreaID overlay (only if area lookup available)
            object? areaIds = null;
            if (areaLookup != null)
            {
                areaIds = AreaIdOverlayBuilder.Build(chunks, version, areaLookup);
            }

            // Combine into single JSON structure
            var combined = new
            {
                map = mapName,
                tile = new { row = tileRow, col = tileCol },
                chunk_size = 32, // Game units per chunk side
                minimap = new { width = 512, height = 512 }, // Minimap dimensions
                layers = new[]
                {
                    new
                    {
                        version,
                        terrain_properties = terrainProps,
                        liquids,
                        holes,
                        area_ids = areaIds
                    }
                }
            };

            // Write to JSON file
            var overlayDir = Path.Combine(outputDir, mapName, "terrain_complete");
            Directory.CreateDirectory(overlayDir);

            var outPath = Path.Combine(overlayDir, $"tile_r{tileRow}_c{tileCol}.json");
            var json = JsonSerializer.Serialize(combined, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(outPath, json);

            tileCount++;
        }

        Console.WriteLine($"Built {tileCount} terrain overlay tiles for {mapName} ({version})");
    }

    private static List<McnkTerrainEntry> ReadCsvFallback(string csvDir, string mapName)
    {
        var terrainCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_terrain.csv");
        if (!File.Exists(terrainCsvPath))
        {
            return new List<McnkTerrainEntry>();
        }

        return McnkTerrainCsvReader.ReadCsv(terrainCsvPath);
    }
}
