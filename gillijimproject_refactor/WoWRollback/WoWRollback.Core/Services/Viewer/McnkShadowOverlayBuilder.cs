using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds shadow map overlay JSON files for the 3D viewer.
/// Converts CSV shadow data to per-tile JSON overlays.
/// </summary>
public static class McnkShadowOverlayBuilder
{
    public static void BuildOverlaysForMap(
        string mapName,
        string csvDir,
        string outputDir,
        string version)
    {
        var shadowCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_shadows.csv");
        if (!File.Exists(shadowCsvPath))
        {
            // No shadow data - silently skip
            return;
        }

        Console.WriteLine($"[shadow] Building shadow overlays for {mapName} ({version})");

        var shadows = ReadShadowCsv(shadowCsvPath);
        if (shadows.Count == 0)
        {
            Console.WriteLine($"[shadow] No shadow data found in CSV");
            return;
        }

        var byTile = shadows
            .Where(s => s.HasShadow) // Only tiles with actual shadow data
            .GroupBy(s => (s.TileRow, s.TileCol));

        int tileCount = 0;
        foreach (var tileGroup in byTile)
        {
            var (tileRow, tileCol) = tileGroup.Key;
            var chunks = tileGroup.ToList();

            // Build JSON overlay
            var overlay = new
            {
                type = "shadow_map",
                version = version,
                map = mapName,
                chunks = chunks.Select(c => new
                {
                    y = c.ChunkY,
                    x = c.ChunkX,
                    // Convert digit string to 64×64 array
                    shadow = ParseShadowDigits(c.ShadowMap)
                })
            };

            var json = JsonSerializer.Serialize(overlay, new JsonSerializerOptions
            {
                WriteIndented = false  // Compact for smaller files
            });

            var overlayDir = Path.Combine(outputDir, mapName, "shadow_map");
            Directory.CreateDirectory(overlayDir);

            var outPath = Path.Combine(overlayDir, $"tile_r{tileRow}_c{tileCol}.json");
            File.WriteAllText(outPath, json);
            tileCount++;
        }

        Console.WriteLine($"[shadow] Built {tileCount} shadow overlay tiles for {mapName} ({version})");
    }

    private static int[][] ParseShadowDigits(string digits)
    {
        if (string.IsNullOrEmpty(digits) || digits.Length != 4096)
        {
            // Return all-lit map as fallback
            var allLit = new int[64][];
            for (int y = 0; y < 64; y++)
            {
                allLit[y] = Enumerable.Repeat(5, 64).ToArray();
            }
            return allLit;
        }

        var result = new int[64][];
        for (int y = 0; y < 64; y++)
        {
            result[y] = new int[64];
            for (int x = 0; x < 64; x++)
            {
                int index = y * 64 + x;
                result[y][x] = digits[index] - '0';  // '0'→0, '5'→5
            }
        }
        return result;
    }

    private static List<ShadowEntry> ReadShadowCsv(string path)
    {
        var entries = new List<ShadowEntry>();

        using var reader = new StreamReader(path);

        // Skip header
        reader.ReadLine();

        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;

            var parts = line.Split(',');
            if (parts.Length < 6) continue;

            try
            {
                entries.Add(new ShadowEntry(
                    MapName: parts[0],
                    TileRow: int.Parse(parts[1]),
                    TileCol: int.Parse(parts[2]),
                    ChunkY: int.Parse(parts[3]),
                    ChunkX: int.Parse(parts[4]),
                    HasShadow: parts.Length > 5 && !string.IsNullOrEmpty(parts[5]),
                    ShadowMap: parts.Length > 5 ? parts[5] : string.Empty
                ));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[shadow:warn] Failed to parse CSV line: {ex.Message}");
            }
        }

        return entries;
    }

    private record ShadowEntry(
        string MapName,
        int TileRow,
        int TileCol,
        int ChunkY,
        int ChunkX,
        bool HasShadow,
        string ShadowMap
    );
}
