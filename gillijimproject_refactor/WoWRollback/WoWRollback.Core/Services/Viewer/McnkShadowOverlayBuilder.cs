using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
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

            var overlayDir = Path.Combine(outputDir, mapName, "shadow_map");
            Directory.CreateDirectory(overlayDir);

            var overviewPngPath = Path.Combine(overlayDir, $"tile_r{tileRow}_c{tileCol}.png");
            using (var overview = RenderTileShadow(chunks))
            {
                overview.Save(overviewPngPath);
            }

            var chunkDir = Path.Combine(overlayDir, "chunks", $"tile_r{tileRow}_c{tileCol}");
            Directory.CreateDirectory(chunkDir);
            foreach (var chunk in chunks)
            {
                var chunkPngPath = Path.Combine(chunkDir, $"chunk_r{chunk.ChunkY}_c{chunk.ChunkX}.png");
                using var chunkImage = RenderChunkShadow(chunk.ShadowMap, chunk.ApplyEdgeFix);
                chunkImage.Save(chunkPngPath);
            }

            var metadata = new
            {
                type = "shadow_map",
                version,
                map = mapName,
                tile = new { row = tileRow, col = tileCol },
                overview = $"tile_r{tileRow}_c{tileCol}.png",
                chunks = chunks.Select(chunk => new
                {
                    row = chunk.ChunkY,
                    col = chunk.ChunkX,
                    path = $"chunks/tile_r{tileRow}_c{tileCol}/chunk_r{chunk.ChunkY}_c{chunk.ChunkX}.png"
                }).ToArray()
            };

            var metadataPath = Path.Combine(overlayDir, $"tile_r{tileRow}_c{tileCol}.json");
            File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, new JsonSerializerOptions
            {
                WriteIndented = true
            }));

            tileCount++;
        }

        Console.WriteLine($"[shadow] Built {tileCount} shadow overlay tiles for {mapName} ({version})");
    }

    private static Image<Rgba32> RenderTileShadow(List<ShadowEntry> chunks)
    {
        var image = new Image<Rgba32>(1024, 1024, Color.Transparent);

        foreach (var chunk in chunks)
        {
            using var chunkImage = RenderChunkShadow(chunk.ShadowMap, chunk.ApplyEdgeFix);
            var location = new Point(chunk.ChunkX * 64, chunk.ChunkY * 64);
            image.Mutate(ctx => ctx.DrawImage(chunkImage, location, 1f));
        }

        return image;
    }

    private static Image<Rgba32> RenderChunkShadow(string digits, bool applyEdgeFix)
    {
        var image = new Image<Rgba32>(64, 64, Color.Transparent);

        if (string.IsNullOrWhiteSpace(digits))
        {
            return image;
        }

        var shadowMap = DecodeDigits(digits);

        if (applyEdgeFix)
        {
            ApplyEdgeFix(shadowMap);
        }

        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var index = y * 64 + x;
                var value = shadowMap[index];
                image[x, y] = value switch
                {
                    ShadowValueLit => new Rgba32(0, 0, 0, 0),
                    ShadowValueShadowed => new Rgba32(0, 0, 0, 170),
                    _ => new Rgba32(0, 0, 0, value)
                };
            }
        }

        return image;
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
            if (parts.Length < 8) continue;

            try
            {
                var hasShadow = bool.TryParse(parts[5], out var parsedHasShadow) && parsedHasShadow;
                var shadowSize = int.TryParse(parts[6], out var parsedSize) ? parsedSize : 0;
                var shadowBase64 = parts.Length > 7 ? parts[7] : string.Empty;
                var applyEdgeFix = shadowSize == 512; // Matches Noggit behaviour

                entries.Add(new ShadowEntry(
                    MapName: parts[0],
                    TileRow: int.Parse(parts[1]),
                    TileCol: int.Parse(parts[2]),
                    ChunkY: int.Parse(parts[3]),
                    ChunkX: int.Parse(parts[4]),
                    HasShadow: hasShadow,
                    ShadowSize: shadowSize,
                    ShadowMap: shadowBase64,
                    ApplyEdgeFix: applyEdgeFix
                ));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[shadow:warn] Failed to parse CSV line: {ex.Message}");
            }
        }

        return entries;
    }

    private static byte[] DecodeDigits(string digits)
    {
        var buffer = new byte[4096];
        if (string.IsNullOrWhiteSpace(digits))
        {
            return buffer;
        }

        int writeIndex = 0;
        foreach (var ch in digits)
        {
            if (char.IsWhiteSpace(ch))
            {
                continue;
            }

            buffer[writeIndex++] = ch == '0' ? ShadowValueShadowed : ShadowValueLit;
            if (writeIndex == buffer.Length)
            {
                break;
            }
        }

        // Any remaining slots default to lit (transparent) pixels
        for (; writeIndex < buffer.Length; writeIndex++)
        {
            buffer[writeIndex] = ShadowValueLit;
        }

        return buffer;
    }

    private static void ApplyEdgeFix(byte[] shadowMap)
    {
        for (int i = 0; i < 64; i++)
        {
            shadowMap[i * 64 + 63] = shadowMap[i * 64 + 62];
            shadowMap[63 * 64 + i] = shadowMap[62 * 64 + i];
        }
        shadowMap[63 * 64 + 63] = shadowMap[62 * 64 + 62];
    }

    private const byte ShadowValueShadowed = 0;
    private const byte ShadowValueLit = 85;

    private record ShadowEntry(
        string MapName,
        int TileRow,
        int TileCol,
        int ChunkY,
        int ChunkX,
        bool HasShadow,
        int ShadowSize,
        string ShadowMap,
        bool ApplyEdgeFix
    );
}
