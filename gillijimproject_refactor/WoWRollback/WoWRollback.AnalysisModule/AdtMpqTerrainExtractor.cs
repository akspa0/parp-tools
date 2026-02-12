using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using WoWRollback.Core.Services.Archive;
using WoWRollback.PM4Module;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts terrain data from ADT files using PM4Module's proven parser.
/// Writes per-tile JSON with heights, textures, layers, and alpha data.
/// </summary>
public sealed class AdtMpqTerrainExtractor
{
    public TerrainExtractionResult ExtractFromArchive(IArchiveSource source, string mapName, string outputDir)
    {
        var terrainDir = Path.Combine(outputDir, "terrain");
        Directory.CreateDirectory(terrainDir);
        
        int tilesProcessed = 0;
        int chunksExtracted = 0;
        
        // Scan 64x64 tile grid
        for (int x = 0; x < 64; x++)
        {
            for (int y = 0; y < 64; y++)
            {
                // Try standard path naming
                var adtPath = $"world/maps/{mapName}/{mapName}_{x}_{y}.adt";
                
                if (!source.FileExists(adtPath))
                    continue;
                
                try
                {
                    using var stream = source.OpenFile(adtPath);
                    using var ms = new MemoryStream();
                    stream.CopyTo(ms);
                    var adtBytes = ms.ToArray();
                    
                    // Use PM4Module's static terrain parser
                    var tileData = AdtTerrainParser.Parse(adtBytes, mapName, x, y);
                    
                    if (tileData != null && tileData.Chunks.Count > 0)
                    {
                        // Convert to JSON-serializable format
                        var jsonData = ConvertToJsonFormat(tileData);
                        
                        var jsonPath = Path.Combine(terrainDir, $"{mapName}_{x}_{y}_terrain.json");
                        var json = JsonSerializer.Serialize(jsonData, new JsonSerializerOptions 
                        { 
                            WriteIndented = true,
                            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                        });
                        File.WriteAllText(jsonPath, json);
                        
                        tilesProcessed++;
                        chunksExtracted += tileData.Chunks.Count;
                        
                        if (tilesProcessed == 1)
                        {
                            var layerCount = tileData.Chunks.Sum(c => c.Layers?.Count ?? 0);
                            var alphaCount = tileData.Chunks.Count(c => c.AlphaRaw != null && c.AlphaRaw.Length > 0);
                            Console.WriteLine($"  [TerrainExtractor] First tile: {mapName}_{x}_{y} ({tileData.Chunks.Count} chunks, {tileData.Textures.Count} textures, {layerCount} total layers, {alphaCount} chunks with alpha)");
                        }
                        
                        if (tilesProcessed % 100 == 0)
                        {
                            Console.WriteLine($"  [TerrainExtractor] Processed {tilesProcessed} tiles...");
                        }
                    }
                }
                catch (Exception ex)
                {
                    if (tilesProcessed < 3)
                    {
                        Console.WriteLine($"  [TerrainExtractor] Warning: {mapName}_{x}_{y}: {ex.Message}");
                    }
                }
            }
        }
        
        return new TerrainExtractionResult(
            Success: true,
            ChunksExtracted: chunksExtracted,
            TilesProcessed: tilesProcessed,
            CsvPath: terrainDir);
    }
    
    /// <summary>
    /// Convert PM4Module's TileTerrainData to a JSON-friendly format with proper serialization.
    /// </summary>
    private object ConvertToJsonFormat(TileTerrainData tileData)
    {
        var chunks = new List<object>();
        
        foreach (var chunk in tileData.Chunks)
        {
            var layers = new List<object>();
            if (chunk.Layers != null)
            {
                foreach (var layer in chunk.Layers)
                {
                    layers.Add(new
                    {
                        textureId = layer.TextureId,
                        flags = layer.Flags,
                        alphaOffset = layer.AlphaOffset
                    });
                }
            }
            
            chunks.Add(new
            {
                idx = chunk.Idx,
                heights = chunk.Heights,
                layers = layers,
                alphaEncoding = DetermineAlphaEncoding(chunk.AlphaRaw?.Length ?? 0, chunk.Layers?.Count ?? 0),
                alpha = chunk.AlphaRaw != null ? Convert.ToBase64String(chunk.AlphaRaw) : null
            });
        }
        
        return new
        {
            map = tileData.Map,
            tileX = tileData.TileX,
            tileY = tileData.TileY,
            textures = tileData.Textures,
            chunks = chunks
        };
    }
    
    private string? DetermineAlphaEncoding(int mcalSize, int layerCount)
    {
        if (mcalSize == 0) return null;
        if (layerCount <= 1) return "noAlpha";
        
        int expectedUncompressed = 2048 * (layerCount - 1); // 64x32 per layer (4-bit packed)
        int expectedBigAlpha = 4096 * (layerCount - 1);     // 64x64 per layer (8-bit)
        
        if (mcalSize == expectedBigAlpha || mcalSize > expectedUncompressed)
            return "bigAlpha";
        if (mcalSize == expectedUncompressed)
            return "uncompressed";
        
        return "compressed"; // RLE or other format
    }
}
