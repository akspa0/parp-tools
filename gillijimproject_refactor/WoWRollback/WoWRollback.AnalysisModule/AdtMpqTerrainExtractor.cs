using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts terrain data (MCVT heights, MTEX textures, MCLY layers, MCAL alphas) from ADT files
/// and writes per-tile JSON for VLM training ground truth.
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
                // Try both monolithic and split ADT paths
                var adtPath = $"world/maps/{mapName}/{mapName}_{x}_{y}.adt";
                
                if (!source.FileExists(adtPath))
                    continue;
                
                try
                {
                    using var stream = source.OpenFile(adtPath);
                    using var ms = new MemoryStream();
                    stream.CopyTo(ms);
                    var adtBytes = ms.ToArray();
                    
                    var tileData = ExtractTileData(adtBytes, mapName, x, y);
                    
                    if (tileData != null && tileData.Chunks.Count > 0)
                    {
                        // Write terrain JSON
                        var jsonPath = Path.Combine(terrainDir, $"{mapName}_{x}_{y}_terrain.json");
                        var json = JsonSerializer.Serialize(tileData, new JsonSerializerOptions 
                        { 
                            WriteIndented = true,
                            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                        });
                        File.WriteAllText(jsonPath, json);
                        
                        tilesProcessed++;
                        chunksExtracted += tileData.Chunks.Count;
                        
                        if (tilesProcessed == 1)
                        {
                            Console.WriteLine($"  [TerrainExtractor] First tile: {mapName}_{x}_{y} ({tileData.Chunks.Count} chunks, {tileData.Textures.Count} textures)");
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
    
    public TileTerrainData? ExtractTileData(byte[] adtBytes, string mapName, int tileX, int tileY)
    {
        var result = new TileTerrainData
        {
            Map = mapName,
            TileX = tileX,
            TileY = tileY,
            Textures = new List<string>(),
            Chunks = new List<ChunkTerrainData>()
        };
        
        // Parse top-level chunks
        var topChunks = ParseChunks(adtBytes, 0);
        
        // Extract MTEX (texture filenames)
        if (topChunks.TryGetValue("MTEX", out var mtexData))
        {
            result.Textures = ParseMtexFilenames(mtexData);
        }
        
        // Find and parse all MCNKs
        int pos = 0;
        int chunkIdx = 0;
        while (pos < adtBytes.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(adtBytes, pos, 4);
            int size = BitConverter.ToInt32(adtBytes, pos + 4);
            
            if (size < 0 || pos + 8 + size > adtBytes.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            if (readable == "MCNK" && size >= 128)
            {
                var mcnkData = new byte[size];
                Buffer.BlockCopy(adtBytes, pos + 8, mcnkData, 0, size);
                
                var chunkData = ExtractChunkData(mcnkData, chunkIdx);
                if (chunkData != null)
                {
                    result.Chunks.Add(chunkData);
                }
                chunkIdx++;
            }
            
            pos += 8 + size;
        }
        
        return result.Chunks.Count > 0 ? result : null;
    }
    
    private ChunkTerrainData? ExtractChunkData(byte[] mcnkData, int chunkIdx)
    {
        if (mcnkData.Length < 128) return null;
        
        var result = new ChunkTerrainData
        {
            Idx = chunkIdx,
            Heights = new List<float>(),
            Layers = new List<TextureLayerInfo>()
        };
        
        // Parse subchunks within MCNK (starting after 128-byte header)
        var subchunks = ParseChunks(mcnkData, 128);
        
        // MCVT - 145 floats (9*9 + 8*8 = 81+64 = 145)
        if (subchunks.TryGetValue("MCVT", out var mcvtData) && mcvtData.Length >= 145 * 4)
        {
            for (int i = 0; i < 145; i++)
            {
                result.Heights.Add(BitConverter.ToSingle(mcvtData, i * 4));
            }
        }
        
        // MCLY - texture layer info (16 bytes per entry)
        if (subchunks.TryGetValue("MCLY", out var mclyData) && mclyData.Length >= 16)
        {
            int layerCount = mclyData.Length / 16;
            for (int i = 0; i < layerCount; i++)
            {
                int offset = i * 16;
                result.Layers.Add(new TextureLayerInfo
                {
                    TextureId = BitConverter.ToUInt32(mclyData, offset),
                    Flags = BitConverter.ToUInt32(mclyData, offset + 4),
                    OffsetInMcal = BitConverter.ToUInt32(mclyData, offset + 8),
                    EffectId = BitConverter.ToInt32(mclyData, offset + 12)
                });
            }
        }
        
        // MCAL - alpha map data (variable size based on flags)
        if (subchunks.TryGetValue("MCAL", out var mcalData) && mcalData.Length > 0)
        {
            result.Alpha = Convert.ToBase64String(mcalData);
            // Determine encoding type based on flags/size
            result.AlphaEncoding = DetermineAlphaEncoding(mcalData.Length, result.Layers.Count);
        }
        
        return result;
    }
    
    private string DetermineAlphaEncoding(int mcalSize, int layerCount)
    {
        if (layerCount <= 1) return "noAlpha";
        
        int expectedUncompressed = 2048 * (layerCount - 1); // 64x32 per layer (4-bit packed)
        int expectedBigAlpha = 4096 * (layerCount - 1);     // 64x64 per layer (8-bit)
        
        if (mcalSize == expectedBigAlpha || mcalSize > expectedUncompressed)
            return "bigAlpha";
        if (mcalSize == expectedUncompressed)
            return "uncompressed";
        
        return "compressed"; // RLE or other format
    }
    
    private Dictionary<string, byte[]> ParseChunks(byte[] data, int startOffset)
    {
        var result = new Dictionary<string, byte[]>();
        int pos = startOffset;
        
        while (pos < data.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);
            
            if (size < 0 || size > 10_000_000 || pos + 8 + size > data.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            // Store first occurrence of each non-MCNK chunk
            if (!result.ContainsKey(readable) && readable != "MCNK")
            {
                var chunkData = new byte[size];
                Buffer.BlockCopy(data, pos + 8, chunkData, 0, size);
                result[readable] = chunkData;
            }
            
            pos += 8 + size;
        }
        
        return result;
    }
    
    private List<string> ParseMtexFilenames(byte[] mtexData)
    {
        var result = new List<string>();
        int start = 0;
        
        for (int i = 0; i < mtexData.Length; i++)
        {
            if (mtexData[i] == 0)
            {
                if (i > start)
                {
                    var filename = Encoding.ASCII.GetString(mtexData, start, i - start);
                    if (!string.IsNullOrWhiteSpace(filename))
                        result.Add(filename);
                }
                start = i + 1;
            }
        }
        
        return result;
    }
}

// Data models for terrain JSON output
public class TileTerrainData
{
    public string Map { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public List<string> Textures { get; set; } = new();
    public List<ChunkTerrainData> Chunks { get; set; } = new();
}

public class ChunkTerrainData
{
    public int Idx { get; set; }
    public List<float> Heights { get; set; } = new();
    public List<TextureLayerInfo> Layers { get; set; } = new();
    public string? AlphaEncoding { get; set; }
    public string? Alpha { get; set; } // Base64 encoded MCAL data
}

public class TextureLayerInfo
{
    public uint TextureId { get; set; }
    public uint Flags { get; set; }
    public uint OffsetInMcal { get; set; }
    public int EffectId { get; set; }
}
