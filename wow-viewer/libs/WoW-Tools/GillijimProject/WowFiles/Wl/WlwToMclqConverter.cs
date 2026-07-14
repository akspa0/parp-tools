using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GillijimProject.WowFiles.Wl;

/// <summary>
/// Converts WL* files (WLW/WLM/WLQ) to Alpha MCLQ format for restoring missing liquids.
/// Maps WL blocks to ADT tiles using vertex world positions.
/// </summary>
public class WlwToMclqConverter
{
    private const float TileSize = 533.333f;
    private const float MapSize = 17066.666f;
    private const float ChunkSize = TileSize / 16f; // ~33.33 units per chunk
    
    /// <summary>
    /// Result of converting a WLW file, grouped by ADT tile.
    /// </summary>
    public class ConversionResult
    {
        public string SourceFile { get; set; } = "";
        public ushort LiquidType { get; set; }
        public Dictionary<(int tileX, int tileY), List<MclqChunkData>> TileData { get; } = new();
    }
    
    /// <summary>
    /// MCLQ chunk data for a specific location within an ADT tile.
    /// </summary>
    public class MclqChunkData
    {
        public int ChunkX { get; set; } // 0-15 within tile
        public int ChunkY { get; set; } // 0-15 within tile
        public float MinHeight { get; set; }
        public float MaxHeight { get; set; }
        public float[] Heights { get; set; } = new float[81]; // 9x9 grid
        public byte[] TileFlags { get; set; } = new byte[64]; // 8x8 tiles
        public byte LiquidType { get; set; }
    }
    
    /// <summary>
    /// Maps unified LiquidType enum to Alpha MCLQ type byte.
    /// </summary>
    private static byte MapLiquidTypeEnum(LiquidType type)
    {
        return type switch
        {
            LiquidType.StillWater => 0x01,
            LiquidType.Ocean => 0x02,
            LiquidType.River => 0x01,
            LiquidType.Magma => 0x03,
            LiquidType.Slime => 0x04,
            LiquidType.FastWater => 0x01,
            _ => 0x01
        };
    }
    
    /// <summary>
    /// Converts a WLW file to MCLQ data grouped by ADT tiles.
    /// </summary>
    public ConversionResult Convert(WlFile wlFile, string sourceFileName)
    {
        var result = new ConversionResult
        {
            SourceFile = sourceFileName,
            LiquidType = wlFile.Header.RawLiquidType
        };
        
        byte mclqType = MapLiquidTypeEnum(wlFile.Header.LiquidType);
        
        foreach (var block in wlFile.Blocks)
        {
            // Get world position from first vertex
            var worldPos = block.Vertices[0];
            
            // Convert to ADT tile index
            // WoW: Y increases north, X increases east
            // Our observed pattern: vx=X, vy=Y (world coords)
            int tileX = (int)Math.Floor((MapSize - worldPos.Y) / TileSize);
            int tileY = (int)Math.Floor((MapSize - worldPos.X) / TileSize);
            
            // Clamp to valid range
            tileX = Math.Clamp(tileX, 0, 63);
            tileY = Math.Clamp(tileY, 0, 63);
            
            // Calculate chunk index within tile (0-15)
            float localX = (MapSize - worldPos.Y) - (tileX * TileSize);
            float localY = (MapSize - worldPos.X) - (tileY * TileSize);
            int chunkX = Math.Clamp((int)(localX / ChunkSize), 0, 15);
            int chunkY = Math.Clamp((int)(localY / ChunkSize), 0, 15);
            
            // Generate MCLQ data for this block
            var mclqData = GenerateMclqChunk(block, mclqType, chunkX, chunkY);
            
            var tileKey = (tileX, tileY);
            if (!result.TileData.ContainsKey(tileKey))
                result.TileData[tileKey] = new List<MclqChunkData>();
            
            result.TileData[tileKey].Add(mclqData);
        }
        
        return result;
    }
    
    /// <summary>
    /// Generates MCLQ chunk data from a WLW block.
    /// Upscales 4x4 vertex grid to 9x9 MCLQ height grid.
    /// </summary>
    private MclqChunkData GenerateMclqChunk(WlBlock block, byte liquidType, int chunkX, int chunkY)
    {
        var mclq = new MclqChunkData
        {
            ChunkX = chunkX,
            ChunkY = chunkY,
            LiquidType = liquidType
        };
        
        // Upscale 4x4 to 9x9 using bilinear interpolation
        // WLW vertex layout (from docs): starts at lower-right, indices 15..0
        // Rearrange to standard row-major order
        var heights4x4 = new float[16];
        for (int i = 0; i < 16; i++)
        {
            // Vertex Z is height (z-up per docs)
            heights4x4[15 - i] = block.Vertices[i].Z;
        }
        
        // Bilinear interpolation from 4x4 to 9x9
        float min = float.MaxValue, max = float.MinValue;
        for (int y = 0; y < 9; y++)
        {
            float v = (y / 8.0f) * 3.0f; // Map [0,8] to [0,3]
            for (int x = 0; x < 9; x++)
            {
                float u = (x / 8.0f) * 3.0f;
                float h = BilinearSample(heights4x4, u, v);
                mclq.Heights[y * 9 + x] = h;
                min = Math.Min(min, h);
                max = Math.Max(max, h);
            }
        }
        
        mclq.MinHeight = min;
        mclq.MaxHeight = max;
        
        // Set all tiles as visible liquid (0x0F = all 4 corner flags set)
        // Data[80] could refine this, but for now assume full coverage
        for (int i = 0; i < 64; i++)
        {
            mclq.TileFlags[i] = (byte)(liquidType | 0x00); // Type + visibility
        }
        
        return mclq;
    }
    
    private static float BilinearSample(float[] grid4x4, float u, float v)
    {
        // Grid is 4x4 (indices 0-3). u,v in [0,3].
        int x0 = (int)Math.Floor(u);
        int y0 = (int)Math.Floor(v);
        int x1 = Math.Min(x0 + 1, 3);
        int y1 = Math.Min(y0 + 1, 3);
        
        float tx = u - x0;
        float ty = v - y0;
        
        float h00 = grid4x4[y0 * 4 + x0];
        float h10 = grid4x4[y0 * 4 + x1];
        float h01 = grid4x4[y1 * 4 + x0];
        float h11 = grid4x4[y1 * 4 + x1];
        
        float lerpX0 = h00 + (h10 - h00) * tx;
        float lerpX1 = h01 + (h11 - h01) * tx;
        return lerpX0 + (lerpX1 - lerpX0) * ty;
    }
    
    /// <summary>
    /// Serializes MCLQ chunk data to binary format for Alpha ADT.
    /// </summary>
    public static byte[] SerializeMclqChunk(MclqChunkData data)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        
        // Min/Max heights
        bw.Write(data.MinHeight);
        bw.Write(data.MaxHeight);
        
        // 9x9 vertices (81 entries)
        // Format depends on liquid type (water vs magma)
        if (data.LiquidType == 0x03) // Magma
        {
            // Magma format: s(u16), t(u16), height(f32) = 8 bytes each
            for (int i = 0; i < 81; i++)
            {
                bw.Write((ushort)0); // s
                bw.Write((ushort)0); // t
                bw.Write(data.Heights[i]);
            }
        }
        else // Water/Ocean
        {
            // Water format: depth(u8), flow0(u8), flow1(u8), filler(u8), height(f32) = 8 bytes each
            for (int i = 0; i < 81; i++)
            {
                bw.Write((byte)128); // depth (0=surface, 255=deep)
                bw.Write((byte)0);   // flow data
                bw.Write((byte)0);
                bw.Write((byte)0);   // filler
                bw.Write(data.Heights[i]);
            }
        }
        
        // 8x8 tile flags (64 bytes)
        bw.Write(data.TileFlags);
        
        return ms.ToArray();
    }
}
