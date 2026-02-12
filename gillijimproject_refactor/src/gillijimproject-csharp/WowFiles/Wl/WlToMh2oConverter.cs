using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GillijimProject.WowFiles.Wl;

/// <summary>
/// Converts WL* files to WotLK MH2O format for liquid restoration in 3.3.5.
/// MH2O is the modern liquid chunk format used in WotLK ADT files.
/// </summary>
public class WlToMh2oConverter
{
    private const float TileSize = 533.333f;
    private const float MapSize = 17066.666f;
    private const float ChunkSize = TileSize / 16f;

    /// <summary>
    /// Result of converting a WL file to MH2O format, grouped by ADT tile.
    /// </summary>
    public class Mh2oConversionResult
    {
        public string SourceFile { get; set; } = "";
        public LiquidType LiquidType { get; set; }
        public Dictionary<(int tileX, int tileY), Mh2oTileData> TileData { get; } = new();
    }

    /// <summary>
    /// MH2O data for an entire ADT tile (16x16 chunks).
    /// </summary>
    public class Mh2oTileData
    {
        public Mh2oChunkData[,] Chunks { get; } = new Mh2oChunkData[16, 16];
        public int ChunkCount => Chunks.Cast<Mh2oChunkData>().Count(c => c != null);
    }

    /// <summary>
    /// MH2O data for a single chunk within an ADT tile.
    /// </summary>
    public class Mh2oChunkData
    {
        public ushort LiquidTypeId { get; set; }  // DB/LiquidType ID
        public ushort Flags { get; set; }
        public float MinHeight { get; set; }
        public float MaxHeight { get; set; }
        public byte XOffset { get; set; }
        public byte YOffset { get; set; }
        public byte Width { get; set; } = 8;
        public byte Height { get; set; } = 8;
        public float[] Heights { get; set; } = new float[81]; // 9x9
        public ulong RenderBitmask { get; set; } = ulong.MaxValue; // All tiles visible
    }

    /// <summary>
    /// Maps unified LiquidType to WotLK MH2O LiquidTypeId (DB/LiquidType).
    /// </summary>
    private static ushort GetMh2oLiquidTypeId(LiquidType type)
    {
        return type switch
        {
            LiquidType.StillWater => 14,  // DB ID for still water
            LiquidType.Ocean => 17,        // Ocean water
            LiquidType.River => 13,        // River
            LiquidType.Magma => 19,        // Magma
            LiquidType.Slime => 20,        // Slime
            LiquidType.FastWater => 13,    // Fast flowing = river
            _ => 14
        };
    }

    /// <summary>
    /// Converts a WL file to MH2O format grouped by ADT tiles.
    /// </summary>
    public Mh2oConversionResult Convert(WlFile wlFile, string sourceFileName)
    {
        var result = new Mh2oConversionResult
        {
            SourceFile = sourceFileName,
            LiquidType = wlFile.Header.LiquidType
        };

        ushort liquidTypeId = GetMh2oLiquidTypeId(wlFile.Header.LiquidType);

        foreach (var block in wlFile.Blocks)
        {
            var worldPos = block.Vertices[0];

            // Convert to ADT tile index
            int tileX = Math.Clamp((int)Math.Floor((MapSize - worldPos.Y) / TileSize), 0, 63);
            int tileY = Math.Clamp((int)Math.Floor((MapSize - worldPos.X) / TileSize), 0, 63);

            // Calculate chunk index within tile
            float localX = (MapSize - worldPos.Y) - (tileX * TileSize);
            float localY = (MapSize - worldPos.X) - (tileY * TileSize);
            int chunkX = Math.Clamp((int)(localX / ChunkSize), 0, 15);
            int chunkY = Math.Clamp((int)(localY / ChunkSize), 0, 15);

            var tileKey = (tileX, tileY);
            if (!result.TileData.ContainsKey(tileKey))
                result.TileData[tileKey] = new Mh2oTileData();

            var tileData = result.TileData[tileKey];

            // Generate MH2O chunk if not already present
            if (tileData.Chunks[chunkX, chunkY] == null)
            {
                tileData.Chunks[chunkX, chunkY] = GenerateMh2oChunk(block, liquidTypeId);
            }
            else
            {
                // Merge overlapping blocks
                MergeMh2oChunk(tileData.Chunks[chunkX, chunkY], block);
            }
        }

        return result;
    }

    private Mh2oChunkData GenerateMh2oChunk(WlBlock block, ushort liquidTypeId)
    {
        var chunk = new Mh2oChunkData
        {
            LiquidTypeId = liquidTypeId,
            Flags = 0,
            XOffset = 0,
            YOffset = 0,
            Width = 8,
            Height = 8
        };

        // Upscale 4x4 to 9x9 using bilinear interpolation
        var heights4x4 = new float[16];
        for (int i = 0; i < 16; i++)
            heights4x4[15 - i] = block.Vertices[i].Z;

        float min = float.MaxValue, max = float.MinValue;
        for (int y = 0; y < 9; y++)
        {
            float v = (y / 8.0f) * 3.0f;
            for (int x = 0; x < 9; x++)
            {
                float u = (x / 8.0f) * 3.0f;
                float h = BilinearSample(heights4x4, u, v);
                chunk.Heights[y * 9 + x] = h;
                min = Math.Min(min, h);
                max = Math.Max(max, h);
            }
        }

        chunk.MinHeight = min;
        chunk.MaxHeight = max;
        chunk.RenderBitmask = ulong.MaxValue; // All 64 tiles visible

        return chunk;
    }

    private void MergeMh2oChunk(Mh2oChunkData existing, WlBlock newBlock)
    {
        // Average heights from overlapping blocks
        var heights4x4 = new float[16];
        for (int i = 0; i < 16; i++)
            heights4x4[15 - i] = newBlock.Vertices[i].Z;

        for (int y = 0; y < 9; y++)
        {
            float v = (y / 8.0f) * 3.0f;
            for (int x = 0; x < 9; x++)
            {
                float u = (x / 8.0f) * 3.0f;
                float h = BilinearSample(heights4x4, u, v);
                // Average with existing
                existing.Heights[y * 9 + x] = (existing.Heights[y * 9 + x] + h) / 2;
            }
        }

        existing.MinHeight = existing.Heights.Min();
        existing.MaxHeight = existing.Heights.Max();
    }

    private static float BilinearSample(float[] grid4x4, float u, float v)
    {
        int x0 = (int)Math.Floor(u), y0 = (int)Math.Floor(v);
        int x1 = Math.Min(x0 + 1, 3), y1 = Math.Min(y0 + 1, 3);
        float tx = u - x0, ty = v - y0;
        float h00 = grid4x4[y0 * 4 + x0], h10 = grid4x4[y0 * 4 + x1];
        float h01 = grid4x4[y1 * 4 + x0], h11 = grid4x4[y1 * 4 + x1];
        return (h00 + (h10 - h00) * tx) + ((h01 + (h11 - h01) * tx) - (h00 + (h10 - h00) * tx)) * ty;
    }

    /// <summary>
    /// Serializes MH2O tile data to binary format for WotLK ADT.
    /// </summary>
    public static byte[] SerializeMh2oTile(Mh2oTileData tileData)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // MH2O structure:
        // 256 headers (16x16 chunks, 24 bytes each)
        // Then instance data for each chunk with liquid

        var instanceOffsets = new int[256];
        var instanceData = new List<byte>();

        int currentOffset = 256 * 24; // After all headers

        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                int idx = cy * 16 + cx;
                var chunk = tileData.Chunks[cx, cy];

                if (chunk == null)
                {
                    instanceOffsets[idx] = 0;
                    continue;
                }

                instanceOffsets[idx] = currentOffset + instanceData.Count;

                // Instance data
                using var chunkMs = new MemoryStream();
                using var chunkBw = new BinaryWriter(chunkMs);

                chunkBw.Write(chunk.LiquidTypeId);
                chunkBw.Write(chunk.Flags);
                chunkBw.Write(chunk.MinHeight);
                chunkBw.Write(chunk.MaxHeight);
                chunkBw.Write(chunk.XOffset);
                chunkBw.Write(chunk.YOffset);
                chunkBw.Write(chunk.Width);
                chunkBw.Write(chunk.Height);

                // Height map offset (relative)
                int heightOffset = (int)chunkMs.Position + 12; // After this header
                chunkBw.Write(heightOffset);

                // Render bitmask
                chunkBw.Write(chunk.RenderBitmask);

                // Height data (9x9 floats for full width/height)
                foreach (var h in chunk.Heights)
                    chunkBw.Write(h);

                instanceData.AddRange(chunkMs.ToArray());
            }
        }

        // Write headers
        for (int i = 0; i < 256; i++)
        {
            if (instanceOffsets[i] == 0)
            {
                bw.Write(new byte[24]); // Empty header
            }
            else
            {
                bw.Write((uint)instanceOffsets[i]); // Offset to instance
                bw.Write((uint)1);  // Layer count
                bw.Write((uint)0);  // Attributes offset (none)
                bw.Write(new byte[12]); // Padding
            }
        }

        // Write instance data
        bw.Write(instanceData.ToArray());

        return ms.ToArray();
    }
}
