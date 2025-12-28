using System.Numerics;

namespace WoWMapConverter.Core.Formats.Liquids;

/// <summary>
/// Converts WL* files (WLW/WLM/WLQ) to MCLQ (Alpha) or MH2O (WotLK+) liquid formats.
/// Used for recovering missing water planes from map data.
/// </summary>
public static class WlToLiquidConverter
{
    private const float TileSize = 533.333f;
    private const float MapSize = 17066.666f;
    private const float ChunkSize = TileSize / 16f; // ~33.33 units per chunk

    #region WL* to MCLQ (Alpha)

    /// <summary>
    /// Converts a WL file to MCLQ chunks grouped by ADT tile.
    /// </summary>
    public static WlToMclqResult ConvertToMclq(WlFile wlFile)
    {
        var result = new WlToMclqResult
        {
            LiquidType = wlFile.Header.LiquidType
        };

        var mclqType = MapWlTypeToMclqType(wlFile.Header.LiquidType);

        foreach (var block in wlFile.Blocks)
        {
            var (tileX, tileY, chunkX, chunkY) = GetTileAndChunkIndices(block.WorldPosition);

            var mclqData = GenerateMclqFromBlock(block, mclqType, chunkX, chunkY);

            var tileKey = (tileX, tileY);
            if (!result.TileData.ContainsKey(tileKey))
                result.TileData[tileKey] = new List<WlMclqChunkData>();

            result.TileData[tileKey].Add(mclqData);
        }

        return result;
    }

    /// <summary>
    /// Generates MCLQ chunk data from a WL block.
    /// Upscales 4x4 vertex grid to 9x9 MCLQ height grid using bilinear interpolation.
    /// </summary>
    private static WlMclqChunkData GenerateMclqFromBlock(WlBlock block, MclqLiquidType liquidType, int chunkX, int chunkY)
    {
        var mclq = new WlMclqChunkData
        {
            ChunkX = chunkX,
            ChunkY = chunkY,
            LiquidType = liquidType
        };

        // Get heights in standard row-major order
        var heights4x4 = block.GetHeights4x4();

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

        // Set all tiles as visible liquid
        byte tileFlag = (byte)liquidType;
        for (int i = 0; i < 64; i++)
            mclq.TileFlags[i] = tileFlag;

        return mclq;
    }

    /// <summary>
    /// Serializes MCLQ chunk data to binary format for Alpha ADT.
    /// </summary>
    public static byte[] SerializeMclqChunk(WlMclqChunkData data)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Min/Max heights (CRange)
        bw.Write(data.MinHeight);
        bw.Write(data.MaxHeight);

        // 9x9 vertices (81 entries, 8 bytes each)
        if (data.LiquidType == MclqLiquidType.Magma)
        {
            // Magma format: s(u16), t(u16), height(f32) = 8 bytes
            for (int i = 0; i < 81; i++)
            {
                bw.Write((ushort)0); // s UV
                bw.Write((ushort)0); // t UV
                bw.Write(data.Heights[i]);
            }
        }
        else
        {
            // Water format: depth(u8), flow0(u8), flow1(u8), filler(u8), height(f32) = 8 bytes
            for (int i = 0; i < 81; i++)
            {
                bw.Write((byte)128); // depth (0=surface, 255=deep)
                bw.Write((byte)0);   // flow0
                bw.Write((byte)0);   // flow1
                bw.Write((byte)0);   // filler
                bw.Write(data.Heights[i]);
            }
        }

        // 8x8 tile flags (64 bytes)
        bw.Write(data.TileFlags);

        // Flow vectors (optional, 2 entries)
        bw.Write(0u); // nFlowvs = 0
        // 2 empty flow vectors (40 bytes each = 80 bytes)
        bw.Write(new byte[80]);

        return ms.ToArray();
    }

    #endregion

    #region WL* to MH2O (WotLK+)

    /// <summary>
    /// Converts a WL file to MH2O instances grouped by ADT tile.
    /// </summary>
    public static WlToMh2oResult ConvertToMh2o(WlFile wlFile)
    {
        var result = new WlToMh2oResult
        {
            LiquidType = wlFile.Header.LiquidType
        };

        ushort liquidTypeId = MapWlTypeToMh2oTypeId(wlFile.Header.LiquidType);

        foreach (var block in wlFile.Blocks)
        {
            var (tileX, tileY, chunkX, chunkY) = GetTileAndChunkIndices(block.WorldPosition);

            var tileKey = (tileX, tileY);
            if (!result.TileData.ContainsKey(tileKey))
                result.TileData[tileKey] = new WlMh2oTileData();

            var tileData = result.TileData[tileKey];

            if (tileData.Chunks[chunkX, chunkY] == null)
            {
                tileData.Chunks[chunkX, chunkY] = GenerateMh2oFromBlock(block, liquidTypeId);
            }
            else
            {
                // Merge overlapping blocks by averaging heights
                MergeMh2oChunk(tileData.Chunks[chunkX, chunkY]!, block);
            }
        }

        return result;
    }

    private static WlMh2oChunkData GenerateMh2oFromBlock(WlBlock block, ushort liquidTypeId)
    {
        var chunk = new WlMh2oChunkData
        {
            LiquidTypeId = liquidTypeId,
            VertexFormat = liquidTypeId == 19 ? Mh2oVertexFormat.HeightUv : Mh2oVertexFormat.HeightDepth,
            XOffset = 0,
            YOffset = 0,
            Width = 8,
            Height = 8
        };

        var heights4x4 = block.GetHeights4x4();

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
        chunk.ExistsBitmap = null; // All tiles exist

        // Set depth map for water types
        if (chunk.VertexFormat == Mh2oVertexFormat.HeightDepth)
        {
            chunk.DepthMap = new byte[81];
            for (int i = 0; i < 81; i++)
                chunk.DepthMap[i] = 128; // Mid-depth
        }

        return chunk;
    }

    private static void MergeMh2oChunk(WlMh2oChunkData existing, WlBlock newBlock)
    {
        var heights4x4 = newBlock.GetHeights4x4();

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

    #endregion

    #region Helpers

    private static (int tileX, int tileY, int chunkX, int chunkY) GetTileAndChunkIndices(Vector3 worldPos)
    {
        // WoW coordinate system: Y increases north, X increases east
        int tileX = Math.Clamp((int)Math.Floor((MapSize - worldPos.Y) / TileSize), 0, 63);
        int tileY = Math.Clamp((int)Math.Floor((MapSize - worldPos.X) / TileSize), 0, 63);

        float localX = (MapSize - worldPos.Y) - (tileX * TileSize);
        float localY = (MapSize - worldPos.X) - (tileY * TileSize);
        int chunkX = Math.Clamp((int)(localX / ChunkSize), 0, 15);
        int chunkY = Math.Clamp((int)(localY / ChunkSize), 0, 15);

        return (tileX, tileY, chunkX, chunkY);
    }

    private static float BilinearSample(float[] grid4x4, float u, float v)
    {
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

    private static MclqLiquidType MapWlTypeToMclqType(WlLiquidType type) => type switch
    {
        WlLiquidType.StillWater => MclqLiquidType.River,
        WlLiquidType.Ocean => MclqLiquidType.Ocean,
        WlLiquidType.River => MclqLiquidType.River,
        WlLiquidType.Magma => MclqLiquidType.Magma,
        WlLiquidType.Slime => MclqLiquidType.Slime,
        WlLiquidType.FastWater => MclqLiquidType.River,
        _ => MclqLiquidType.River
    };

    private static ushort MapWlTypeToMh2oTypeId(WlLiquidType type) => type switch
    {
        WlLiquidType.StillWater => 14, // DB/LiquidType ID for still water
        WlLiquidType.Ocean => 17,      // Ocean
        WlLiquidType.River => 13,      // River
        WlLiquidType.Magma => 19,      // Magma
        WlLiquidType.Slime => 20,      // Slime
        WlLiquidType.FastWater => 13,  // Fast = river
        _ => 14
    };

    #endregion
}

#region Result Types

/// <summary>
/// Result of converting WL* to MCLQ format.
/// </summary>
public class WlToMclqResult
{
    public WlLiquidType LiquidType { get; set; }
    public Dictionary<(int tileX, int tileY), List<WlMclqChunkData>> TileData { get; } = new();
}

/// <summary>
/// MCLQ chunk data generated from WL* block.
/// </summary>
public class WlMclqChunkData
{
    public int ChunkX { get; set; }
    public int ChunkY { get; set; }
    public float MinHeight { get; set; }
    public float MaxHeight { get; set; }
    public float[] Heights { get; set; } = new float[81]; // 9x9 grid
    public byte[] TileFlags { get; set; } = new byte[64]; // 8x8 tiles
    public MclqLiquidType LiquidType { get; set; }
}

/// <summary>
/// Result of converting WL* to MH2O format.
/// </summary>
public class WlToMh2oResult
{
    public WlLiquidType LiquidType { get; set; }
    public Dictionary<(int tileX, int tileY), WlMh2oTileData> TileData { get; } = new();
}

/// <summary>
/// MH2O data for an entire ADT tile (16x16 chunks).
/// </summary>
public class WlMh2oTileData
{
    public WlMh2oChunkData?[,] Chunks { get; } = new WlMh2oChunkData?[16, 16];
    public int ChunkCount => Chunks.Cast<WlMh2oChunkData?>().Count(c => c != null);
}

/// <summary>
/// MH2O chunk data generated from WL* block.
/// </summary>
public class WlMh2oChunkData
{
    public ushort LiquidTypeId { get; set; }
    public Mh2oVertexFormat VertexFormat { get; set; }
    public float MinHeight { get; set; }
    public float MaxHeight { get; set; }
    public byte XOffset { get; set; }
    public byte YOffset { get; set; }
    public byte Width { get; set; } = 8;
    public byte Height { get; set; } = 8;
    public float[] Heights { get; set; } = new float[81]; // 9x9
    public byte[]? DepthMap { get; set; }
    public byte[]? ExistsBitmap { get; set; }
}

#endregion
