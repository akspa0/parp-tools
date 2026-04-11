using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.Formats.Liquids;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Liquid data service - handles MH2O (WotLK+) and MCLQ (legacy) formats.
/// Based on Noggit-Red's MapHeaders.h and TileWater structures.
/// </summary>
public static class LiquidService
{
    /// <summary>
    /// Liquid type enum (from liquidtype.dbc basic types).
    /// </summary>
    public enum LiquidBasicType
    {
        Water = 0,
        Ocean = 1,
        Magma = 2,
        Slime = 3
    }

    /// <summary>
    /// Extract liquid data from MH2O chunk for a single chunk.
    /// Uses the active MH2O parser and selects the largest visible layer for this chunk.
    /// </summary>
    public static VlmLiquidData? ExtractMH2O(byte[] mh2oData, int chunkIndex, int baseOffset)
    {
        _ = baseOffset;

        if (mh2oData == null || mh2oData.Length == 0)
            return null;

        var mh2o = Mh2oChunk.Parse(mh2oData);
        var instance = mh2o.GetInstancesForChunk(chunkIndex)
            .OrderByDescending(GetVisibleTileCount)
            .ThenByDescending(inst => inst.HeightMap?.Length ?? 0)
            .FirstOrDefault();

        return instance == null ? null : CreateMh2oLiquid(instance);
    }

    public static VlmLiquidData? CreateMh2oLiquid(Mh2oInstance instance)
    {
        int width = instance.Width;
        if (width < 1) width = 1;
        else if (width > 8) width = 8;

        int height = instance.Height;
        if (height < 1) height = 1;
        else if (height > 8) height = 8;
        if (width <= 0 || height <= 0)
            return null;

        float[]? heights = instance.HeightMap;
        if ((heights == null || heights.Length == 0) && instance.VertexFormat == Mh2oVertexFormat.DepthOnly)
        {
            heights = new float[(width + 1) * (height + 1)];
            Array.Fill(heights, instance.MinHeightLevel);
        }

        string? existsBitmapBase64 = instance.ExistsBitmap is { Length: > 0 }
            ? Convert.ToBase64String(instance.ExistsBitmap)
            : null;

        return new VlmLiquidData(
            instance.ChunkIndex,
            MapLiquidTypeIdToBasicType(instance.LiquidTypeId),
            instance.MinHeightLevel,
            instance.MaxHeightLevel,
            null,
            heights,
            instance.XOffset,
            instance.YOffset,
            width,
            height,
            existsBitmapBase64);
    }

    /// <summary>
    /// Extract liquid data from MCLQ chunk (legacy pre-WotLK format).
    /// </summary>
    public static VlmLiquidData? ExtractMCLQ(byte[] mclqData, int chunkIndex)
    {
        if (mclqData.Length < 8)
            return null;

        float minHeight = BitConverter.ToSingle(mclqData, 0);
        float maxHeight = BitConverter.ToSingle(mclqData, 4);

        // 9×9 vertices with height data (4 bytes per vertex for water)
        int vertexStart = 8;
        float[]? heights = null;
        
        if (mclqData.Length >= vertexStart + 81 * 8)  // mclq_vertex is 8 bytes
        {
            heights = new float[81];
            for (int i = 0; i < 81; i++)
            {
                // Height is at offset 4 within each vertex struct
                heights[i] = BitConverter.ToSingle(mclqData, vertexStart + i * 8 + 4);
            }
        }

        // Determine liquid type from tile flags (at offset 8 + 81*8)
        int tileStart = vertexStart + 81 * 8;
        int liquidType = 0;  // Default water
        string? existsBitmapBase64 = null;
        
        if (tileStart + 64 <= mclqData.Length)
        {
            byte[] tileFlags = new byte[64];
            Array.Copy(mclqData, tileStart, tileFlags, 0, tileFlags.Length);
            liquidType = InferLiquidTypeFromTileFlags(tileFlags);

            byte[]? existsBitmap = BuildExistsBitmapFromTileFlags(tileFlags);
            if (existsBitmap != null)
                existsBitmapBase64 = Convert.ToBase64String(existsBitmap);
        }

        return new VlmLiquidData(
            chunkIndex,
            liquidType,
            minHeight,
            maxHeight,
            null,
            heights,
            0,
            0,
            8,
            8,
            existsBitmapBase64
        );
    }

    public static int MapLiquidTypeIdToBasicType(ushort liquidTypeId)
    {
        return liquidTypeId switch
        {
            17 => (int)LiquidBasicType.Ocean,
            19 => (int)LiquidBasicType.Magma,
            20 => (int)LiquidBasicType.Slime,
            _ => (int)LiquidBasicType.Water,
        };
    }

    private static int GetVisibleTileCount(Mh2oInstance instance)
    {
        int width = instance.Width;
        if (width < 1) width = 1;
        else if (width > 8) width = 8;

        int height = instance.Height;
        if (height < 1) height = 1;
        else if (height > 8) height = 8;
        if (instance.ExistsBitmap == null)
            return width * height;

        int visibleTiles = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int tileIndex = (y * width) + x;
                int byteIndex = tileIndex / 8;
                if (byteIndex >= instance.ExistsBitmap.Length)
                    continue;

                int bitIndex = tileIndex % 8;
                if ((instance.ExistsBitmap[byteIndex] & (1 << bitIndex)) != 0)
                    visibleTiles++;
            }
        }

        return visibleTiles;
    }

    private static int InferLiquidTypeFromTileFlags(byte[] tileFlags)
    {
        foreach (byte tileFlag in tileFlags)
        {
            if ((tileFlag & 0x0F) == 0x0F)
                continue;

            return tileFlag & 0x07;
        }

        return 0;
    }

    private static byte[]? BuildExistsBitmapFromTileFlags(byte[] tileFlags)
    {
        byte[] bitmap = new byte[8];
        bool allVisible = true;

        for (int tileIndex = 0; tileIndex < 64 && tileIndex < tileFlags.Length; tileIndex++)
        {
            bool visible = (tileFlags[tileIndex] & 0x0F) != 0x0F;
            if (!visible)
            {
                allVisible = false;
                continue;
            }

            bitmap[tileIndex / 8] |= (byte)(1 << (tileIndex % 8));
        }

        return allVisible ? null : bitmap;
    }

    /// <summary>
    /// Generate 8×8 liquid mask PNG (1 = has liquid, 0 = no liquid).
    /// </summary>
    public static byte[] GenerateMaskPng(byte[] mask8x8)
    {
        using var image = new Image<L8>(8, 8);
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                int idx = y * 8 + x;
                byte value = idx < mask8x8.Length ? (byte)(mask8x8[idx] != 0 ? 255 : 0) : (byte)0;
                image[x, y] = new L8(value);
            }
        }

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    /// <summary>
    /// Generate 9×9 liquid height PNG (normalized 0-255 range).
    /// </summary>
    public static byte[] GenerateHeightPng(float[] heights, float minHeight, float maxHeight)
    {
        using var image = new Image<L8>(9, 9);
        float range = Math.Max(0.001f, maxHeight - minHeight);
        
        for (int y = 0; y < 9; y++)
        {
            for (int x = 0; x < 9; x++)
            {
                int idx = y * 9 + x;
                float h = idx < heights.Length ? heights[idx] : minHeight;
                byte value = (byte)Math.Clamp((h - minHeight) / range * 255f, 0, 255);
                image[x, y] = new L8(value);
            }
        }

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }
}
