using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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
    /// MH2O uses per-chunk headers with variable data.
    /// </summary>
    public static VlmLiquidData? ExtractMH2O(byte[] mh2oData, int chunkIndex, int baseOffset)
    {
        // MH2O structure: 256 headers (16 bytes each), then variable data
        int headerOffset = chunkIndex * 16;  // sizeof(MH2O_Header)
        
        if (headerOffset + 12 > mh2oData.Length)
            return null;

        // Read MH2O_Header
        int ofsInformation = BitConverter.ToInt32(mh2oData, headerOffset);
        int nLayers = BitConverter.ToInt32(mh2oData, headerOffset + 4);
        int ofsAttributes = BitConverter.ToInt32(mh2oData, headerOffset + 8);

        if (nLayers == 0 || ofsInformation == 0)
            return null;

        // Read MH2O_Information (relative to baseOffset)
        int infoOffset = ofsInformation;
        if (infoOffset + 24 > mh2oData.Length)
            return null;

        ushort liquidId = BitConverter.ToUInt16(mh2oData, infoOffset);
        ushort liquidVertexFormat = BitConverter.ToUInt16(mh2oData, infoOffset + 2);
        float minHeight = BitConverter.ToSingle(mh2oData, infoOffset + 4);
        float maxHeight = BitConverter.ToSingle(mh2oData, infoOffset + 8);
        byte xOffset = mh2oData[infoOffset + 12];
        byte yOffset = mh2oData[infoOffset + 13];
        byte width = mh2oData[infoOffset + 14];
        byte height = mh2oData[infoOffset + 15];
        int ofsInfoMask = BitConverter.ToInt32(mh2oData, infoOffset + 16);
        int ofsHeightMap = BitConverter.ToInt32(mh2oData, infoOffset + 20);

        // Read height data if present
        float[]? heights = null;
        if (ofsHeightMap != 0 && liquidVertexFormat == 0)
        {
            int heightDataOffset = ofsHeightMap;
            int vertCount = (width + 1) * (height + 1);
            heights = new float[vertCount];
            
            for (int i = 0; i < vertCount && heightDataOffset + i * 4 < mh2oData.Length; i++)
            {
                heights[i] = BitConverter.ToSingle(mh2oData, heightDataOffset + i * 4);
            }
        }

        return new VlmLiquidData(
            chunkIndex,
            (int)(liquidId & 3),  // Basic type mask
            minHeight,
            maxHeight,
            null,  // Mask path set later
            heights
        );
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
        
        if (tileStart < mclqData.Length)
        {
            byte tileFlags = mclqData[tileStart];
            liquidType = tileFlags & 0x07;  // Lower 3 bits
        }

        return new VlmLiquidData(
            chunkIndex,
            liquidType,
            minHeight,
            maxHeight,
            null,
            heights
        );
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
