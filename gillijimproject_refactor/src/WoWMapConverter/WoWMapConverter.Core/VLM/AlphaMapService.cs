using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// MCAL alpha map service - handles reading, writing, and PNG generation.
/// Based on Noggit-Red's Alphamap.cpp implementation.
/// Supports: RLE compressed, 4-bit packed, big (uncompressed 4096).
/// </summary>
public static class AlphaMapService
{
    private const int AlphaSize = 64 * 64;  // 4096 bytes uncompressed

    /// <summary>
    /// Read alpha data from MCAL chunk, handling all format variants.
    /// </summary>
    /// <param name="data">Raw MCAL data for this layer</param>
    /// <param name="flags">MCLY flags for this layer</param>
    /// <param name="useBigAlphamaps">MPHD flag 0x0004</param>
    /// <param name="doNotFixAlphaMap">MCNK flag</param>
    /// <returns>64×64 alpha values</returns>
    public static byte[] ReadAlpha(byte[] data, int offset, uint flags, bool useBigAlphamaps, bool doNotFixAlphaMap)
    {
        var amap = new byte[AlphaSize];

        if (useBigAlphamaps)
        {
            // Can only compress big alpha (flag 0x200)
            if ((flags & 0x200) != 0)
            {
                ReadCompressed(data, offset, amap);
            }
            else
            {
                ReadBigAlpha(data, offset, amap);
            }
        }
        else
        {
            ReadNotCompressed(data, offset, amap, doNotFixAlphaMap);
        }

        return amap;
    }

    /// <summary>
    /// Read RLE compressed alpha data.
    /// </summary>
    private static void ReadCompressed(byte[] data, int offset, byte[] amap)
    {
        int readPos = offset;
        int writePos = 0;

        while (writePos < AlphaSize && readPos < data.Length)
        {
            byte header = data[readPos++];
            int count = header & 0x7F;
            bool isFill = (header & 0x80) != 0;

            if (count == 0) continue;
            if (writePos + count > AlphaSize) count = AlphaSize - writePos;

            if (isFill)
            {
                // Fill mode: repeat value[0] count times
                byte value = readPos < data.Length ? data[readPos++] : (byte)0;
                for (int i = 0; i < count; i++)
                    amap[writePos++] = value;
            }
            else
            {
                // Copy mode: copy count bytes
                for (int i = 0; i < count && readPos < data.Length; i++)
                    amap[writePos++] = data[readPos++];
            }
        }
    }

    /// <summary>
    /// Read uncompressed big alpha (4096 bytes direct copy).
    /// </summary>
    private static void ReadBigAlpha(byte[] data, int offset, byte[] amap)
    {
        int copyLen = Math.Min(AlphaSize, data.Length - offset);
        if (copyLen > 0)
            Array.Copy(data, offset, amap, 0, copyLen);
    }

    /// <summary>
    /// Read 4-bit packed alpha (2048 bytes → 4096 bytes).
    /// </summary>
    private static void ReadNotCompressed(byte[] data, int offset, byte[] amap, bool doNotFixAlphaMap)
    {
        // 4-bit packed: lower 4 bits, upper 4 bits
        for (int x = 0; x < 64 && offset < data.Length; x++)
        {
            for (int y = 0; y < 64; y += 2)
            {
                if (offset >= data.Length) break;
                byte packed = data[offset++];
                byte lower = (byte)((packed & 0x0F) | ((packed & 0x0F) << 4));
                byte upper = (byte)((packed >> 4) | (packed & 0xF0));
                amap[x * 64 + y] = lower;
                amap[x * 64 + y + 1] = upper;
            }
        }

        // Fix last row/column (Noggit legacy fix)
        if (!doNotFixAlphaMap)
        {
            for (int i = 0; i < 64; i++)
            {
                amap[i * 64 + 63] = amap[i * 64 + 62];
                amap[63 * 64 + i] = amap[62 * 64 + i];
            }
            amap[63 * 64 + 63] = amap[62 * 64 + 62];
        }
    }

    /// <summary>
    /// Compress alpha data using RLE (Noggit-style).
    /// </summary>
    public static byte[] Compress(byte[] amap)
    {
        var result = new List<byte>();
        int pos = 0;

        while (pos < amap.Length)
        {
            // Try fill mode (consecutive identical bytes)
            int fillCount = 1;
            while (pos + fillCount < amap.Length && 
                   amap[pos + fillCount] == amap[pos] && 
                   fillCount < 127)
            {
                fillCount++;
            }

            if (fillCount > 1)
            {
                // Fill mode: header (count | 0x80), value
                result.Add((byte)(fillCount | 0x80));
                result.Add(amap[pos]);
                pos += fillCount;
            }
            else
            {
                // Copy mode: count consecutive different bytes
                int copyStart = pos;
                int copyCount = 1;
                while (pos + copyCount < amap.Length && 
                       copyCount < 127 &&
                       (pos + copyCount + 1 >= amap.Length || amap[pos + copyCount] != amap[pos + copyCount + 1]))
                {
                    copyCount++;
                }

                result.Add((byte)copyCount);
                for (int i = 0; i < copyCount; i++)
                    result.Add(amap[pos + i]);
                pos += copyCount;
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Generate 64×64 grayscale PNG from alpha data.
    /// </summary>
    public static byte[] ToPng(byte[] amap)
    {
        using var image = new Image<L8>(64, 64);
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                image[x, y] = new L8(amap[y * 64 + x]);
            }
        }

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    /// <summary>
    /// Load alpha data from 64×64 grayscale PNG.
    /// </summary>
    public static byte[] FromPng(byte[] pngData)
    {
        using var ms = new MemoryStream(pngData);
        using var image = Image.Load<L8>(ms);
        
        var amap = new byte[AlphaSize];
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                amap[y * 64 + x] = image[x, y].PackedValue;
            }
        }
        return amap;
    }
}
