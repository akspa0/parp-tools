using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// MCSH shadow map service - converts shadow data to/from PNG.
/// Shadow maps are 64×64 bitmaps (8 bytes × 64 rows = 512 bytes).
/// </summary>
public static class ShadowMapService
{
    private const int ShadowSize = 64 * 64;
    private const int RawShadowSize = 64 * 8;  // 512 bytes (64 rows × 8 bytes/row)

    /// <summary>
    /// Read shadow bitmap from MCSH data.
    /// Each row is 8 bytes = 64 bits, one bit per pixel.
    /// </summary>
    public static byte[] ReadShadow(byte[] data, int offset = 0)
    {
        var shadow = new byte[ShadowSize];
        
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int byteIndex = offset + y * 8 + (x / 8);
                int bitIndex = x % 8;
                
                if (byteIndex < data.Length)
                {
                    bool isShadowed = (data[byteIndex] & (1 << bitIndex)) != 0;
                    shadow[y * 64 + x] = isShadowed ? (byte)0 : (byte)255;
                }
                else
                {
                    shadow[y * 64 + x] = 255;  // No shadow
                }
            }
        }
        
        return shadow;
    }

    /// <summary>
    /// Write shadow data back to MCSH format (512 bytes).
    /// </summary>
    public static byte[] WriteShadow(byte[] shadow)
    {
        var data = new byte[RawShadowSize];
        
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int byteIndex = y * 8 + (x / 8);
                int bitIndex = x % 8;
                
                // 0-127 = shadowed, 128-255 = not shadowed
                bool isShadowed = shadow[y * 64 + x] < 128;
                if (isShadowed)
                {
                    data[byteIndex] |= (byte)(1 << bitIndex);
                }
            }
        }
        
        return data;
    }

    /// <summary>
    /// Generate 64×64 grayscale PNG from shadow data.
    /// Black pixels = shadowed, white = lit.
    /// </summary>
    public static byte[] ToPng(byte[] shadow)
    {
        using var image = new Image<L8>(64, 64);
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                image[x, y] = new L8(shadow[y * 64 + x]);
            }
        }

        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    /// <summary>
    /// Load shadow data from 64×64 grayscale PNG.
    /// </summary>
    public static byte[] FromPng(byte[] pngData)
    {
        using var ms = new MemoryStream(pngData);
        using var image = Image.Load<L8>(ms);
        
        var shadow = new byte[ShadowSize];
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                shadow[y * 64 + x] = image[x, y].PackedValue;
            }
        }
        return shadow;
    }
}
