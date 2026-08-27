using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Blp;

/// <summary>
/// Writes standard Blizzard BLP2 container files with DXT1 compression (e.g. for minimap tiles).
/// </summary>
public static class Blp2Writer
{
    private const uint Blp2Magic = 0x32504C42; // "BLP2"
    private const uint Blp2Type = 1;          // DirectX Compression
    private const byte ColorEncodingDxtc = 2; // DirectX Compressed
    private const byte AlphaDepthNone = 0;    // 0 = no alpha
    private const byte PixelFormatDxt1 = 0;   // 0 = DXT1
    private const byte HasMipsFalse = 0;      // 1 mip level only
    private const int HeaderSize = 148;       // 20 bytes fixed header + 64 bytes offsets + 64 bytes sizes

    /// <summary>
    /// Encodes an Image to a standalone BLP2 DXT1 byte buffer.
    /// </summary>
    public static byte[] EncodeDxt1(Image<Rgba32> image)
    {
        ArgumentNullException.ThrowIfNull(image);

        int width = image.Width;
        int height = image.Height;
        byte[] rgba = new byte[width * height * 4];
        image.CopyPixelDataTo(rgba);

        return EncodeDxt1(rgba, width, height);
    }

    /// <summary>
    /// Encodes raw 32-bit RGBA pixel bytes to a standalone BLP2 DXT1 byte buffer.
    /// </summary>
    public static byte[] EncodeDxt1(byte[] rgba, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgba);
        if (width <= 0 || (width & (width - 1)) != 0)
            throw new ArgumentException("Width must be a positive power of two.", nameof(width));
        if (height <= 0 || (height & (height - 1)) != 0)
            throw new ArgumentException("Height must be a positive power of two.", nameof(height));

        byte[] dxt1Data = Dxt1TileCodec.EncodeDxt1(rgba, width, height);

        byte[] output = new byte[HeaderSize + dxt1Data.Length];
        using var ms = new MemoryStream(output);
        using var writer = new BinaryWriter(ms);

        // 1. Magic
        writer.Write(Blp2Magic);
        // 2. Type (1 = DirectDraw Surface)
        writer.Write(Blp2Type);
        // 3. Compression / ColorEncoding
        writer.Write(ColorEncodingDxtc);
        // 4. Alpha Size
        writer.Write(AlphaDepthNone);
        // 5. Preferred Format (DXT1)
        writer.Write(PixelFormatDxt1);
        // 6. Has Mipmaps
        writer.Write(HasMipsFalse);
        // 7. Width & Height
        writer.Write(width);
        writer.Write(height);

        // 8. Mipmap offsets (16 entries)
        writer.Write((uint)HeaderSize);
        for (int i = 1; i < 16; i++)
            writer.Write(0u);

        // 9. Mipmap sizes (16 entries)
        writer.Write((uint)dxt1Data.Length);
        for (int i = 1; i < 16; i++)
            writer.Write(0u);

        // 10. Data (Mip 0)
        writer.Write(dxt1Data);

        return output;
    }
}
