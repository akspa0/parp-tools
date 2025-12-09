using BCnEncoder.Encoder;
using BCnEncoder.Shared;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BlpResizer;

/// <summary>
/// Writes BLP1 files (Alpha 0.5.3 compatible) with DXT compression.
/// Supports DXT1 (no alpha / 1-bit alpha) and DXT3/DXT5 (full alpha).
/// </summary>
public static class BlpWriter
{
    private const int BLP1Magic = 0x31504c42; // "BLP1"

    /// <summary>
    /// Write an RGBA image as a BLP1 file with DXT compression (Alpha client compatible).
    /// </summary>
    /// <param name="outputPath">Output file path</param>
    /// <param name="image">Source image (will be cloned, not modified)</param>
    /// <param name="hasAlpha">If true, uses DXT3; otherwise DXT1</param>
    /// <param name="generateMipmaps">If true, generates mipmaps down to 1x1</param>
    public static void WriteBlp1(string outputPath, Image<Rgba32> image, bool hasAlpha = true, bool generateMipmaps = true)
    {
        using var fs = File.Create(outputPath);
        using var bw = new BinaryWriter(fs);

        var width = image.Width;
        var height = image.Height;

        // Generate mipmaps
        var mipmaps = new List<byte[]>();
        var mipImages = GenerateMipmaps(image, generateMipmaps);

        // Compress each mipmap to DXT
        // Alpha uses DXT3 (not DXT5) for BLP1 compatibility
        var encoder = new BcEncoder
        {
            OutputOptions =
            {
                GenerateMipMaps = false, // We handle mipmaps ourselves
                Quality = CompressionQuality.Balanced,
                Format = hasAlpha ? CompressionFormat.Bc2 : CompressionFormat.Bc1 // DXT3 or DXT1
            }
        };

        foreach (var mip in mipImages)
        {
            var rgba = GetRgbaBytes(mip);
            var compressed = encoder.EncodeToRawBytes(rgba, mip.Width, mip.Height, PixelFormat.Rgba32, 0, out _, out _);
            mipmaps.Add(compressed);
            if (mip != image) mip.Dispose();
        }

        // BLP1 header layout:
        // 4 bytes: magic "BLP1"
        // 4 bytes: compression (0=JPEG, 1=palette/DXT)
        // 4 bytes: alpha bit depth (0, 1, 4, or 8)
        // 4 bytes: width
        // 4 bytes: height
        // 4 bytes: flags/type (pictureType for DXT: 4=DXT1, 5=DXT3)
        // 4 bytes: hasMips (0 or 1)
        // 64 bytes: mipmap offsets (16 x 4 bytes)
        // 64 bytes: mipmap sizes (16 x 4 bytes)
        // Total header: 156 bytes

        const int headerSize = 156;
        var offsets = new uint[16];
        var sizes = new uint[16];

        uint currentOffset = headerSize;
        for (int i = 0; i < mipmaps.Count && i < 16; i++)
        {
            offsets[i] = currentOffset;
            sizes[i] = (uint)mipmaps[i].Length;
            currentOffset += sizes[i];
        }

        // Write BLP1 header
        bw.Write(BLP1Magic);                    // Magic "BLP1"
        bw.Write(1);                            // Compression: 1 = DXT
        bw.Write(hasAlpha ? 8 : 0);             // Alpha bit depth
        bw.Write(width);                        // Width
        bw.Write(height);                       // Height
        bw.Write(hasAlpha ? 5 : 4);             // PictureType: 4=DXT1, 5=DXT3
        bw.Write(generateMipmaps ? 1 : 0);      // Has mipmaps

        // Mipmap offsets (16 entries)
        for (int i = 0; i < 16; i++)
            bw.Write(offsets[i]);

        // Mipmap sizes (16 entries)
        for (int i = 0; i < 16; i++)
            bw.Write(sizes[i]);

        // Write mipmap data
        foreach (var mip in mipmaps)
        {
            bw.Write(mip);
        }
    }

    /// <summary>
    /// Write an RGBA image as a BLP2 file with DXT compression (WotLK+ compatible).
    /// </summary>
    public static void WriteBlp2(string outputPath, Image<Rgba32> image, bool hasAlpha = true, bool generateMipmaps = true)
    {
        const int BLP2Magic = 0x32504c42; // "BLP2"

        using var fs = File.Create(outputPath);
        using var bw = new BinaryWriter(fs);

        var width = image.Width;
        var height = image.Height;

        // Generate mipmaps
        var mipmaps = new List<byte[]>();
        var mipImages = GenerateMipmaps(image, generateMipmaps);

        // Compress each mipmap to DXT
        var encoder = new BcEncoder
        {
            OutputOptions =
            {
                GenerateMipMaps = false,
                Quality = CompressionQuality.Balanced,
                Format = hasAlpha ? CompressionFormat.Bc3 : CompressionFormat.Bc1 // DXT5 or DXT1
            }
        };

        foreach (var mip in mipImages)
        {
            var rgba = GetRgbaBytes(mip);
            var compressed = encoder.EncodeToRawBytes(rgba, mip.Width, mip.Height, PixelFormat.Rgba32, 0, out _, out _);
            mipmaps.Add(compressed);
            if (mip != image) mip.Dispose();
        }

        // BLP2 header: 148 bytes
        const int headerSize = 148;
        var offsets = new uint[16];
        var sizes = new uint[16];

        uint currentOffset = headerSize;
        for (int i = 0; i < mipmaps.Count && i < 16; i++)
        {
            offsets[i] = currentOffset;
            sizes[i] = (uint)mipmaps[i].Length;
            currentOffset += sizes[i];
        }

        // Write BLP2 header
        bw.Write(BLP2Magic);
        bw.Write((uint)1);                          // Version
        bw.Write((byte)2);                          // Color encoding: DXT
        bw.Write((byte)(hasAlpha ? 8 : 0));         // Alpha bit depth
        bw.Write((byte)(hasAlpha ? 7 : 0));         // Preferred format: 7=DXT5, 0=DXT1
        bw.Write((byte)(generateMipmaps ? 1 : 0));  // Has mipmaps
        bw.Write(width);
        bw.Write(height);

        for (int i = 0; i < 16; i++)
            bw.Write(offsets[i]);
        for (int i = 0; i < 16; i++)
            bw.Write(sizes[i]);

        foreach (var mip in mipmaps)
            bw.Write(mip);
    }

    /// <summary>
    /// Generate mipmaps from an image, halving dimensions each level until 1x1.
    /// </summary>
    private static List<Image<Rgba32>> GenerateMipmaps(Image<Rgba32> source, bool generateMips)
    {
        var result = new List<Image<Rgba32>> { source.Clone() };

        if (!generateMips)
            return result;

        int w = source.Width;
        int h = source.Height;

        while (w > 1 || h > 1)
        {
            w = Math.Max(1, w / 2);
            h = Math.Max(1, h / 2);

            var mip = source.Clone();
            mip.Mutate(x => x.Resize(w, h));
            result.Add(mip);
        }

        return result;
    }

    /// <summary>
    /// Extract RGBA bytes from an ImageSharp image.
    /// </summary>
    private static byte[] GetRgbaBytes(Image<Rgba32> image)
    {
        var bytes = new byte[image.Width * image.Height * 4];
        image.CopyPixelDataTo(bytes);
        return bytes;
    }
}
