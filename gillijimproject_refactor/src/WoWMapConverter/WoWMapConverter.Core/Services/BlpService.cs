using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using BCnEncoder.Encoder;
using BCnEncoder.Shared;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// BLP texture handling service for Alpha compatibility.
/// Alpha 0.5.3 requires textures to be max 256x256.
/// </summary>
public static class BlpService
{
    /// <summary>
    /// Maximum texture dimension for Alpha 0.5.3 client.
    /// </summary>
    public const int AlphaMaxDimension = 256;

    /// <summary>
    /// Check if a texture needs resizing for Alpha compatibility.
    /// </summary>
    public static bool NeedsResize(int width, int height)
    {
        return width > AlphaMaxDimension || height > AlphaMaxDimension;
    }

    /// <summary>
    /// Calculate the target dimensions for Alpha-compatible resize.
    /// Maintains aspect ratio while ensuring max dimension is 256.
    /// </summary>
    public static (int width, int height) CalculateAlphaDimensions(int width, int height)
    {
        if (!NeedsResize(width, height))
            return (width, height);

        float scale = Math.Min(
            (float)AlphaMaxDimension / width,
            (float)AlphaMaxDimension / height);

        int newWidth = Math.Max(1, (int)(width * scale));
        int newHeight = Math.Max(1, (int)(height * scale));

        // Ensure power of 2 for DXT compression
        newWidth = NextPowerOfTwo(newWidth);
        newHeight = NextPowerOfTwo(newHeight);

        // Clamp to max
        newWidth = Math.Min(newWidth, AlphaMaxDimension);
        newHeight = Math.Min(newHeight, AlphaMaxDimension);

        return (newWidth, newHeight);
    }

    /// <summary>
    /// Get the next power of two >= value.
    /// </summary>
    private static int NextPowerOfTwo(int value)
    {
        int power = 1;
        while (power < value) power *= 2;
        return power;
    }

    /// <summary>
    /// Resize a BLP file for Alpha compatibility (max 256x256).
    /// </summary>
    /// <param name="inputPath">Source BLP file</param>
    /// <param name="outputPath">Output BLP file</param>
    /// <returns>True if resized, false if already compatible (copied as-is)</returns>
    public static bool ResizeBlp(string inputPath, string outputPath)
    {
        using var blp = new BlpFile(File.OpenRead(inputPath));
        var bmp = blp.GetBitmap(0);

        if (!NeedsResize(bmp.Width, bmp.Height))
        {
            // Already compatible, copy as-is
            File.Copy(inputPath, outputPath, overwrite: true);
            return false;
        }

        var (newWidth, newHeight) = CalculateAlphaDimensions(bmp.Width, bmp.Height);

        // Convert to ImageSharp for resizing
        using var image = Image.LoadPixelData<Bgra32>(BitmapToBytes(bmp), bmp.Width, bmp.Height);
        image.Mutate(x => x.Resize(newWidth, newHeight));

        // Convert to Rgba32 for encoding
        using var rgba = image.CloneAs<Rgba32>();

        // Write as BLP2 (Alpha 0.5.3 actually uses BLP2 format)
        WriteBlp2(outputPath, rgba, hasAlpha: true);

        return true;
    }

    /// <summary>
    /// Batch process a directory of BLP files for Alpha compatibility.
    /// </summary>
    public static (int processed, int resized, int copied, int errors) ProcessDirectory(
        string inputDir,
        string outputDir,
        string pattern = "*.blp",
        bool recursive = true,
        bool verbose = false)
    {
        int processed = 0, resized = 0, copied = 0, errors = 0;

        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var files = Directory.GetFiles(inputDir, pattern, searchOption);

        foreach (var file in files)
        {
            try
            {
                var relativePath = Path.GetRelativePath(inputDir, file);
                var outputPath = Path.Combine(outputDir, relativePath);
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

                if (ResizeBlp(file, outputPath))
                {
                    resized++;
                    if (verbose) Console.WriteLine($"  Resized: {relativePath}");
                }
                else
                {
                    copied++;
                }
                processed++;
            }
            catch (Exception ex)
            {
                errors++;
                if (verbose) Console.Error.WriteLine($"  Error: {file}: {ex.Message}");
            }
        }

        return (processed, resized, copied, errors);
    }

    private static byte[] BitmapToBytes(System.Drawing.Bitmap bmp)
    {
        var rect = new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height);
        var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, 
            System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        try
        {
            var bytes = new byte[data.Stride * data.Height];
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);
            return bytes;
        }
        finally
        {
            bmp.UnlockBits(data);
        }
    }

    private static void WriteBlp2(string outputPath, Image<Rgba32> image, bool hasAlpha)
    {
        const int BLP2Magic = 0x32504c42; // "BLP2"

        using var fs = File.Create(outputPath);
        using var bw = new BinaryWriter(fs);

        var width = image.Width;
        var height = image.Height;

        // Generate mipmaps
        var mipmaps = new List<byte[]>();
        var mipImages = GenerateMipmaps(image);

        // Compress each mipmap to DXT
        var encoder = new BcEncoder
        {
            OutputOptions =
            {
                GenerateMipMaps = false,
                Quality = CompressionQuality.Balanced,
                Format = hasAlpha ? CompressionFormat.Bc3 : CompressionFormat.Bc1
            }
        };

        foreach (var mip in mipImages)
        {
            var rgba = GetRgbaBytes(mip);
            var compressed = encoder.EncodeToRawBytes(rgba, mip.Width, mip.Height, PixelFormat.Rgba32, 0, out _, out _);
            mipmaps.Add(compressed);
            if (mip != image) mip.Dispose();
        }

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

        bw.Write(BLP2Magic);
        bw.Write((uint)1);
        bw.Write((byte)2);
        bw.Write((byte)(hasAlpha ? 8 : 0));
        bw.Write((byte)(hasAlpha ? 7 : 0));
        bw.Write((byte)1);
        bw.Write(width);
        bw.Write(height);

        for (int i = 0; i < 16; i++) bw.Write(offsets[i]);
        for (int i = 0; i < 16; i++) bw.Write(sizes[i]);

        foreach (var mip in mipmaps) bw.Write(mip);
    }

    private static List<Image<Rgba32>> GenerateMipmaps(Image<Rgba32> source)
    {
        var result = new List<Image<Rgba32>> { source.Clone() };
        int w = source.Width, h = source.Height;

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

    private static byte[] GetRgbaBytes(Image<Rgba32> image)
    {
        var bytes = new byte[image.Width * image.Height * 4];
        image.CopyPixelDataTo(bytes);
        return bytes;
    }
}
