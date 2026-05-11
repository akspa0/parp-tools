using BCnEncoder.Encoder;
using BCnEncoder.Shared;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WowViewer.Core.IO.Blp;

public sealed record AlphaBlpCompatibilityResult(
    byte[] Data,
    bool Rewritten,
    bool Resized,
    bool SpecularReencoded,
    int SourceWidth,
    int SourceHeight,
    int OutputWidth,
    int OutputHeight);

public static class AlphaBlpCompatibilityService
{
    public const int AlphaMaxDimension = 256;

    public static AlphaBlpCompatibilityResult NormalizeForAlphaClient(string virtualPath, byte[] sourceData)
    {
        ArgumentNullException.ThrowIfNull(virtualPath);
        ArgumentNullException.ThrowIfNull(sourceData);

        string fileName = Path.GetFileNameWithoutExtension(virtualPath);
        bool isSpecular = fileName.EndsWith("_s", StringComparison.OrdinalIgnoreCase);

        try
        {
            using MemoryStream stream = new(sourceData, writable: false);
            using BlpFile blp = new(stream);
            using Image<Rgba32> image = blp.GetImage(0);

            bool needsResize = image.Width > AlphaMaxDimension || image.Height > AlphaMaxDimension;
            bool shouldRewrite = needsResize || isSpecular;
            if (!shouldRewrite)
            {
                return new AlphaBlpCompatibilityResult(
                    sourceData,
                    Rewritten: false,
                    Resized: false,
                    SpecularReencoded: false,
                    SourceWidth: image.Width,
                    SourceHeight: image.Height,
                    OutputWidth: image.Width,
                    OutputHeight: image.Height);
            }

            using Image<Rgba32> working = image.Clone();
            int outputWidth = working.Width;
            int outputHeight = working.Height;
            if (needsResize)
            {
                (outputWidth, outputHeight) = CalculateAlphaDimensions(working.Width, working.Height);
                working.Mutate(ctx => ctx.Resize(outputWidth, outputHeight));
            }

            byte[] rewritten = EncodeBlp2(working, hasAlpha: true, generateMipmaps: true);
            return new AlphaBlpCompatibilityResult(
                rewritten,
                Rewritten: true,
                Resized: needsResize,
                SpecularReencoded: isSpecular,
                SourceWidth: image.Width,
                SourceHeight: image.Height,
                OutputWidth: outputWidth,
                OutputHeight: outputHeight);
        }
        catch
        {
            return new AlphaBlpCompatibilityResult(
                sourceData,
                Rewritten: false,
                Resized: false,
                SpecularReencoded: false,
                SourceWidth: 0,
                SourceHeight: 0,
                OutputWidth: 0,
                OutputHeight: 0);
        }
    }

    public static byte[] EncodeBlp2(Image<Rgba32> image, bool hasAlpha = true, bool generateMipmaps = true)
    {
        ArgumentNullException.ThrowIfNull(image);

        const int blp2Magic = 0x32504c42;
        const int headerSize = 148;

        List<byte[]> mipmaps = [];
        List<Image<Rgba32>> mipImages = GenerateMipmaps(image, generateMipmaps);
        try
        {
            BcEncoder encoder = new()
            {
                OutputOptions =
                {
                    GenerateMipMaps = false,
                    Quality = CompressionQuality.Balanced,
                    Format = hasAlpha ? CompressionFormat.Bc3 : CompressionFormat.Bc1,
                }
            };

            foreach (Image<Rgba32> mip in mipImages)
            {
                byte[] rgba = new byte[mip.Width * mip.Height * 4];
                mip.CopyPixelDataTo(rgba);
                byte[] compressed = encoder.EncodeToRawBytes(rgba, mip.Width, mip.Height, PixelFormat.Rgba32, 0, out _, out _);
                mipmaps.Add(compressed);
            }

            uint[] offsets = new uint[16];
            uint[] sizes = new uint[16];
            uint currentOffset = headerSize;
            for (int i = 0; i < mipmaps.Count && i < offsets.Length; i++)
            {
                offsets[i] = currentOffset;
                sizes[i] = (uint)mipmaps[i].Length;
                currentOffset += sizes[i];
            }

            using MemoryStream output = new();
            using BinaryWriter writer = new(output, System.Text.Encoding.UTF8, leaveOpen: true);
            writer.Write(blp2Magic);
            writer.Write((uint)1);
            writer.Write((byte)2);
            writer.Write((byte)(hasAlpha ? 8 : 0));
            writer.Write((byte)(hasAlpha ? 7 : 0));
            writer.Write((byte)(generateMipmaps ? 1 : 0));
            writer.Write(image.Width);
            writer.Write(image.Height);

            for (int i = 0; i < offsets.Length; i++)
                writer.Write(offsets[i]);

            for (int i = 0; i < sizes.Length; i++)
                writer.Write(sizes[i]);

            foreach (byte[] mip in mipmaps)
                writer.Write(mip);

            writer.Flush();
            return output.ToArray();
        }
        finally
        {
            foreach (Image<Rgba32> mip in mipImages)
                mip.Dispose();
        }
    }

    public static (int Width, int Height) CalculateAlphaDimensions(int width, int height)
    {
        if (width <= AlphaMaxDimension && height <= AlphaMaxDimension)
            return (width, height);

        float scale = Math.Min((float)AlphaMaxDimension / width, (float)AlphaMaxDimension / height);
        int newWidth = Math.Max(1, (int)(width * scale));
        int newHeight = Math.Max(1, (int)(height * scale));
        newWidth = Math.Min(NextPowerOfTwo(newWidth), AlphaMaxDimension);
        newHeight = Math.Min(NextPowerOfTwo(newHeight), AlphaMaxDimension);
        return (newWidth, newHeight);
    }

    private static int NextPowerOfTwo(int value)
    {
        int power = 1;
        while (power < value)
            power *= 2;

        return power;
    }

    private static List<Image<Rgba32>> GenerateMipmaps(Image<Rgba32> source, bool generateMipmaps)
    {
        List<Image<Rgba32>> mipmaps = [source.Clone()];
        if (!generateMipmaps)
            return mipmaps;

        int width = source.Width;
        int height = source.Height;
        while (width > 1 || height > 1)
        {
            width = Math.Max(1, width / 2);
            height = Math.Max(1, height / 2);

            Image<Rgba32> mip = source.Clone();
            mip.Mutate(ctx => ctx.Resize(width, height));
            mipmaps.Add(mip);
        }

        return mipmaps;
    }
}