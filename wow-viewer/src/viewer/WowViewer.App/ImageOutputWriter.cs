using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.App;

internal static class ImageOutputWriter
{
    public static void WriteRgbImage(string outputPath, int width, int height, byte[] rgbPixels)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(rgbPixels);

        string resolvedOutputPath = PrepareOutputPath(outputPath);
        using FileStream stream = File.Create(resolvedOutputPath);
        string extension = Path.GetExtension(resolvedOutputPath);
        if (string.IsNullOrWhiteSpace(extension) || extension.Equals(".bmp", StringComparison.OrdinalIgnoreCase))
        {
            WriteRgbBitmap(stream, width, height, rgbPixels);
            return;
        }

        if (extension.Equals(".png", StringComparison.OrdinalIgnoreCase))
        {
            using Image<Rgb24> image = Image.LoadPixelData<Rgb24>(rgbPixels, width, height);
            image.Save(stream, new PngEncoder());
            return;
        }

        throw new NotSupportedException($"Unsupported image output extension '{extension}'. Use .bmp or .png.");
    }

    public static void WriteRgbaImage(string outputPath, int width, int height, byte[] rgbaPixels, bool sourceOriginBottomLeft)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(rgbaPixels);

        string resolvedOutputPath = PrepareOutputPath(outputPath);
        using FileStream stream = File.Create(resolvedOutputPath);
        string extension = Path.GetExtension(resolvedOutputPath);
        if (string.IsNullOrWhiteSpace(extension) || extension.Equals(".bmp", StringComparison.OrdinalIgnoreCase))
        {
            WriteRgbaBitmap(stream, width, height, rgbaPixels);
            return;
        }

        if (extension.Equals(".png", StringComparison.OrdinalIgnoreCase))
        {
            byte[] pngPixels = sourceOriginBottomLeft
                ? FlipRgbaRows(width, height, rgbaPixels)
                : rgbaPixels;
            using Image<Rgba32> image = Image.LoadPixelData<Rgba32>(pngPixels, width, height);
            image.Save(stream, new PngEncoder());
            return;
        }

        throw new NotSupportedException($"Unsupported image output extension '{extension}'. Use .bmp or .png.");
    }

    private static string PrepareOutputPath(string outputPath)
    {
        string resolvedOutputPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(resolvedOutputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        return resolvedOutputPath;
    }

    private static byte[] FlipRgbaRows(int width, int height, byte[] rgbaPixels)
    {
        int rowStride = checked(width * 4);
        byte[] flipped = new byte[rgbaPixels.Length];
        for (int row = 0; row < height; row++)
        {
            int sourceOffset = row * rowStride;
            int targetOffset = (height - 1 - row) * rowStride;
            Buffer.BlockCopy(rgbaPixels, sourceOffset, flipped, targetOffset, rowStride);
        }

        return flipped;
    }

    private static void WriteRgbBitmap(Stream stream, int width, int height, byte[] rgbPixels)
    {
        int rowStride = ((width * 3) + 3) & ~3;
        int pixelDataSize = checked(rowStride * height);
        int fileSize = 14 + 40 + pixelDataSize;

        using BinaryWriter writer = new(stream, System.Text.Encoding.ASCII, leaveOpen: true);
        writer.Write((byte)'B');
        writer.Write((byte)'M');
        writer.Write(fileSize);
        writer.Write(0);
        writer.Write(54);
        writer.Write(40);
        writer.Write(width);
        writer.Write(height);
        writer.Write((short)1);
        writer.Write((short)24);
        writer.Write(0);
        writer.Write(pixelDataSize);
        writer.Write(2835);
        writer.Write(2835);
        writer.Write(0);
        writer.Write(0);

        byte[] padding = new byte[rowStride - (width * 3)];
        for (int row = height - 1; row >= 0; row--)
        {
            int rowOffset = row * width * 3;
            for (int column = 0; column < width; column++)
            {
                int offset = rowOffset + (column * 3);
                writer.Write(rgbPixels[offset + 2]);
                writer.Write(rgbPixels[offset + 1]);
                writer.Write(rgbPixels[offset + 0]);
            }

            if (padding.Length > 0)
                writer.Write(padding);
        }
    }

    private static void WriteRgbaBitmap(Stream stream, int width, int height, byte[] rgbaPixels)
    {
        int rowStride = width * 4;
        int pixelDataLength = rowStride * height;
        int fileSize = 14 + 40 + pixelDataLength;

        using BinaryWriter writer = new(stream, System.Text.Encoding.ASCII, leaveOpen: true);
        writer.Write((byte)'B');
        writer.Write((byte)'M');
        writer.Write(fileSize);
        writer.Write(0);
        writer.Write(14 + 40);
        writer.Write(40);
        writer.Write(width);
        writer.Write(height);
        writer.Write((short)1);
        writer.Write((short)32);
        writer.Write(0);
        writer.Write(pixelDataLength);
        writer.Write(2835);
        writer.Write(2835);
        writer.Write(0);
        writer.Write(0);

        byte[] bgraRow = new byte[rowStride];
        for (int row = 0; row < height; row++)
        {
            int sourceOffset = row * rowStride;
            for (int column = 0; column < width; column++)
            {
                int sourcePixel = sourceOffset + (column * 4);
                int targetPixel = column * 4;
                bgraRow[targetPixel + 0] = rgbaPixels[sourcePixel + 2];
                bgraRow[targetPixel + 1] = rgbaPixels[sourcePixel + 1];
                bgraRow[targetPixel + 2] = rgbaPixels[sourcePixel + 0];
                bgraRow[targetPixel + 3] = 255;
            }

            writer.Write(bgraRow);
        }
    }
}