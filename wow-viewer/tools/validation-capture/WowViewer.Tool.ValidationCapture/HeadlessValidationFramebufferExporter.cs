using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Tools.ValidationCapture;

internal static class HeadlessValidationFramebufferExporter
{
    public static void WriteImage(
        string outputPath,
        int width,
        int height,
        byte[] rgbaPixels,
        bool sourceOriginBottomLeft)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentNullException.ThrowIfNull(rgbaPixels);
        if (rgbaPixels.Length != checked(width * height * 4))
            throw new ArgumentException("RGBA payload length must match width * height * 4.", nameof(rgbaPixels));

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        byte[] imagePixels = sourceOriginBottomLeft ? FlipVertical(width, height, rgbaPixels) : rgbaPixels;
        using Image<Rgba32> image = Image.LoadPixelData<Rgba32>(imagePixels, width, height);
        image.SaveAsPng(outputPath);
    }

    public static void WriteMaskImage(
        string outputPath,
        int width,
        int height,
        byte[] l8Pixels)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentNullException.ThrowIfNull(l8Pixels);
        if (l8Pixels.Length != checked(width * height))
            throw new ArgumentException("L8 payload length must match width * height.", nameof(l8Pixels));

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using Image<L8> image = Image.LoadPixelData<L8>(l8Pixels, width, height);
        image.SaveAsPng(outputPath);
    }

    public static byte[] ReadRgbaImage(string inputPath, out int width, out int height)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);

        using Image<Rgba32> image = Image.Load<Rgba32>(inputPath);
        width = image.Width;
        height = image.Height;
        byte[] rgbaPixels = new byte[checked(width * height * 4)];
        image.CopyPixelDataTo(rgbaPixels);
        return rgbaPixels;
    }

    private static byte[] FlipVertical(int width, int height, byte[] rgbaPixels)
    {
        int rowLength = checked(width * 4);
        byte[] flipped = new byte[rgbaPixels.Length];
        for (int y = 0; y < height; y++)
        {
            int sourceOffset = y * rowLength;
            int destinationOffset = (height - 1 - y) * rowLength;
            Buffer.BlockCopy(rgbaPixels, sourceOffset, flipped, destinationOffset, rowLength);
        }

        return flipped;
    }
}