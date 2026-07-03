using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Blp;

/// <summary>
/// Minimal BLP-to-RGB decoder. Returns contiguous RGB uint8 pixel data
/// (no alpha). Wraps SereniaBLPLib for the actual BLP decode.
/// </summary>
public static class BlpRgbReader
{
    /// <summary>
    /// Decode a BLP byte array to RGB pixels.
    /// </summary>
    /// <param name="source">Raw BLP file bytes.</param>
    /// <param name="virtualPath">The virtual path for diagnostics (e.g. "Tileset/Generic/Black.blp").</param>
    /// <returns>A result tuple: (Width, Height, RgbBytes). Width/Height are 0 on failure.</returns>
    public static BlpRgbResult ReadRgb(byte[] source, string virtualPath)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);

        if (source.Length < 4)
            return new BlpRgbResult(0, 0, null, LoadError: 1);

        try
        {
            using var stream = new MemoryStream(source, writable: false);
            using var blp = new BlpFile(stream);
            using Image<Rgba32> image = blp.GetImage(0);

            int width = image.Width;
            int height = image.Height;

            if (width <= 0 || height <= 0)
                return new BlpRgbResult(0, 0, null, LoadError: 1);

            byte[] rgb = new byte[width * height * 3];
            image.CopyPixelDataTo(new Span<byte>(rgb));
            // CopyPixelDataTo writes RGBA. We need RGB only. Walk in place:
            // Shift each pixel's R,G,B over the A byte.
            // stride = width * 4 (RGBA). Output stride = width * 3.
            // We iterate rows and compact each row in-place.
            // Simplified: allocate new array, copy R,G,B only.
            byte[] rgbOnly = new byte[width * height * 3];
            for (int y = 0; y < height; y++)
            {
                int srcRow = y * width * 4;
                int dstRow = y * width * 3;
                for (int x = 0; x < width; x++)
                {
                    rgbOnly[dstRow + x * 3 + 0] = rgb[srcRow + x * 4 + 0]; // R
                    rgbOnly[dstRow + x * 3 + 1] = rgb[srcRow + x * 4 + 1]; // G
                    rgbOnly[dstRow + x * 3 + 2] = rgb[srcRow + x * 4 + 2]; // B
                }
            }

            return new BlpRgbResult(width, height, rgbOnly, LoadError: 0);
        }
        catch
        {
            return new BlpRgbResult(0, 0, null, LoadError: 1);
        }
    }
}

/// <summary>
/// Result of a BLP decode. <see cref="LoadError"/> is 0 on success, 1 on failure.
/// On failure, <see cref="Width"/> and <see cref="Height"/> are 0 and <see cref="Rgb"/> is null.
/// </summary>
/// <param name="Width">Pixel width of the decoded texture.</param>
/// <param name="Height">Pixel height of the decoded texture.</param>
/// <param name="Rgb">Contiguous flat RGB bytes. Length = Width * Height * 3. Null on failure.</param>
/// <param name="LoadError">0 = success, 1 = decode failure.</param>
public sealed record BlpRgbResult(int Width, int Height, byte[]? Rgb, int LoadError);