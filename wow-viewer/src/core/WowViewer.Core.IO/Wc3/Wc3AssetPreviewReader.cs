using System.Buffers.Binary;
using System.Text;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Memory;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Blp;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Mdx;

namespace WowViewer.Core.IO.Wc3;

public sealed record Wc3AssetPreview(int Width, int Height, byte[]? Rgba, string? Text, string? Error, string Format,
    MdxFile? Model = null);

/// <summary>Bounded source previews, independent of raw-file extraction.</summary>
public static class Wc3AssetPreviewReader
{
    private const int MaxInputBytes = 64 * 1024 * 1024;
    private const int MaxTextBytes = 256 * 1024;
    private const int MaxImagePixels = 4 * 1024 * 1024;
    private const int MaxImageDimension = 8192;
    private static readonly HashSet<string> TextExtensions = new(StringComparer.OrdinalIgnoreCase)
        { ".txt", ".slk", ".ini", ".lst", ".log", ".mak", ".bat", ".xml", ".wts", ".j", ".jass", ".fdf", ".toc" };
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
        { ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".pbm", ".pgm", ".ppm", ".qoi" };
    private static readonly DecoderOptions ImageOptions = CreateImageOptions();

    public static Wc3AssetPreview Read(byte[] bytes, string virtualPath, Func<string, byte[]?>? readCompanion = null)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);
        string extension = Path.GetExtension(virtualPath.Replace('\\', '/'));
        string format = extension.Length > 1 ? extension[1..].ToUpperInvariant() : "Binary";
        if (bytes.Length > MaxInputBytes)
            return Failure(format, $"Preview input exceeds the {MaxInputBytes / 1024 / 1024} MiB limit. Raw extraction is still available.");

        try
        {
            if (extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
            {
                MdxFile model = Wc3MdxAdapter.Read(bytes);
                return new(0, 0, null,
                    $"{model.Model.Name}\n{model.Geosets.Count} geosets; {model.Textures.Count} textures; {model.Sequences.Count} animations",
                    null, $"MDX v{model.Version}", model);
            }
            if (Wc3Blp0Decoder.IsMipPath(virtualPath) || bytes.AsSpan().StartsWith("BLP0"u8))
                return Wc3Blp0Decoder.Read(bytes, virtualPath, readCompanion);
            bool blpSignature = bytes.Length >= 4 && bytes[0] == (byte)'B' && bytes[1] == (byte)'L' && bytes[2] == (byte)'P';
            if (blpSignature || extension.Equals(".blp", StringComparison.OrdinalIgnoreCase))
                return ReadBlp(bytes, virtualPath);
            if (TextExtensions.Contains(extension))
                return ReadText(bytes, format);
            if (ImageExtensions.Contains(extension))
            {
                using MemoryStream stream = new(bytes, writable: false);
                ImageInfo info = Image.Identify(ImageOptions, stream);
                format = info.Metadata.DecodedImageFormat?.Name ?? format;
                CheckDimensions(info.Width, info.Height);
                stream.Position = 0;
                using Image<Rgba32> image = Image.Load<Rgba32>(ImageOptions, stream);
                return Pixels(image, format);
            }
            return new Wc3AssetPreview(0, 0, null, $"Binary/unsupported asset: {bytes.Length:N0} bytes ({format}).",
                "No image/text preview is supported for this file. Raw extraction is still available.", format);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return Failure(format, $"Preview unavailable: {ex.Message} Raw extraction is still available.");
        }
    }

    private static Wc3AssetPreview ReadBlp(byte[] bytes, string virtualPath)
    {
        string signature = bytes.Length >= 4 ? Encoding.ASCII.GetString(bytes, 0, 4) : "BLP";
        // The shared library accepts the BLP0 magic as BLP1; that is not proof of
        // beta external-mipmap support. Never turn missing companions into an image.
        if (signature == "BLP0")
            return Failure("BLP0", "BLP0 beta/external-mipmap preview is unsupported. Supply an independently decoded image for viewing; keep the original and companion bytes for extraction.");
        using MemoryStream stream = new(bytes, writable: false);
        BlpSummary summary = BlpSummaryReader.Read(stream, virtualPath);
        CheckDimensions(summary.Width, summary.Height);
        BlpMipMapEntry? mip = summary.MipMaps.FirstOrDefault(static entry => entry.Level == 0);
        if (mip is null || !mip.IsInBounds || summary.OutOfBoundsMipLevelCount != 0)
            return Failure(signature, "BLP mip data is missing/out of bounds; external companions or a corrupt file may be involved.", summary.Width, summary.Height);
        int offsetsBase = summary.Format == BlpFormat.Blp2 ? 20 : 28;
        bool terminated = false;
        for (int i = 0; i < 16; i++)
            if (BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offsetsBase + i * 4, 4)) == 0) terminated = true;
        if (!terminated)
            return Failure(signature, "BLP has no terminated mip chain; the existing decoder cannot safely handle this header.");

        int pixels = checked(summary.Width * summary.Height);
        long minimumBytes;
        switch ((uint)summary.Compression)
        {
            case 0: // JPEG: validate the *embedded* dimensions before shared decoding.
                long jpegEnd = summary.HeaderSizeBytes + 4L + summary.JpegHeaderSizeBytes;
                if (summary.JpegHeaderSizeBytes < 0 || jpegEnd > bytes.Length || mip.Offset < jpegEnd)
                    throw new InvalidDataException("BLP JPEG header is truncated or overlaps mip data.");
                using (MemoryStream jpeg = new(checked(summary.JpegHeaderSizeBytes + (int)mip.SizeBytes)))
                {
                    jpeg.Write(bytes, summary.HeaderSizeBytes + 4, summary.JpegHeaderSizeBytes);
                    jpeg.Write(bytes, checked((int)mip.Offset), checked((int)mip.SizeBytes));
                    jpeg.Position = 0;
                    ImageInfo info = Image.Identify(ImageOptions, jpeg);
                    CheckDimensions(info.Width, info.Height);
                    if (info.Width != summary.Width || info.Height != summary.Height || info.FrameMetadataCollection.Count > 1)
                        throw new InvalidDataException("BLP JPEG dimensions/frame count disagree with its header.");
                }
                minimumBytes = 1;
                break;
            case 1: // Palette + separate packed alpha, without normalization.
                if (summary.PaletteSizeBytes != 1024 || mip.Offset < summary.HeaderSizeBytes + 1024)
                    throw new InvalidDataException("BLP palette is missing or overlaps mip data.");
                if (summary.AlphaDepthBits is not (0 or 1 or 4 or 8))
                    throw new NotSupportedException($"Unsupported BLP palette alpha depth {summary.AlphaDepthBits}.");
                minimumBytes = pixels + ((long)pixels * summary.AlphaDepthBits + 7) / 8;
                break;
            case 2: // Existing decoder supports DXT1/3/5, not every pixel-format enum.
                uint pixelFormat = (uint)summary.PixelFormat;
                if (summary.AlphaDepthBits is not (0 or 1 or 4 or 8) || pixelFormat is not (0 or 1 or 7))
                    throw new NotSupportedException("Unsupported BLP DXT alpha/pixel format.");
                minimumBytes = ((summary.Width + 3L) / 4) * ((summary.Height + 3L) / 4) * (summary.AlphaDepthBits > 1 ? 16 : 8);
                break;
            case 3:
                minimumBytes = (long)pixels * 4;
                break;
            default:
                throw new NotSupportedException($"Unsupported BLP compression {(uint)summary.Compression}.");
        }
        if (mip.Offset < summary.HeaderSizeBytes || mip.SizeBytes < minimumBytes)
            throw new InvalidDataException("BLP mip 0 is too short or overlaps its header.");
        stream.Position = 0;
        using BlpFile blp = new(stream);
        using Image<Rgba32> decoded = blp.GetImage(0);
        if (decoded.Width != summary.Width || decoded.Height != summary.Height)
            throw new InvalidDataException("BLP decoded dimensions disagree with its header.");
        // The frozen palette decoder expands nibbles by << 4 (max 240), not
        // multiplication by 17 (max 255). Expose that limitation, don't fix or
        // normalize the source through this opt-in file-browser adapter.
        string format = (uint)summary.Compression == 1 && summary.AlphaDepthBits == 4
            ? $"{signature} (4-bit alpha: shared decoder expands 0-15 to 0-240, not 0-255)"
            : signature;
        return Pixels(decoded, format);
    }

    /// <summary>Bounded text decode for any text-like file (BOM-aware, Windows-1252 fallback).</summary>
    public static Wc3AssetPreview ReadText(byte[] bytes, string format)
    {
        Encoding encoding = new UTF8Encoding(false, true);
        string encodingName = "UTF-8";
        int bomLength = 0;
        if (bytes.AsSpan().StartsWith(new byte[] { 0xff, 0xfe, 0, 0 }))
        { encoding = new UTF32Encoding(false, false, true); encodingName = "UTF-32 LE BOM"; bomLength = 4; }
        else if (bytes.AsSpan().StartsWith(new byte[] { 0, 0, 0xfe, 0xff }))
        { encoding = new UTF32Encoding(true, false, true); encodingName = "UTF-32 BE BOM"; bomLength = 4; }
        else if (bytes.AsSpan().StartsWith(new byte[] { 0xff, 0xfe }))
        { encoding = new UnicodeEncoding(false, false, true); encodingName = "UTF-16 LE BOM"; bomLength = 2; }
        else if (bytes.AsSpan().StartsWith(new byte[] { 0xfe, 0xff }))
        { encoding = new UnicodeEncoding(true, false, true); encodingName = "UTF-16 BE BOM"; bomLength = 2; }
        else if (bytes.AsSpan().StartsWith(new byte[] { 0xef, 0xbb, 0xbf }))
        { encodingName = "UTF-8 BOM"; bomLength = 3; }

        int length = Math.Min(bytes.Length - bomLength, MaxTextBytes);
        bool truncated = length < bytes.Length - bomLength;
        ReadOnlySpan<byte> content = bytes.AsSpan(bomLength, length);
        char[] chars = new char[encoding.GetMaxCharCount(length)];
        int written;
        try
        {
            encoding.GetDecoder().Convert(content, chars, flush: !truncated, out _, out written, out _);
        }
        catch (DecoderFallbackException) when (bomLength == 0)
        {
            encoding = CodePagesEncodingProvider.Instance.GetEncoding(1252)!;
            encodingName = "Windows-1252 fallback (encoding inferred)";
            written = encoding.GetChars(content, chars);
        }
        string text = new(chars, 0, written);
        if (text.Any(static c => char.IsControl(c) && c is not ('\r' or '\n' or '\t' or '\f')))
            return Failure(format, "This text-named file contains binary/control data; text preview is unsupported.");
        if (truncated)
            text += $"\n\n[Preview truncated after {MaxTextBytes:N0} input bytes; extract the original for complete content.]";
        return new Wc3AssetPreview(0, 0, null, text, null, $"{format} ({encodingName}{(truncated ? "; truncated" : "")})");
    }

    private static DecoderOptions CreateImageOptions()
    {
        Configuration configuration = Configuration.Default.Clone();
        configuration.MemoryAllocator = MemoryAllocator.Create(new MemoryAllocatorOptions
            { MaximumPoolSizeMegabytes = 32, AllocationLimitMegabytes = 64 });
        return new DecoderOptions { Configuration = configuration, MaxFrames = 1, SkipMetadata = true };
    }

    private static void CheckDimensions(int width, int height)
    {
        if (width <= 0 || height <= 0 || width > MaxImageDimension || height > MaxImageDimension || (long)width * height > MaxImagePixels)
            throw new NotSupportedException($"Image {width}x{height} exceeds preview bounds ({MaxImagePixels:N0} pixels, {MaxImageDimension} per dimension).");
    }

    private static Wc3AssetPreview Pixels(Image<Rgba32> image, string format)
    {
        CheckDimensions(image.Width, image.Height);
        byte[] rgba = new byte[checked(image.Width * image.Height * 4)];
        image.CopyPixelDataTo(rgba);
        return new Wc3AssetPreview(image.Width, image.Height, rgba, null, null, format);
    }

    private static Wc3AssetPreview Failure(string format, string error, int width = 0, int height = 0)
        => new(width, height, null, null, error, format);
}
