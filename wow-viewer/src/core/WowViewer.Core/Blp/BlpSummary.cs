namespace WowViewer.Core.Blp;

public sealed class BlpSummary
{
    public BlpSummary(
        string sourcePath,
        BlpFormat format,
        string signature,
        uint? version,
        BlpCompressionType compression,
        byte alphaDepthBits,
        BlpPixelFormat pixelFormat,
        uint mipMapTypeRaw,
        int width,
        int height,
        int headerSizeBytes,
        int paletteSizeBytes,
        int jpegHeaderSizeBytes,
        IReadOnlyList<BlpMipMapEntry> mipMaps,
        int inBoundsMipLevelCount,
        int outOfBoundsMipLevelCount,
        long maxMipEndOffset)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(signature);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegative(headerSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(paletteSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(jpegHeaderSizeBytes);
        ArgumentNullException.ThrowIfNull(mipMaps);
        ArgumentOutOfRangeException.ThrowIfNegative(inBoundsMipLevelCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfBoundsMipLevelCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxMipEndOffset);

        SourcePath = sourcePath;
        Format = format;
        Signature = signature;
        Version = version;
        Compression = compression;
        AlphaDepthBits = alphaDepthBits;
        PixelFormat = pixelFormat;
        MipMapTypeRaw = mipMapTypeRaw;
        Width = width;
        Height = height;
        HeaderSizeBytes = headerSizeBytes;
        PaletteSizeBytes = paletteSizeBytes;
        JpegHeaderSizeBytes = jpegHeaderSizeBytes;
        MipMaps = mipMaps;
        InBoundsMipLevelCount = inBoundsMipLevelCount;
        OutOfBoundsMipLevelCount = outOfBoundsMipLevelCount;
        MaxMipEndOffset = maxMipEndOffset;
    }

    public string SourcePath { get; }

    public BlpFormat Format { get; }

    public string Signature { get; }

    public uint? Version { get; }

    public BlpCompressionType Compression { get; }

    public byte AlphaDepthBits { get; }

    public BlpPixelFormat PixelFormat { get; }

    public uint MipMapTypeRaw { get; }

    public int Width { get; }

    public int Height { get; }

    public int HeaderSizeBytes { get; }

    public int PaletteSizeBytes { get; }

    public int JpegHeaderSizeBytes { get; }

    public IReadOnlyList<BlpMipMapEntry> MipMaps { get; }

    public int InBoundsMipLevelCount { get; }

    public int OutOfBoundsMipLevelCount { get; }

    public long MaxMipEndOffset { get; }
}