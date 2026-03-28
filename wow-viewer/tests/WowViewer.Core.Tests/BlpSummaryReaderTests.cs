using System.Buffers.Binary;
using WowViewer.Core.Blp;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class BlpSummaryReaderTests
{
    [Fact]
    public void Read_SyntheticBlp2Header_ProducesExpectedSummary()
    {
        byte[] bytes = CreateBlp2Bytes(
            width: 64,
            height: 32,
            compressionRaw: (byte)BlpCompressionType.Dxtc,
            alphaDepthBits: 8,
            pixelFormatRaw: (byte)BlpPixelFormat.Dxt5,
            mipMapTypeRaw: 1,
            mipOffsets: [148u, 1172u, 1428u],
            mipSizes: [1024u, 256u, 64u]);

        using MemoryStream stream = new(bytes);
        BlpSummary summary = BlpSummaryReader.Read(stream, "synthetic.blp");

        Assert.Equal(BlpFormat.Blp2, summary.Format);
        Assert.Equal("BLP2", summary.Signature);
        Assert.Equal(1u, summary.Version);
        Assert.Equal(BlpCompressionType.Dxtc, summary.Compression);
        Assert.Equal(8, summary.AlphaDepthBits);
        Assert.Equal(BlpPixelFormat.Dxt5, summary.PixelFormat);
        Assert.Equal(1u, summary.MipMapTypeRaw);
        Assert.Equal(64, summary.Width);
        Assert.Equal(32, summary.Height);
        Assert.Equal(148, summary.HeaderSizeBytes);
        Assert.Equal(0, summary.PaletteSizeBytes);
        Assert.Equal(0, summary.JpegHeaderSizeBytes);
        Assert.Equal(3, summary.MipMaps.Count);
        Assert.Equal(3, summary.InBoundsMipLevelCount);
        Assert.Equal(0, summary.OutOfBoundsMipLevelCount);
        Assert.Equal(1492, summary.MaxMipEndOffset);

        BlpMipMapEntry firstMip = summary.MipMaps[0];
        Assert.Equal(0, firstMip.Level);
        Assert.Equal(64, firstMip.Width);
        Assert.Equal(32, firstMip.Height);
        Assert.Equal(148u, firstMip.Offset);
        Assert.Equal(1024u, firstMip.SizeBytes);
        Assert.True(firstMip.IsInBounds);

        BlpMipMapEntry thirdMip = summary.MipMaps[2];
        Assert.Equal(16, thirdMip.Width);
        Assert.Equal(8, thirdMip.Height);
        Assert.Equal(1428u, thirdMip.Offset);
        Assert.Equal(64u, thirdMip.SizeBytes);
    }

    [Fact]
    public void Read_SyntheticBlp1PalettizedHeader_ProducesExpectedSummary()
    {
        byte[] bytes = CreateBlp1Bytes(
            width: 128,
            height: 64,
            compressionRaw: (uint)BlpCompressionType.Palettized,
            alphaDepthBits: 8,
            pixelFormatRaw: (uint)BlpPixelFormat.Palettized,
            mipMapTypeRaw: 1,
            mipOffsets: [1180u, 5276u],
            mipSizes: [4096u, 1024u],
            appendPalette: true,
            jpegHeaderBytes: 0);

        using MemoryStream stream = new(bytes);
        BlpSummary summary = BlpSummaryReader.Read(stream, "legacy.blp");

        Assert.Equal(BlpFormat.Blp1, summary.Format);
        Assert.Equal("BLP1", summary.Signature);
        Assert.Null(summary.Version);
        Assert.Equal(BlpCompressionType.Palettized, summary.Compression);
        Assert.Equal(8, summary.AlphaDepthBits);
        Assert.Equal(BlpPixelFormat.Palettized, summary.PixelFormat);
        Assert.Equal(128, summary.Width);
        Assert.Equal(64, summary.Height);
        Assert.Equal(156, summary.HeaderSizeBytes);
        Assert.Equal(1024, summary.PaletteSizeBytes);
        Assert.Equal(0, summary.JpegHeaderSizeBytes);
        Assert.Equal(2, summary.MipMaps.Count);
        Assert.Equal(2, summary.InBoundsMipLevelCount);
        Assert.Equal(0, summary.OutOfBoundsMipLevelCount);
        Assert.Equal(6300, summary.MaxMipEndOffset);

        Assert.Equal(128, summary.MipMaps[0].Width);
        Assert.Equal(64, summary.MipMaps[0].Height);
        Assert.Equal(64, summary.MipMaps[1].Width);
        Assert.Equal(32, summary.MipMaps[1].Height);
    }

    [Fact]
    public void Read_RealStandardArchiveBlp_ProducesExpectedSignals()
    {
        if (!Directory.Exists(BlpTestPaths.Standard060DataPath) || !File.Exists(BlpTestPaths.ListfilePath))
            return;

        using IArchiveCatalog catalog = new MpqArchiveCatalog();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, [BlpTestPaths.Standard060DataPath], BlpTestPaths.ListfilePath);

        string? virtualPath = null;
        foreach (string candidate in bootstrap.AllFiles)
        {
            if (!candidate.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                continue;

            if (!catalog.FileExists(candidate))
                continue;

            virtualPath = candidate;
            break;
        }

        Assert.False(string.IsNullOrWhiteSpace(virtualPath));
        byte[]? bytes = catalog.ReadFile(virtualPath!);
        Assert.NotNull(bytes);

        using MemoryStream detectionStream = new(bytes);
        WowFileDetection detection = WowFileDetector.Detect(detectionStream, virtualPath!);
        Assert.Equal(WowFileKind.Blp, detection.Kind);

        using MemoryStream summaryStream = new(bytes);
        BlpSummary summary = BlpSummaryReader.Read(summaryStream, virtualPath!);

        Assert.True(summary.Width > 0);
        Assert.True(summary.Height > 0);
        Assert.NotEmpty(summary.MipMaps);
        Assert.True(summary.InBoundsMipLevelCount > 0);
        Assert.True(summary.MaxMipEndOffset <= bytes.Length);
    }

    private static byte[] CreateBlp2Bytes(
        int width,
        int height,
        byte compressionRaw,
        byte alphaDepthBits,
        byte pixelFormatRaw,
        byte mipMapTypeRaw,
        IReadOnlyList<uint> mipOffsets,
        IReadOnlyList<uint> mipSizes)
    {
        byte[] bytes = new byte[checked((int)(mipOffsets.Last() + mipSizes.Last()))];
        WriteAscii(bytes, 0, "BLP2");
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), 1u);
        bytes[8] = compressionRaw;
        bytes[9] = alphaDepthBits;
        bytes[10] = pixelFormatRaw;
        bytes[11] = mipMapTypeRaw;
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12), (uint)width);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16), (uint)height);

        for (int index = 0; index < mipOffsets.Count; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(20 + (index * 4)), mipOffsets[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(84 + (index * 4)), mipSizes[index]);
        }

        return bytes;
    }

    private static byte[] CreateBlp1Bytes(
        int width,
        int height,
        uint compressionRaw,
        uint alphaDepthBits,
        uint pixelFormatRaw,
        uint mipMapTypeRaw,
        IReadOnlyList<uint> mipOffsets,
        IReadOnlyList<uint> mipSizes,
        bool appendPalette,
        int jpegHeaderBytes)
    {
        int headerBytes = 156;
        int paletteBytes = appendPalette ? 1024 : 0;
        int extraJpegBytes = jpegHeaderBytes > 0 ? 4 + jpegHeaderBytes : 0;
        byte[] bytes = new byte[Math.Max(headerBytes + paletteBytes + extraJpegBytes, checked((int)(mipOffsets.Last() + mipSizes.Last())))];
        WriteAscii(bytes, 0, "BLP1");
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), compressionRaw);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8), alphaDepthBits);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12), (uint)width);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16), (uint)height);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(20), pixelFormatRaw);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24), mipMapTypeRaw);

        for (int index = 0; index < mipOffsets.Count; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28 + (index * 4)), mipOffsets[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(92 + (index * 4)), mipSizes[index]);
        }

        if (jpegHeaderBytes > 0)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(headerBytes), (uint)jpegHeaderBytes);

        return bytes;
    }

    private static void WriteAscii(byte[] buffer, int offset, string text)
    {
        for (int index = 0; index < text.Length; index++)
            buffer[offset + index] = (byte)text[index];
    }
}

internal static class BlpTestPaths
{
    public static string Standard060DataPath => Path.Combine(GetWowViewerRoot(), "testdata", "0.6.0", "World of Warcraft", "Data");

    public static string ListfilePath => Path.Combine(GetWowViewerRoot(), "libs", "wowdev", "wow-listfile", "listfile.txt");

    private static string GetWowViewerRoot()
    {
        string? current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            if (File.Exists(Path.Combine(current, "WowViewer.slnx")))
                return current;

            current = Directory.GetParent(current)?.FullName;
        }

        throw new DirectoryNotFoundException("Could not locate wow-viewer workspace root from the current test context.");
    }
}