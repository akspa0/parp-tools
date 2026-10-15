using System.Buffers.Binary;
using BitMiracle.LibJpeg.Classic;
using WowViewer.Core.IO.Wc3;

namespace WowViewer.Core.Tests;

public sealed class Wc3Blp0Tests
{
    [Fact]
    public void SplitJpegPreservesRawBgraInsteadOfConvertingCmyk()
    {
        byte[] jpeg = FourChannelJpeg();
        // Split across a real JPEG marker, like the beta shared-table/mip split.
        int split = Array.FindIndex(jpeg, 2, b => b == 0xff);
        Assert.True(split > 0);
        byte[] header = Header(0, 8, 2, 1, 32 + split);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(28), split);
        jpeg.AsSpan(0, split).CopyTo(header.AsSpan(32));
        byte[] mip = jpeg[split..];
        Wc3AssetPreview result = Wc3AssetPreviewReader.Read(header, "Textures\\demo.blp",
            p => p == "Textures\\demo.b00" ? mip : null);
        Assert.Null(result.Error);
        Assert.Equal(2, result.Width);
        Assert.Equal(1, result.Height);
        Assert.NotNull(result.Rgba);
        for (int i = 0; i < 2; i++)
        {
            Assert.InRange(result.Rgba[i * 4], (byte)197, (byte)203);
            Assert.InRange(result.Rgba[i * 4 + 1], (byte)37, (byte)43);
            Assert.InRange(result.Rgba[i * 4 + 2], (byte)7, (byte)13);
            Assert.InRange(result.Rgba[i * 4 + 3], (byte)87, (byte)93);
        }
    }

    [Fact]
    public void ExternalPaletteSupportsPackedAlphaAndSelectingMipDirectly()
    {
        byte[] header = Header(1, 4, 4, 2, 28 + 1024);
        header[28] = 10; header[29] = 20; header[30] = 30;
        header[32] = 40; header[33] = 50; header[34] = 60;
        byte[] mip = [0, 1, 0xf0]; // Level 1: 2 x 1 pixels, alpha 0 / 255.
        var preview = Wc3AssetPreviewReader.Read(mip, "terrain.b01", p => p == "terrain.blp" ? header : null);
        Assert.Null(preview.Error);
        Assert.Equal(2, preview.Width);
        Assert.Equal(1, preview.Height);
        Assert.Equal(new byte[] { 30, 20, 10, 0, 60, 50, 40, 255 }, preview.Rgba);
    }

    [Fact]
    public void MissingCompanionAndTruncatedPaletteAreActionableFailures()
    {
        byte[] header = Header(1, 8, 2, 1, 28 + 1024);
        var missing = Wc3AssetPreviewReader.Read(header, "terrain.blp", _ => null);
        Assert.Contains("terrain.b00", missing.Error);
        var truncated = Wc3AssetPreviewReader.Read(header, "terrain.blp", _ => new byte[] { 1 });
        Assert.NotNull(truncated.Error);
        Assert.Null(truncated.Rgba);
    }

    [Fact]
    public void RejectsInvalidHeaderAndOversizedDimensionsBeforeReadingMip()
    {
        byte[] header = Header(0, 0, 100000, 1, 32);
        bool read = false;
        var result = Wc3AssetPreviewReader.Read(header, "huge.blp", _ => { read = true; return []; });
        Assert.NotNull(result.Error);
        Assert.False(read);
    }

    private static byte[] Header(uint compression, uint alpha, uint width, uint height, int size)
    {
        byte[] result = new byte[size];
        "BLP0"u8.CopyTo(result);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(4), compression);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(8), alpha);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(12), width);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(16), height);
        return result;
    }

    private static byte[] FourChannelJpeg()
    {
        using MemoryStream output = new();
        jpeg_compress_struct encoder = new();
        encoder.jpeg_stdio_dest(output);
        encoder.Image_width = 2;
        encoder.Image_height = 1;
        encoder.Input_components = 4;
        encoder.In_color_space = J_COLOR_SPACE.JCS_CMYK;
        encoder.jpeg_set_defaults();
        encoder.jpeg_set_quality(100, true);
        encoder.jpeg_start_compress(true);
        encoder.jpeg_write_scanlines([[10, 40, 200, 90, 10, 40, 200, 90]], 1);
        encoder.jpeg_finish_compress();
        return output.ToArray();
    }
}
