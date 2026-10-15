using System.Buffers.Binary;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Wc3;

namespace WowViewer.Core.Tests;

public sealed class Wc3AssetPreviewTests
{
    [Theory]
    [InlineData("txt")]
    [InlineData("slk")]
    [InlineData("ini")]
    [InlineData("lst")]
    [InlineData("log")]
    [InlineData("mak")]
    [InlineData("bat")]
    [InlineData("xml")]
    [InlineData("wts")]
    [InlineData("j")]
    [InlineData("jass")]
    [InlineData("fdf")]
    [InlineData("toc")]
    public void TextAndTableExtensionsUseBoundedUtf8WithoutChangingNamesOrBytes(string extension)
    {
        byte[] bytes = Encoding.UTF8.GetBytes("original -> SHORT~1\r\néclair\t値");
        byte[] original = bytes.ToArray();
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(bytes, $"Build\\SHORT~1.{extension.ToUpperInvariant()}");
        Assert.Null(preview.Error);
        Assert.Null(preview.Rgba);
        Assert.Equal(Encoding.UTF8.GetString(bytes), preview.Text);
        Assert.Contains("UTF-8", preview.Format);
        Assert.Equal(original, bytes);
    }

    [Theory]
    [InlineData("utf8")]
    [InlineData("utf16le")]
    [InlineData("utf16be")]
    [InlineData("utf32le")]
    [InlineData("utf32be")]
    public void BomEncodingIsHonoredAndBomIsNotDisplayed(string variant)
    {
        Encoding encoding = variant switch
        {
            "utf8" => new UTF8Encoding(true),
            "utf16le" => new UnicodeEncoding(false, true),
            "utf16be" => new UnicodeEncoding(true, true),
            "utf32le" => new UTF32Encoding(false, true),
            _ => new UTF32Encoding(true, true)
        };
        byte[] bytes = encoding.GetPreamble().Concat(encoding.GetBytes("Text: é世界")).ToArray();
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(bytes, "build.txt");
        Assert.Null(preview.Error);
        Assert.Equal("Text: é世界", preview.Text);
        Assert.Contains("BOM", preview.Format);
    }

    [Fact]
    public void InvalidUtf8WithoutBomUsesExplicitCodepageFallback()
    {
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read([0x93, 0x63, 0x61, 0x66, 0xe9, 0x94], "beta.mak");
        Assert.Null(preview.Error);
        Assert.Equal("“café”", preview.Text);
        Assert.Contains("1252 fallback", preview.Format);
    }

    [Fact]
    public void TextIsTruncatedWithoutMistakingASplitUtf8SequenceForLegacyEncoding()
    {
        byte[] bytes = Encoding.UTF8.GetBytes(new string('a', 256 * 1024 - 1) + "世界");
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(bytes, "large.log");
        Assert.Null(preview.Error);
        Assert.Contains("UTF-8", preview.Format);
        Assert.Contains("truncated", preview.Format);
        Assert.Contains("Preview truncated", preview.Text);
        Assert.DoesNotContain("ä", preview.Text);
        Assert.True(preview.Text!.Length < 263000);
    }

    [Fact]
    public void BinaryIsMetadataOrAnExplicitFailureNeverAFalseImage()
    {
        Wc3AssetPreview unknown = Wc3AssetPreviewReader.Read([0, 1, 255], "map.w3e");
        Assert.Null(unknown.Rgba);
        Assert.Equal(0, unknown.Width);
        Assert.NotNull(unknown.Error);
        Assert.Contains("3 bytes", unknown.Text);
        Wc3AssetPreview binaryText = Wc3AssetPreviewReader.Read([0, 1, 2], "not-text.txt");
        Assert.NotNull(binaryText.Error);
        Assert.Null(binaryText.Text);
    }

    [Fact]
    public void PngPreviewPreservesDimensionsOrientationColorsAndAlpha()
    {
        using Image<Rgba32> image = new(2, 2);
        image[0, 0] = new Rgba32(250, 10, 30, 255);
        image[1, 0] = new Rgba32(3, 4, 5, 127);
        image[0, 1] = new Rgba32(20, 30, 40, 0);
        image[1, 1] = new Rgba32(60, 70, 80, 255);
        using MemoryStream stream = new();
        image.SaveAsPng(stream);
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(stream.ToArray(), "Textures/atlas.png");
        Assert.Null(preview.Error);
        Assert.Equal(2, preview.Width);
        Assert.Equal(2, preview.Height);
        Assert.Equal(new byte[] { 250, 10, 30, 255, 3, 4, 5, 127, 20, 30, 40, 0, 60, 70, 80, 255 }, preview.Rgba);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(8)]
    public void Blp1PaletteUsesDirectDecoderRgbaWithoutAlphaNormalization(int alphaBits)
    {
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(PaletteBlp(alphaBits), "Terrain/atlas.blp");
        Assert.Null(preview.Error);
        Assert.StartsWith("BLP1", preview.Format);
        if (alphaBits == 4) Assert.Contains("0-240", preview.Format);
        Assert.Equal(2, preview.Width);
        Assert.Equal(2, preview.Height);
        byte[] alpha = alphaBits switch
        {
            0 => [255, 255, 255, 255],
            1 => [255, 0, 255, 0],
            // Preserve the frozen decoder's nibble << 4 behavior; this test is
            // not a claim of independently verified 4-bit beta alpha fidelity.
            4 => [0, 80, 160, 240],
            _ => [0, 64, 128, 255]
        };
        Assert.Equal(new byte[] { 30, 20, 10, alpha[0], 60, 50, 40, alpha[1], 30, 20, 10, alpha[2], 60, 50, 40, alpha[3] }, preview.Rgba);
    }

    [Fact]
    public void Blp0ExternalMipVariantIsExplicitlyUnsupportedEvenWithBlp1ShapedData()
    {
        byte[] bytes = PaletteBlp(8);
        bytes[3] = (byte)'0';
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(bytes, "beta.blp");
        Assert.Equal("BLP0", preview.Format);
        Assert.Null(preview.Rgba);
        Assert.Contains("external-mipmap", preview.Error);
    }

    [Fact]
    public void TruncatedInvalidAndOversizedBlpsReturnErrorsWithoutDecoding()
    {
        Assert.NotNull(Wc3AssetPreviewReader.Read("BLP1"u8.ToArray(), "broken.blp").Error);
        Assert.NotNull(Wc3AssetPreviewReader.Read([1, 2, 3], "broken.png").Error);
        byte[] truncated = PaletteBlp(8)[..^1];
        Assert.NotNull(Wc3AssetPreviewReader.Read(truncated, "truncated.blp").Error);
        byte[] oversized = PaletteBlp(8);
        BinaryPrimitives.WriteUInt32LittleEndian(oversized.AsSpan(12), 100000);
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(oversized, "huge.blp");
        Assert.Null(preview.Rgba);
        Assert.Contains("exceeds preview bounds", preview.Error);
        byte[] shortMip = PaletteBlp(8);
        BinaryPrimitives.WriteUInt32LittleEndian(shortMip.AsSpan(92), 1);
        Assert.NotNull(Wc3AssetPreviewReader.Read(shortMip, "short.blp").Error);
    }

    [Fact]
    public void InputLimitReturnsExtractionNotice()
    {
        Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(new byte[64 * 1024 * 1024 + 1], "huge.bin");
        Assert.Null(preview.Rgba);
        Assert.Contains("64 MiB limit", preview.Error);
        Assert.Contains("extraction", preview.Error);
    }

    private static byte[] PaletteBlp(int alphaBits)
    {
        byte[] alpha = alphaBits switch { 1 => [0b0101], 4 => [0x50, 0xfa], 8 => [0, 64, 128, 255], _ => [] };
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream);
        writer.Write("BLP1"u8.ToArray());
        writer.Write(1u);
        writer.Write((uint)alphaBits);
        writer.Write(2u);
        writer.Write(2u);
        writer.Write(5u);
        writer.Write(0u);
        for (int i = 0; i < 16; i++) writer.Write(i == 0 ? 1180u : 0u);
        for (int i = 0; i < 16; i++) writer.Write(i == 0 ? (uint)(4 + alpha.Length) : 0u);
        for (int i = 0; i < 256; i++) writer.Write(i == 1 ? 0x003c3228u : 0x001e140au);
        writer.Write(new byte[] { 0, 1, 0, 1 });
        writer.Write(alpha);
        return stream.ToArray();
    }
}
