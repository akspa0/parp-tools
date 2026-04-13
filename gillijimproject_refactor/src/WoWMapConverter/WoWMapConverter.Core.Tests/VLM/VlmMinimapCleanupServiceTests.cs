using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.VLM;
using Xunit;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class VlmMinimapCleanupServiceTests
{
    [Fact]
    public void DecodeStoredMccvColor_BgraInput_ReturnsExpectedRgba()
    {
        byte[] stored = [30, 20, 10, 255];

        Rgba32 decoded = VlmMinimapCleanupService.DecodeStoredMccvColor(stored, 0);

        Assert.Equal(new Rgba32(10, 20, 30, 255), decoded);
    }

    [Fact]
    public void DecodeStoredMccvColor_RawPngPixel_ReturnsExpectedRgba()
    {
        Rgba32 stored = new(30, 20, 10, 255);

        Rgba32 decoded = VlmMinimapCleanupService.DecodeStoredMccvColor(stored);

        Assert.Equal(new Rgba32(10, 20, 30, 255), decoded);
    }

    [Fact]
    public void RemoveMccvTint_NeutralOverlay_MatchesInverseShaderTint()
    {
        using Image<Rgba32> source = CreateSolidImage(120, 140, 160);
        using Image<Rgba32> neutral = CreateSolidImage(127, 127, 127);

        byte[] outputBytes = VlmMinimapCleanupService.RemoveMccvTint(source, neutral);

        using Image<Rgba32> output = Image.Load<Rgba32>(outputBytes);
        Assert.Equal(new Rgba32(120, 141, 161, 255), output[0, 0]);
    }

    [Fact]
    public void RemoveMccvTint_RawStoredOrderTintInvertsShaderTint()
    {
        using Image<Rgba32> source = CreateSolidImage(120, 140, 160);
        using Image<Rgba32> tint = CreateSolidImage(127, 117, 147);

        byte[] outputBytes = VlmMinimapCleanupService.RemoveMccvTint(source, tint);

        using Image<Rgba32> output = Image.Load<Rgba32>(outputBytes);
        Assert.Equal(new Rgba32(104, 153, 161, 255), output[0, 0]);
    }

    private static Image<Rgba32> CreateSolidImage(byte r, byte g, byte b)
    {
        Image<Rgba32> image = new(4, 4);
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
                image[x, y] = new Rgba32(r, g, b, 255);
        }

        return image;
    }
}