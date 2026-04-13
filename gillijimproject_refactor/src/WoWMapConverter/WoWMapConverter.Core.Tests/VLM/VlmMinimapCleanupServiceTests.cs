using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.VLM;
using Xunit;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class VlmMinimapCleanupServiceTests
{
    [Fact]
    public void RemoveMccvTint_NeutralOverlay_LeavesImageUnchanged()
    {
        using Image<Rgba32> source = CreateSolidImage(120, 140, 160);
        using Image<Rgba32> neutral = CreateSolidImage(127, 127, 127);

        byte[] outputBytes = VlmMinimapCleanupService.RemoveMccvTint(source, neutral);

        using Image<Rgba32> output = Image.Load<Rgba32>(outputBytes);
        Assert.Equal(new Rgba32(120, 140, 160, 255), output[0, 0]);
    }

    [Fact]
    public void RemoveMccvTint_PositiveTintSubtractsFromSource()
    {
        using Image<Rgba32> source = CreateSolidImage(120, 140, 160);
        using Image<Rgba32> tint = CreateSolidImage(147, 117, 127);

        byte[] outputBytes = VlmMinimapCleanupService.RemoveMccvTint(source, tint);

        using Image<Rgba32> output = Image.Load<Rgba32>(outputBytes);
        Assert.Equal(new Rgba32(100, 150, 160, 255), output[0, 0]);
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