using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Blp;

namespace WowViewer.Core.Tests;

public sealed class AlphaBlpCompatibilityServiceTests
{
    [Fact]
    public void NormalizeForAlphaClient_LeavesCompatibleNonSpecularTextureUnchanged()
    {
        using Image<Rgba32> image = new(64, 64, new Rgba32(12, 34, 56, 255));
        byte[] source = AlphaBlpCompatibilityService.EncodeBlp2(image);

        AlphaBlpCompatibilityResult result = AlphaBlpCompatibilityService.NormalizeForAlphaClient("Tileset\\test.blp", source);

        Assert.False(result.Rewritten);
        Assert.False(result.Resized);
        Assert.False(result.SpecularReencoded);
        Assert.Equal(source, result.Data);
    }

    [Fact]
    public void NormalizeForAlphaClient_ReencodesSpecularTextureEvenWhenAlreadySmall()
    {
        using Image<Rgba32> image = new(64, 64, new Rgba32(90, 180, 20, 200));
        byte[] source = AlphaBlpCompatibilityService.EncodeBlp2(image);

        AlphaBlpCompatibilityResult result = AlphaBlpCompatibilityService.NormalizeForAlphaClient("Tileset\\test_s.blp", source);

        Assert.True(result.Rewritten);
        Assert.False(result.Resized);
        Assert.True(result.SpecularReencoded);

        using MemoryStream stream = new(result.Data, writable: false);
        using BlpFile blp = new(stream);
        using Image<Rgba32> decoded = blp.GetImage(0);

        Assert.Equal(64, decoded.Width);
        Assert.Equal(64, decoded.Height);
    }

    [Fact]
    public void NormalizeForAlphaClient_ResizesOversizedTextureToAlphaLimits()
    {
        using Image<Rgba32> image = new(512, 300, new Rgba32(10, 200, 90, 255));
        byte[] source = AlphaBlpCompatibilityService.EncodeBlp2(image);

        AlphaBlpCompatibilityResult result = AlphaBlpCompatibilityService.NormalizeForAlphaClient("Tileset\\big_s.blp", source);

        Assert.True(result.Rewritten);
        Assert.True(result.Resized);
        Assert.True(result.SpecularReencoded);
        Assert.Equal(256, result.OutputWidth);
        Assert.Equal(256, result.OutputHeight);

        using MemoryStream stream = new(result.Data, writable: false);
        using BlpFile blp = new(stream);
        using Image<Rgba32> decoded = blp.GetImage(0);

        Assert.Equal(256, decoded.Width);
        Assert.Equal(256, decoded.Height);
    }
}