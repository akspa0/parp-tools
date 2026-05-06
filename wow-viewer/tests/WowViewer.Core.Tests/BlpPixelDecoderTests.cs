using SereniaBLPLib;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.Tests;

public sealed class BlpPixelDecoderTests
{
    [Fact]
    public void DecodeRealBlpFile_ProducesValidPixels()
    {
        string blpPath = Path.Combine(GetWowViewerRoot(), "test_data", "0.5.3", "tree", "Dungeons", "Textures", "decoration", "JLO_BBAY_NET.blp");
        if (!File.Exists(blpPath))
            return;

        using FileStream stream = File.OpenRead(blpPath);
        using BlpFile blp = new(stream);

        byte[] pixels = blp.GetPixels(0, out int w, out int h, bgra: false);

        Assert.True(w > 0);
        Assert.True(h > 0);
        Assert.Equal(w * h * 4, pixels.Length);

        bool hasNonZeroAlpha = false;
        bool hasNonBlackColor = false;
        for (int i = 0; i < pixels.Length; i += 4)
        {
            if (pixels[i + 3] > 0)
                hasNonZeroAlpha = true;
            if (pixels[i] > 0 || pixels[i + 1] > 0 || pixels[i + 2] > 0)
                hasNonBlackColor = true;
            if (hasNonZeroAlpha && hasNonBlackColor)
                break;
        }

        Assert.True(hasNonZeroAlpha);
        Assert.True(hasNonBlackColor);
    }

    [Fact]
    public void DecodeBlpToImageSharp_ProducesValidImage()
    {
        string blpPath = Path.Combine(GetWowViewerRoot(), "test_data", "0.5.3", "tree", "Dungeons", "Textures", "decoration", "JLO_BBAY_NET.blp");
        if (!File.Exists(blpPath))
            return;

        using FileStream stream = File.OpenRead(blpPath);
        using BlpFile blp = new(stream);

        SixLabors.ImageSharp.Image<Rgba32> image = blp.GetImage(0);

        Assert.True(image.Width > 0);
        Assert.True(image.Height > 0);
        Assert.NotEqual(new Rgba32(0, 0, 0, 0), image[0, 0]);

        image.Dispose();
    }

    private static string GetWowViewerRoot()
    {
        string? current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            if (File.Exists(Path.Combine(current, "WowViewer.slnx")))
                return current;

            current = Directory.GetParent(current)?.FullName;
        }

        throw new DirectoryNotFoundException("Could not locate wow-viewer workspace root.");
    }
}
