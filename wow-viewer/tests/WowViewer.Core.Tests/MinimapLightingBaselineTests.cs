using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class MinimapLightingBaselineTests
{
    private const int Size = 8;

    private static byte[,,] FlatTile(byte value)
    {
        var tile = new byte[Size, Size, 3];
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                tile[y, x, 0] = value;
                tile[y, x, 1] = value;
                tile[y, x, 2] = value;
            }

        return tile;
    }

    private static byte[,,] GradientTile(byte baseValue)
    {
        var tile = new byte[Size, Size, 3];
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                byte v = (byte)Math.Clamp(baseValue + (x * 4), 0, 255);
                tile[y, x, 0] = v;
                tile[y, x, 1] = v;
                tile[y, x, 2] = v;
            }

        return tile;
    }

    [Fact]
    public void Survey_DetectsSharedBaseline_WhenTilesAgree()
    {
        // All tiles share a common brightness (small cross-tile spread relative to within-tile).
        byte[][,,] tiles =
        [
            GradientTile(100),
            GradientTile(104),
            GradientTile(108),
        ];

        LightingBaselineResult result = MinimapLightingBaseline.Survey("map", "build", tiles, buildRecognised: true);

        Assert.True(result.BaselinePresent);
        Assert.True(result.BuildRecognised);
    }

    [Fact]
    public void Survey_DoesNotDetectBaseline_WhenTilesHaveIndependentExposures()
    {
        // Tiles span a wide brightness range -> no shared baseline.
        byte[][,,] tiles =
        [
            FlatTile(20),
            FlatTile(120),
            FlatTile(220),
        ];

        LightingBaselineResult result = MinimapLightingBaseline.Survey("map", "build", tiles, buildRecognised: true);

        Assert.False(result.BaselinePresent);
    }

    [Fact]
    public void Survey_FlagsUnrecognisedBuild()
    {
        byte[][,,] tiles = [GradientTile(100)];
        LightingBaselineResult result = MinimapLightingBaseline.Survey("map", "build", tiles, buildRecognised: false);
        Assert.False(result.BuildRecognised);
    }

    [Fact]
    public void NormalizeToBaseline_BringsMeanToBaseline()
    {
        // A synthetic tile that is much brighter than the baseline.
        using var synthetic = new Image<Rgba32>(Size, Size);
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
                synthetic[x, y] = new Rgba32(200, 200, 200, 255);

        var baseline = new LightingBaselineResult("map", "build", MeanLuma: 0.3f, StdLuma: 0.05f, BaselinePresent: true, BuildRecognised: true);

        using Image<Rgba32> normalised = MinimapLightingBaseline.NormalizeToBaseline(synthetic, baseline);

        // The normalised tile's mean luma should be near the baseline mean (0.3).
        double sum = 0;
        int count = 0;
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                Rgba32 p = normalised[x, y];
                sum += (0.2126f * p.R / 255f) + (0.7152f * p.G / 255f) + (0.0722f * p.B / 255f);
                count++;
            }

        double mean = sum / count;
        Assert.InRange(mean, 0.25, 0.35);
    }
}
