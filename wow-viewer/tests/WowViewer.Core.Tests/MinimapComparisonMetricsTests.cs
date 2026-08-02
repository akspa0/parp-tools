using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class MinimapComparisonMetricsTests
{
    private const int Size = 16;

    /// <summary>A gradient, so the image has real contrast and structure to measure.</summary>
    private static byte[,,] AuthoredGradient(float scale = 1f, float offset = 0f)
    {
        var authored = new byte[Size, Size, 3];
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                byte value = (byte)Math.Clamp(((x / (float)(Size - 1) * scale) + offset) * 255f, 0f, 255f);
                authored[y, x, 0] = value;
                authored[y, x, 1] = value;
                authored[y, x, 2] = value;
            }
        }

        return authored;
    }

    private static Image<Rgba32> SyntheticFrom(byte[,,] source)
    {
        var image = new Image<Rgba32>(Size, Size);
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
                image[x, y] = new Rgba32(source[y, x, 0], source[y, x, 1], source[y, x, 2], 255);
        }

        return image;
    }

    [Fact]
    public void Compare_IdenticalImagesScorePerfectly()
    {
        byte[,,] authored = AuthoredGradient();
        using Image<Rgba32> synthetic = SyntheticFrom(authored);

        MinimapComparisonMetrics metrics = MinimapComparisonMetrics.Compare(authored, synthetic);

        Assert.Equal(1f, metrics.MeanRatio, 3);
        Assert.Equal(1f, metrics.ContrastRatio, 3);
        Assert.Equal(1f, metrics.LumaCorrelation, 3);
        Assert.Equal(0f, metrics.MeanAbsoluteError, 3);
        Assert.Equal(1f, metrics.Score, 3);
    }

    /// <summary>
    /// The exact failure the exposure-20 calibration missed: a render whose MEAN matches the authored
    /// image perfectly while its contrast has collapsed. Mean ratio alone calls this a success, so
    /// the score must not.
    /// </summary>
    [Fact]
    public void Compare_FlatRenderWithCorrectMeanIsNotScoredAsAMatch()
    {
        byte[,,] authored = AuthoredGradient();

        // Constant image at the authored mean: perfect brightness, zero contrast, no structure.
        var flat = new Image<Rgba32>(Size, Size, new Rgba32(127, 127, 127, 255));
        using Image<Rgba32> synthetic = flat;

        MinimapComparisonMetrics metrics = MinimapComparisonMetrics.Compare(authored, synthetic);

        Assert.InRange(metrics.MeanRatio, 0.95f, 1.05f);
        Assert.True(metrics.ContrastRatio < 0.05f, $"Flat render must show near-zero contrast ratio, got {metrics.ContrastRatio}.");
        Assert.True(metrics.Score < 0.1f, $"A flat render must not score well, got {metrics.Score}.");
    }

    [Fact]
    public void Compare_HalfContrastRenderIsReportedAsHalfContrast()
    {
        byte[,,] authored = AuthoredGradient();
        // Same mean, half the spread around it.
        byte[,,] halved = AuthoredGradient(scale: 0.5f, offset: 0.25f);
        using Image<Rgba32> synthetic = SyntheticFrom(halved);

        MinimapComparisonMetrics metrics = MinimapComparisonMetrics.Compare(authored, synthetic);

        Assert.InRange(metrics.MeanRatio, 0.95f, 1.05f);
        Assert.InRange(metrics.ContrastRatio, 0.45f, 0.55f);
        // Structure is preserved even though contrast is not, which is why they are separate metrics.
        Assert.Equal(1f, metrics.LumaCorrelation, 2);
    }

    /// <summary>
    /// Structure is independent of brightness and contrast scaling: a mirrored render can match both
    /// ratios exactly while putting every shadow on the wrong side. Only correlation catches it,
    /// which is what makes it the metric that responds to sun direction.
    /// </summary>
    [Fact]
    public void Compare_MirroredRenderMatchesBothRatiosButLosesCorrelation()
    {
        byte[,,] authored = AuthoredGradient();
        var mirrored = new byte[Size, Size, 3];
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                for (int channel = 0; channel < 3; channel++)
                    mirrored[y, x, channel] = authored[y, Size - 1 - x, channel];
            }
        }

        using Image<Rgba32> synthetic = SyntheticFrom(mirrored);
        MinimapComparisonMetrics metrics = MinimapComparisonMetrics.Compare(authored, synthetic);

        Assert.Equal(1f, metrics.MeanRatio, 3);
        Assert.Equal(1f, metrics.ContrastRatio, 3);
        Assert.True(metrics.LumaCorrelation < -0.9f, $"A mirrored gradient must anti-correlate, got {metrics.LumaCorrelation}.");
        Assert.Equal(0f, metrics.Score, 3);
    }

    /// <summary>Luma is channel-weighted, so a pure hue shift can hide in it. Channel ratios must not.</summary>
    [Fact]
    public void Compare_ChannelRatiosExposeAHueShift()
    {
        byte[,,] authored = AuthoredGradient();
        var image = new Image<Rgba32>(Size, Size);
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                byte value = authored[y, x, 0];
                image[x, y] = new Rgba32(value, value, (byte)Math.Min(255, value * 2), 255);
            }
        }

        using Image<Rgba32> synthetic = image;
        MinimapComparisonMetrics metrics = MinimapComparisonMetrics.Compare(authored, synthetic);

        Assert.InRange(metrics.ChannelRatios.X, 0.95f, 1.05f);
        Assert.True(metrics.ChannelRatios.Z > 1.3f, $"Blue was doubled; ratio should exceed 1.3, got {metrics.ChannelRatios.Z}.");
    }

    [Fact]
    public void Score_PenalisesBrightnessAndContrastMultiplicativelyNotOnAverage()
    {
        // Correct brightness, no contrast. An averaging score would still award ~0.5 here.
        var flatButBright = new MinimapComparisonMetrics(
            0.5f, 0.5f, MeanRatio: 1f,
            0.2f, 0f, ContrastRatio: 0f,
            LumaCorrelation: 1f,
            MeanAbsoluteError: 0.1f,
            ChannelRatios: System.Numerics.Vector3.One,
            PixelCount: 256);

        Assert.Equal(0f, flatButBright.Score, 5);
    }

    [Fact]
    public void CsvRow_MatchesTheHeaderColumnCount()
    {
        byte[,,] authored = AuthoredGradient();
        using Image<Rgba32> synthetic = SyntheticFrom(authored);
        MinimapComparisonMetrics metrics = MinimapComparisonMetrics.Compare(authored, synthetic);

        int headerColumns = MinimapComparisonMetrics.CsvHeader.Split(',').Length;
        int rowColumns = metrics.ToCsvRow(12, 34, "current").Split(',').Length;

        Assert.Equal(headerColumns, rowColumns);
    }
}
