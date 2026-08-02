using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Result of a per-map lighting-baseline survey.
/// </summary>
/// <param name="Map">Map name.</param>
/// <param name="Build">Build identity.</param>
/// <param name="MeanLuma">Cross-tile mean luma (0..1).</param>
/// <param name="StdLuma">Cross-tile luma standard deviation (0..1).</param>
/// <param name="BaselinePresent">True when cross-tile variance is small relative to within-tile variance.</param>
/// <param name="BuildRecognised">Whether the build was recognised by era-gating (FR-006).</param>
public sealed record LightingBaselineResult(
    string Map,
    string Build,
    float MeanLuma,
    float StdLuma,
    bool BaselinePresent,
    bool BuildRecognised);

/// <summary>
/// Tests the global lighting normalisation hypothesis — whether the authored tiles of a map share a
/// common lighting baseline (FR-016).
///
/// WHY THIS EXISTS: the authored tiles of a map may share a common brightness/contrast baseline from
/// the client's minimap renderer. If so, per-tile lighting differences between our synthesizer and the
/// authored tiles are not purely codec damage; they are a separate systematic offset that must be
/// measured and removed before any per-tile comparison or calibration decision is trusted.
/// </summary>
public static class MinimapLightingBaseline
{
    /// <summary>
    /// Survey a set of authored tiles for a shared lighting baseline. A baseline is present when
    /// cross-tile luma variance is small relative to within-tile variance.
    /// </summary>
    public static LightingBaselineResult Survey(
        string map,
        string build,
        IEnumerable<byte[,,]> authoredTiles,
        bool buildRecognised)
    {
        ArgumentNullException.ThrowIfNull(authoredTiles);

        // Per-tile means and within-tile variances. The baseline question is whether the tile MEANS
        // agree (small cross-tile spread) relative to how much each tile varies internally.
        var tileMeans = new List<double>();
        double withinVarianceSum = 0.0;
        double overallSum = 0.0;
        int pixelCount = 0;

        foreach (byte[,,] tile in authoredTiles)
        {
            int height = tile.GetLength(0);
            int width = tile.GetLength(1);
            if (height <= 0 || width <= 0 || tile.GetLength(2) < 3)
                continue;

            double tileSum = 0.0, tileSumSquares = 0.0;
            int tilePixels = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float luma = Luma(
                        tile[y, x, 0] / 255f,
                        tile[y, x, 1] / 255f,
                        tile[y, x, 2] / 255f);
                    tileSum += luma;
                    tileSumSquares += luma * luma;
                    tilePixels++;
                }
            }

            if (tilePixels == 0)
                continue;

            double tileMean = tileSum / tilePixels;
            double tileVariance = Math.Max((tileSumSquares / tilePixels) - (tileMean * tileMean), 0.0);

            tileMeans.Add(tileMean);
            withinVarianceSum += tileVariance;
            overallSum += tileSum;
            pixelCount += tilePixels;
        }

        if (tileMeans.Count == 0 || pixelCount == 0)
            return new LightingBaselineResult(map, build, 0f, 0f, BaselinePresent: false, buildRecognised);

        // Cross-tile variance = variance of the tile means (how much tiles disagree on brightness).
        double meanOfMeans = tileMeans.Average();
        double crossVariance = tileMeans.Sum(m => (m - meanOfMeans) * (m - meanOfMeans)) / tileMeans.Count;
        double crossStd = Math.Sqrt(crossVariance);
        double withinVariance = withinVarianceSum / tileMeans.Count;

        // A shared baseline is present when the tile means agree (small cross-tile spread) relative
        // to how much each tile varies internally. Conservative default: cross-std under 1/4 of
        // within-std.
        bool baselinePresent = withinVariance > 1e-9 && crossVariance < withinVariance * 0.25;

        return new LightingBaselineResult(
            map,
            build,
            (float)meanOfMeans,
            (float)crossStd,
            baselinePresent,
            buildRecognised);
    }

    /// <summary>
    /// Normalise a synthetic tile to the authored baseline (mean/std match) before scoring, so
    /// comparison attributes only residual (non-baseline) differences.
    /// </summary>
    public static Image<Rgba32> NormalizeToBaseline(Image<Rgba32> synthetic, LightingBaselineResult baseline)
    {
        ArgumentNullException.ThrowIfNull(synthetic);
        ArgumentNullException.ThrowIfNull(baseline);

        // Compute the synthetic tile's current mean/std luma.
        double sum = 0.0, sumSquares = 0.0;
        int count = 0;
        for (int y = 0; y < synthetic.Height; y++)
        {
            for (int x = 0; x < synthetic.Width; x++)
            {
                Rgba32 p = synthetic[x, y];
                float luma = Luma(p.R / 255f, p.G / 255f, p.B / 255f);
                sum += luma;
                sumSquares += luma * luma;
                count++;
            }
        }

        if (count == 0)
            return synthetic.Clone();

        double mean = sum / count;
        double variance = Math.Max((sumSquares / count) - (mean * mean), 0.0);
        double std = Math.Sqrt(variance);

        var output = new Image<Rgba32>(synthetic.Width, synthetic.Height);
        for (int y = 0; y < synthetic.Height; y++)
        {
            for (int x = 0; x < synthetic.Width; x++)
            {
                Rgba32 p = synthetic[x, y];
                float luma = Luma(p.R / 255f, p.G / 255f, p.B / 255f);

                // Standardise to zero-mean/unit-std, then rescale to the baseline mean/std.
                float normalised = std > 1e-9
                    ? (float)((luma - mean) / std) * baseline.StdLuma + baseline.MeanLuma
                    : baseline.MeanLuma;

                // Preserve the per-pixel hue by scaling RGB by the luma ratio.
                float ratio = luma > 1e-9 ? normalised / luma : 1f;
                byte r = ClampByte(p.R * ratio);
                byte g = ClampByte(p.G * ratio);
                byte b = ClampByte(p.B * ratio);
                output[x, y] = new Rgba32(r, g, b, p.A);
            }
        }

        return output;
    }

    private static float Luma(float r, float g, float b) => (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

    private static byte ClampByte(float value) => (byte)Math.Clamp((int)Math.Round(value), 0, 255);
}
