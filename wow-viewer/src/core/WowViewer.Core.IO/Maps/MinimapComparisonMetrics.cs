using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Quantitative agreement between a synthesized minimap tile and its authored counterpart.
///
/// WHY MORE THAN ONE NUMBER: this project has already been burned once by calibrating on mean
/// brightness alone. Exposure 20 was fitted to a 0.990 mean ratio and scored as a success while it
/// was flattening the terrain hillshade into 12.8% of albedo -- the mean was right and the image was
/// wrong. Any claim that a render got "better" has to move contrast and structure too, so those are
/// first-class fields here rather than something to check later.
/// </summary>
/// <param name="MeanRatio">Synthetic mean luma / authored mean luma. 1.0 is a match.</param>
/// <param name="ContrastRatio">
/// Synthetic luma standard deviation / authored. 1.0 is a match; below 1.0 means the render is
/// flatter than the real minimap, which is exactly the failure the tone map produced.
/// </param>
/// <param name="LumaCorrelation">
/// Pearson correlation of the luma fields, in [-1, 1]. Measures whether light and dark land in the
/// same PLACES, independent of overall brightness or contrast scaling -- so it responds to shadow
/// direction and terrain structure when the two ratios above cannot.
/// </param>
/// <param name="MeanAbsoluteError">Mean per-channel absolute difference in 0..1 sRGB.</param>
/// <param name="ChannelRatios">Per-channel synthetic/authored mean ratios; catches hue drift that luma hides.</param>
public readonly record struct MinimapComparisonMetrics(
    float AuthoredMeanLuma,
    float SyntheticMeanLuma,
    float MeanRatio,
    float AuthoredStdLuma,
    float SyntheticStdLuma,
    float ContrastRatio,
    float LumaCorrelation,
    float MeanAbsoluteError,
    Vector3 ChannelRatios,
    int PixelCount)
{
    /// <summary>
    /// Single 0..1 score combining the three things that must all hold: right brightness, right
    /// contrast, right structure. Deliberately a product of the three penalties rather than an
    /// average, so a render cannot score well by nailing brightness while losing all its contrast.
    /// </summary>
    public float Score
    {
        get
        {
            float brightness = RatioCloseness(MeanRatio);
            float contrast = RatioCloseness(ContrastRatio);
            float structure = Math.Clamp(LumaCorrelation, 0f, 1f);
            return brightness * contrast * structure;
        }
    }

    /// <summary>Maps a ratio to 0..1 closeness, where 1.0 means exactly 1:1 and 0.5x or 2x scores 0.5.</summary>
    private static float RatioCloseness(float ratio)
    {
        if (!float.IsFinite(ratio) || ratio <= 0f)
            return 0f;
        return ratio > 1f ? 1f / ratio : ratio;
    }

    public static MinimapComparisonMetrics Compare(byte[,,] authoredRgb, Image<Rgba32> synthetic)
    {
        ArgumentNullException.ThrowIfNull(authoredRgb);
        ArgumentNullException.ThrowIfNull(synthetic);

        int height = Math.Min(authoredRgb.GetLength(0), synthetic.Height);
        int width = Math.Min(authoredRgb.GetLength(1), synthetic.Width);
        if (height <= 0 || width <= 0 || authoredRgb.GetLength(2) < 3)
            return default;

        double authoredSum = 0.0, syntheticSum = 0.0;
        double authoredSquares = 0.0, syntheticSquares = 0.0, crossProducts = 0.0;
        double absoluteError = 0.0;
        Vector3 authoredChannelSum = Vector3.Zero;
        Vector3 syntheticChannelSum = Vector3.Zero;
        int count = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Rgba32 pixel = synthetic[x, y];
                var syntheticRgb = new Vector3(pixel.R / 255f, pixel.G / 255f, pixel.B / 255f);
                var authored = new Vector3(
                    authoredRgb[y, x, 0] / 255f,
                    authoredRgb[y, x, 1] / 255f,
                    authoredRgb[y, x, 2] / 255f);

                float authoredLuma = Luma(authored);
                float syntheticLuma = Luma(syntheticRgb);

                authoredSum += authoredLuma;
                syntheticSum += syntheticLuma;
                authoredSquares += authoredLuma * authoredLuma;
                syntheticSquares += syntheticLuma * syntheticLuma;
                crossProducts += authoredLuma * syntheticLuma;
                absoluteError += (Math.Abs(authored.X - syntheticRgb.X)
                    + Math.Abs(authored.Y - syntheticRgb.Y)
                    + Math.Abs(authored.Z - syntheticRgb.Z)) / 3.0;
                authoredChannelSum += authored;
                syntheticChannelSum += syntheticRgb;
                count++;
            }
        }

        if (count == 0)
            return default;

        double authoredMean = authoredSum / count;
        double syntheticMean = syntheticSum / count;
        double authoredVariance = Math.Max((authoredSquares / count) - (authoredMean * authoredMean), 0.0);
        double syntheticVariance = Math.Max((syntheticSquares / count) - (syntheticMean * syntheticMean), 0.0);
        double authoredStd = Math.Sqrt(authoredVariance);
        double syntheticStd = Math.Sqrt(syntheticVariance);

        double covariance = (crossProducts / count) - (authoredMean * syntheticMean);
        double denominator = authoredStd * syntheticStd;
        double correlation = denominator > 1e-9 ? Math.Clamp(covariance / denominator, -1.0, 1.0) : 0.0;

        Vector3 authoredChannelMean = authoredChannelSum / count;
        Vector3 syntheticChannelMean = syntheticChannelSum / count;

        return new MinimapComparisonMetrics(
            (float)authoredMean,
            (float)syntheticMean,
            authoredMean > 1e-6 ? (float)(syntheticMean / authoredMean) : 0f,
            (float)authoredStd,
            (float)syntheticStd,
            authoredStd > 1e-6 ? (float)(syntheticStd / authoredStd) : 0f,
            (float)correlation,
            (float)(absoluteError / count),
            SafeRatio(syntheticChannelMean, authoredChannelMean),
            count);
    }

    private static Vector3 SafeRatio(Vector3 numerator, Vector3 denominator) => new(
        denominator.X > 1e-6f ? numerator.X / denominator.X : 0f,
        denominator.Y > 1e-6f ? numerator.Y / denominator.Y : 0f,
        denominator.Z > 1e-6f ? numerator.Z / denominator.Z : 0f);

    private static float Luma(Vector3 rgb) => (0.2126f * rgb.X) + (0.7152f * rgb.Y) + (0.0722f * rgb.Z);

    public static string CsvHeader =>
        "tile_x,tile_y,variant,pixels,authored_mean,synthetic_mean,mean_ratio," +
        "authored_std,synthetic_std,contrast_ratio,luma_correlation,mae," +
        "ratio_r,ratio_g,ratio_b,score";

    public string ToCsvRow(int tileX, int tileY, string variant) =>
        $"{tileX},{tileY},{variant},{PixelCount}," +
        $"{AuthoredMeanLuma:0.#####},{SyntheticMeanLuma:0.#####},{MeanRatio:0.#####}," +
        $"{AuthoredStdLuma:0.#####},{SyntheticStdLuma:0.#####},{ContrastRatio:0.#####}," +
        $"{LumaCorrelation:0.#####},{MeanAbsoluteError:0.#####}," +
        $"{ChannelRatios.X:0.#####},{ChannelRatios.Y:0.#####},{ChannelRatios.Z:0.#####},{Score:0.#####}";
}
