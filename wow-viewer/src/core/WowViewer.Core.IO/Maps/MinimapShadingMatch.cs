using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Determines which time-of-day best explains an authored 0.5.3.3368 minimap's shading pattern, by
/// sweeping <see cref="TerrainMinimapCompositor"/> candidates through the production
/// <see cref="TerrainSolarDirection"/> path and correlating luma value patterns -- a signal
/// independent of the tint-ratio inference in <see cref="MinimapLightingProvenance"/>. This is an
/// inference, not proof of the historical capture time; see
/// <c>specs/111-minimap-lighting-calibration/contracts/minimap-lighting-calibration-contract.md</c>.
/// </summary>
public static class MinimapShadingMatch
{
    public const string RequiredBuildFingerprint = "0.5.3.3368";
    public const string EvidenceDirectionalStructureMatch = "directional_structure_match_not_capture_proof";

    private const string MatchedStatus = "matched";
    private const string LowConfidenceAmbiguousStatus = "low_confidence_ambiguous";
    private const string LowConfidenceFlatTerrainStatus = "low_confidence_flat_terrain";
    private const string NotEvaluatedStatus = "not_evaluated";

    public static MinimapLightingProvenance Evaluate(
        MinimapLightingProvenance existing,
        TerrainTileTensorPack pack,
        IReadOnlyDictionary<int, byte[,,]> texturesById,
        byte[,,] authoredMinimapRgb,
        string buildFingerprint,
        MinimapShadingMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(existing);
        ArgumentNullException.ThrowIfNull(pack);
        ArgumentNullException.ThrowIfNull(texturesById);
        ArgumentNullException.ThrowIfNull(authoredMinimapRgb);
        options ??= MinimapShadingMatchOptions.Default;

        // Scope gate: this feature is explicitly 0.5.3.3368-only (spec 111 FR-006). A tile from any
        // other build stays not_evaluated without rendering a single candidate.
        if (!IsRequiredBuild(buildFingerprint))
            return NotEvaluated(existing, buildFingerprint);

        if (authoredMinimapRgb.GetLength(2) < 3 || authoredMinimapRgb.GetLength(0) <= 0 || authoredMinimapRgb.GetLength(1) <= 0)
            return NotEvaluated(existing, buildFingerprint);

        if (pack.McnrNormalXyz is null)
        {
            // No decoded ground-truth terrain normals: the compositor cannot produce a meaningful
            // shading candidate, so this tile is excluded with an explicit reason rather than scored
            // against a degenerate (flat/white) render.
            return NotEvaluated(existing, buildFingerprint);
        }

        int resolution = authoredMinimapRgb.GetLength(0);
        float[,]? mcshMask = pack.McshShadowMask256;

        // TerrainSolarDirection holds bearing fixed all day and only cycles elevation, so any two
        // hours with the same elevation render an IDENTICAL candidate image: the whole "night" span
        // clamps to one shared elevation floor, and every daytime hour has an exact mirror on the
        // other side of solar noon (elevation(hour) == elevation(24-hour)). A naive best-vs-runner-up
        // margin is therefore comparing the winner against its own structural twin on most real
        // inputs, producing a near-zero margin regardless of match quality. Track each candidate's
        // elevation and require a genuinely distinct elevation before it can count as the runner-up.
        var candidates = new List<(float Hour, float Elevation, float Score01, float ExcludedFraction, float SignalStrength)>();
        for (float hour = 0f; hour < 24f; hour += options.TimeStepHours)
        {
            float gameTime = hour / 24f;
            float elevation = TerrainSolarDirection.EvaluateElevation(gameTime);
            var candidateOptions = new TerrainMinimapCompositionOptions(
                resolution,
                TerrainMinimapLighting.CreateWhiteTopEdge(gameTime));

            using Image<Rgba32> candidate = TerrainMinimapCompositor.Compose(pack, texturesById, candidateOptions);

            float correlation = ScoreCandidate(
                candidate,
                authoredMinimapRgb,
                mcshMask,
                options.McshExclusionThreshold,
                out float signalStrength,
                out float excludedFraction);
            float score01 = Math.Clamp((correlation + 1f) * 0.5f, 0f, 1f);
            candidates.Add((hour, elevation, score01, excludedFraction, signalStrength));
        }

        (float Hour, float Elevation, float Score01, float ExcludedFraction, float SignalStrength) best =
            candidates.MaxBy(static c => c.Score01);
        float bestScore = best.Score01;
        float bestTimeHours = best.Hour;
        float bestExcludedFraction = best.ExcludedFraction;
        float bestSignalStrength = best.SignalStrength;

        float secondBestScore = float.NegativeInfinity;
        foreach ((float Hour, float Elevation, float Score01, float ExcludedFraction, float SignalStrength) candidate in candidates)
        {
            bool distinctFromBest = MathF.Abs(candidate.Elevation - best.Elevation) > options.ElevationDistinctnessEpsilon;
            if (distinctFromBest && candidate.Score01 > secondBestScore)
                secondBestScore = candidate.Score01;
        }

        MatchClassification classification = Classify(
            bestScore,
            secondBestScore,
            bestSignalStrength,
            options.MinimumSignalStrength,
            options);

        return existing with
        {
            ShadingMatchStatus = classification.Status,
            ShadingMatchedTimeOfDayHours = classification.Status == LowConfidenceFlatTerrainStatus ? null : bestTimeHours,
            ShadingMatchConfidence = classification.Confidence,
            ShadingMatchEvidence = EvidenceDirectionalStructureMatch,
            ShadingMatchExcludedMcshFraction = bestExcludedFraction,
            ShadingMatchBuildFingerprint = buildFingerprint,
        };
    }

    /// <summary>
    /// Pure decision logic, isolated from image rendering so the matched/ambiguous/flat-terrain
    /// boundaries can be tested directly with contrived scores rather than pixel fixtures tuned to
    /// hit exact thresholds.
    /// </summary>
    public static MatchClassification Classify(
        float bestScore,
        float secondBestScore,
        float signalStrength,
        float minimumSignalStrength,
        MinimapShadingMatchOptions options)
    {
        if (signalStrength <= 0f || signalStrength < minimumSignalStrength)
            return new MatchClassification(LowConfidenceFlatTerrainStatus, 0f);

        float margin = bestScore - MathF.Max(secondBestScore, 0f);
        if (margin < options.AmbiguousMarginThreshold)
        {
            return new MatchClassification(
                LowConfidenceAmbiguousStatus,
                Math.Clamp(margin / options.AmbiguousMarginThreshold, 0f, 1f));
        }

        return new MatchClassification(
            MatchedStatus,
            Math.Clamp(margin / (options.AmbiguousMarginThreshold * 4f), 0f, 1f));
    }

    public readonly record struct MatchClassification(string Status, float Confidence);

    public static bool IsRequiredBuild(string? buildFingerprint) =>
        ClientBuildKey.TryParse(buildFingerprint, out ClientBuildKey key)
            && key == ClientBuildKey.FromVersion(RequiredBuildFingerprint);

    private static MinimapLightingProvenance NotEvaluated(MinimapLightingProvenance existing, string buildFingerprint) =>
        existing with
        {
            ShadingMatchStatus = NotEvaluatedStatus,
            ShadingMatchedTimeOfDayHours = null,
            ShadingMatchConfidence = null,
            ShadingMatchEvidence = null,
            ShadingMatchExcludedMcshFraction = null,
            ShadingMatchBuildFingerprint = buildFingerprint,
        };

    /// <summary>
    /// Pearson correlation, in [-1, 1], between the candidate's and the authored minimap's luma
    /// value fields over non-excluded pixels. For a single flat material, the compositor's output
    /// luma is (material luma) x (lighting value) -- a positive multiplicative transform of the
    /// underlying Lambert field -- so this correlation is exactly invariant to material/tint
    /// differences while remaining sensitive to how the lit *pattern* itself changes with elevation
    /// (different-tilted normals respond differently to a change in light elevation, even though the
    /// light's azimuth/bearing is fixed all day). That is the property gradient-direction comparison
    /// does not have: azimuth is fixed by <see cref="TerrainSolarDirection"/>, so the *direction* of
    /// shading barely changes across hours, only its value/contrast does.
    /// </summary>
    private static float ScoreCandidate(
        Image<Rgba32> candidate,
        byte[,,] authoredMinimapRgb,
        float[,]? mcshExclusionMask,
        float mcshExclusionThreshold,
        out float signalStrength,
        out float excludedMcshFraction)
    {
        int height = candidate.Height;
        int width = candidate.Width;
        int totalCount = height * width;

        var candidateLuma = new List<float>(totalCount);
        var authoredLuma = new List<float>(totalCount);
        int excludedCount = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (IsExcludedByMcsh(mcshExclusionMask, mcshExclusionThreshold, y, x, height, width))
                {
                    excludedCount++;
                    continue;
                }

                candidateLuma.Add(LumaFromRgba32(candidate[x, y]));
                authoredLuma.Add(LumaFromByteArray(authoredMinimapRgb, y, x));
            }
        }

        excludedMcshFraction = totalCount == 0 ? 0f : excludedCount / (float)totalCount;

        if (candidateLuma.Count < 4)
        {
            signalStrength = 0f;
            return 0f;
        }

        double meanCandidate = Average(candidateLuma);
        double meanAuthored = Average(authoredLuma);
        double covariance = 0.0;
        double varianceCandidate = 0.0;
        double varianceAuthored = 0.0;
        for (int i = 0; i < candidateLuma.Count; i++)
        {
            double dc = candidateLuma[i] - meanCandidate;
            double da = authoredLuma[i] - meanAuthored;
            covariance += dc * da;
            varianceCandidate += dc * dc;
            varianceAuthored += da * da;
        }

        double denominator = Math.Sqrt(varianceCandidate * varianceAuthored);
        signalStrength = (float)(denominator / candidateLuma.Count);
        return denominator > 1e-9 ? (float)Math.Clamp(covariance / denominator, -1.0, 1.0) : 0f;
    }

    private static double Average(List<float> values)
    {
        double sum = 0.0;
        foreach (float value in values)
            sum += value;
        return sum / values.Count;
    }

    private static bool IsExcludedByMcsh(
        float[,]? mcshExclusionMask,
        float mcshExclusionThreshold,
        int y,
        int x,
        int sourceHeight,
        int sourceWidth)
    {
        if (mcshExclusionMask is null || mcshExclusionMask.GetLength(0) <= 0 || mcshExclusionMask.GetLength(1) <= 0)
            return false;

        int shadowY = TerrainMinimapCompositor.ScaleCoordinate(y, sourceHeight, mcshExclusionMask.GetLength(0));
        int shadowX = TerrainMinimapCompositor.ScaleCoordinate(x, sourceWidth, mcshExclusionMask.GetLength(1));
        float shadowValue = mcshExclusionMask[shadowY, shadowX];
        return float.IsFinite(shadowValue) && shadowValue >= mcshExclusionThreshold;
    }

    private static float LumaFromRgba32(Rgba32 pixel) =>
        ((pixel.R * 0.2126f) + (pixel.G * 0.7152f) + (pixel.B * 0.0722f)) / 255f;

    private static float LumaFromByteArray(byte[,,] rgb, int y, int x) =>
        ((rgb[y, x, 0] * 0.2126f) + (rgb[y, x, 1] * 0.7152f) + (rgb[y, x, 2] * 0.0722f)) / 255f;
}

/// <summary>Tunable thresholds for <see cref="MinimapShadingMatch"/>; defaults are conservative.</summary>
public sealed record MinimapShadingMatchOptions(
    float TimeStepHours = 1f,
    float McshExclusionThreshold = 0.5f,
    float AmbiguousMarginThreshold = 0.05f,
    float MinimumSignalStrength = 0.001f,
    float ElevationDistinctnessEpsilon = 1e-3f)
{
    public static MinimapShadingMatchOptions Default { get; } = new();
}
