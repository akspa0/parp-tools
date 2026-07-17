namespace WowViewer.Core.Maps;

/// <summary>
/// A build-scoped candidate for an authored minimap's baked lighting tint. The candidate is used
/// only for downstream bucketing; it does not prove the client captured the minimap at this time.
/// </summary>
public sealed record MinimapLightingTimeCandidate(
    float TimeOfDayHours,
    float Red,
    float Green,
    float Blue,
    string Source);

/// <summary>
/// Provenance derived by comparing an authored minimap with an unlit terrain-only reconstruction.
/// MCSH is retained as an independent terrain signal; it is only correlated here to identify the
/// exceptional minimaps that appear to have static terrain shadows baked into their RGB.
/// </summary>
public sealed record MinimapLightingProvenance(
    string ContractVersion,
    string InferenceStatus,
    int ValidPixelCount,
    float? TintRed,
    float? TintGreen,
    float? TintBlue,
    float? TintStrength,
    float? TintFit,
    float? McshDarkeningCorrelation,
    float? EstimatedTimeOfDayHours,
    float? TimeOfDayConfidence,
    string TimeOfDayEvidence,
    string? TimeOfDayCandidateSource,
    string ShadingMatchStatus = "not_evaluated",
    float? ShadingMatchedTimeOfDayHours = null,
    float? ShadingMatchConfidence = null,
    string? ShadingMatchEvidence = null,
    float? ShadingMatchExcludedMcshFraction = null,
    string? ShadingMatchBuildFingerprint = null)
{
    public const string CurrentContractVersion = "minimap-lighting-provenance-v1";

    // Spec 111: the shading-match fields above are additive to the v1 tint-based contract. They
    // describe a distinct, geometric signal (which direction the terrain shadows fall) that the
    // tint-ratio inference below cannot see, and are populated independently by
    // WowViewer.Core.IO.Maps.MinimapShadingMatch. A tile can be tint-matched and
    // shading-not_evaluated, or vice versa -- the two signals do not imply each other.

    public IReadOnlyDictionary<string, object?> ToMetadata() => new Dictionary<string, object?>
    {
        ["contract_version"] = ContractVersion,
        ["inference_status"] = InferenceStatus,
        ["valid_pixel_count"] = ValidPixelCount,
        ["tint_rgb"] = TintRed is null || TintGreen is null || TintBlue is null
            ? null
            : new[] { TintRed.Value, TintGreen.Value, TintBlue.Value },
        ["tint_strength"] = TintStrength,
        ["tint_fit"] = TintFit,
        ["mcsh_darkening_correlation"] = McshDarkeningCorrelation,
        ["estimated_time_of_day_hours"] = EstimatedTimeOfDayHours,
        ["time_of_day_confidence"] = TimeOfDayConfidence,
        ["time_of_day_evidence"] = TimeOfDayEvidence,
        ["time_of_day_candidate_source"] = TimeOfDayCandidateSource,
        ["shading_match_status"] = ShadingMatchStatus,
        ["shading_matched_time_of_day_hours"] = ShadingMatchedTimeOfDayHours,
        ["shading_match_confidence"] = ShadingMatchConfidence,
        ["shading_match_evidence"] = ShadingMatchEvidence,
        ["shading_match_excluded_mcsh_fraction"] = ShadingMatchExcludedMcshFraction,
        ["shading_match_build_fingerprint"] = ShadingMatchBuildFingerprint,
    };

    public static MinimapLightingProvenance NotEvaluated(string reason) => new(
        CurrentContractVersion,
        reason,
        0,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        "not_inferred",
        null);

    public static MinimapLightingProvenance Infer(
        byte[,,] authoredMinimapRgb,
        byte[,,] unlitTerrainRgb,
        float[,]? mcshShadowMask256,
        IReadOnlyList<MinimapLightingTimeCandidate>? timeCandidates = null)
    {
        ArgumentNullException.ThrowIfNull(authoredMinimapRgb);
        ArgumentNullException.ThrowIfNull(unlitTerrainRgb);
        ValidateRgbPair(authoredMinimapRgb, unlitTerrainRgb);

        var redRatios = new List<float>();
        var greenRatios = new List<float>();
        var blueRatios = new List<float>();
        int height = authoredMinimapRgb.GetLength(0);
        int width = authoredMinimapRgb.GetLength(1);

        // A very dark terrain baseline does not carry reliable colour-ratio information.
        const float minimumBaselineLuma = 24f / 255f;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float baseR = unlitTerrainRgb[y, x, 0] / 255f;
                float baseG = unlitTerrainRgb[y, x, 1] / 255f;
                float baseB = unlitTerrainRgb[y, x, 2] / 255f;
                if (Luma(baseR, baseG, baseB) < minimumBaselineLuma)
                    continue;

                redRatios.Add(ClampRatio(authoredMinimapRgb[y, x, 0] / 255f, baseR));
                greenRatios.Add(ClampRatio(authoredMinimapRgb[y, x, 1] / 255f, baseG));
                blueRatios.Add(ClampRatio(authoredMinimapRgb[y, x, 2] / 255f, baseB));
            }
        }

        if (redRatios.Count < 64)
            return NotEvaluated("insufficient_unlit_baseline_coverage");

        float tintRed = Median(redRatios);
        float tintGreen = Median(greenRatios);
        float tintBlue = Median(blueRatios);
        float tintStrength = (MathF.Abs(MathF.Log2(tintRed))
            + MathF.Abs(MathF.Log2(tintGreen))
            + MathF.Abs(MathF.Log2(tintBlue))) / 3f;

        float absoluteError = 0f;
        int fitSamples = 0;
        var shadowValues = new List<float>();
        var darkeningValues = new List<float>();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float baseR = unlitTerrainRgb[y, x, 0] / 255f;
                float baseG = unlitTerrainRgb[y, x, 1] / 255f;
                float baseB = unlitTerrainRgb[y, x, 2] / 255f;
                if (Luma(baseR, baseG, baseB) < minimumBaselineLuma)
                    continue;

                float authoredR = authoredMinimapRgb[y, x, 0] / 255f;
                float authoredG = authoredMinimapRgb[y, x, 1] / 255f;
                float authoredB = authoredMinimapRgb[y, x, 2] / 255f;
                float predictedR = Math.Clamp(baseR * tintRed, 0f, 1f);
                float predictedG = Math.Clamp(baseG * tintGreen, 0f, 1f);
                float predictedB = Math.Clamp(baseB * tintBlue, 0f, 1f);
                absoluteError += (MathF.Abs(authoredR - predictedR)
                    + MathF.Abs(authoredG - predictedG)
                    + MathF.Abs(authoredB - predictedB)) / 3f;
                fitSamples++;

                if (mcshShadowMask256 is not null && mcshShadowMask256.GetLength(0) > 0 && mcshShadowMask256.GetLength(1) > 0)
                {
                    int shadowY = ScaleCoordinate(y, height, mcshShadowMask256.GetLength(0));
                    int shadowX = ScaleCoordinate(x, width, mcshShadowMask256.GetLength(1));
                    float shadow = mcshShadowMask256[shadowY, shadowX];
                    if (float.IsFinite(shadow))
                    {
                        shadowValues.Add(Math.Clamp(shadow, 0f, 1f));
                        darkeningValues.Add(Luma(predictedR, predictedG, predictedB) - Luma(authoredR, authoredG, authoredB));
                    }
                }
            }
        }

        float tintFit = fitSamples == 0 ? 0f : Math.Clamp(1f - (absoluteError / fitSamples), 0f, 1f);
        float? mcshCorrelation = PearsonCorrelation(shadowValues, darkeningValues);
        bool tintLikely = tintStrength >= 0.075f && tintFit >= 0.55f;
        bool mcshLikely = mcshCorrelation is >= 0.30f;

        (float? timeHours, float? timeConfidence, string timeEvidence, string? candidateSource) =
            InferTimeOfDay(tintRed, tintGreen, tintBlue, tintLikely, tintFit, timeCandidates);

        string status = (tintLikely, mcshLikely) switch
        {
            (true, true) => "baked_tint_and_mcsh_likely",
            (true, false) => "baked_tint_likely",
            (false, true) => "baked_mcsh_likely",
            _ => "unlit_or_unclassified",
        };

        return new MinimapLightingProvenance(
            CurrentContractVersion,
            status,
            redRatios.Count,
            tintRed,
            tintGreen,
            tintBlue,
            tintStrength,
            tintFit,
            mcshCorrelation,
            timeHours,
            timeConfidence,
            timeEvidence,
            candidateSource);
    }

    private static (float? TimeHours, float? Confidence, string Evidence, string? CandidateSource) InferTimeOfDay(
        float tintRed,
        float tintGreen,
        float tintBlue,
        bool tintLikely,
        float tintFit,
        IReadOnlyList<MinimapLightingTimeCandidate>? candidates)
    {
        if (!tintLikely)
            return (null, null, "no_baked_tint_detected", null);
        if (candidates is null || candidates.Count == 0)
            return (null, null, "no_build_scoped_lighting_candidates", null);

        (float observedR, float observedG, float observedB) = NormalizeChromaticity(tintRed, tintGreen, tintBlue);
        MinimapLightingTimeCandidate? best = null;
        float bestDistance = float.PositiveInfinity;
        foreach (MinimapLightingTimeCandidate candidate in candidates)
        {
            if (!float.IsFinite(candidate.TimeOfDayHours)
                || !float.IsFinite(candidate.Red)
                || !float.IsFinite(candidate.Green)
                || !float.IsFinite(candidate.Blue))
            {
                continue;
            }

            (float candidateR, float candidateG, float candidateB) = NormalizeChromaticity(candidate.Red, candidate.Green, candidate.Blue);
            float distance = MathF.Sqrt(
                ((observedR - candidateR) * (observedR - candidateR))
                + ((observedG - candidateG) * (observedG - candidateG))
                + ((observedB - candidateB) * (observedB - candidateB)));
            if (distance < bestDistance)
            {
                best = candidate;
                bestDistance = distance;
            }
        }

        const float maximumChromaDistance = 0.12f;
        if (best is null || bestDistance > maximumChromaDistance)
            return (null, null, "no_lighting_chroma_match", null);

        float confidence = Math.Clamp(tintFit * (1f - (bestDistance / maximumChromaDistance)), 0f, 1f);
        return (
            best.TimeOfDayHours,
            confidence,
            "inferred_global_lighting_chroma_match_not_capture_proof",
            best.Source);
    }

    private static void ValidateRgbPair(byte[,,] authored, byte[,,] baseline)
    {
        if (authored.GetLength(2) < 3 || baseline.GetLength(2) < 3
            || authored.GetLength(0) != baseline.GetLength(0)
            || authored.GetLength(1) != baseline.GetLength(1))
        {
            throw new ArgumentException("Authored minimap and unlit terrain baseline must have matching RGB dimensions.");
        }
    }

    private static float ClampRatio(float numerator, float denominator) =>
        Math.Clamp(numerator / MathF.Max(denominator, 1f / 255f), 0.25f, 4f);

    private static float Median(List<float> values)
    {
        values.Sort();
        int middle = values.Count / 2;
        return (values.Count & 1) == 0
            ? (values[middle - 1] + values[middle]) * 0.5f
            : values[middle];
    }

    private static float? PearsonCorrelation(IReadOnlyList<float> x, IReadOnlyList<float> y)
    {
        if (x.Count < 64 || x.Count != y.Count)
            return null;

        float meanX = x.Average();
        float meanY = y.Average();
        float covariance = 0f;
        float varianceX = 0f;
        float varianceY = 0f;
        for (int i = 0; i < x.Count; i++)
        {
            float dx = x[i] - meanX;
            float dy = y[i] - meanY;
            covariance += dx * dy;
            varianceX += dx * dx;
            varianceY += dy * dy;
        }

        float denominator = MathF.Sqrt(varianceX * varianceY);
        return denominator > 1e-8f ? Math.Clamp(covariance / denominator, -1f, 1f) : null;
    }

    private static (float R, float G, float B) NormalizeChromaticity(float red, float green, float blue)
    {
        float magnitude = MathF.Sqrt((red * red) + (green * green) + (blue * blue));
        return magnitude > 1e-8f ? (red / magnitude, green / magnitude, blue / magnitude) : (0f, 0f, 0f);
    }

    private static float Luma(float red, float green, float blue) =>
        (red * 0.2126f) + (green * 0.7152f) + (blue * 0.0722f);

    private static int ScaleCoordinate(int coordinate, int sourceSize, int targetSize)
    {
        if (targetSize <= 1 || sourceSize <= 1)
            return 0;

        return Math.Clamp((int)MathF.Round(coordinate * (targetSize - 1f) / (sourceSize - 1f)), 0, targetSize - 1);
    }
}
