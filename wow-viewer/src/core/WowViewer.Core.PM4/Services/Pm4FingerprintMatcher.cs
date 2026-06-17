using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4FingerprintMatcher
{
    public static IReadOnlyList<Pm4FingerprintMatchResult> Match(
        IReadOnlyList<Pm4FingerprintRecord> pm4Fingerprints,
        Pm4FingerprintDatabase wmoDatabase,
        Pm4FingerprintMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(pm4Fingerprints);
        ArgumentNullException.ThrowIfNull(wmoDatabase);

        Pm4FingerprintMatchOptions resolvedOptions = options ?? Pm4FingerprintMatchOptions.Default;
        IReadOnlyList<Pm4FingerprintRecord> wmoRecords = wmoDatabase.WmoRecords;

        List<Pm4FingerprintMatchResult> results = new(pm4Fingerprints.Count);

        for (int i = 0; i < pm4Fingerprints.Count; i++)
        {
            Pm4FingerprintMatchResult result = MatchOne(pm4Fingerprints[i], wmoRecords, resolvedOptions);
            results.Add(result);
        }

        return results;
    }

    public static Pm4FingerprintMatchResult MatchOne(
        Pm4FingerprintRecord pm4Fingerprint,
        IReadOnlyList<Pm4FingerprintRecord> wmoRecords,
        Pm4FingerprintMatchOptions options)
    {
        ArgumentNullException.ThrowIfNull(pm4Fingerprint);
        ArgumentNullException.ThrowIfNull(wmoRecords);
        ArgumentNullException.ThrowIfNull(options);

        string? expectedKind = ResolveExpectedAssetKind(pm4Fingerprint.Ck24Type);
        List<string> rationale = new(4)
        {
            $"ck24Type=0x{pm4Fingerprint.Ck24Type:X2} surfaces={pm4Fingerprint.SurfaceCount} dims={pm4Fingerprint.SortedDim0:F0}x{pm4Fingerprint.SortedDim1:F0}x{pm4Fingerprint.SortedDim2:F0}",
        };

        if (expectedKind is null)
        {
            rationale.Add($"ck24Type 0x{pm4Fingerprint.Ck24Type:X2} is not WMO/M2-matchable.");
            return BuildResult(pm4Fingerprint, Pm4FingerprintMatchStatus.Ineligible, true, rationale, []);
        }

        if (!string.Equals(expectedKind, "wmo", StringComparison.OrdinalIgnoreCase))
        {
            rationale.Add($"ck24Type 0x{pm4Fingerprint.Ck24Type:X2} maps to '{expectedKind}' — M2 fingerprint DB not yet built.");
            return BuildResult(pm4Fingerprint, Pm4FingerprintMatchStatus.Ineligible, true, rationale, []);
        }

        List<Pm4FingerprintRecord> prefiltered = PrefilterByDimensions(pm4Fingerprint, wmoRecords, options.DimPrefilterTolerance);
        rationale.Add($"dimension prefilter: {wmoRecords.Count} WMOs -> {prefiltered.Count} survivors (tolerance={options.DimPrefilterTolerance:F2})");

        if (prefiltered.Count == 0)
        {
            rationale.Add("no WMO candidates survived dimension prefilter.");
            return BuildResult(pm4Fingerprint, Pm4FingerprintMatchStatus.Unresolved, true, rationale, []);
        }

        List<CandidateEvaluation> evaluations = new(prefiltered.Count);
        for (int i = 0; i < prefiltered.Count; i++)
        {
            CandidateEvaluation? eval = EvaluateCandidate(pm4Fingerprint, prefiltered[i], options);
            if (eval is not null)
                evaluations.Add(eval.Value);
        }

        evaluations.Sort(static (a, b) => b.OverallScore.CompareTo(a.OverallScore));

        int resolvedMaxCandidates = Math.Max(1, options.MaxCandidates);
        List<Pm4FingerprintMatchCandidate> candidates = new(Math.Min(evaluations.Count, resolvedMaxCandidates));

        double topScore = evaluations.Count > 0 ? evaluations[0].OverallScore : 0;
        double secondScore = evaluations.Count > 1 ? evaluations[1].OverallScore : double.NegativeInfinity;

        Pm4FingerprintMatchStatus status;
        if (evaluations.Count == 0 || topScore < options.MinScore)
        {
            status = Pm4FingerprintMatchStatus.Unresolved;
            rationale.Add($"best score {topScore:F3} below minimum {options.MinScore:F2}.");
        }
        else if (Math.Abs(topScore - secondScore) <= options.AmbiguousWindow)
        {
            status = Pm4FingerprintMatchStatus.Ambiguous;
            rationale.Add($"top candidates too close: {topScore:F3} vs {secondScore:F3}.");
        }
        else
        {
            status = Pm4FingerprintMatchStatus.Matched;
            rationale.Add($"top candidate '{evaluations[0].Candidate.AssetPath}' score={topScore:F3}.");
        }

        for (int i = 0; i < evaluations.Count && i < resolvedMaxCandidates; i++)
        {
            CandidateEvaluation eval = evaluations[i];
            Pm4FingerprintMatchStatus candidateStatus = ResolveCandidateStatus(status, i, eval.OverallScore, topScore, secondScore, options);
            candidates.Add(new Pm4FingerprintMatchCandidate(
                eval.Candidate,
                i + 1,
                candidateStatus,
                eval.Metrics.FootprintOverlapRatio,
                eval.Metrics.FootprintAreaRatio,
                eval.Metrics.FootprintDistance,
                eval.Metrics.PlanarGap,
                eval.Metrics.VerticalGap,
                eval.Metrics.CenterDistance,
                eval.Metrics.PlanarOverlapRatio,
                eval.Metrics.VolumeOverlapRatio,
                eval.OverallScore,
                eval.Rationale));
        }

        bool reviewRequired = status != Pm4FingerprintMatchStatus.Matched;
        return BuildResult(pm4Fingerprint, status, reviewRequired, rationale, candidates);
    }

    private static List<Pm4FingerprintRecord> PrefilterByDimensions(
        Pm4FingerprintRecord pm4Fp,
        IReadOnlyList<Pm4FingerprintRecord> wmoRecords,
        float tolerance)
    {
        List<Pm4FingerprintRecord> survivors = new(wmoRecords.Count);

        float pm4D0 = pm4Fp.SortedDim0;
        float pm4D1 = pm4Fp.SortedDim1;
        float pm4D2 = pm4Fp.SortedDim2;

        for (int i = 0; i < wmoRecords.Count; i++)
        {
            Pm4FingerprintRecord wmo = wmoRecords[i];

            if (wmo.SortedDim0 <= 0 || wmo.SortedDim1 <= 0 || wmo.SortedDim2 <= 0)
                continue;

            if (!IsDimensionCompatible(pm4D0, wmo.SortedDim0, tolerance))
                continue;
            if (!IsDimensionCompatible(pm4D1, wmo.SortedDim1, tolerance))
                continue;
            if (!IsDimensionCompatible(pm4D2, wmo.SortedDim2, tolerance))
                continue;

            survivors.Add(wmo);
        }

        return survivors;
    }

    private static bool IsDimensionCompatible(float pm4Dim, float wmoDim, float tolerance)
    {
        if (pm4Dim <= 0 || wmoDim <= 0)
            return false;

        float maxDim = MathF.Max(pm4Dim, wmoDim);
        float ratio = MathF.Min(pm4Dim, wmoDim) / maxDim;
        return ratio >= (1f - tolerance);
    }

    private static CandidateEvaluation? EvaluateCandidate(
        Pm4FingerprintRecord pm4Fp,
        Pm4FingerprintRecord wmoFp,
        Pm4FingerprintMatchOptions options)
    {
        if (pm4Fp.NormalizedFootprintHull.Count < 3 || wmoFp.NormalizedFootprintHull.Count < 3)
            return null;

        Vector2[] pm4Hull = pm4Fp.FootprintHullAsVectors.ToArray();
        Vector2[] wmoHull = wmoFp.FootprintHullAsVectors.ToArray();

        Pm4CorrelationMetrics bestMetrics = default;
        double bestScore = double.NegativeInfinity;

        foreach ((bool flipX, bool flipY) in FlipCombinations)
        {
            Vector2[] flippedPm4 = FlipHullPoints(pm4Hull, flipX, flipY);

            Pm4CorrelationMetrics metrics = Pm4CorrelationMath.EvaluateMetrics(
                wmoFp.NormalizedBounds.Min, wmoFp.NormalizedBounds.Max,
                new Vector3(wmoFp.NormalizedCenter.X, wmoFp.NormalizedCenter.Y, 0f),
                wmoFp.NormalizedFootprintHull.Select(static p => p.AsVector2()).ToList(),
                wmoFp.FootprintArea,
                pm4Fp.NormalizedBounds.Min, pm4Fp.NormalizedBounds.Max,
                new Vector3(pm4Fp.NormalizedCenter.X, pm4Fp.NormalizedCenter.Y, 0f),
                flippedPm4,
                pm4Fp.FootprintArea);

            double score = ComputeOverallScore(metrics, pm4Fp, wmoFp);
            if (score > bestScore)
            {
                bestScore = score;
                bestMetrics = metrics;
            }
        }

        List<string> rationale =
        [
            $"footprintOverlap={bestMetrics.FootprintOverlapRatio:F3}",
            $"footprintDist={bestMetrics.FootprintDistance:F1}",
            $"planarGap={bestMetrics.PlanarGap:F1}",
            $"volumeOverlap={bestMetrics.VolumeOverlapRatio:F3}",
        ];

        return new CandidateEvaluation(wmoFp, bestMetrics, bestScore, rationale);
    }

    private static double ComputeOverallScore(Pm4CorrelationMetrics metrics, Pm4FingerprintRecord pm4Fp, Pm4FingerprintRecord wmoFp)
    {
        double footprintWeight = 0.45;
        double volumeWeight = 0.20;
        double areaWeight = 0.15;
        double distanceWeight = 0.10;
        double planarWeight = 0.05;
        double verticalWeight = 0.05;

        double footprintScore = metrics.FootprintOverlapRatio;
        double volumeScore = metrics.VolumeOverlapRatio;
        double areaScore = metrics.FootprintAreaRatio;
        double distanceScore = ScoreDistance(metrics.FootprintDistance);
        double planarScore = ScoreOverlap(metrics.PlanarOverlapRatio);
        double verticalScore = 1f - Math.Clamp(metrics.VerticalGap / Math.Max(1f, pm4Fp.SortedDim2), 0f, 1f);

        return footprintScore * footprintWeight
            + volumeScore * volumeWeight
            + areaScore * areaWeight
            + distanceScore * distanceWeight
            + planarScore * planarWeight
            + verticalScore * verticalWeight;
    }

    private static double ScoreDistance(float distance)
    {
        if (!float.IsFinite(distance) || distance <= 0)
            return 1.0;
        return 1.0 / (1.0 + distance / 50.0);
    }

    private static double ScoreOverlap(float ratio)
    {
        return Math.Clamp(ratio, 0f, 1f);
    }

    private static Vector2[] FlipHullPoints(Vector2[] hull, bool flipX, bool flipY)
    {
        if (!flipX && !flipY)
            return hull;

        Vector2[] flipped = new Vector2[hull.Length];
        for (int i = 0; i < hull.Length; i++)
        {
            flipped[i] = new Vector2(
                flipX ? -hull[i].X : hull[i].X,
                flipY ? -hull[i].Y : hull[i].Y);
        }

        return Pm4CorrelationMath.BuildConvexHull(flipped);
    }

    private static readonly (bool, bool)[] FlipCombinations =
    [
        (false, false),
        (true, false),
        (false, true),
        (true, true),
    ];

    private static string? ResolveExpectedAssetKind(byte ck24Type)
    {
        return ck24Type switch
        {
            0x42 or 0x43 or 0xC0 or 0xC1 or 0xC2 or 0xC3 => "wmo",
            0x40 or 0x41 => "m2",
            _ => null,
        };
    }

    private static Pm4FingerprintMatchStatus ResolveCandidateStatus(
        Pm4FingerprintMatchStatus segmentStatus, int index,
        double evalScore, double topScore, double secondScore,
        Pm4FingerprintMatchOptions options)
    {
        if (segmentStatus == Pm4FingerprintMatchStatus.Unresolved)
            return Pm4FingerprintMatchStatus.Unresolved;

        if (segmentStatus == Pm4FingerprintMatchStatus.Ambiguous)
            return Math.Abs(topScore - evalScore) <= options.AmbiguousWindow
                ? Pm4FingerprintMatchStatus.Ambiguous
                : Pm4FingerprintMatchStatus.Unresolved;

        return index == 0 ? Pm4FingerprintMatchStatus.Matched : Pm4FingerprintMatchStatus.Unresolved;
    }

    private static Pm4FingerprintMatchResult BuildResult(
        Pm4FingerprintRecord pm4Fp,
        Pm4FingerprintMatchStatus status,
        bool reviewRequired,
        IReadOnlyList<string> rationale,
        IReadOnlyList<Pm4FingerprintMatchCandidate> candidates)
    {
        return new Pm4FingerprintMatchResult(
            pm4Fp.AssetId,
            pm4Fp.AssetPath,
            pm4Fp.Ck24Type,
            0,
            pm4Fp.SurfaceCount,
            pm4Fp.VertexCount,
            pm4Fp.IndexCount,
            pm4Fp.SortedDim0,
            pm4Fp.SortedDim1,
            pm4Fp.SortedDim2,
            status,
            reviewRequired,
            rationale,
            candidates);
    }

    private readonly record struct CandidateEvaluation(
        Pm4FingerprintRecord Candidate,
        Pm4CorrelationMetrics Metrics,
        double OverallScore,
        IReadOnlyList<string> Rationale);
}
