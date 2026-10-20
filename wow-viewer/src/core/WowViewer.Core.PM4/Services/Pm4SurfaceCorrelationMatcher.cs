using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4SurfaceCorrelationMatcher
{
    public static IReadOnlyList<SurfaceMatchResult> Match(
        IReadOnlyList<SurfaceCorrelationFingerprint> pm4Fingerprints,
        SurfaceCorrelationDatabase database,
        SurfaceMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(pm4Fingerprints);
        ArgumentNullException.ThrowIfNull(database);

        SurfaceMatchOptions opts = options ?? SurfaceMatchOptions.Default;
        List<SurfaceMatchResult> results = new(pm4Fingerprints.Count);

        foreach (SurfaceCorrelationFingerprint pm4 in pm4Fingerprints)
        {
            SurfaceMatchResult result = MatchOne(pm4, database.Records, opts);
            results.Add(result);
        }

        return results;
    }

    public static SurfaceMatchResult MatchOne(
        SurfaceCorrelationFingerprint pm4,
        SurfaceCorrelationDatabase database,
        SurfaceMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(database);
        SurfaceMatchOptions opts = options ?? new SurfaceMatchOptions(
            EdgeBinSize: database.EdgeBinSize,
            AreaBinSize: database.AreaBinSize,
            NormalAlignmentBinSize: database.NormalAlignmentBinSize,
            PlanarOffsetBinSize: database.PlanarOffsetBinSize);
        return MatchOne(pm4, database.Records, opts);
    }

    public static SurfaceMatchResult MatchOne(
        SurfaceCorrelationFingerprint pm4,
        IReadOnlyList<SurfaceCorrelationFingerprint> referenceRecords,
        SurfaceMatchOptions options)
    {
        ArgumentNullException.ThrowIfNull(pm4);
        ArgumentNullException.ThrowIfNull(referenceRecords);
        ArgumentNullException.ThrowIfNull(options);

        List<string> rationale = new(4)
        {
            $"ck24Type=0x{pm4.Ck24Type:X2} surfaces={pm4.SurfaceCount} triangles={pm4.TriangleCount} verts={pm4.VertexCount}",
        };

        string? expectedKind = ResolveExpectedAssetKind(pm4.Ck24Type);
        if (expectedKind is null)
        {
            rationale.Add($"ck24Type 0x{pm4.Ck24Type:X2} is not WMO/M2-matchable.");
            return BuildResult(pm4, "Ineligible", true, rationale, []);
        }

        bool hasMatchingKind = false;
        List<CandidateEvaluation> evaluations = new(referenceRecords.Count);
        for (int i = 0; i < referenceRecords.Count; i++)
        {
            SurfaceCorrelationFingerprint candidate = referenceRecords[i];
            if (!string.IsNullOrEmpty(expectedKind) &&
                !string.Equals(candidate.AssetKind, expectedKind, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            hasMatchingKind = true;
            CandidateEvaluation? eval = EvaluateCandidate(pm4, candidate, options);
            if (eval is not null)
                evaluations.Add(eval.Value);
        }

        if (!hasMatchingKind)
        {
            rationale.Add($"ck24Type 0x{pm4.Ck24Type:X2} maps to '{expectedKind}' — no '{expectedKind}' reference records found in database.");
            return BuildResult(pm4, "Ineligible", true, rationale, []);
        }

        evaluations.Sort(static (a, b) => b.SymmetricScore.CompareTo(a.SymmetricScore));

        int maxCandidates = Math.Max(1, options.MaxCandidates);
        double topScore = evaluations.Count > 0 ? evaluations[0].SymmetricScore : 0;
        double secondScore = evaluations.Count > 1 ? evaluations[1].SymmetricScore : double.NegativeInfinity;

        string status;
        if (evaluations.Count == 0 || topScore < options.MinScore)
        {
            status = "Unresolved";
            rationale.Add($"best score {topScore:F3} below minimum {options.MinScore:F2}.");
        }
        else if (Math.Abs(topScore - secondScore) <= options.AmbiguousWindow)
        {
            status = "Ambiguous";
            rationale.Add($"top candidates too close: {topScore:F3} vs {secondScore:F3}.");
        }
        else
        {
            status = "Matched";
            rationale.Add($"top candidate '{evaluations[0].Candidate.AssetPath}' score={topScore:F3} pm4Coverage={evaluations[0].Pm4Coverage:F2} candidateCoverage={evaluations[0].CandidateCoverage:F2}.");
        }

        List<SurfaceMatchCandidate> candidates = new(Math.Min(evaluations.Count, maxCandidates));
        for (int i = 0; i < evaluations.Count && i < maxCandidates; i++)
        {
            CandidateEvaluation eval = evaluations[i];
            string candidateStatus = ResolveCandidateStatus(status, i, eval.SymmetricScore, topScore, secondScore, options);
            candidates.Add(new SurfaceMatchCandidate(
                eval.Candidate,
                i + 1,
                eval.Pm4TrianglesMatched,
                eval.Pm4TriangleTotal,
                eval.CandidateTrianglesMatched,
                eval.CandidateTriangleTotal,
                eval.Pm4Coverage,
                eval.CandidateCoverage,
                eval.SymmetricScore,
                candidateStatus,
                eval.Rationale));
        }

        bool reviewRequired = status != "Matched";
        return BuildResult(pm4, status, reviewRequired, rationale, candidates);
    }

    private static CandidateEvaluation? EvaluateCandidate(
        SurfaceCorrelationFingerprint pm4,
        SurfaceCorrelationFingerprint candidate,
        SurfaceMatchOptions options)
    {
        if (pm4.TriangleHistogram.Count == 0 || candidate.TriangleHistogram.Count == 0)
            return null;

        if (pm4.TriangleCount < 1 || candidate.TriangleCount < 1)
            return null;

        int pm4Matched = 0;
        int pm4Total = pm4.TriangleCount;
        int candidateMatched = 0;
        int candidateTotal = candidate.TriangleCount;

        foreach (var kv in pm4.TriangleHistogram)
        {
            if (candidate.TriangleHistogram.TryGetValue(kv.Key, out int candidateCount))
            {
                int matched = Math.Min(kv.Value, candidateCount);
                pm4Matched += matched;
                candidateMatched += matched;
            }
        }

        double pm4Coverage = (double)pm4Matched / pm4Total;
        double candidateCoverage = (double)candidateMatched / candidateTotal;
        double symmetricScore = pm4Coverage * candidateCoverage > 0
            ? 2.0 * pm4Coverage * candidateCoverage / (pm4Coverage + candidateCoverage)
            : 0;

        List<string> rationale =
        [
            $"pm4Coverage={pm4Coverage:F3} ({pm4Matched}/{pm4Total} triangles matched)",
            $"{candidate.AssetKind}Coverage={candidateCoverage:F3} ({candidateMatched}/{candidateTotal} triangles matched)",
            $"symmetricF1={symmetricScore:F3}",
        ];

        return new CandidateEvaluation(candidate, pm4Matched, pm4Total, candidateMatched, candidateTotal, pm4Coverage, candidateCoverage, symmetricScore, rationale);
    }

    private static string ResolveExpectedAssetKind(byte ck24Type)
    {
        return ck24Type switch
        {
            0x42 or 0x43 or 0xC0 or 0xC1 or 0xC2 or 0xC3 => "wmo",
            0x40 or 0x41 => "m2",
            _ => "",
        };
    }

    private static string ResolveCandidateStatus(
        string segmentStatus, int index,
        double evalScore, double topScore, double secondScore,
        SurfaceMatchOptions options)
    {
        if (segmentStatus == "Unresolved")
            return "Unresolved";

        if (segmentStatus == "Ambiguous")
            return Math.Abs(topScore - evalScore) <= options.AmbiguousWindow
                ? "Ambiguous"
                : "Unresolved";

        return index == 0 ? "Matched" : "Unresolved";
    }

    private static SurfaceMatchResult BuildResult(
        SurfaceCorrelationFingerprint pm4,
        string status,
        bool reviewRequired,
        IReadOnlyList<string> rationale,
        IReadOnlyList<SurfaceMatchCandidate> candidates)
    {
        return new SurfaceMatchResult(
            pm4.AssetId,
            pm4.AssetPath,
            pm4.Ck24Type,
            pm4.SurfaceCount,
            pm4.TriangleCount,
            pm4.VertexCount,
            status,
            reviewRequired,
            rationale,
            candidates);
    }

    private readonly record struct CandidateEvaluation(
        SurfaceCorrelationFingerprint Candidate,
        int Pm4TrianglesMatched,
        int Pm4TriangleTotal,
        int CandidateTrianglesMatched,
        int CandidateTriangleTotal,
        double Pm4Coverage,
        double CandidateCoverage,
        double SymmetricScore,
        IReadOnlyList<string> Rationale);
}
