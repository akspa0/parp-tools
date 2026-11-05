using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WowViewer.Core.PM4.Footprint;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// Pure, UI-free analysis that turns PM4 footprint prisms into ranked, reviewable object
/// proposals. The operator's accept/reject/replace decision is the commit gate; this service
/// only proposes and explains.
/// </summary>
public static class Pm4ShapeMatchAnalysisService
{
    /// <summary>Below this score a candidate is not credible enough to propose.</summary>
    public const double MinimumCredibleScore = 0.55;

    /// <summary>Below this score a credible proposal is flagged for closer review.</summary>
    public const double LowConfidenceThreshold = 0.70;

    /// <summary>Two proposals of the same model within this distance are a conflict.</summary>
    public const float ConflictDistance = 2.0f;

    public static Pm4ShapeMatchProposal Analyze(
        uint hullKey,
        Pm4FootprintPrism prism,
        IReadOnlyList<Pm4ModelCandidate> candidates,
        Pm4CentroidCalibration calibration,
        int maxCandidates = 5)
    {
        ArgumentNullException.ThrowIfNull(prism);
        ArgumentNullException.ThrowIfNull(candidates);

        List<Pm4DirectMatchResult> matches = Pm4DirectModelMatcher.MatchPrism(prism, candidates, maxCandidates);
        var ranked = new List<Pm4ShapeCandidate>(matches.Count);
        foreach (Pm4DirectMatchResult match in matches)
        {
            if (match.Score < MinimumCredibleScore)
                continue;

            ranked.Add(new Pm4ShapeCandidate(
                match.Candidate.ModelPath,
                match.Candidate.ModelName,
                match.Candidate.AssetKind,
                match.Score,
                match.SolvedScale,
                match.SolvedRotation,
                calibration.Apply(match.SolvedPosition),
                match.Explanation));
        }

        double confidence = ranked.Count > 0 ? ranked[0].Score : 0.0;
        bool unmatched = ranked.Count == 0;
        bool lowConfidence = !unmatched && confidence < LowConfidenceThreshold;

        return new Pm4ShapeMatchProposal(
            hullKey,
            prism.Centroid,
            new Vector3(prism.Centroid.X - prism.UnrotatedLength * 0.5f, prism.Centroid.Y - prism.UnrotatedWidth * 0.5f, prism.BaseZ),
            new Vector3(prism.Centroid.X + prism.UnrotatedLength * 0.5f, prism.Centroid.Y + prism.UnrotatedWidth * 0.5f, prism.TopZ),
            ranked,
            confidence,
            lowConfidence,
            unmatched,
            1);
    }

    public static IReadOnlyList<Pm4ShapeMatchProposal> AnalyzeBatch(
        IEnumerable<(uint HullKey, Pm4FootprintPrism Prism)> hulls,
        IReadOnlyList<Pm4ModelCandidate> candidates,
        Pm4CentroidCalibration calibration,
        int maxCandidates = 5)
    {
        ArgumentNullException.ThrowIfNull(hulls);
        var proposals = new List<Pm4ShapeMatchProposal>();
        foreach ((uint hullKey, Pm4FootprintPrism prism) in hulls)
            proposals.Add(Analyze(hullKey, prism, candidates, calibration, maxCandidates));

        IReadOnlyList<Pm4ShapeMatchConflict> conflicts = FindConflicts(proposals);
        if (conflicts.Count == 0)
            return proposals;

        var conflictCounts = new Dictionary<uint, int>();
        foreach (Pm4ShapeMatchConflict conflict in conflicts)
        {
            conflictCounts[conflict.HullKeyA] = conflictCounts.GetValueOrDefault(conflict.HullKeyA) + 1;
            conflictCounts[conflict.HullKeyB] = conflictCounts.GetValueOrDefault(conflict.HullKeyB) + 1;
        }

        return proposals
            .Select(proposal => proposal with
            {
                ConflictCount = 1 + conflictCounts.GetValueOrDefault(proposal.HullKey),
            })
            .ToList();
    }

    /// <summary>
    /// Finds pairs of proposals whose top candidates are the same model and solve to nearly the
    /// same world position. This is the duplicate-claim signal the review queue must surface.
    /// </summary>
    public static IReadOnlyList<Pm4ShapeMatchConflict> FindConflicts(
        IReadOnlyList<Pm4ShapeMatchProposal> proposals,
        float maxDistance = ConflictDistance)
    {
        ArgumentNullException.ThrowIfNull(proposals);
        var conflicts = new List<Pm4ShapeMatchConflict>();
        float maxDistanceSq = maxDistance * maxDistance;

        for (int i = 0; i < proposals.Count; i++)
        {
            Pm4ShapeCandidate? a = proposals[i].TopCandidate;
            if (a is null)
                continue;

            for (int j = i + 1; j < proposals.Count; j++)
            {
                Pm4ShapeCandidate? b = proposals[j].TopCandidate;
                if (b is null)
                    continue;
                if (!string.Equals(a.ModelPath, b.ModelPath, StringComparison.OrdinalIgnoreCase))
                    continue;

                float distanceSq = Vector3.DistanceSquared(a.SolvedPosition, b.SolvedPosition);
                if (distanceSq > maxDistanceSq)
                    continue;

                conflicts.Add(new Pm4ShapeMatchConflict(
                    proposals[i].HullKey,
                    proposals[j].HullKey,
                    a.ModelPath,
                    MathF.Sqrt(distanceSq)));
            }
        }

        return conflicts;
    }

    /// <summary>
    /// Measures the correction that would move a solved proposal onto an authored position.
    /// The result is the value the operator would store as the calibration.
    /// </summary>
    public static Pm4CentroidCalibration MeasureOffset(
        Vector3 solvedPosition,
        Vector3 authoredPosition,
        string source)
        => new(authoredPosition - solvedPosition, source);

    /// <summary>
    /// Aggregates solved-vs-authored samples into a mean offset and its observed range. This is
    /// the measurement harness the operator runs on a validation tile before accepting a
    /// calibration value.
    /// </summary>
    public static Pm4CentroidMeasurement MeasureCalibration(
        IEnumerable<(Vector3 Solved, Vector3 Authored)> samples,
        string source)
    {
        ArgumentNullException.ThrowIfNull(samples);

        int count = 0;
        Vector3 sum = Vector3.Zero;
        Vector3 min = new(float.PositiveInfinity);
        Vector3 max = new(float.NegativeInfinity);
        foreach ((Vector3 solved, Vector3 authored) in samples)
        {
            Vector3 offset = authored - solved;
            sum += offset;
            min = Vector3.Min(min, offset);
            max = Vector3.Max(max, offset);
            count++;
        }

        if (count == 0)
            return new Pm4CentroidMeasurement(0, Vector3.Zero, Vector3.Zero, Vector3.Zero, source);

        return new Pm4CentroidMeasurement(count, sum / count, min, max, source);
    }
}
