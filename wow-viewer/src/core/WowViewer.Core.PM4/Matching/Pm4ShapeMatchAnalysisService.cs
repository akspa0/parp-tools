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

    public static Pm4ShapeMatchProposal Analyze(
        Pm4HullShapeInput hull,
        IReadOnlyList<Pm4ModelShapeCandidate> candidates,
        Pm4CentroidCalibration? calibration = null,
        Pm4ShapeMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(hull);
        ArgumentNullException.ThrowIfNull(candidates);

        Pm4CentroidCalibration activeCal = calibration ?? Pm4CentroidCalibration.None;
        Pm4ShapeMatchOptions opts = options ?? new Pm4ShapeMatchOptions();
        Vector3 uncalibrated = hull.Descriptor.PrismCentroid;
        Vector3 calibrated = activeCal.Apply(uncalibrated, calibrationApplied: false);
        var criteria = new Pm4ShapeMatchingCriteria();

        var scored = new List<Pm4ShapeCandidate>();
        foreach (Pm4ModelShapeCandidate cand in candidates)
        {
            if (string.IsNullOrWhiteSpace(cand.ModelPath))
                continue;

            float dist = Pm4DirectModelMatcher.CompareShapes(hull.Descriptor, cand.Descriptor, criteria);
            if (float.IsInfinity(dist) || dist > opts.MaxCredibleShapeDistance)
                continue;

            double score = 1.0 / (1.0 + MathF.Max(0f, dist));
            scored.Add(new Pm4ShapeCandidate(
                cand.ModelPath,
                System.IO.Path.GetFileName(cand.ModelPath),
                cand.AssetKind,
                score,
                1.0f,
                Vector3.Zero,
                calibrated,
                $"ShapeDistance={dist:F3}",
                dist));
        }

        var ranked = scored
            .OrderBy(static c => c.ShapeDistance)
            .ThenByDescending(static c => c.Score)
            .Take(Math.Max(1, opts.MaxCandidates))
            .ToArray();

        double confidence = ranked.Length > 0 ? ranked[0].Score : 0.0;
        bool unmatched = ranked.Length == 0;
        bool lowConfidence = !unmatched && confidence < opts.LowConfidenceThreshold;

        Vector3 half = hull.Descriptor.BoundingBoxDimensions * 0.5f;
        return new Pm4ShapeMatchProposal(
            HullKey: hull.Ck24 != 0 ? hull.Ck24 : (uint)hull.ObjectPartId,
            HullCentroid: uncalibrated,
            HullBoundsMin: uncalibrated - half,
            HullBoundsMax: uncalibrated + half,
            Candidates: ranked,
            ConfidenceScore: confidence,
            IsLowConfidence: lowConfidence,
            IsUnmatched: unmatched,
            ConflictCount: 1,
            Hull: hull,
            ExplicitCalibratedCentroid: calibrated);
    }

    public static Pm4ShapeBatchResult AnalyzeBatch(
        IReadOnlyList<Pm4HullShapeInput> hulls,
        IReadOnlyList<Pm4ModelShapeCandidate> candidates,
        Pm4CentroidCalibration? calibration = null,
        Pm4ShapeMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(hulls);
        ArgumentNullException.ThrowIfNull(candidates);

        Pm4ShapeMatchOptions opts = options ?? new Pm4ShapeMatchOptions();
        var proposals = new List<Pm4ShapeMatchProposal>(hulls.Count);
        foreach (Pm4HullShapeInput hull in hulls)
            proposals.Add(Analyze(hull, candidates, calibration, opts));

        IReadOnlyList<Pm4ProposalConflict> conflicts = FindHullConflicts(proposals, opts.ConflictDistance);
        return new Pm4ShapeBatchResult(proposals, conflicts);
    }

    public static IReadOnlyList<Pm4ProposalConflict> FindHullConflicts(
        IReadOnlyList<Pm4ShapeMatchProposal> proposals,
        float conflictDistance)
    {
        ArgumentNullException.ThrowIfNull(proposals);
        var list = new List<Pm4ProposalConflict>();
        float maxDistSq = conflictDistance * conflictDistance;

        for (int i = 0; i < proposals.Count; i++)
        {
            Pm4ShapeCandidate? a = proposals[i].TopCandidate;
            if (a == null) continue;

            for (int j = i + 1; j < proposals.Count; j++)
            {
                Pm4ShapeCandidate? b = proposals[j].TopCandidate;
                if (b == null) continue;

                if (!string.Equals(a.ModelPath, b.ModelPath, StringComparison.OrdinalIgnoreCase))
                    continue;

                float distSq = Vector3.DistanceSquared(proposals[i].CalibratedCentroid, proposals[j].CalibratedCentroid);
                if (distSq > maxDistSq)
                    continue;

                Pm4HullShapeInput hullA = proposals[i].Hull ?? new Pm4HullShapeInput(0, 0, proposals[i].HullKey, i, 0, null!);
                Pm4HullShapeInput hullB = proposals[j].Hull ?? new Pm4HullShapeInput(0, 0, proposals[j].HullKey, j, 0, null!);
                list.Add(new Pm4ProposalConflict(hullA, hullB, a.ModelPath, MathF.Sqrt(distSq)));
            }
        }

        return list;
    }
}
