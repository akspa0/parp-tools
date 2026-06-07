using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4AssetMatchScorer
{
    public const double MinimumMatchedScore = 0.45d;
    public const double AmbiguousScoreWindow = 0.03d;
    public const string CurrentReferenceSignalVersion = "pm4-asset-reference-signal-v1";

    public static IReadOnlyList<Pm4SegmentMatchResult> ScoreSegments(
        IReadOnlyList<Pm4BuiltObjectSegment> segments,
        IReadOnlyList<Pm4AssetReferenceSignalRecord> assetReferences,
        int maxCandidates = 10)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(assetReferences);

        return segments
            .Select(segment => ScoreSegment(segment, assetReferences, maxCandidates))
            .ToList();
    }

    public static Pm4SegmentMatchResult ScoreSegment(
        Pm4BuiltObjectSegment segment,
        IReadOnlyList<Pm4AssetReferenceSignalRecord> assetReferences,
        int maxCandidates = 10)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(assetReferences);

        int resolvedMaxCandidates = Math.Max(1, maxCandidates);
        List<string> rationale = BuildSegmentRationale(segment);
        string? expectedAssetKind = ResolveExpectedAssetKind(segment.Segment.Ck24Type);

        if (expectedAssetKind is null)
        {
            rationale.Add($"ck24Type 0x{segment.Segment.Ck24Type:X2} is not currently treated as WMO/M2-matchable.");
            return new Pm4SegmentMatchResult(
                segment,
                null,
                Pm4AssetMatchStatus.Ineligible,
                true,
                rationale,
                Array.Empty<Pm4AssetMatchCandidate>());
        }

        if (segment.Signal.Bounds is null)
        {
            rationale.Add("segment has no usable bounds, so geometry scoring is not possible.");
            return new Pm4SegmentMatchResult(
                segment,
                expectedAssetKind,
                Pm4AssetMatchStatus.Unresolved,
                true,
                rationale,
                Array.Empty<Pm4AssetMatchCandidate>());
        }

        List<CandidateEvaluation> evaluations = assetReferences
            .Where(asset => string.Equals(asset.AssetKind, expectedAssetKind, StringComparison.OrdinalIgnoreCase))
            .Select(asset => EvaluateCandidate(segment, asset))
            .Where(static evaluation => evaluation is not null)
            .Select(static evaluation => evaluation!)
            .OrderByDescending(static evaluation => evaluation.OverallScore)
            .ThenByDescending(static evaluation => evaluation.Metrics.FootprintOverlapRatio)
            .ThenByDescending(static evaluation => evaluation.Metrics.PlanarOverlapRatio)
            .ThenBy(static evaluation => evaluation.AnchorPlanarGap)
            .ThenBy(static evaluation => evaluation.Asset.AssetPath, StringComparer.OrdinalIgnoreCase)
            .ThenBy(static evaluation => evaluation.Asset.AssetId, StringComparer.OrdinalIgnoreCase)
            .Take(resolvedMaxCandidates)
            .ToList();

        if (evaluations.Count == 0)
        {
            rationale.Add($"no {expectedAssetKind} validation references were available to score against this segment.");
            return new Pm4SegmentMatchResult(
                segment,
                expectedAssetKind,
                Pm4AssetMatchStatus.Unresolved,
                true,
                rationale,
                Array.Empty<Pm4AssetMatchCandidate>());
        }

        double topScore = evaluations[0].OverallScore;
        double secondScore = evaluations.Count > 1 ? evaluations[1].OverallScore : double.NegativeInfinity;
        Pm4AssetMatchStatus status = topScore < MinimumMatchedScore
            ? Pm4AssetMatchStatus.Unresolved
            : Math.Abs(topScore - secondScore) <= AmbiguousScoreWindow
                ? Pm4AssetMatchStatus.Ambiguous
                : Pm4AssetMatchStatus.Matched;

        switch (status)
        {
            case Pm4AssetMatchStatus.Matched:
                rationale.Add($"top {expectedAssetKind} candidate '{evaluations[0].Asset.AssetPath}' cleared the score floor at {topScore:F3}.");
                break;
            case Pm4AssetMatchStatus.Ambiguous:
                rationale.Add($"top {expectedAssetKind} candidates are too close to separate confidently ({topScore:F3} vs {secondScore:F3}).");
                break;
            case Pm4AssetMatchStatus.Unresolved:
                rationale.Add($"best {expectedAssetKind} candidate scored only {topScore:F3}, below the {MinimumMatchedScore:F2} acceptance floor.");
                break;
        }

        List<Pm4AssetMatchCandidate> candidates = new(evaluations.Count);
        for (int index = 0; index < evaluations.Count; index++)
        {
            CandidateEvaluation evaluation = evaluations[index];
            candidates.Add(new Pm4AssetMatchCandidate(
                evaluation.Asset.AssetId,
                evaluation.Asset.AssetPath,
                evaluation.Asset.AssetKind,
                index + 1,
                evaluation.OverallScore,
                ResolveCandidateStatus(status, index, evaluation, topScore),
                evaluation.ScoreBreakdown,
                evaluation.Rationale));
        }

        bool reviewRequired = status != Pm4AssetMatchStatus.Matched || segment.Segment.ConfidenceFlags != Pm4SegmentConfidenceFlags.None;
        return new Pm4SegmentMatchResult(segment, expectedAssetKind, status, reviewRequired, rationale, candidates);
    }

    private static CandidateEvaluation? EvaluateCandidate(Pm4BuiltObjectSegment segment, Pm4AssetReferenceSignalRecord asset)
    {
        if (asset.Bounds is null)
            return null;

        Vector3 referenceCenter = segment.CorrelationState.Center;
        Vector3 candidateCenter = asset.Center;
        Pm4CorrelationMetrics metrics = Pm4CorrelationMath.EvaluateMetrics(
            segment.CorrelationState.BoundsMin,
            segment.CorrelationState.BoundsMax,
            referenceCenter,
            segment.CorrelationState.FootprintHull,
            segment.CorrelationState.FootprintArea,
            asset.Bounds.Min,
            asset.Bounds.Max,
            candidateCenter,
            asset.FootprintHull,
            asset.FootprintArea);

        bool sameTile = asset.TileCoordinates.Count > 0
            && asset.TileCoordinates.Intersect(segment.Segment.TileCoordinates, StringComparer.OrdinalIgnoreCase).Any();

        float anchorPlanarGap = ComputeAnchorPlanarGap(segment.AnchorPlanarPoints, asset.ReferencePosition ?? asset.Center);
        double footprintDistanceScore = ScoreDistance(metrics.FootprintDistance, 64f);
        double planarGapScore = ScoreDistance(metrics.PlanarGap, 48f);
        double verticalGapScore = ScoreDistance(metrics.VerticalGap, 16f);
        double anchorGapScore = float.IsFinite(anchorPlanarGap)
            ? ScoreDistance(anchorPlanarGap, 48f)
            : 0.5d;
        double sameTileScore = sameTile ? 1d : 0d;

        Dictionary<string, double> scoreBreakdown = new(StringComparer.Ordinal)
        {
            ["footprintOverlap"] = metrics.FootprintOverlapRatio,
            ["planarOverlap"] = metrics.PlanarOverlapRatio,
            ["volumeOverlap"] = metrics.VolumeOverlapRatio,
            ["footprintAreaRatio"] = metrics.FootprintAreaRatio,
            ["footprintDistanceScore"] = footprintDistanceScore,
            ["planarGapScore"] = planarGapScore,
            ["verticalGapScore"] = verticalGapScore,
            ["anchorGapScore"] = anchorGapScore,
            ["sameTileScore"] = sameTileScore,
        };

        double overallScore =
            metrics.FootprintOverlapRatio * 0.28d +
            metrics.PlanarOverlapRatio * 0.16d +
            metrics.VolumeOverlapRatio * 0.10d +
            metrics.FootprintAreaRatio * 0.12d +
            footprintDistanceScore * 0.10d +
            planarGapScore * 0.08d +
            verticalGapScore * 0.05d +
            anchorGapScore * 0.08d +
            sameTileScore * 0.03d;

        List<string> rationale =
        [
            $"footprint overlap {metrics.FootprintOverlapRatio:F3}",
            $"planar overlap {metrics.PlanarOverlapRatio:F3}",
            $"footprint area ratio {metrics.FootprintAreaRatio:F3}",
            float.IsFinite(anchorPlanarGap)
                ? $"anchor planar gap {anchorPlanarGap:F1}"
                : "no usable anchor-planar comparison",
        ];

        return new CandidateEvaluation(asset, metrics, anchorPlanarGap, overallScore, scoreBreakdown, rationale);
    }

    private static Pm4AssetMatchStatus ResolveCandidateStatus(
        Pm4AssetMatchStatus segmentStatus,
        int index,
        CandidateEvaluation evaluation,
        double topScore)
    {
        if (segmentStatus == Pm4AssetMatchStatus.Unresolved)
            return Pm4AssetMatchStatus.Unresolved;

        if (segmentStatus == Pm4AssetMatchStatus.Ambiguous)
            return Math.Abs(topScore - evaluation.OverallScore) <= AmbiguousScoreWindow
                ? Pm4AssetMatchStatus.Ambiguous
                : Pm4AssetMatchStatus.Unresolved;

        return index == 0
            ? Pm4AssetMatchStatus.Matched
            : Pm4AssetMatchStatus.Unresolved;
    }

    private static List<string> BuildSegmentRationale(Pm4BuiltObjectSegment segment)
    {
        List<string> rationale =
        [
            $"segment family ck24Type=0x{segment.Segment.Ck24Type:X2}",
            $"surfaces={segment.Segment.SurfaceCount} indices={segment.Segment.TotalIndexCount}",
        ];

        if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.ZeroCk24Seed))
            rationale.Add("segment came from the zero-CK24 fallback seed path.");
        if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.UsedConnectivityFallback))
            rationale.Add("segment required connectivity fallback splitting.");
        if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.MissingPositionRefs))
            rationale.Add("segment is missing linked position refs.");
        if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.MultipleLinkGroupIds))
            rationale.Add("segment spans multiple link-group ids.");
        if (segment.Segment.ConfidenceFlags.HasFlag(Pm4SegmentConfidenceFlags.ReusedLow16ObjectId))
            rationale.Add("segment reuses a low16 object id across multiple CK24 families.");

        return rationale;
    }

    private static string? ResolveExpectedAssetKind(byte ck24Type)
    {
        return ck24Type switch
        {
            0x42 or 0x43 => "wmo",
            0x40 or 0x41 or 0xC0 or 0xC1 or 0xC2 or 0xC3 => "m2",
            _ => null,
        };
    }

    private static double ScoreDistance(float distance, float scale)
    {
        if (!float.IsFinite(distance))
            return 0d;

        if (distance <= 0f)
            return 1d;

        return 1d / (1d + distance / Math.Max(0.001f, scale));
    }

    private static float ComputeAnchorPlanarGap(IReadOnlyList<Vector2> anchorPlanarPoints, Vector3 referencePosition)
    {
        if (anchorPlanarPoints.Count == 0)
            return float.PositiveInfinity;

        Vector2 target = new(referencePosition.X, referencePosition.Y);
        float bestDistanceSquared = float.PositiveInfinity;
        for (int index = 0; index < anchorPlanarPoints.Count; index++)
        {
            float distanceSquared = Vector2.DistanceSquared(anchorPlanarPoints[index], target);
            if (distanceSquared < bestDistanceSquared)
                bestDistanceSquared = distanceSquared;
        }

        return float.IsFinite(bestDistanceSquared) ? MathF.Sqrt(bestDistanceSquared) : float.PositiveInfinity;
    }

    private sealed record CandidateEvaluation(
        Pm4AssetReferenceSignalRecord Asset,
        Pm4CorrelationMetrics Metrics,
        float AnchorPlanarGap,
        double OverallScore,
        IReadOnlyDictionary<string, double> ScoreBreakdown,
        IReadOnlyList<string> Rationale);
}
