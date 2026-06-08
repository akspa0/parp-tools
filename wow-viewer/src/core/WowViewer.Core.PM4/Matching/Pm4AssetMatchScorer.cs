using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4AssetMatchScorer
{
    public const double MinimumMatchedScore = 0.45d;
    public const double AmbiguousScoreWindow = 0.03d;

    public const string CurrentReferenceSignalVersion = "pm4-asset-reference-signal-v1";

    // Known MSLK.TypeFlags values from real-data inspection
    private const byte TypeFlag_M2Top = 0x03;
    private const byte TypeFlag_InteriorFloor = 0x10;
    private const byte TypeFlag_ExteriorSolid = 0x12;

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
            return new Pm4SegmentMatchResult(segment, null, Pm4AssetMatchStatus.Ineligible, true, rationale, []);
        }

        if (segment.Signal.Bounds is null)
        {
            rationale.Add("segment has no usable bounds, so geometry scoring is not possible.");
            return new Pm4SegmentMatchResult(segment, expectedAssetKind, Pm4AssetMatchStatus.Unresolved, true, rationale, []);
        }

        // Build TypeFlags profile from typed bounds
        Dictionary<byte, Pm4Bounds3> typedBounds = segment.Signal.TypedBounds is not null
            ? new Dictionary<byte, Pm4Bounds3>(segment.Signal.TypedBounds)
            : [];
        bool hasTypeFlagsData = typedBounds.Count > 0;
        bool hasExteriorSolid = typedBounds.ContainsKey(TypeFlag_ExteriorSolid);
        bool hasInteriorFloor = typedBounds.ContainsKey(TypeFlag_InteriorFloor);
        bool hasM2Top = typedBounds.ContainsKey(TypeFlag_M2Top);

        string typeProfile = DescribeTypeProfile(typedBounds);
        rationale.Add($"TypeFlags profile: {typeProfile}");

        // Determine expected TypeFlags for this asset kind
        bool profileMatchesExpectedKind = expectedAssetKind switch
        {
            "wmo" => hasExteriorSolid || hasInteriorFloor,
            "m2" => hasM2Top,
            _ => false,
        };

        if (profileMatchesExpectedKind)
            rationale.Add($"TypeFlags profile is consistent with {expectedAssetKind} expectation.");
        else if (hasTypeFlagsData)
            rationale.Add($"TypeFlags profile does not match typical {expectedAssetKind} pattern — scoring with reduced weight.");
        else
            rationale.Add("no TypeFlags surface classification available — scoring on shape only.");

        List<CandidateEvaluation> evaluations = assetReferences
            .Where(asset => string.Equals(asset.AssetKind, expectedAssetKind, StringComparison.OrdinalIgnoreCase))
            .Select(asset => EvaluateTypedCandidate(segment, asset, typedBounds, profileMatchesExpectedKind, hasTypeFlagsData))
            .Where(static evaluation => evaluation is not null)
            .Select(static evaluation => evaluation!)
            .OrderByDescending(static evaluation => evaluation.OverallScore)
            .ThenByDescending(static evaluation => evaluation.TypedOverlapScore)
            .ThenByDescending(static evaluation => evaluation.ShapeScore)
            .Take(resolvedMaxCandidates)
            .ToList();

        if (evaluations.Count == 0)
        {
            rationale.Add($"no {expectedAssetKind} validation references were available to score against this segment.");
            return new Pm4SegmentMatchResult(segment, expectedAssetKind, Pm4AssetMatchStatus.Unresolved, true, rationale, []);
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

    private static CandidateEvaluation? EvaluateTypedCandidate(
        Pm4BuiltObjectSegment segment,
        Pm4AssetReferenceSignalRecord asset,
        Dictionary<byte, Pm4Bounds3> typedBounds,
        bool profileMatchesExpectedKind,
        bool hasTypeFlagsData)
    {
        if (asset.Bounds is null || segment.Signal.Bounds is null)
            return null;

        // 1. TypeFlags profile match score
        double profileScore = profileMatchesExpectedKind ? 1.0 : (hasTypeFlagsData ? 0.3 : 0.5);

        // 2. Per-type-class overlap against asset bounds
        double typedOverlapScore = 0;
        int typedCount = 0;
        foreach (KeyValuePair<byte, Pm4Bounds3> kv in typedBounds)
        {
            if (kv.Value.Min == kv.Value.Max)
                continue;

            double overlap = ComputeBoundsOverlapRatio(kv.Value, asset.Bounds);
            typedOverlapScore += overlap;
            typedCount++;
        }
        typedOverlapScore = typedCount > 0 ? typedOverlapScore / typedCount : 0.5;

        // 3. Shape similarity (sorted span ratios, footprint, volume)
        Vector3 segmentSpan = segment.Signal.Bounds.Max - segment.Signal.Bounds.Min;
        Vector3 assetSpan = asset.Bounds.Max - asset.Bounds.Min;

        double[] sortedSegmentSpans = [segmentSpan.X, segmentSpan.Y, segmentSpan.Z];
        double[] sortedAssetSpans = [assetSpan.X, assetSpan.Y, assetSpan.Z];
        Array.Sort(sortedSegmentSpans);
        Array.Reverse(sortedSegmentSpans);
        Array.Sort(sortedAssetSpans);
        Array.Reverse(sortedAssetSpans);

        double spanScore0 = ScoreRatio(sortedSegmentSpans[0], sortedAssetSpans[0]);
        double spanScore1 = ScoreRatio(sortedSegmentSpans[1], sortedAssetSpans[1]);
        double spanScore2 = ScoreRatio(sortedSegmentSpans[2], sortedAssetSpans[2]);
        double sortedSpanScore = (spanScore0 + spanScore1 + spanScore2) / 3d;

        // Same-tile bonus for validation placements with position overlap
        double sameTileBonus = 0;
        if (asset.TileCoordinates.Count > 0 && asset.ReferencePosition.HasValue)
        {
            bool sharesTile = asset.TileCoordinates
                .Intersect(segment.Segment.TileCoordinates, StringComparer.OrdinalIgnoreCase)
                .Any();
            if (sharesTile)
            {
                // Compute center distance overlap ratio
                double centerDist = Vector3.Distance(
                    segment.CorrelationState.Center,
                    asset.Center);
                sameTileBonus = ScoreDistance(centerDist, 64f);
            }
        }

        double segmentFootprint = Math.Max(0d, segment.CorrelationState.FootprintArea);
        double assetFootprint = Math.Max(0d, asset.FootprintArea);
        double footprintScore = ScoreRatio(segmentFootprint, assetFootprint);

        double segmentVolume = Math.Max(0d, segmentSpan.X) * Math.Max(0d, segmentSpan.Y) * Math.Max(0d, segmentSpan.Z);
        double assetVolume = Math.Max(0d, assetSpan.X) * Math.Max(0d, assetSpan.Y) * Math.Max(0d, assetSpan.Z);
        double volumeScore = ScoreRatio(segmentVolume, assetVolume);

        double segmentDiagonal = Math.Sqrt(segmentSpan.X * segmentSpan.X + segmentSpan.Y * segmentSpan.Y);
        double assetDiagonal = Math.Sqrt(assetSpan.X * assetSpan.X + assetSpan.Y * assetSpan.Y);
        double diagonalScore = ScoreRatio(segmentDiagonal, assetDiagonal);

        double heightScore = ScoreRatio(segmentSpan.Z, assetSpan.Z);
        double segmentAspect = segmentSpan.Y > 0 ? segmentSpan.X / segmentSpan.Y : 0;
        double assetAspect = assetSpan.Y > 0 ? assetSpan.X / assetSpan.Y : 0;
        double aspectScore = ScoreRatio(segmentAspect, assetAspect);

        double shapeScore = sortedSpanScore * 0.25 + footprintScore * 0.15 + volumeScore * 0.15 + diagonalScore * 0.12 + heightScore * 0.10 + aspectScore * 0.08 + sameTileBonus * 0.15;

        // 4. Combined score: type overlap + shape + profile
        double typeWeight = hasTypeFlagsData ? 0.35 : 0.0;
        double profileWeight = hasTypeFlagsData ? 0.15 : 0.0;
        double shapeWeight = 1.0 - typeWeight - profileWeight;

        double overallScore = typedOverlapScore * typeWeight + profileScore * profileWeight + shapeScore * shapeWeight;

        Dictionary<string, double> scoreBreakdown = new(StringComparer.Ordinal)
        {
            ["typeProfileScore"] = profileScore,
            ["typedOverlapScore"] = typedOverlapScore,
            ["sortedSpanScore"] = sortedSpanScore,
            ["footprintAreaScore"] = footprintScore,
            ["volumeScore"] = volumeScore,
            ["diagonalScore"] = diagonalScore,
            ["heightScore"] = heightScore,
            ["aspectScore"] = aspectScore,
            ["shapeScore"] = shapeScore,
            ["typeWeight"] = typeWeight,
            ["profileWeight"] = profileWeight,
            ["shapeWeight"] = shapeWeight,
        };

        List<string> rationale =
        [
            $"typed overlap {typedOverlapScore:F3} (typeWeight={typeWeight:F2})",
            $"shape score {shapeScore:F3} (shapeWeight={shapeWeight:F2})",
            $"type profile {profileScore:F3} (profileWeight={profileWeight:F2})",
        ];

        return new CandidateEvaluation(asset, typedOverlapScore, shapeScore, overallScore, scoreBreakdown, rationale);
    }

    private static double ComputeBoundsOverlapRatio(Pm4Bounds3 left, Pm4Bounds3 right)
    {
        // Axis-aligned bounding box overlap in XY (footprint plane)
        double overlapX = Math.Max(0, Math.Min(left.Max.X, right.Max.X) - Math.Max(left.Min.X, right.Min.X));
        double overlapY = Math.Max(0, Math.Min(left.Max.Y, right.Max.Y) - Math.Max(left.Min.Y, right.Min.Y));
        double leftArea = (left.Max.X - left.Min.X) * (left.Max.Y - left.Min.Y);
        double rightArea = (right.Max.X - right.Min.X) * (right.Max.Y - right.Min.Y);
        double intersectionArea = overlapX * overlapY;

        if (leftArea <= 0 || rightArea <= 0)
            return 0;

        // Jaccard-like: intersection / union
        double unionArea = leftArea + rightArea - intersectionArea;
        return unionArea > 0 ? intersectionArea / unionArea : 0;
    }

    private static string DescribeTypeProfile(Dictionary<byte, Pm4Bounds3> typedBounds)
    {
        if (typedBounds.Count == 0)
            return "none";

        List<string> parts = [];
        foreach (byte typeFlag in typedBounds.Keys.Order())
        {
            string label = typeFlag switch
            {
                TypeFlag_M2Top => "m2-top(0x03)",
                TypeFlag_InteriorFloor => "interior-floor(0x10)",
                TypeFlag_ExteriorSolid => "exterior-solid(0x12)",
                _ => $"0x{typeFlag:X2}",
            };
            parts.Add(label);
        }
        return string.Join(", ", parts);
    }

    private static Pm4AssetMatchStatus ResolveCandidateStatus(
        Pm4AssetMatchStatus segmentStatus, int index,
        CandidateEvaluation evaluation, double topScore)
    {
        if (segmentStatus == Pm4AssetMatchStatus.Unresolved)
            return Pm4AssetMatchStatus.Unresolved;

        if (segmentStatus == Pm4AssetMatchStatus.Ambiguous)
            return Math.Abs(topScore - evaluation.OverallScore) <= AmbiguousScoreWindow
                ? Pm4AssetMatchStatus.Ambiguous
                : Pm4AssetMatchStatus.Unresolved;

        return index == 0 ? Pm4AssetMatchStatus.Matched : Pm4AssetMatchStatus.Unresolved;
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

    private static double ScoreDistance(double distance, double scale)
    {
        if (!double.IsFinite(distance))
            return 0d;
        if (distance <= 0d)
            return 1d;
        return 1d / (1d + distance / Math.Max(0.001, scale));
    }

    private static double ScoreRatio(double left, double right)
    {
        if (!double.IsFinite(left) || !double.IsFinite(right) || left <= 0d || right <= 0d)
            return 0d;
        double min = Math.Min(left, right);
        double max = Math.Max(left, right);
        return max > 0d ? min / max : 0d;
    }

    private sealed record CandidateEvaluation(
        Pm4AssetReferenceSignalRecord Asset,
        double TypedOverlapScore,
        double ShapeScore,
        double OverallScore,
        IReadOnlyDictionary<string, double> ScoreBreakdown,
        IReadOnlyList<string> Rationale);
}
