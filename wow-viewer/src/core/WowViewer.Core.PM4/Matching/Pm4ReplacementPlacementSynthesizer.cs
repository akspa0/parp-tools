using System.Security.Cryptography;
using System.Text;
using System.Numerics;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4ReplacementPlacementSynthesizer
{
    public static IReadOnlyList<Pm4ReplacementPlacementProposal> Synthesize(
        IReadOnlyList<Pm4SegmentMatchResult> matchResults,
        IReadOnlyList<Pm4AssetReferenceSignalRecord> assetReferences,
        IReadOnlyCollection<string>? targetTileCoordinates = null)
    {
        ArgumentNullException.ThrowIfNull(matchResults);
        ArgumentNullException.ThrowIfNull(assetReferences);

        HashSet<string>? normalizedTargetTiles = targetTileCoordinates is null
            ? null
            : new HashSet<string>(targetTileCoordinates.Where(static tile => !string.IsNullOrWhiteSpace(tile)), StringComparer.OrdinalIgnoreCase);
        Dictionary<string, Pm4AssetReferenceSignalRecord> assetsById = assetReferences.ToDictionary(static asset => asset.AssetId, StringComparer.OrdinalIgnoreCase);
        List<Pm4ReplacementPlacementProposal> proposals = [];

        foreach (Pm4SegmentMatchResult matchResult in matchResults)
        {
            if (matchResult.Status is Pm4AssetMatchStatus.Ineligible or Pm4AssetMatchStatus.Unresolved)
                continue;

            if (matchResult.Candidates.Count == 0)
                continue;

            IReadOnlyList<string> targetTiles = FilterTargetTiles(matchResult.Segment.Segment.TileCoordinates, normalizedTargetTiles);
            if (targetTiles.Count == 0)
                continue;

            Pm4AssetMatchCandidate selectedCandidate = matchResult.Candidates[0];
            if (!assetsById.TryGetValue(selectedCandidate.AssetId, out Pm4AssetReferenceSignalRecord? assetReference))
                continue;

            bool usedFallbackPosition = assetReference.ReferencePosition is null;
            bool usedFallbackRotation = assetReference.ReferenceRotation is null;
            bool usedFallbackScale = !assetReference.ReferenceScale.HasValue;

            Vector3? worldPosition = assetReference.ReferencePosition ?? matchResult.Segment.CorrelationState.Center;
            Vector3? worldRotation = assetReference.ReferenceRotation ?? BuildFallbackRotation(matchResult.Segment);
            float? worldScale = assetReference.ReferenceScale ?? 1f;
            double confidence = Math.Clamp(selectedCandidate.OverallScore, 0d, 1d);
            bool reviewRequired = matchResult.ReviewRequired
                || matchResult.Status != Pm4AssetMatchStatus.Matched
                || usedFallbackPosition
                || usedFallbackRotation
                || usedFallbackScale;

            List<string> provenance =
            [
                $"segment:{matchResult.Segment.Segment.SegmentId}",
                $"asset:{selectedCandidate.AssetId}",
                $"match-status:{FormatStatus(matchResult.Status)}",
                $"candidate-rank:{selectedCandidate.Rank}",
                $"candidate-score:{selectedCandidate.OverallScore:F4}",
                usedFallbackPosition ? "position:pm4-center-fallback" : "position:asset-reference",
                usedFallbackRotation ? "rotation:pm4-heading-fallback" : "rotation:asset-reference",
                usedFallbackScale ? "scale:unit-fallback" : "scale:asset-reference",
            ];

            proposals.Add(new Pm4ReplacementPlacementProposal(
                BuildProposalId(matchResult.Segment.Segment.SegmentId, selectedCandidate.AssetId, targetTiles),
                matchResult.Segment.Segment.SegmentId,
                selectedCandidate.AssetId,
                targetTiles,
                worldPosition,
                worldRotation,
                worldScale,
                confidence,
                reviewRequired,
                provenance));
        }

        return proposals;
    }

    private static IReadOnlyList<string> FilterTargetTiles(
        IReadOnlyList<string> segmentTiles,
        HashSet<string>? normalizedTargetTiles)
    {
        if (normalizedTargetTiles is null || normalizedTargetTiles.Count == 0)
            return segmentTiles;

        return segmentTiles
            .Where(normalizedTargetTiles.Contains)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static Vector3? BuildFallbackRotation(Pm4BuiltObjectSegment segment)
    {
        float yawDegrees = segment.Signal.AnchorSignals.HeadingMeanDegrees ?? segment.FrameYawDegrees;
        if (!float.IsFinite(yawDegrees))
            return null;

        return new Vector3(0f, 0f, yawDegrees);
    }

    private static string BuildProposalId(string segmentId, string assetId, IReadOnlyList<string> targetTiles)
    {
        StringBuilder builder = new();
        builder.Append(segmentId);
        builder.Append('|');
        builder.Append(assetId);
        builder.Append('|');
        foreach (string tile in targetTiles)
        {
            builder.Append(tile);
            builder.Append(',');
        }

        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
        return $"proposal-{Convert.ToHexString(digest[..8]).ToLowerInvariant()}";
    }

    private static string FormatStatus(Pm4AssetMatchStatus status)
    {
        return status switch
        {
            Pm4AssetMatchStatus.Matched => "matched",
            Pm4AssetMatchStatus.Ambiguous => "ambiguous",
            Pm4AssetMatchStatus.Unresolved => "unresolved",
            Pm4AssetMatchStatus.Ineligible => "ineligible",
            _ => status.ToString().ToLowerInvariant(),
        };
    }
}
