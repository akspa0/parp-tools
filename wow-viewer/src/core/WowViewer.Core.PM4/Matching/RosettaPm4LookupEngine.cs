using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// Options controlling the deterministic PM4 lookup engine.
/// </summary>
public sealed record RosettaLookupOptions(
    float ToleranceFactor = 0.5f,
    int MaxCandidates = 10,
    double MinimumMatchedScore = Pm4AssetMatchScorer.MinimumMatchedScore,
    double AmbiguousScoreWindow = Pm4AssetMatchScorer.AmbiguousScoreWindow);

/// <summary>
/// High-level result of looking up a PM4 segment in the <see cref="RosettaReferenceLibrary"/>.
/// </summary>
public sealed record RosettaPm4LookupResult(
    Pm4BuiltObjectSegment Segment,
    string? ExpectedAssetKind,
    Pm4AssetMatchStatus Status,
    bool ReviewRequired,
    IReadOnlyList<string> Rationale,
    IReadOnlyList<Pm4AssetMatchCandidate> Candidates,
    IReadOnlyDictionary<string, string> SignalAgreements,
    IReadOnlyDictionary<string, string> SignalDisagreements,
    Pm4SegmentMatchResult? LegacyScorerResult = null,
    bool? AgreesWithLegacy = null)
{
    /// <summary>
    /// Gets the top ranked candidate, or null if no candidates were found.
    /// </summary>
    public Pm4AssetMatchCandidate? TopCandidate => Candidates.Count > 0 ? Candidates[0] : null;

    /// <summary>
    /// Returns true if the segment was confidently identified.
    /// </summary>
    public bool IsIdentified => Status == Pm4AssetMatchStatus.Matched;

    /// <summary>
    /// Returns true if the segment was ambiguous between multiple near-equal candidates.
    /// </summary>
    public bool IsAmbiguous => Status == Pm4AssetMatchStatus.Ambiguous;

    /// <summary>
    /// Returns true if no candidate in the library was found or cleared the acceptance floor.
    /// </summary>
    public bool IsNoReference => Status == Pm4AssetMatchStatus.Unresolved;

    /// <summary>
    /// Returns true if the segment CK24 type is ineligible for geometry matching.
    /// </summary>
    public bool IsIneligible => Status == Pm4AssetMatchStatus.Ineligible;
}

/// <summary>
/// Deterministic, non-LLM PM4 object identification lookup engine (Spec 190 User Story 3).
/// Evaluates real PM4 geometry segments against the complete <see cref="RosettaReferenceLibrary"/>,
/// producing structured classifications (Identified, Ambiguous, NoReference, Ineligible) with
/// detailed signal agreement and disagreement evidence.
/// </summary>
public static class RosettaPm4LookupEngine
{
    /// <summary>
    /// Looks up a single PM4 object segment in the reference library.
    /// </summary>
    public static RosettaPm4LookupResult LookupSegment(
        Pm4BuiltObjectSegment segment,
        RosettaReferenceLibrary library,
        RosettaLookupOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(library);

        options ??= new RosettaLookupOptions();
        var rationale = new List<string>();

        // 1. Resolve expected asset kind from CK24
        string? expectedAssetKind = ResolveExpectedAssetKind(segment.Segment.Ck24Type);
        if (expectedAssetKind is null)
        {
            rationale.Add($"ck24Type 0x{segment.Segment.Ck24Type:X2} is not currently treated as a matchable 3D model or world model.");
            return new RosettaPm4LookupResult(
                segment,
                null,
                Pm4AssetMatchStatus.Ineligible,
                ReviewRequired: true,
                rationale,
                Candidates: [],
                SignalAgreements: new Dictionary<string, string>(),
                SignalDisagreements: new Dictionary<string, string>());
        }

        // 2. Verify usable segment bounds
        if (segment.Signal.Bounds is null)
        {
            rationale.Add("segment has no usable bounds, so geometry lookup is not possible.");
            return new RosettaPm4LookupResult(
                segment,
                expectedAssetKind,
                Pm4AssetMatchStatus.Unresolved,
                ReviewRequired: true,
                rationale,
                Candidates: [],
                SignalAgreements: new Dictionary<string, string>(),
                SignalDisagreements: new Dictionary<string, string>());
        }

        // 3. Fast spatial & bounding candidate filtering from the library
        IReadOnlyList<RosettaCandidateMatch> candidateMatches = library.FindCandidatesByBounds(
            segment.Signal.Bounds,
            expectedAssetKind,
            options.ToleranceFactor,
            maxCandidates: Math.Max(options.MaxCandidates * 3, 20));

        var filteredAssets = candidateMatches.Select(static m => m.Asset).ToList();

        IReadOnlyList<Pm4AssetReferenceSignalRecord> referenceSignals;
        if (filteredAssets.Count > 0)
        {
            referenceSignals = filteredAssets.Select(static a => a.ToAssetReferenceSignalRecord()).ToList();
        }
        else
        {
            // Try all library assets of the expected kind
            referenceSignals = library.Assets
                .Where(a => string.Equals(a.AssetKind, expectedAssetKind, StringComparison.OrdinalIgnoreCase))
                .Select(static a => a.ToAssetReferenceSignalRecord())
                .ToList();
        }

        if (referenceSignals.Count == 0)
        {
            rationale.Add($"no {expectedAssetKind} reference assets exist in library '{library.LibraryId}'.");
            return new RosettaPm4LookupResult(
                segment,
                expectedAssetKind,
                Pm4AssetMatchStatus.Unresolved,
                ReviewRequired: true,
                rationale,
                Candidates: [],
                SignalAgreements: new Dictionary<string, string>(),
                SignalDisagreements: new Dictionary<string, string>());
        }

        // 4. Run multi-signal geometric scoring
        Pm4SegmentMatchResult scoreResult = Pm4AssetMatchScorer.ScoreSegment(
            segment,
            referenceSignals,
            options.MaxCandidates);

        // 5. Evaluate signal agreements / disagreements against top candidate
        var agreements = new Dictionary<string, string>(StringComparer.Ordinal);
        var disagreements = new Dictionary<string, string>(StringComparer.Ordinal);

        if (scoreResult.Candidates.Count > 0)
        {
            Pm4AssetMatchCandidate top = scoreResult.Candidates[0];
            if (library.TryGetAssetById(top.AssetId, out RosettaReferenceAsset? matchedAsset) && matchedAsset is not null)
            {
                EvaluateSignalEvidence(segment, matchedAsset, agreements, disagreements);
            }
        }

        rationale.AddRange(scoreResult.Rationale);

        return new RosettaPm4LookupResult(
            segment,
            expectedAssetKind,
            scoreResult.Status,
            scoreResult.ReviewRequired,
            rationale,
            scoreResult.Candidates,
            agreements,
            disagreements);
    }

    /// <summary>
    /// Looks up a list of PM4 segments in the reference library in batch.
    /// </summary>
    public static IReadOnlyList<RosettaPm4LookupResult> LookupSegments(
        IReadOnlyList<Pm4BuiltObjectSegment> segments,
        RosettaReferenceLibrary library,
        RosettaLookupOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(segments);
        ArgumentNullException.ThrowIfNull(library);

        options ??= new RosettaLookupOptions();
        return segments
            .Select(segment => LookupSegment(segment, library, options))
            .ToList();
    }

    /// <summary>
    /// Compares Rosetta global lookup with the legacy local ADT self-corpus scorer,
    /// surfacing agreements and disagreements.
    /// </summary>
    public static RosettaPm4LookupResult CompareWithLegacyScorer(
        Pm4BuiltObjectSegment segment,
        IReadOnlyList<Pm4AssetReferenceSignalRecord> legacyCorpus,
        RosettaReferenceLibrary library,
        RosettaLookupOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(legacyCorpus);
        ArgumentNullException.ThrowIfNull(library);

        options ??= new RosettaLookupOptions();

        // 1. Run Rosetta global lookup
        RosettaPm4LookupResult rosettaResult = LookupSegment(segment, library, options);

        // 2. Run legacy local scorer
        Pm4SegmentMatchResult legacyResult = Pm4AssetMatchScorer.ScoreSegment(
            segment,
            legacyCorpus,
            options.MaxCandidates);

        // 3. Determine agreement
        bool? agreesWithLegacy = null;
        if (rosettaResult.TopCandidate is not null && legacyResult.Candidates.Count > 0)
        {
            agreesWithLegacy = string.Equals(
                rosettaResult.TopCandidate.AssetPath,
                legacyResult.Candidates[0].AssetPath,
                StringComparison.OrdinalIgnoreCase);
        }

        return rosettaResult with
        {
            LegacyScorerResult = legacyResult,
            AgreesWithLegacy = agreesWithLegacy
        };
    }

    /// <summary>
    /// Evaluates signal features between a PM4 segment and a reference asset,
    /// populating agreement and disagreement dictionaries.
    /// </summary>
    public static void EvaluateSignalEvidence(
        Pm4BuiltObjectSegment segment,
        RosettaReferenceAsset reference,
        Dictionary<string, string> agreements,
        Dictionary<string, string> disagreements)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(reference);
        ArgumentNullException.ThrowIfNull(agreements);
        ArgumentNullException.ThrowIfNull(disagreements);

        if (segment.Signal.Bounds is null)
            return;

        Vector3 segSpan = segment.Signal.Bounds.Max - segment.Signal.Bounds.Min;
        Vector3 refSpan = reference.Span;

        // Aspect ratio comparison (sorted spans)
        float[] segSorted = [segSpan.X, segSpan.Y, segSpan.Z];
        float[] refSorted = [refSpan.X, refSpan.Y, refSpan.Z];
        Array.Sort(segSorted);
        Array.Reverse(segSorted);
        Array.Sort(refSorted);
        Array.Reverse(refSorted);

        float segAspect = segSorted[0] > 0.01f ? segSorted[1] / segSorted[0] : 1f;
        float refAspect = refSorted[0] > 0.01f ? refSorted[1] / refSorted[0] : 1f;
        float aspectDiff = MathF.Abs(segAspect - refAspect);

        if (aspectDiff <= 0.15f)
            agreements["AspectRatio"] = $"aspect ratio matches closely (seg: {segAspect:F2}, ref: {refAspect:F2}, diff: {aspectDiff:F2})";
        else
            disagreements["AspectRatio"] = $"aspect ratio differs (seg: {segAspect:F2}, ref: {refAspect:F2}, diff: {aspectDiff:F2})";

        // Span scale ratio
        float maxSpanRatio = MathF.Max(segSorted[0], refSorted[0]) > 0.01f
            ? MathF.Min(segSorted[0], refSorted[0]) / MathF.Max(segSorted[0], refSorted[0])
            : 1f;

        if (maxSpanRatio >= 0.70f)
            agreements["MajorSpan"] = $"primary dimension matches ({segSorted[0]:F1}m vs {refSorted[0]:F1}m, ratio: {maxSpanRatio:P0})";
        else
            disagreements["MajorSpan"] = $"primary dimension mismatch ({segSorted[0]:F1}m vs {refSorted[0]:F1}m, ratio: {maxSpanRatio:P0})";

        // Volume ratio
        float segVol = MathF.Max(0.01f, segSpan.X * segSpan.Y * segSpan.Z);
        float refVol = MathF.Max(0.01f, reference.Volume);
        float volRatio = MathF.Min(segVol, refVol) / MathF.Max(segVol, refVol);

        if (volRatio >= 0.50f)
            agreements["Volume"] = $"bounding volume is consistent (seg: {segVol:F1}m³, ref: {refVol:F1}m³, ratio: {volRatio:P0})";
        else
            disagreements["Volume"] = $"bounding volume differs significantly (seg: {segVol:F1}m³, ref: {refVol:F1}m³, ratio: {volRatio:P0})";

        // Footprint area ratio
        float segFootprint = MathF.Max(0.01f, segSpan.X * segSpan.Y);
        float refFootprint = MathF.Max(0.01f, reference.FootprintArea);
        float footprintRatio = MathF.Min(segFootprint, refFootprint) / MathF.Max(segFootprint, refFootprint);

        if (footprintRatio >= 0.50f)
            agreements["Footprint"] = $"footprint area is consistent (seg: {segFootprint:F1}m², ref: {refFootprint:F1}m², ratio: {footprintRatio:P0})";
        else
            disagreements["Footprint"] = $"footprint area differs (seg: {segFootprint:F1}m², ref: {refFootprint:F1}m², ratio: {footprintRatio:P0})";
    }

    private static string? ResolveExpectedAssetKind(byte ck24Type)
        => ck24Type switch
        {
            0x42 or 0x43 => "wmo",
            0x40 or 0x41 or 0xC0 or 0xC1 or 0xC2 or 0xC3 => "m2",
            _ => null,
        };
}
