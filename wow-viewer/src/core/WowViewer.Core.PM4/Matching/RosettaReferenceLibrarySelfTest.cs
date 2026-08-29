using System.Diagnostics;
using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Matching;

public sealed record RosettaSelfTestOptions(
    float Tolerance = 0.35f,
    int TopK = 10,
    bool IncludePerturbations = false,
    float PerturbationJitterPercent = 0.03f,
    int? MaxItemsToTest = null);

public sealed record RosettaSelfTestDefect(
    string AssetPath,
    string ExpectedAssetId,
    string? ActualTop1AssetId,
    string? ActualTop1AssetPath,
    double ActualTop1Score,
    double TrueAssetScore,
    int TrueAssetRank,
    string Reason);

public sealed record RosettaSelfTestResult(
    string LibraryId,
    int TotalTested,
    int Top1Matches,
    int Top3Matches,
    int AmbiguousMatches,
    int Misses,
    double Top1AccuracyPercent,
    double Top3AccuracyPercent,
    IReadOnlyList<RosettaSelfTestDefect> Defects,
    bool Passed,
    double ExecutionDurationMs)
{
    public string Summary =>
        $"[Rosetta Self-Test] Library={LibraryId}, Tested={TotalTested}, Top1={Top1Matches} ({Top1AccuracyPercent:F2}%), " +
        $"Top3={Top3Matches} ({Top3AccuracyPercent:F2}%), Ambiguous={AmbiguousMatches}, Misses={Misses}, " +
        $"Passed={Passed} (Duration: {ExecutionDurationMs:F1}ms)";
}

/// <summary>
/// Automated self-test runner for the Rosetta Reference Library (Spec 190 US2 / Acceptance Scenario 3).
/// Synthesizes queries from library assets and verifies >= 99.0% top-1 identification accuracy.
/// </summary>
public static class RosettaReferenceLibrarySelfTest
{
    public const double TargetAccuracyPercent = 99.0;

    /// <summary>
    /// Runs the self-test verification suite over the provided reference library.
    /// </summary>
    public static RosettaSelfTestResult Run(
        RosettaReferenceLibrary library,
        RosettaSelfTestOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(library);
        options ??= new RosettaSelfTestOptions();

        var stopwatch = Stopwatch.StartNew();
        library.IndexAssets();

        IReadOnlyList<RosettaReferenceAsset> testSet = library.Assets;
        if (options.MaxItemsToTest.HasValue && options.MaxItemsToTest.Value > 0)
            testSet = testSet.Take(options.MaxItemsToTest.Value).ToList();

        int totalTested = 0;
        int top1Matches = 0;
        int top3Matches = 0;
        int ambiguousMatches = 0;
        int misses = 0;
        var defects = new List<RosettaSelfTestDefect>();

        foreach (RosettaReferenceAsset asset in testSet)
        {
            totalTested++;

            Pm4Bounds3 queryBounds = asset.Bounds;

            if (options.IncludePerturbations && options.PerturbationJitterPercent > 0f)
            {
                // Apply subtle bounding jitter to test real-world noise resilience
                float jitter = options.PerturbationJitterPercent;
                Vector3 span = asset.Span;
                Vector3 delta = span * jitter * 0.5f;
                queryBounds = new Pm4Bounds3(asset.Bounds.Min - delta, asset.Bounds.Max + delta);
            }

            IReadOnlyList<RosettaCandidateMatch> candidates = library.FindCandidatesByBounds(
                queryBounds,
                assetKind: asset.AssetKind,
                tolerance: options.Tolerance,
                maxCandidates: options.TopK);

            if (candidates.Count == 0)
            {
                misses++;
                defects.Add(new RosettaSelfTestDefect(
                    asset.AssetPath,
                    asset.AssetId,
                    ActualTop1AssetId: null,
                    ActualTop1AssetPath: null,
                    ActualTop1Score: 0.0,
                    TrueAssetScore: 0.0,
                    TrueAssetRank: -1,
                    Reason: "No candidate returned within tolerance."));
                continue;
            }

            RosettaCandidateMatch top1 = candidates[0];
            bool isTop1 = string.Equals(top1.Asset.AssetId, asset.AssetId, StringComparison.OrdinalIgnoreCase)
                || string.Equals(top1.Asset.NormalizedPath, asset.NormalizedPath, StringComparison.OrdinalIgnoreCase);

            int trueRank = -1;
            double trueScore = 0.0;
            for (int i = 0; i < candidates.Count; i++)
            {
                if (string.Equals(candidates[i].Asset.AssetId, asset.AssetId, StringComparison.OrdinalIgnoreCase)
                    || string.Equals(candidates[i].Asset.NormalizedPath, asset.NormalizedPath, StringComparison.OrdinalIgnoreCase))
                {
                    trueRank = i + 1;
                    trueScore = candidates[i].OverallScore;
                    break;
                }
            }

            if (isTop1)
            {
                top1Matches++;
                top3Matches++;

                // Check for ambiguity (Top-2 score within 0.005 of Top-1)
                if (candidates.Count > 1 && Math.Abs(candidates[0].OverallScore - candidates[1].OverallScore) < 0.005)
                {
                    ambiguousMatches++;
                }
            }
            else
            {
                if (trueRank is > 1 and <= 3)
                    top3Matches++;

                misses++;
                defects.Add(new RosettaSelfTestDefect(
                    asset.AssetPath,
                    asset.AssetId,
                    top1.Asset.AssetId,
                    top1.Asset.AssetPath,
                    top1.OverallScore,
                    trueScore,
                    trueRank,
                    $"Top-1 mismatch. Expected '{asset.AssetPath}', got '{top1.Asset.AssetPath}' (TopScore={top1.OverallScore:F3}, TrueScore={trueScore:F3}, TrueRank={trueRank})"));
            }
        }

        stopwatch.Stop();

        double top1Accuracy = totalTested > 0 ? (top1Matches / (double)totalTested) * 100.0 : 0.0;
        double top3Accuracy = totalTested > 0 ? (top3Matches / (double)totalTested) * 100.0 : 0.0;
        bool passed = top1Accuracy >= TargetAccuracyPercent;

        return new RosettaSelfTestResult(
            library.LibraryId,
            totalTested,
            top1Matches,
            top3Matches,
            ambiguousMatches,
            misses,
            top1Accuracy,
            top3Accuracy,
            defects,
            passed,
            stopwatch.Elapsed.TotalMilliseconds);
    }
}
