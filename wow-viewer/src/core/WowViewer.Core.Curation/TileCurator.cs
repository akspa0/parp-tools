using WowViewer.Core.Curation.Buckets;
using WowViewer.Core.Curation.Mismatch;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation;

/// <summary>
/// The single per-tile entry point: runs every canonical check (difficulty/coverage/lighting
/// buckets, height-normal mismatch, non-finite values, has-flag truthfulness) over one already-built
/// <see cref="TerrainTileTensorPack"/> and returns its <see cref="TileCurationRecord"/> plus every
/// <see cref="MismatchFinding"/> it produced. This is the one place "which checks exist and how
/// their results combine into a tile's record" is decided -- <c>curate</c>'s orchestration (in
/// <c>WowViewer.Tool.Harvest</c>) only builds tensor packs and calls this, it does not decide
/// curation logic itself.
/// </summary>
public static class TileCurator
{
    /// <summary>Names of every check this method can run, for the run record's <c>checks_run</c>
    /// list. Synthetic-fidelity is intentionally absent here -- it is wired in separately (User
    /// Story 3) since it requires an authored-vs-synthesized minimap pair, not just the base pack.</summary>
    public static readonly IReadOnlyList<string> KnownChecks =
    [
        "difficulty_bucket",
        "coverage_bucket",
        "lighting_bucket",
        WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
        WowViewer.Core.Curation.MismatchCategory.NonFiniteValue,
        WowViewer.Core.Curation.MismatchCategory.HasFlagMismatch,
        WowViewer.Core.Curation.MismatchCategory.SyntheticFidelityGap,
    ];

    public static (TileCurationRecord Record, IReadOnlyList<MismatchFinding> Findings) Classify(
        TerrainTileTensorPack pack,
        string build,
        string map,
        int tileX,
        int tileY,
        long tileId,
        string curationRunId)
    {
        ArgumentNullException.ThrowIfNull(pack);

        string difficultyBucket = DifficultyBucketClassifier.Classify(pack);
        string coverageBucket = CoverageBucketClassifier.Classify(pack);
        string lightingBucket = LightingBucketClassifier.Classify(pack);

        var findings = new List<MismatchFinding>();

        MismatchFinding? heightNormalFinding = HeightNormalMismatchDetector.Detect(
            pack, build, map, tileX, tileY, tileId, curationRunId);
        if (heightNormalFinding is not null)
            findings.Add(heightNormalFinding);

        findings.AddRange(NonFiniteSignalDetector.Detect(pack, build, map, tileX, tileY, tileId, curationRunId));
        findings.AddRange(HasFlagTruthfulnessDetector.Detect(pack, build, map, tileX, tileY, tileId, curationRunId));

        (string fidelityStatus, float? fidelityScore) = SyntheticFidelityDetector.Classify(pack);
        MismatchFinding? fidelityFinding = SyntheticFidelityDetector.Detect(
            pack, build, map, tileX, tileY, tileId, curationRunId);
        if (fidelityFinding is not null)
            findings.Add(fidelityFinding);

        var record = new TileCurationRecord(
            build, map, tileX, tileY, tileId,
            difficultyBucket, coverageBucket, lightingBucket,
            SyntheticFidelityStatus: fidelityStatus,
            SyntheticFidelityScore: fidelityScore,
            FindingCount: findings.Count,
            curationRunId);

        return (record, findings);
    }
}
