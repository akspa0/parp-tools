using WowViewer.Core.Curation;

namespace WowViewer.Core.Curation.Tests;

/// <summary>
/// Not a behavioral test of <see cref="CurationManifestWriter"/> itself (that's
/// <see cref="CurationManifestWriterTests"/>) -- this produces the checked-in fixture Parquet files
/// that <c>data-harvester/tests/test_curation_store.py</c> reads with <c>pyarrow</c>, proving the
/// real cross-language contract: a manifest written by this C# writer is readable, with the exact
/// documented column names/dtypes, from the Python side (data-model.md's actual proof requirement,
/// tasks.md T011).
/// </summary>
public class CrossLanguageFixtureGeneratorTests
{
    [Fact]
    public void GenerateFixture_ForPythonCrossLanguageReadTest()
    {
        string repoRoot = FindRepoRoot(AppContext.BaseDirectory);
        string fixtureStoreRoot = Path.Combine(repoRoot, "data-harvester", "tests", "fixtures", "spec122_curation_manifest");

        if (Directory.Exists(Path.Combine(fixtureStoreRoot, "curation")))
            Directory.Delete(Path.Combine(fixtureStoreRoot, "curation"), recursive: true);
        Directory.CreateDirectory(fixtureStoreRoot);

        const string runId = "0_5_3_3368-fixture-20260730T000000000Z";
        var records = new List<TileCurationRecord>
        {
            new("alpha", "Kalimdor", 19, 12, 0,
                WowViewer.Core.Curation.DifficultyBucket.Easy,
                WowViewer.Core.Curation.CoverageBucket.WellCovered,
                WowViewer.Core.Curation.LightingBucket.Matched,
                WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated, 0.91f, 0, runId),
            new("alpha", "Kalimdor", 19, 13, 1,
                WowViewer.Core.Curation.DifficultyBucket.Pathological,
                WowViewer.Core.Curation.CoverageBucket.Blank,
                WowViewer.Core.Curation.LightingBucket.NotEvaluated,
                WowViewer.Core.Curation.SyntheticFidelityStatus.NotEvaluable, null, 2, runId),
            new("alpha", "Kalimdor", 19, 14, 2,
                WowViewer.Core.Curation.DifficultyBucket.Hard,
                WowViewer.Core.Curation.CoverageBucket.LowCoverage,
                WowViewer.Core.Curation.LightingBucket.LowConfidenceAmbiguous,
                WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated, 0.34f, 0, runId),
        };
        var findings = new List<MismatchFinding>
        {
            new("alpha", "Kalimdor", 19, 13, 1,
                WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
                WowViewer.Core.Curation.MismatchSeverity.High,
                "height_flat_vs_normal_varied",
                WowViewer.Core.Curation.Evaluability.Evaluated,
                "normal_xyz", runId),
            new("alpha", "Kalimdor", 19, 13, 1,
                WowViewer.Core.Curation.MismatchCategory.NonFiniteValue,
                WowViewer.Core.Curation.MismatchSeverity.High,
                "nan_detected_in_height_257",
                WowViewer.Core.Curation.Evaluability.Evaluated,
                "height_257", runId),
        };

        var run = new CurationRunRecord
        {
            CurationRunId = runId,
            StorePath = fixtureStoreRoot,
            BuildFingerprint = "0_5_3_3368",
            ChecksRun = ["difficulty_bucket", "coverage_bucket", "lighting_bucket", "height_normal_mismatch", "non_finite_value"],
            TileCount = records.Count,
            BucketCounts = new Dictionary<string, IReadOnlyDictionary<string, int>>
            {
                ["difficulty_bucket"] = new Dictionary<string, int> { ["easy"] = 1, ["hard"] = 1, ["pathological"] = 1 },
                ["coverage_bucket"] = new Dictionary<string, int> { ["well_covered"] = 1, ["low_coverage"] = 1, ["blank"] = 1 },
            },
            FindingCounts = new Dictionary<string, int>
            {
                ["height_normal_mismatch"] = 1,
                ["non_finite_value"] = 1,
            },
            ToolVersion = "0.0.0-fixture",
            CreatedAt = new DateTimeOffset(2026, 7, 30, 0, 0, 0, TimeSpan.Zero),
        };

        string runDir = CurationManifestWriter.Write(fixtureStoreRoot, run, records, findings);
        Assert.True(File.Exists(Path.Combine(runDir, "curation_manifest.parquet")));
        Assert.True(File.Exists(Path.Combine(runDir, "curation_findings.parquet")));
    }

    private static string FindRepoRoot(string startDirectory)
    {
        var dir = new DirectoryInfo(startDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "WowViewer.slnx")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new InvalidOperationException($"Could not find WowViewer.slnx walking up from {startDirectory}");
    }
}
