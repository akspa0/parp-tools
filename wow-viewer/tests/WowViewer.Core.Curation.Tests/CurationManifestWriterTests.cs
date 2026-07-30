using WowViewer.Core.Curation;

namespace WowViewer.Core.Curation.Tests;

public class CurationManifestWriterTests : IDisposable
{
    private readonly string _tempRoot;

    public CurationManifestWriterTests()
    {
        _tempRoot = Path.Combine(Path.GetTempPath(), "curation-writer-tests-" + Guid.NewGuid());
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempRoot, recursive: true); } catch { /* best-effort cleanup */ }
    }

    private static (IReadOnlyList<TileCurationRecord> Records, IReadOnlyList<MismatchFinding> Findings) MakeFixture(string runId)
    {
        var records = new List<TileCurationRecord>
        {
            new("alpha", "Kalimdor", 19, 12, 0,
                WowViewer.Core.Curation.DifficultyBucket.Easy,
                WowViewer.Core.Curation.CoverageBucket.WellCovered,
                WowViewer.Core.Curation.LightingBucket.Matched,
                WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated, 0.9f, 0, runId),
            new("alpha", "Kalimdor", 19, 13, 1,
                WowViewer.Core.Curation.DifficultyBucket.Pathological,
                WowViewer.Core.Curation.CoverageBucket.Blank,
                WowViewer.Core.Curation.LightingBucket.NotEvaluated,
                WowViewer.Core.Curation.SyntheticFidelityStatus.NotEvaluable, null, 1, runId),
        };
        var findings = new List<MismatchFinding>
        {
            new("alpha", "Kalimdor", 19, 13, 1,
                WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
                WowViewer.Core.Curation.MismatchSeverity.High,
                "height_flat_vs_normal_varied",
                WowViewer.Core.Curation.Evaluability.Evaluated,
                "normal_xyz", runId),
        };
        return (records, findings);
    }

    private static CurationRunRecord MakeRunRecord(string runId, int tileCount) => new()
    {
        CurationRunId = runId,
        StorePath = "fixture-store",
        BuildFingerprint = "0_5_3_3368",
        ChecksRun = ["difficulty_bucket", "coverage_bucket", "lighting_bucket", "height_normal_mismatch"],
        TileCount = tileCount,
        BucketCounts = new Dictionary<string, IReadOnlyDictionary<string, int>>(),
        FindingCounts = new Dictionary<string, int> { ["height_normal_mismatch"] = 1 },
        ToolVersion = "0.0.0-test",
        CreatedAt = DateTimeOffset.UtcNow,
    };

    [Fact]
    public void Write_ProducesBothParquetFilesWithCorrectRowCounts()
    {
        (IReadOnlyList<TileCurationRecord> records, IReadOnlyList<MismatchFinding> findings) = MakeFixture("run-a");
        CurationRunRecord run = MakeRunRecord("run-a", records.Count);

        string runDir = CurationManifestWriter.Write(_tempRoot, run, records, findings);

        string manifestPath = Path.Combine(runDir, "curation_manifest.parquet");
        string findingsPath = Path.Combine(runDir, "curation_findings.parquet");
        string runJsonPath = Path.Combine(runDir, "curation_run.json");

        Assert.True(File.Exists(manifestPath));
        Assert.True(File.Exists(findingsPath));
        Assert.True(File.Exists(runJsonPath));

        Assert.Equal(records.Count, CurationManifestWriter.ReadManifestRowCount(manifestPath));
        Assert.Equal(findings.Count, CurationManifestWriter.ReadManifestRowCount(findingsPath));
    }

    [Fact]
    public void Write_UpdatesLatestPointer_ToTheJustWrittenRunId()
    {
        (IReadOnlyList<TileCurationRecord> records, IReadOnlyList<MismatchFinding> findings) = MakeFixture("run-b");
        CurationRunRecord run = MakeRunRecord("run-b", records.Count);

        CurationManifestWriter.Write(_tempRoot, run, records, findings);

        string latestPointer = Path.Combine(_tempRoot, "curation", "latest");
        Assert.True(File.Exists(latestPointer));
        Assert.Equal("run-b", File.ReadAllText(latestPointer).Trim());
    }

    [Fact]
    public void Write_SecondRun_DoesNotOverwriteOrMutateTheFirstRunsDirectory()
    {
        (IReadOnlyList<TileCurationRecord> recordsA, IReadOnlyList<MismatchFinding> findingsA) = MakeFixture("run-1");
        CurationManifestWriter.Write(_tempRoot, MakeRunRecord("run-1", recordsA.Count), recordsA, findingsA);

        string firstManifestPath = Path.Combine(_tempRoot, "curation", "run-1", "curation_manifest.parquet");
        DateTime firstWriteTime = File.GetLastWriteTimeUtc(firstManifestPath);

        (IReadOnlyList<TileCurationRecord> recordsB, IReadOnlyList<MismatchFinding> findingsB) = MakeFixture("run-2");
        CurationManifestWriter.Write(_tempRoot, MakeRunRecord("run-2", recordsB.Count), recordsB, findingsB);

        // The first run's directory and file must still exist, untouched.
        Assert.True(File.Exists(firstManifestPath));
        Assert.Equal(firstWriteTime, File.GetLastWriteTimeUtc(firstManifestPath));
        Assert.True(File.Exists(Path.Combine(_tempRoot, "curation", "run-2", "curation_manifest.parquet")));

        // "latest" now points at the second run, not the first.
        Assert.Equal("run-2", File.ReadAllText(Path.Combine(_tempRoot, "curation", "latest")).Trim());
    }

    [Fact]
    public void Write_Throws_WhenRunRecordTileCountDoesNotMatchRecordCount()
    {
        (IReadOnlyList<TileCurationRecord> records, IReadOnlyList<MismatchFinding> findings) = MakeFixture("run-mismatch");
        CurationRunRecord run = MakeRunRecord("run-mismatch", tileCount: records.Count + 1);

        Assert.Throws<InvalidOperationException>(() => CurationManifestWriter.Write(_tempRoot, run, records, findings));
        // Nothing should have been written for a run that fails its own tile-count gate.
        Assert.False(Directory.Exists(Path.Combine(_tempRoot, "curation", "run-mismatch")));
    }
}
