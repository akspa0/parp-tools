using WowViewer.Core.Curation;

namespace WowViewer.Core.Curation.Tests;

public class CurationRunRecordTests
{
    private static CurationRunRecord MakeRecord(int tileCount = 3) => new()
    {
        CurationRunId = "0_5_3_3368-Kalimdor.zarr-20260730T000000000Z",
        StorePath = @"I:\store\0_5_3_3368-Kalimdor.zarr",
        BuildFingerprint = "0_5_3_3368",
        ChecksRun = ["difficulty_bucket", "coverage_bucket", "lighting_bucket", "height_normal_mismatch"],
        TileCount = tileCount,
        BucketCounts = new Dictionary<string, IReadOnlyDictionary<string, int>>
        {
            ["difficulty_bucket"] = new Dictionary<string, int> { ["easy"] = 2, ["hard"] = 1 },
        },
        FindingCounts = new Dictionary<string, int> { ["height_normal_mismatch"] = 1 },
        ToolVersion = "0.0.0-test",
        CreatedAt = DateTimeOffset.UtcNow,
    };

    [Fact]
    public void ToJson_FromJson_RoundTripsEveryField()
    {
        CurationRunRecord original = MakeRecord();
        string json = original.ToJson();
        CurationRunRecord roundTripped = CurationRunRecord.FromJson(json);

        Assert.Equal(original.Schema, roundTripped.Schema);
        Assert.Equal(original.CurationRunId, roundTripped.CurationRunId);
        Assert.Equal(original.StorePath, roundTripped.StorePath);
        Assert.Equal(original.BuildFingerprint, roundTripped.BuildFingerprint);
        Assert.Equal(original.ChecksRun, roundTripped.ChecksRun);
        Assert.Equal(original.TileCount, roundTripped.TileCount);
        Assert.Equal(original.FindingCounts["height_normal_mismatch"], roundTripped.FindingCounts["height_normal_mismatch"]);
        Assert.Equal(original.BucketCounts["difficulty_bucket"]["easy"], roundTripped.BucketCounts["difficulty_bucket"]["easy"]);
        Assert.Equal(CurationRunRecord.CurrentSchema, roundTripped.Schema);
    }

    [Fact]
    public void Verify_Passes_WhenTileCountMatches()
    {
        CurationRunRecord record = MakeRecord(tileCount: 951);
        record.Verify(951); // must not throw
    }

    [Fact]
    public void Verify_Throws_WhenTileCountMismatches()
    {
        CurationRunRecord record = MakeRecord(tileCount: 950);
        var ex = Assert.Throws<InvalidOperationException>(() => record.Verify(951));
        Assert.Contains("950", ex.Message);
        Assert.Contains("951", ex.Message);
    }

    [Fact]
    public void GenerateRunId_IsDeterministicForTheSameInputs()
    {
        var now = new DateTimeOffset(2026, 7, 30, 0, 0, 0, TimeSpan.Zero);
        string a = CurationRunRecord.GenerateRunId("0_5_3_3368", @"I:\store\0_5_3_3368-Kalimdor.zarr", now);
        string b = CurationRunRecord.GenerateRunId("0_5_3_3368", @"I:\store\0_5_3_3368-Kalimdor.zarr", now);
        Assert.Equal(a, b);
        Assert.Contains("0_5_3_3368-Kalimdor.zarr", a);
    }

    [Fact]
    public void GenerateRunId_DiffersForDifferentTimestamps()
    {
        var t1 = new DateTimeOffset(2026, 7, 30, 0, 0, 0, TimeSpan.Zero);
        var t2 = t1.AddSeconds(1);
        string a = CurationRunRecord.GenerateRunId("0_5_3_3368", @"I:\store\x.zarr", t1);
        string b = CurationRunRecord.GenerateRunId("0_5_3_3368", @"I:\store\x.zarr", t2);
        Assert.NotEqual(a, b);
    }
}
