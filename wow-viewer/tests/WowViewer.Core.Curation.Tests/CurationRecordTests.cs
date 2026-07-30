using WowViewer.Core.Curation;

namespace WowViewer.Core.Curation.Tests;

public class CurationRecordTests
{
    [Fact]
    public void TileCurationRecord_RoundTripsEveryField()
    {
        var record = new TileCurationRecord(
            Build: "alpha",
            Map: "Kalimdor",
            TileX: 19,
            TileY: 12,
            TileId: 0,
            DifficultyBucket: WowViewer.Core.Curation.DifficultyBucket.Easy,
            CoverageBucket: WowViewer.Core.Curation.CoverageBucket.WellCovered,
            LightingBucket: WowViewer.Core.Curation.LightingBucket.Matched,
            SyntheticFidelityStatus: WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated,
            SyntheticFidelityScore: 0.82f,
            FindingCount: 0,
            CurationRunId: "run-1");

        Assert.Equal("alpha", record.Build);
        Assert.Equal("Kalimdor", record.Map);
        Assert.Equal(19, record.TileX);
        Assert.Equal(12, record.TileY);
        Assert.Equal(0, record.TileId);
        Assert.Equal(WowViewer.Core.Curation.DifficultyBucket.Easy, record.DifficultyBucket);
        Assert.Equal(WowViewer.Core.Curation.CoverageBucket.WellCovered, record.CoverageBucket);
        Assert.Equal(WowViewer.Core.Curation.LightingBucket.Matched, record.LightingBucket);
        Assert.Equal(WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated, record.SyntheticFidelityStatus);
        Assert.Equal(0.82f, record.SyntheticFidelityScore);
        Assert.Equal(0, record.FindingCount);
        Assert.Equal("run-1", record.CurationRunId);
    }

    [Fact]
    public void TileCurationRecord_AllowsNullSyntheticFidelityScore_WhenNotEvaluable()
    {
        var record = new TileCurationRecord(
            "alpha", "Kalimdor", 19, 12, 0,
            WowViewer.Core.Curation.DifficultyBucket.Medium,
            WowViewer.Core.Curation.CoverageBucket.LowCoverage,
            WowViewer.Core.Curation.LightingBucket.NotEvaluated,
            WowViewer.Core.Curation.SyntheticFidelityStatus.NotEvaluable,
            SyntheticFidelityScore: null,
            FindingCount: 1,
            CurationRunId: "run-1");

        Assert.Null(record.SyntheticFidelityScore);
    }

    [Fact]
    public void MismatchFinding_RoundTripsEveryField()
    {
        var finding = new MismatchFinding(
            Build: "alpha",
            Map: "Kalimdor",
            TileX: 19,
            TileY: 12,
            TileId: 0,
            Category: WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
            Severity: WowViewer.Core.Curation.MismatchSeverity.High,
            Reason: "height_flat_vs_normal_varied",
            Evaluability: WowViewer.Core.Curation.Evaluability.Evaluated,
            Signal: "normal_xyz",
            CurationRunId: "run-1");

        Assert.Equal(WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch, finding.Category);
        Assert.Equal(WowViewer.Core.Curation.MismatchSeverity.High, finding.Severity);
        Assert.Equal("height_flat_vs_normal_varied", finding.Reason);
        Assert.Equal(WowViewer.Core.Curation.Evaluability.Evaluated, finding.Evaluability);
        Assert.Equal("normal_xyz", finding.Signal);
    }

    [Fact]
    public void MismatchFinding_NotEvaluable_NeverCarriesAPlainSeverity()
    {
        // Data-model.md validation rule: a not_evaluable finding must use the literal
        // "not_evaluable" severity sentinel, never "none" -- so a consumer filtering on severity
        // alone cannot conflate "not checked" with "checked, no problem".
        var finding = new MismatchFinding(
            "alpha", "Kalimdor", 19, 12, 0,
            WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
            WowViewer.Core.Curation.MismatchSeverity.NotEvaluable,
            "insufficient_normal_coverage",
            WowViewer.Core.Curation.Evaluability.NotEvaluable,
            Signal: "normal_mask",
            CurationRunId: "run-1");

        Assert.Equal(WowViewer.Core.Curation.Evaluability.NotEvaluable, finding.Evaluability);
        Assert.Equal(WowViewer.Core.Curation.MismatchSeverity.NotEvaluable, finding.Severity);
        Assert.NotEqual(WowViewer.Core.Curation.MismatchSeverity.None, finding.Severity);
    }
}
