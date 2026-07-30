using WowViewer.Core.Curation.Mismatch;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Tests;

public class SyntheticFidelityDetectorTests
{
    private static TerrainTileTensorPack PackWithProvenance(MinimapLightingProvenance? provenance)
    {
        var pack = TestFixtures.HighReliefWellPaintedPack();
        pack.MinimapLightingProvenance = provenance;
        return pack;
    }

    [Fact]
    public void NoProvenance_IsNotEvaluable()
    {
        (string status, float? score) = SyntheticFidelityDetector.Classify(PackWithProvenance(null));
        Assert.Equal(WowViewer.Core.Curation.SyntheticFidelityStatus.NotEvaluable, status);
        Assert.Null(score);
    }

    [Fact]
    public void NoProvenance_ProducesANotEvaluableFinding()
    {
        MismatchFinding? finding = SyntheticFidelityDetector.Detect(
            PackWithProvenance(null), "alpha", "Kalimdor", 20, 12, 1, "run-1");

        Assert.NotNull(finding);
        Assert.Equal(WowViewer.Core.Curation.Evaluability.NotEvaluable, finding!.Evaluability);
        Assert.Equal(WowViewer.Core.Curation.MismatchSeverity.NotEvaluable, finding.Severity);
    }

    [Fact]
    public void ExplicitlyNotEvaluatedStatus_IsNotEvaluable()
    {
        var provenance = MinimapLightingProvenance.NotEvaluated("insufficient_unlit_baseline_coverage");
        (string status, float? score) = SyntheticFidelityDetector.Classify(PackWithProvenance(provenance));
        Assert.Equal(WowViewer.Core.Curation.SyntheticFidelityStatus.NotEvaluable, status);
        Assert.Null(score);
    }

    [Fact]
    public void HighCorrelation_IsEvaluated_AndProducesNoFinding()
    {
        var provenance = new MinimapLightingProvenance(
            MinimapLightingProvenance.CurrentContractVersion, "baked_tint_and_mcsh_likely", 1000,
            1f, 1f, 1f, 0f, 1f, 0.9f, 12f, 0.9f, "evidence", "candidate",
            ShadingMatchStatus: "matched",
            ShadingMatchedTimeOfDayHours: 12f,
            ShadingMatchConfidence: 0.85f);

        var pack = PackWithProvenance(provenance);

        (string status, float? score) = SyntheticFidelityDetector.Classify(pack);
        Assert.Equal(WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated, status);
        Assert.Equal(0.85f, score);

        MismatchFinding? finding = SyntheticFidelityDetector.Detect(pack, "alpha", "Kalimdor", 20, 12, 1, "run-1");
        Assert.Null(finding); // Evaluated and acceptable -- no finding row.
    }

    [Fact]
    public void LowCorrelation_IsEvaluated_AndProducesAGapFinding()
    {
        var provenance = new MinimapLightingProvenance(
            MinimapLightingProvenance.CurrentContractVersion, "unlit_or_unclassified", 1000,
            null, null, null, null, null, null, null, null, "no_baked_tint_detected", null,
            ShadingMatchStatus: "low_confidence_ambiguous",
            ShadingMatchedTimeOfDayHours: null,
            ShadingMatchConfidence: 0.05f);

        var pack = PackWithProvenance(provenance);

        (string status, float? score) = SyntheticFidelityDetector.Classify(pack);
        Assert.Equal(WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated, status);
        Assert.Equal(0.05f, score);

        MismatchFinding? finding = SyntheticFidelityDetector.Detect(pack, "alpha", "Kalimdor", 20, 12, 1, "run-1");
        Assert.NotNull(finding);
        Assert.Equal(WowViewer.Core.Curation.MismatchCategory.SyntheticFidelityGap, finding!.Category);
        Assert.Equal(WowViewer.Core.Curation.Evaluability.Evaluated, finding.Evaluability);
        Assert.Equal(WowViewer.Core.Curation.MismatchSeverity.High, finding.Severity); // 0.05 < 0.10
    }

    [Fact]
    public void ModeratelyLowCorrelation_IsMediumSeverity()
    {
        var provenance = new MinimapLightingProvenance(
            MinimapLightingProvenance.CurrentContractVersion, "unlit_or_unclassified", 1000,
            null, null, null, null, null, null, null, null, "no_baked_tint_detected", null,
            ShadingMatchStatus: "low_confidence_ambiguous",
            ShadingMatchedTimeOfDayHours: null,
            ShadingMatchConfidence: 0.20f);

        var pack = PackWithProvenance(provenance);
        MismatchFinding? finding = SyntheticFidelityDetector.Detect(pack, "alpha", "Kalimdor", 20, 12, 1, "run-1");

        Assert.NotNull(finding);
        Assert.Equal(WowViewer.Core.Curation.MismatchSeverity.Medium, finding!.Severity); // 0.10 <= 0.20 < 0.30
    }
}
