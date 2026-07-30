using WowViewer.Core.Curation.Mismatch;

namespace WowViewer.Core.Curation.Tests;

public class NonFiniteSignalDetectorTests
{
    [Fact]
    public void CleanPack_ProducesNoFindings()
    {
        var findings = NonFiniteSignalDetector.Detect(TestFixtures.HighReliefWellPaintedPack(), "alpha", "Kalimdor", 20, 12, 1, "run-1");
        Assert.Empty(findings);
    }

    [Fact]
    public void InjectedNaNInHeight_IsFlagged()
    {
        var pack = TestFixtures.FlatBlankPack();
        pack.Height257![10, 10] = float.NaN;

        var findings = NonFiniteSignalDetector.Detect(pack, "alpha", "Kalimdor", 19, 12, 0, "run-1");

        Assert.Contains(findings, f =>
            f.Category == WowViewer.Core.Curation.MismatchCategory.NonFiniteValue
            && f.Signal == "height_257"
            && f.Severity == WowViewer.Core.Curation.MismatchSeverity.High
            && f.Evaluability == WowViewer.Core.Curation.Evaluability.Evaluated);
    }

    [Fact]
    public void InjectedInfinityInNormals_IsFlagged()
    {
        var pack = TestFixtures.HighReliefWellPaintedPack();
        pack.McnrNormalXyz![5, 5, 0] = float.PositiveInfinity;

        var findings = NonFiniteSignalDetector.Detect(pack, "alpha", "Kalimdor", 20, 12, 1, "run-1");

        Assert.Contains(findings, f => f.Signal == "normal_xyz");
    }
}
