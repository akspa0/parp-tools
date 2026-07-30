using WowViewer.Core.Curation.Mismatch;

namespace WowViewer.Core.Curation.Tests;

public class HasFlagTruthfulnessDetectorTests
{
    [Fact]
    public void FlagTrue_BackingDataPresent_ProducesNoFinding()
    {
        var pack = TestFixtures.HighReliefWellPaintedPack(); // claims has_normal_xyz/has_alpha_256/has_mcly_texture_ids, all genuinely backed
        var findings = HasFlagTruthfulnessDetector.Detect(pack, "alpha", "Kalimdor", 20, 12, 1, "run-1");
        Assert.Empty(findings);
    }

    [Fact]
    public void FlagTrue_BackingDataAbsent_IsFlagged()
    {
        var pack = new WowViewer.Core.Maps.TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 0,
            TileY = 0,
            McnrNormalXyz = null, // contradicts the flag below
            AvailableSignals = new HashSet<string> { "has_normal_xyz" },
        };

        var findings = HasFlagTruthfulnessDetector.Detect(pack, "alpha", "Kalimdor", 0, 0, 0, "run-1");

        Assert.Contains(findings, f =>
            f.Category == WowViewer.Core.Curation.MismatchCategory.HasFlagMismatch
            && f.Signal == "normal_xyz"
            && f.Evaluability == WowViewer.Core.Curation.Evaluability.Evaluated);
    }

    [Fact]
    public void FlagNotClaimed_ProducesNoFinding_EvenIfDataAbsent()
    {
        var pack = new WowViewer.Core.Maps.TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 0,
            TileY = 0,
            McnrNormalXyz = null,
            AvailableSignals = new HashSet<string>(), // never claimed present -- nothing to contradict
        };

        var findings = HasFlagTruthfulnessDetector.Detect(pack, "alpha", "Kalimdor", 0, 0, 0, "run-1");
        Assert.Empty(findings);
    }
}
