using WowViewer.Core.Curation.Mismatch;

namespace WowViewer.Core.Curation.Tests;

public class HeightNormalMismatchDetectorTests
{
    [Fact]
    public void FlatHeightWithVariedNormals_ProducesAnEvaluatedFinding()
    {
        MismatchFinding? finding = HeightNormalMismatchDetector.Detect(
            TestFixtures.HeightNormalMismatchPack(), "alpha", "Kalimdor", 21, 12, 3, "run-1");

        Assert.NotNull(finding);
        Assert.Equal(WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch, finding!.Category);
        Assert.Equal(WowViewer.Core.Curation.Evaluability.Evaluated, finding.Evaluability);
        Assert.Equal("height_flat_vs_normal_varied", finding.Reason);
        Assert.NotEqual(WowViewer.Core.Curation.MismatchSeverity.None, finding.Severity);
        Assert.NotEqual(WowViewer.Core.Curation.MismatchSeverity.NotEvaluable, finding.Severity);
    }

    [Fact]
    public void FlatHeightFlatNormals_ProducesNoFinding()
    {
        // Normals genuinely present and fully covered, but flat (straight-up everywhere) -- height
        // is also flat, so this is consistent, not a mismatch ("flat_normals" branch).
        var normals = new float[257, 257, 3];
        var normalMask = new bool[257, 257];
        for (int y = 0; y < 257; y++)
        {
            for (int x = 0; x < 257; x++)
            {
                normals[y, x, 2] = 1f;
                normalMask[y, x] = true;
            }
        }
        var pack = new WowViewer.Core.Maps.TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 19,
            TileY = 12,
            Height257 = new float[257, 257],
            McnrNormalXyz = normals,
            McnrMask257 = normalMask,
        };

        MismatchFinding? finding = HeightNormalMismatchDetector.Detect(pack, "alpha", "Kalimdor", 19, 12, 0, "run-1");

        Assert.Null(finding); // "flat_normals": consistent, not a mismatch.
    }

    [Fact]
    public void HighReliefHeightWithVariedNormals_ProducesNoFinding()
    {
        // Both height and normals vary -- consistent, not a mismatch ("sufficient_height_range").
        MismatchFinding? finding = HeightNormalMismatchDetector.Detect(
            TestFixtures.HighReliefWellPaintedPack(), "alpha", "Kalimdor", 20, 12, 1, "run-1");

        Assert.Null(finding);
    }

    [Fact]
    public void MissingNormalData_ProducesNotEvaluableFinding()
    {
        var pack = new WowViewer.Core.Maps.TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 0,
            TileY = 0,
            Height257 = new float[257, 257],
            McnrNormalXyz = null,
        };

        MismatchFinding? finding = HeightNormalMismatchDetector.Detect(pack, "alpha", "Kalimdor", 0, 0, 0, "run-1");

        Assert.NotNull(finding);
        Assert.Equal(WowViewer.Core.Curation.Evaluability.NotEvaluable, finding!.Evaluability);
        Assert.Equal(WowViewer.Core.Curation.MismatchSeverity.NotEvaluable, finding.Severity);
        Assert.Equal("no_normal_data", finding.Reason);
    }

    [Fact]
    public void InsufficientNormalCoverage_ProducesNotEvaluableFinding_NotAFalseNegative()
    {
        var normals = new float[257, 257, 3];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                normals[y, x, 2] = 1f;

        var sparseMask = new bool[257, 257]; // almost entirely false -> coverage well under 0.10
        sparseMask[0, 0] = true;

        var pack = new WowViewer.Core.Maps.TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 0,
            TileY = 0,
            Height257 = new float[257, 257],
            McnrNormalXyz = normals,
            McnrMask257 = sparseMask,
        };

        MismatchFinding? finding = HeightNormalMismatchDetector.Detect(pack, "alpha", "Kalimdor", 0, 0, 0, "run-1");

        Assert.NotNull(finding);
        Assert.Equal(WowViewer.Core.Curation.Evaluability.NotEvaluable, finding!.Evaluability);
        Assert.Equal("insufficient_normal_coverage", finding.Reason);
    }
}
