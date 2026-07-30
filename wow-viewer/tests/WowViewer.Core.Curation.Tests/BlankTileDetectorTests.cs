using WowViewer.Core.Curation.Buckets;

namespace WowViewer.Core.Curation.Tests;

public class BlankTileDetectorTests
{
    [Fact]
    public void FlatZeroCoverageTile_IsBlank()
    {
        Assert.True(BlankTileDetector.IsBlank(TestFixtures.FlatBlankPack()));
    }

    [Fact]
    public void HighReliefWellPaintedTile_IsNotBlank()
    {
        Assert.False(BlankTileDetector.IsBlank(TestFixtures.HighReliefWellPaintedPack()));
    }

    [Fact]
    public void FlatBlankTile_CoverageBucket_IsBlank()
    {
        string bucket = CoverageBucketClassifier.Classify(TestFixtures.FlatBlankPack());
        Assert.Equal(WowViewer.Core.Curation.CoverageBucket.Blank, bucket);
    }

    [Fact]
    public void WellPaintedTile_CoverageBucket_IsWellCovered()
    {
        string bucket = CoverageBucketClassifier.Classify(TestFixtures.HighReliefWellPaintedPack());
        Assert.Equal(WowViewer.Core.Curation.CoverageBucket.WellCovered, bucket);
    }

    [Fact]
    public void EmptyPack_DoesNotThrow_AndIsBlank()
    {
        var emptyPack = new WowViewer.Core.Maps.TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 0,
            TileY = 0,
        };
        Assert.True(BlankTileDetector.IsBlank(emptyPack));
    }
}
