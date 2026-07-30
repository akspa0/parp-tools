using WowViewer.Core.Curation.Buckets;

namespace WowViewer.Core.Curation.Tests;

public class DifficultyBucketClassifierTests
{
    [Fact]
    public void FlatBlankTile_ClassifiesEasy()
    {
        string bucket = DifficultyBucketClassifier.Classify(TestFixtures.FlatBlankPack());
        Assert.Equal(WowViewer.Core.Curation.DifficultyBucket.Easy, bucket);
    }

    [Fact]
    public void HighReliefWellPaintedTile_ClassifiesHardOrPathological()
    {
        string bucket = DifficultyBucketClassifier.Classify(TestFixtures.HighReliefWellPaintedPack());
        Assert.True(
            bucket is WowViewer.Core.Curation.DifficultyBucket.Hard or WowViewer.Core.Curation.DifficultyBucket.Pathological,
            $"Expected hard or pathological for a high-relief, well-painted fixture, got '{bucket}'.");
    }

    [Fact]
    public void EmptyPack_DoesNotThrow_AndClassifiesEasy()
    {
        var emptyPack = new WowViewer.Core.Maps.TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 0,
            TileY = 0,
        };

        string bucket = DifficultyBucketClassifier.Classify(emptyPack);
        Assert.Equal(WowViewer.Core.Curation.DifficultyBucket.Easy, bucket);
    }
}
