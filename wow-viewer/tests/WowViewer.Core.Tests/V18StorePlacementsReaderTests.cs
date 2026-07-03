using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public class V18StorePlacementsReaderTests
{
    [Fact]
    public void ReadPlacements_MissingParquet_ReturnsEmptyInventory()
    {
        string tmpDir = Path.Combine(Path.GetTempPath(), "v22enrich_test_empty_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmpDir);
        try
        {
            var inventory = V18StorePlacementsReader.ReadPlacements(tmpDir);
            Assert.Empty(inventory.UniqueM2Paths);
            Assert.Empty(inventory.UniqueWmoPaths);
            Assert.Empty(inventory.UniqueBlpPaths);
        }
        finally
        {
            if (Directory.Exists(tmpDir))
                Directory.Delete(tmpDir, recursive: true);
        }
    }

    [Fact]
    public void ReadBlpPathsFromMetadata_ReturnsCorrectPaths()
    {
        var paths = new List<string>
        {
            "Tileset\\Generic\\Dirt.blp",
            "Tileset\\Generic\\Grass.blp",
            "World\\M2\\SomeModel.m2",
            "Tileset\\Generic\\Rock.blp",
            "",
            null!,  // null should be handled gracefully
        };

        var result = V18StorePlacementsReader.ReadBlpPathsFromMetadata(paths!);
        Assert.Equal(3, result.Count);
        Assert.Contains("tileset/generic/dirt.blp", result.Select(p => p.ToLowerInvariant()));
        Assert.Contains("tileset/generic/rock.blp", result.Select(p => p.ToLowerInvariant()));
        Assert.DoesNotContain(result, p => p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase));
    }
}