using MdxViewer.Terrain;
using Xunit;

namespace MdxViewer.Tests.Terrain;

public sealed class BuildVersionCatalogTests
{
    [Fact]
    public void LoadOptionsFromMapDbd_ParsesBuildLines_AndSortsByVersion()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "MdxViewer_BuildVersionCatalogTests_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            string mapDbdPath = Path.Combine(tempDir, "Map.dbd");
            File.WriteAllText(mapDbdPath,
                "BUILD 3.3.5.12340, 0.8.0.3734\n" +
                "BUILD 0.9.0.3807-0.9.1.3810\n" +
                "BUILD 0.5.3.3368\n");

            var options = BuildVersionCatalog.LoadOptionsFromMapDbd(tempDir);
            var builds = options.Select(o => o.BuildVersion).ToList();

            Assert.Contains("0.5.3.3368", builds);
            Assert.Contains("0.8.0.3734", builds);
            Assert.Contains("0.9.0.3807", builds);
            Assert.Contains("0.9.1.3810", builds);
            Assert.Contains("3.3.5.12340", builds);

            Assert.Equal("0.5.3.3368", builds.First());
            Assert.Equal("3.3.5.12340", builds.Last());
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void TryInferBuildIndexFromPath_PrefersExactBuildMatch()
    {
        var options = new List<ClientBuildOption>
        {
            new("Alpha (0.x) - 0.8.0.3734", "0.8.0.3734"),
            new("Alpha (0.x) - 0.9.1.3810", "0.9.1.3810"),
            new("Wrath (3.x) - 3.3.5.12340", "3.3.5.12340")
        };

        bool inferred = BuildVersionCatalog.TryInferBuildIndexFromPath(options, @"C:\Games\WoW\3.3.5.12340\Data", out int index);

        Assert.True(inferred);
        Assert.Equal(2, index);
    }

    [Fact]
    public void TryInferBuildIndexFromPath_UsesShortVersionWhenFullBuildMissing()
    {
        var options = new List<ClientBuildOption>
        {
            new("Alpha (0.x) - 0.9.0.3807", "0.9.0.3807"),
            new("Alpha (0.x) - 0.9.1.3810", "0.9.1.3810"),
            new("Wrath (3.x) - 3.3.5.12340", "3.3.5.12340")
        };

        bool inferred = BuildVersionCatalog.TryInferBuildIndexFromPath(options, @"D:\wow\alpha\0.9.1\client", out int index);

        Assert.True(inferred);
        Assert.Equal(1, index);
    }
}