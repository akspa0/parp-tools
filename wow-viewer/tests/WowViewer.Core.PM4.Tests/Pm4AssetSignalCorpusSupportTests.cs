namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4AssetSignalCorpusSupportTests
{
    [Fact]
    public void LoadSeedPlacementAssetPaths_DevelopmentDirectory_IsStableUniqueAndWorldPrioritized()
    {
        IReadOnlyList<string> first = Pm4AssetSignalCorpusSupport.LoadSeedPlacementAssetPaths(Pm4TestPaths.DevelopmentDirectoryPath);
        IReadOnlyList<string> second = Pm4AssetSignalCorpusSupport.LoadSeedPlacementAssetPaths(Pm4TestPaths.DevelopmentDirectoryPath);

        Assert.NotEmpty(first);
        Assert.Equal(first.Count, first.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.True(first.SequenceEqual(second, StringComparer.OrdinalIgnoreCase));
        Assert.Contains(first, path => path.Equals(@"WORLD\WMO\NORTHREND\WINTERGRASP\WG_GATE01.WMO", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(first, path => !path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase));

        int firstDoodadsIndex = first
            .Select((path, index) => (path, index))
            .Where(static entry => entry.path.StartsWith("DOODADS\\", StringComparison.OrdinalIgnoreCase))
            .Select(static entry => entry.index)
            .DefaultIfEmpty(-1)
            .First();
        int lastWorldWmoIndex = first
            .Select((path, index) => (path, index))
            .Where(static entry => entry.path.StartsWith("WORLD\\WMO\\", StringComparison.OrdinalIgnoreCase))
            .Select(static entry => entry.index)
            .DefaultIfEmpty(-1)
            .Last();

        if (firstDoodadsIndex >= 0 && lastWorldWmoIndex >= 0)
            Assert.True(lastWorldWmoIndex < firstDoodadsIndex);
    }

}
