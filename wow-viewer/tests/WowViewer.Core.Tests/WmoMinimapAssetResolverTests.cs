using WowViewer.Core.IO.Wmo;

namespace WowViewer.Core.Tests;

public class WmoMinimapAssetResolverTests
{
    [Fact]
    public void ResolveStemToAssetPath_FindsMatch()
    {
        var stemMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["deadmines"] = @"World\wmo\dungeon\deadmines\deadmines.wmo",
            ["stockades"] = @"World\wmo\dungeon\stockades\stockades.wmo",
        };

        string? result = WmoMinimapAssetResolver.ResolveStemToAssetPath(stemMap, "deadmines");

        Assert.Equal(@"World\wmo\dungeon\deadmines\deadmines.wmo", result);
    }

    [Fact]
    public void ResolveStemToAssetPath_CaseInsensitive()
    {
        var stemMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["deadmines"] = @"World\wmo\dungeon\deadmines\deadmines.wmo",
        };

        string? result = WmoMinimapAssetResolver.ResolveStemToAssetPath(stemMap, "Deadmines");

        Assert.Equal(@"World\wmo\dungeon\deadmines\deadmines.wmo", result);
    }

    [Fact]
    public void ResolveStemToAssetPath_ReturnsNull_WhenNotFound()
    {
        var stemMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["deadmines"] = @"World\wmo\dungeon\deadmines\deadmines.wmo",
        };

        string? result = WmoMinimapAssetResolver.ResolveStemToAssetPath(stemMap, "stockades");

        Assert.Null(result);
    }

    [Fact]
    public void ResolveStemToAssetPath_ReturnsNull_WhenStemIsEmpty()
    {
        var stemMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        string? result = WmoMinimapAssetResolver.ResolveStemToAssetPath(stemMap, "");

        Assert.Null(result);
    }
}
