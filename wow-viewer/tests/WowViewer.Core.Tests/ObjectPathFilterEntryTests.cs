using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

public sealed class ObjectPathFilterEntryTests
{
    [Theory]
    [InlineData("World/Generic/Human", "World\\Generic\\Human")]
    [InlineData("\\World\\Generic\\Human\\", "World\\Generic\\Human")]
    [InlineData("  World\\Generic\\Human\\House.wmo  ", "World\\Generic\\Human\\House.wmo")]
    public void NormalizePrefix_NormalizesSeparatorsAndOuterSlashes(string rawPath, string expected)
    {
        string normalized = ObjectPathFilterEntry.NormalizePrefix(rawPath);

        Assert.Equal(expected, normalized);
    }

    [Fact]
    public void MatchesModelPath_MatchesFolderPrefixAndExactAssetPath()
    {
        ObjectPathFilterEntry folderFilter = new("World\\Generic\\Human", AppliesToWmo: true, AppliesToMdx: false);
        ObjectPathFilterEntry exactFilter = new("World\\Generic\\Human\\House.wmo", AppliesToWmo: true, AppliesToMdx: false);

        Assert.True(folderFilter.MatchesModelPath("World/Generic/Human/House.wmo"));
        Assert.True(exactFilter.MatchesModelPath("World\\Generic\\Human\\House.wmo"));
        Assert.False(folderFilter.MatchesModelPath("World\\Generic\\Humanoid\\House.wmo"));
    }

    [Fact]
    public void MatchesModelPath_RespectsWmoAndMdxFamilyFlags()
    {
        ObjectPathFilterEntry wmoOnly = new("World\\Buildings", AppliesToWmo: true, AppliesToMdx: false);
        ObjectPathFilterEntry mdxOnly = new("World\\Buildings", AppliesToWmo: false, AppliesToMdx: true);

        Assert.True(wmoOnly.MatchesModelPath("World\\Buildings\\Tower.wmo"));
        Assert.False(wmoOnly.MatchesModelPath("World\\Buildings\\Torch.mdx"));
        Assert.True(mdxOnly.MatchesModelPath("World\\Buildings\\Torch.mdx"));
        Assert.False(mdxOnly.MatchesModelPath("World\\Buildings\\Tower.wmo"));
    }
}