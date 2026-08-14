using WowViewer.Core.Lit;

namespace WowViewer.Core.Tests;

public sealed class LitSourcePathResolverTests
{
    [Fact]
    public void DiscoversAllDirectMapVariantsAcrossSupportedWorldRoots()
    {
        IReadOnlyList<string> paths = LitSourcePathResolver.Resolve(
            [
                "World\\Maps\\Azeroth\\lights.lit",
                "World\\Maps\\Azeroth\\lights_night.lit",
                "World\\Azeroth\\custom.lit",
                "World\\Maps\\Kalimdor\\lights.lit",
                "World\\Maps\\Azeroth\\nested\\ignored.lit",
                "World\\Maps\\Azeroth\\not-a-lit.txt",
            ],
            "Azeroth");

        Assert.Equal(
            [
                "World\\Azeroth\\lights.lit",
                "World\\Maps\\Azeroth\\lights.lit",
                "World\\Azeroth\\areatest.lit",
                "World\\Maps\\Azeroth\\areatest.lit",
                "World\\Azeroth\\light.lit",
                "World\\Maps\\Azeroth\\light.lit",
                "World\\Azeroth\\custom.lit",
                "World\\Maps\\Azeroth\\lights_night.lit",
            ],
            paths);
    }

    [Fact]
    public void KeepsConventionalCandidatesWhenTheFileListDoesNotExposeThem()
    {
        IReadOnlyList<string> paths = LitSourcePathResolver.Resolve([], "Azeroth");

        Assert.Equal(6, paths.Count);
        Assert.Equal("World\\Azeroth\\lights.lit", paths[0]);
        Assert.Equal("World\\Maps\\Azeroth\\lights.lit", paths[1]);
        Assert.Equal("World\\Maps\\Azeroth\\light.lit", paths[^1]);
    }
}
