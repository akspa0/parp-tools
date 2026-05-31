using WowViewer.Core.IO.Wmo;

namespace WowViewer.Core.Tests;

public class WmoMinimapBlpReaderTests
{
    [Theory]
    [InlineData("deadmines_000_00_00.blp", "deadmines", 0, 0, 0)]
    [InlineData("stockades_001_01_02.blp", "stockades", 1, 1, 2)]
    [InlineData("ragefire_003_00_01.blp", "ragefire", 3, 0, 1)]
    [InlineData("my-wmo-name_000_00_00.blp", "my-wmo-name", 0, 0, 0)]
    [InlineData("dungeon_000_00_00.blp", "dungeon", 0, 0, 0)]
    public void TryParseFilename_ValidPattern_ExtractsComponents(
        string filename,
        string expectedStem,
        int expectedGroup,
        int expectedQuadY,
        int expectedQuadX)
    {
        bool result = WmoMinimapBlpReader.TryParseFilename(
            filename, out string? stem, out int group, out int quadY, out int quadX);

        Assert.True(result);
        Assert.Equal(expectedStem, stem);
        Assert.Equal(expectedGroup, group);
        Assert.Equal(expectedQuadY, quadY);
        Assert.Equal(expectedQuadX, quadX);
    }

    [Theory]
    [InlineData("terrain_minimap.blp")]
    [InlineData("deadmines.blp")]
    [InlineData("deadmines_00.blp")]
    [InlineData("deadmines_000_00.blp")]
    [InlineData("deadmines_000_00_00_00.blp")]
    [InlineData("deadmines_abc_00_00.blp")]
    [InlineData("")]
    public void TryParseFilename_InvalidPattern_ReturnsFalse(string filename)
    {
        bool result = WmoMinimapBlpReader.TryParseFilename(
            filename, out string? stem, out int group, out int quadY, out int quadX);

        Assert.False(result);
        Assert.Null(stem);
    }

    [Theory]
    [InlineData("deadmines_000_00_00.blp", true)]
    [InlineData("stockades_001_01_02.blp", true)]
    [InlineData("terrain_map_3_4.blp", false)]
    [InlineData("not_a_wmo.blp", false)]
    public void IsWmoMinimapBlp_ReturnsCorrectResult(string filename, bool expected)
    {
        Assert.Equal(expected, WmoMinimapBlpReader.IsWmoMinimapBlp(filename));
    }

    [Fact]
    public void ParseGroupIndex_HandlesTwoDigit()
    {
        Assert.Equal(5, WmoMinimapBlpReader.ParseGroupIndex("05"));
        Assert.Equal(0, WmoMinimapBlpReader.ParseGroupIndex("00"));
        Assert.Equal(99, WmoMinimapBlpReader.ParseGroupIndex("99"));
    }

    [Fact]
    public void ParseGroupIndex_HandlesThreeDigit()
    {
        Assert.Equal(100, WmoMinimapBlpReader.ParseGroupIndex("100"));
        Assert.Equal(256, WmoMinimapBlpReader.ParseGroupIndex("256"));
    }

    [Fact]
    public void TryParseFilename_CaseInsensitive()
    {
        bool result = WmoMinimapBlpReader.TryParseFilename(
            "Deadmines_000_00_00.BLP", out string? stem, out _, out _, out _);

        Assert.True(result);
        Assert.Equal("Deadmines", stem);
    }
}
