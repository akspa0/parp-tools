using WowViewer.Core.Runtime.World.Wdl;

namespace WowViewer.Core.Tests;

public sealed class WorldWdlTileBuilderTests
{
    [Fact]
    public void Read_DevelopmentWdl_ProducesExpectedTileSignals()
    {
        WorldWdlTileData tileData = WorldWdlTileBuilder.Read(MapTestPaths.DevelopmentWdlPath, 0, 0);

        Assert.True(tileData.SourceFound);
        Assert.True(tileData.HasData);
        Assert.Equal((uint)18, tileData.Version);
        Assert.Equal(0, tileData.TileX);
        Assert.Equal(0, tileData.TileY);
        Assert.Equal(289, tileData.OuterHeightCount);
        Assert.Equal(256, tileData.InnerHeightCount);
        Assert.True(tileData.MinHeight <= tileData.MaxHeight);
        Assert.NotNull(tileData.CenterHeight);
    }

    [Fact]
    public void Read_SyntheticReadableChunkTags_ProducesExpectedAggregateSignals()
    {
        byte[] bytes = WdlTestDataBuilder.CreateSingleTileWdl(tileX: 5, tileY: 7, useReadableTags: true, startHeight: 10);

        using MemoryStream stream = new(bytes, writable: false);
        WorldWdlTileData tileData = WorldWdlTileBuilder.Read(stream, "synthetic.wdl", 5, 7);

        Assert.True(tileData.SourceFound);
        Assert.True(tileData.HasData);
        Assert.Equal((uint)18, tileData.Version);
        Assert.Equal((short)10, tileData.MinHeight);
        Assert.Equal((short)554, tileData.MaxHeight);
        Assert.Equal((short)154, tileData.CenterHeight);
        Assert.Equal((short)10, tileData.NorthWestHeight);
        Assert.Equal((short)26, tileData.NorthEastHeight);
        Assert.Equal((short)282, tileData.SouthWestHeight);
        Assert.Equal((short)298, tileData.SouthEastHeight);
        Assert.Equal(289, tileData.OuterHeightCount);
        Assert.Equal(256, tileData.InnerHeightCount);
    }
}