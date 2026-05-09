using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public class WdlWriterTests
{
    [Fact]
    public void ExtractTileHeightsFromAlpha_SamplesLocalTileHeightmap()
    {
        float[,] heightmap = new float[257, 257];
        for (int y = 0; y < 257; y++)
        {
            for (int x = 0; x < 257; x++)
                heightmap[y, x] = y * 10 + x;
        }

        WdlHeightTile tile = WdlWriter.ExtractTileHeightsFromAlpha(heightmap, 32, 48);

        Assert.Equal(32, tile.TileX);
        Assert.Equal(48, tile.TileY);
        Assert.Equal(0, tile.OuterHeights[0]);
        Assert.Equal(256 * 10 + 256, tile.OuterHeights[^1]);
        Assert.Equal(8 * 10 + 8, tile.InnerHeights[0]);
        Assert.Equal(248 * 10 + 248, tile.InnerHeights[^1]);
    }
}
