using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public class WdlWriterTests
{
    [Fact]
    public void Build_WritesStrictAlphaChunkOrder_MverThenMaof()
    {
        var tiles = new List<WdlHeightTile>
        {
            new(0, 0, new short[17 * 17], new short[16 * 16])
        };

        byte[] bytes = WdlWriter.Build(tiles);

        int firstChunk = 0;
        Assert.Equal("MVER", ReadChunkId(bytes, firstChunk));

        int firstSize = BitConverter.ToInt32(bytes, firstChunk + 4);
        int secondChunk = firstChunk + 8 + firstSize;
        Assert.Equal("MAOF", ReadChunkId(bytes, secondChunk));
    }

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

    private static string ReadChunkId(byte[] data, int offset)
    {
        return FourCC.FromFileBytes(data.AsSpan(offset, 4)).ToString();
    }
}
