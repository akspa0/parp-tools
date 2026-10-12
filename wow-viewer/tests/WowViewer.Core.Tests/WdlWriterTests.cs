using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Terrain;
using WowViewer.Core.Maps;

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

    [Fact]
    public void ExtractTileHeightsFromLk_SamplesChunkGridCornersAndCenters()
    {
        var chunks = new List<LkMcnkData>(256);
        for (int i = 0; i < 256; i++)
        {
            var heights = new float[145];
            heights[0] = 10f + i;      // top-left
            heights[9] = 20f + i;      // center
            heights[8] = 15f + i;      // top-right
            heights[136] = 25f + i;    // bottom-left
            heights[144] = 30f + i;    // bottom-right

            chunks.Add(new LkMcnkData
            {
                IndexX = i % 16,
                IndexY = i / 16,
                PosZ = 100f,
                Heights = heights
            });
        }

        var adt = new LkAdtData
        {
            TileX = 30,
            TileY = 40,
            Chunks = chunks
        };

        var tile = WdlWriter.ExtractTileHeightsFromLk(adt);

        Assert.Equal(30, tile.TileX);
        Assert.Equal(40, tile.TileY);
        Assert.Equal(17 * 17, tile.OuterHeights.Length);
        Assert.Equal(16 * 16, tile.InnerHeights.Length);
        // Chunk (0, 0): base 100 + h0 (10) = 110
        Assert.Equal(110, tile.OuterHeights[0]);
        // Chunk (0, 0) center: base 100 + hCenter (20) = 120
        Assert.Equal(120, tile.InnerHeights[0]);
    }

    private static string ReadChunkId(byte[] data, int offset)
    {
        return FourCC.FromFileBytes(data.AsSpan(offset, 4)).ToString();
    }
}
