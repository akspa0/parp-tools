using System.Numerics;
using WoWMapConverter.Core.VLM;
using Xunit;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class TerrainTileBakeServiceTests
{
    [Fact]
    public void BuildTileHeightmap257_NonInterleavedChunkData_UsesAlphaMcvtOrdering()
    {
        float[] rawChunkHeights = new float[145];
        for (int i = 0; i < 81; i++)
            rawChunkHeights[i] = i;

        for (int i = 0; i < 64; i++)
            rawChunkHeights[81 + i] = 1000 + i;

        var tileHeightmap = TerrainTileBakeService.BuildTileHeightmap257(
            new Dictionary<int, float[]> { [0] = rawChunkHeights },
            isInterleaved: false);

        Assert.Equal(0f, tileHeightmap.Heights[0]);
        Assert.Equal(1f, tileHeightmap.Heights[2]);
        Assert.Equal(9f, tileHeightmap.Heights[2 * TerrainTileBakeService.TileHeightmapSize]);
        Assert.Equal(1000f, tileHeightmap.Heights[TerrainTileBakeService.TileHeightmapSize + 1]);
        Assert.Equal(1063f, tileHeightmap.Heights[(15 * TerrainTileBakeService.TileHeightmapSize) + 15]);
        Assert.Equal(80f, tileHeightmap.Heights[(16 * TerrainTileBakeService.TileHeightmapSize) + 16]);
    }

    [Fact]
    public void BuildTileNormals257_FlatTile_ProducesUpNormals()
    {
        var heightsByChunk = Enumerable.Range(0, 256)
            .ToDictionary(
                chunkIndex => chunkIndex,
                _ => Enumerable.Repeat(123.45f, 145).ToArray());

        var tileHeightmap = TerrainTileBakeService.BuildTileHeightmap257(heightsByChunk, isInterleaved: true);
        Vector3[] normals = TerrainTileBakeService.BuildTileNormals257(tileHeightmap.Heights);

        Vector3 center = normals[(128 * TerrainTileBakeService.TileHeightmapSize) + 128];
        Assert.InRange(center.X, -0.001f, 0.001f);
        Assert.InRange(center.Y, -0.001f, 0.001f);
        Assert.InRange(center.Z, 0.999f, 1.001f);
    }
}