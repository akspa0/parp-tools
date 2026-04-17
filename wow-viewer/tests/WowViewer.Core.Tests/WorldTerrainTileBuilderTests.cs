using System.Buffers.Binary;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Tests;

public sealed class WorldTerrainTileBuilderTests
{
    [Fact]
    public void Read_DevelopmentRootAdt_ProducesExpectedChunkInventory()
    {
        WorldTerrainTileData terrainTile = WorldTerrainTileBuilder.Read(MapTestPaths.DevelopmentRootAdtPath);

        Assert.Equal(MapFileKind.Adt, terrainTile.Kind);
        Assert.Equal(256, terrainTile.ChunkCount);
        Assert.Equal(256, terrainTile.ChunksWithHeights);
        Assert.Equal(10, terrainTile.HoleChunkCount);
        Assert.Equal(0, terrainTile.LiquidFlagChunkCount);
        Assert.Equal(0, terrainTile.VertexColorChunkCount);
        Assert.Equal(1, terrainTile.DistinctAreaIdCount);
        Assert.NotNull(terrainTile.Heightmap);
        Assert.Equal(257, terrainTile.Heightmap!.Width);
        Assert.Equal(257, terrainTile.Heightmap.Height);
        Assert.True(terrainTile.Heightmap.AuthoritativeSampleCount > 0);
        Assert.True(terrainTile.Heightmap.MaxHeight > terrainTile.Heightmap.MinHeight);
    }

    [Fact]
    public void Read_SyntheticRootAdt_ProducesExpectedChunkSignals()
    {
        float[] heights = Enumerable.Range(0, 145).Select(static value => (float)value).ToArray();
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateRootMcnkPayload(flags: 0x44, indexX: 0, indexY: 0, areaId: 123, holes: 0x000F, layerCount: 3, heights)),
        ];

        using MemoryStream stream = new(bytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        stream.Position = 0;

        WorldTerrainTileData terrainTile = WorldTerrainTileBuilder.Read(stream, fileSummary);

        WorldTerrainChunkData chunk = Assert.Single(terrainTile.Chunks);
        Assert.Equal(0, chunk.ChunkIndex);
        Assert.Equal(0, chunk.IndexX);
        Assert.Equal(0, chunk.IndexY);
        Assert.Equal((uint)123, chunk.AreaId);
        Assert.Equal((uint)0x44, chunk.Flags);
        Assert.Equal(3, chunk.LayerCount);
        Assert.True(chunk.HasHoles);
        Assert.Equal((ushort)0x000F, chunk.HoleMask);
        Assert.True(chunk.HasLiquidFlags);
        Assert.True(chunk.HasVertexColors);
        Assert.True(chunk.HasHeights);
        Assert.NotNull(chunk.Heights);
        Assert.Equal(0f, chunk.Heights![0]);
        Assert.Equal(144f, chunk.Heights[144]);

        WorldTerrainHeightmapData heightmap = Assert.IsType<WorldTerrainHeightmapData>(terrainTile.Heightmap);
        Assert.Equal(145, heightmap.AuthoritativeSampleCount);
        Assert.Equal(0f, heightmap.GetHeight(0, 0));
        Assert.Equal(1f, heightmap.GetHeight(2, 0));
        Assert.Equal(9f, heightmap.GetHeight(1, 1));
        Assert.Equal(17f, heightmap.GetHeight(0, 2));
        Assert.Equal(0f, heightmap.MinHeight);
        Assert.Equal(144f, heightmap.MaxHeight);
    }

    private static byte[] CreateRootMcnkPayload(uint flags, uint indexX, uint indexY, uint areaId, ushort holes, uint layerCount, float[] heights)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x0C, 4), layerCount);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), areaId);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x3C, 2), holes);

        byte[] mcvtPayload = new byte[heights.Length * sizeof(float)];
        for (int index = 0; index < heights.Length; index++)
            BinaryPrimitives.WriteSingleLittleEndian(mcvtPayload.AsSpan(index * sizeof(float), sizeof(float)), heights[index]);

        return
        [
            .. header,
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCVT", mcvtPayload),
        ];
    }
}