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
        Assert.Equal(10, terrainTile.HoleChunkCount);
        Assert.Equal(0, terrainTile.LiquidFlagChunkCount);
        Assert.Equal(0, terrainTile.VertexColorChunkCount);
        Assert.Equal(1, terrainTile.DistinctAreaIdCount);
    }

    [Fact]
    public void Read_SyntheticRootAdt_ProducesExpectedChunkSignals()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateRootMcnkPayload(flags: 0x44, indexX: 7, indexY: 9, areaId: 123, holes: 0x000F, layerCount: 3)),
        ];

        using MemoryStream stream = new(bytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        stream.Position = 0;

        WorldTerrainTileData terrainTile = WorldTerrainTileBuilder.Read(stream, fileSummary);

        WorldTerrainChunkData chunk = Assert.Single(terrainTile.Chunks);
        Assert.Equal(0, chunk.ChunkIndex);
        Assert.Equal(7, chunk.IndexX);
        Assert.Equal(9, chunk.IndexY);
        Assert.Equal((uint)123, chunk.AreaId);
        Assert.Equal((uint)0x44, chunk.Flags);
        Assert.Equal(3, chunk.LayerCount);
        Assert.True(chunk.HasHoles);
        Assert.True(chunk.HasLiquidFlags);
        Assert.True(chunk.HasVertexColors);
    }

    private static byte[] CreateRootMcnkPayload(uint flags, uint indexX, uint indexY, uint areaId, ushort holes, uint layerCount)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x0C, 4), layerCount);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), areaId);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x3C, 2), holes);
        return header;
    }
}