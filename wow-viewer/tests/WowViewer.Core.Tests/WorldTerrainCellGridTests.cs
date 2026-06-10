using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Tests;

public sealed class WorldTerrainCellGridTests
{
    [Fact]
    public void CreateDefault_BuildsAll64CellsWithExpectedVertexIndices()
    {
        WorldTerrainCellGrid grid = WorldTerrainCellGrid.CreateDefault(0);

        Assert.Equal(64, grid.Cells.Count);

        ref readonly WorldTerrainCell first = ref grid.GetCell(0, 0);
        Assert.Equal(0, first.CellX);
        Assert.Equal(0, first.CellY);
        Assert.False(first.IsHoled);
        Assert.Equal(0, first.TopLeftVertexIndex);
        Assert.Equal(1, first.TopRightVertexIndex);
        Assert.Equal(17, first.BottomLeftVertexIndex);
        Assert.Equal(18, first.BottomRightVertexIndex);
        Assert.Equal(9, first.CenterVertexIndex);

        ref readonly WorldTerrainCell last = ref grid.GetCell(7, 7);
        Assert.Equal(7, last.CellX);
        Assert.Equal(7, last.CellY);
        Assert.Equal(126, last.TopLeftVertexIndex);
        Assert.Equal(127, last.TopRightVertexIndex);
        Assert.Equal(143, last.BottomLeftVertexIndex);
        Assert.Equal(144, last.BottomRightVertexIndex);
        Assert.Equal(135, last.CenterVertexIndex);
    }

    [Fact]
    public void HoleMaskState_UsesNative2x2GroupingAcross8x8Cells()
    {
        WorldTerrainHoleMask holeMask = new(0x000F);

        Assert.True(holeMask.IsCellHoled(0, 0));
        Assert.True(holeMask.IsCellHoled(7, 1));
        Assert.False(holeMask.IsCellHoled(0, 2));
        Assert.True(holeMask.IsHoleGroupSet(0, 0));
        Assert.True(holeMask.IsHoleGroupSet(3, 0));
        Assert.False(holeMask.IsHoleGroupSet(0, 1));
    }

    [Fact]
    public void Read_SyntheticRootAdt_ExposesRuntimeCellKnowledgeOnChunkData()
    {
        float[] heights = Enumerable.Range(0, 145).Select(static value => (float)value).ToArray();
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateRootMcnkPayload(flags: 0, indexX: 0, indexY: 0, areaId: 123, holes: 0x000F, layerCount: 1, heights)),
        ];

        using MemoryStream stream = new(bytes, writable: false);
        var fileSummary = MapFileSummaryReader.Read(stream, "synthetic_runtime_cell_0_0.adt");
        stream.Position = 0;

        WorldTerrainTileData terrainTile = WorldTerrainTileBuilder.Read(stream, fileSummary);
        WorldTerrainChunkData chunk = Assert.Single(terrainTile.Chunks);

        Assert.Equal((ushort)0x000F, chunk.HoleMaskState.RawValue);
        Assert.Equal(64, chunk.CellGrid.Cells.Count);
        Assert.True(chunk.HoleMaskState.IsCellHoled(0, 0));
        Assert.True(chunk.HoleMaskState.IsCellHoled(6, 1));
        Assert.False(chunk.HoleMaskState.IsCellHoled(0, 2));

        ref readonly WorldTerrainCell topLeft = ref chunk.CellGrid.GetCell(0, 0);
        Assert.True(topLeft.IsHoled);
        Assert.Equal(9, topLeft.CenterVertexIndex);
    }

    private static byte[] CreateRootMcnkPayload(uint flags, uint indexX, uint indexY, uint areaId, ushort holes, uint layerCount, float[] heights, float baseHeight = 0f)
    {
        byte[] header = new byte[128];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x0C, 4), layerCount);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), areaId);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x3C, 2), holes);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(0x70, 4), baseHeight);

        byte[] mcvtPayload = new byte[heights.Length * sizeof(float)];
        for (int index = 0; index < heights.Length; index++)
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(mcvtPayload.AsSpan(index * sizeof(float), sizeof(float)), heights[index]);

        return
        [
            .. header,
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCVT", mcvtPayload),
        ];
    }
}
