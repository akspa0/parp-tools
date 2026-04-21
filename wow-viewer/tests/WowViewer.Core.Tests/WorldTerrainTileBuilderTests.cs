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

    [Fact]
    public void Read_SyntheticRootAdt_AppliesBaseHeightFromVerifiedHeaderOffset()
    {
        float[] heights = Enumerable.Range(0, 145).Select(static value => (float)value).ToArray();
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateRootMcnkPayload(flags: 0, indexX: 0, indexY: 0, areaId: 1, holes: 0, layerCount: 1, heights, baseHeight: 100f)),
        ];

        using MemoryStream stream = new(bytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_baseheight_0_0.adt");
        stream.Position = 0;

        WorldTerrainTileData terrainTile = WorldTerrainTileBuilder.Read(stream, fileSummary);

        WorldTerrainChunkData chunk = Assert.Single(terrainTile.Chunks);
        Assert.NotNull(chunk.Heights);
        Assert.Equal(100f, chunk.Heights![0]);
        Assert.Equal(244f, chunk.Heights[144]);

        WorldTerrainHeightmapData heightmap = Assert.IsType<WorldTerrainHeightmapData>(terrainTile.Heightmap);
        Assert.Equal(100f, heightmap.MinHeight);
        Assert.Equal(244f, heightmap.MaxHeight);
    }

    [Fact]
    public void Read_SyntheticRootAdt_FallsBackToMcinWhenTopLevelSummaryIsIncomplete()
    {
        float[] heights0 = Enumerable.Range(0, 145).Select(static value => (float)value).ToArray();
        float[] heights1 = Enumerable.Range(0, 145).Select(static value => 1000f + value).ToArray();

        byte[] firstMcnk = CreateChunk("MCNK", CreateRootMcnkPayload(flags: 0, indexX: 0, indexY: 0, areaId: 1, holes: 0, layerCount: 1, heights0));
        byte[] secondMcnk = CreateChunk("MCNK", CreateRootMcnkPayload(flags: 0, indexX: 1, indexY: 0, areaId: 2, holes: 0, layerCount: 1, heights1));

        byte[] mver = CreateChunk("MVER", CreateUInt32Payload(18));
        byte[] mhdr = CreateChunk("MHDR", new byte[64]);
        int firstMcnkHeaderOffset = mver.Length + mhdr.Length + (8 + 4096);
        int secondMcnkHeaderOffset = firstMcnkHeaderOffset + firstMcnk.Length + (8 + 4);
        byte[] mcin = CreateChunk("MCIN", CreateMcinPayload(
            firstMcnkHeaderOffset,
            firstMcnk.Length,
            secondMcnkHeaderOffset,
            secondMcnk.Length));
        byte[] invalid = CreateInvalidEmbeddedChunk("MMDX", 4096, trailingBytes: 4);

        byte[] bytes =
        [
            .. mver,
            .. mhdr,
            .. mcin,
            .. firstMcnk,
            .. invalid,
            .. secondMcnk,
        ];

        using MemoryStream stream = new(bytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_mcin_fallback_0_0.adt");
        Assert.Equal(1, fileSummary.CountChunks(MapChunkIds.Mcnk));
        stream.Position = 0;

        WorldTerrainTileData terrainTile = WorldTerrainTileBuilder.Read(stream, fileSummary);

        Assert.Equal(2, terrainTile.ChunkCount);
        Assert.Equal(2, terrainTile.ChunksWithHeights);
        Assert.Equal(0f, terrainTile.Chunks[0].Heights![0]);
        Assert.Equal(1000f, terrainTile.Chunks[1].Heights![0]);
        Assert.NotNull(terrainTile.Heightmap);
        Assert.Equal(0f, terrainTile.Heightmap!.GetHeight(0, 0));
    }

    private static byte[] CreateRootMcnkPayload(uint flags, uint indexX, uint indexY, uint areaId, ushort holes, uint layerCount, float[] heights, float baseHeight = 0f)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x0C, 4), layerCount);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), areaId);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x3C, 2), holes);
        BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(0x70, 4), baseHeight);

        byte[] mcvtPayload = new byte[heights.Length * sizeof(float)];
        for (int index = 0; index < heights.Length; index++)
            BinaryPrimitives.WriteSingleLittleEndian(mcvtPayload.AsSpan(index * sizeof(float), sizeof(float)), heights[index]);

        return
        [
            .. header,
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCVT", mcvtPayload),
        ];
    }

    private static byte[] CreateMcinPayload(int firstChunkHeaderOffset, int firstChunkSize, int secondChunkHeaderOffset, int secondChunkSize)
    {
        byte[] payload = new byte[4096];
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0, 4), (uint)firstChunkHeaderOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(4, 4), (uint)firstChunkSize);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(16, 4), (uint)secondChunkHeaderOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(20, 4), (uint)secondChunkSize);
        return payload;
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateInvalidEmbeddedChunk(string id, uint declaredSize, int trailingBytes)
    {
        byte[] bytes = new byte[8 + trailingBytes];
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), declaredSize);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }
}