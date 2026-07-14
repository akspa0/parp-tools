using System.Buffers.Binary;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtLiquidReaderTests
{
    [Fact]
    public void Read_SyntheticRootAdt_ParsesMh2oLayers()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MH2O", CreateMh2oPayload()),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateMinimalRootMcnkPayload(indexX: 0, indexY: 0)),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, summary);

        Assert.Equal(MapFileKind.Adt, liquidFile.Kind);
        Assert.Equal(256, liquidFile.Chunks.Count);

        AdtLiquidChunk chunk = liquidFile.Chunks[5];
        AdtLiquidLayer layer = Assert.Single(chunk.Layers);
        Assert.Equal((ushort)17, layer.LiquidTypeId);
        Assert.Equal(AdtLiquidBasicType.Ocean, layer.BasicType);
        Assert.Equal(AdtLiquidVertexFormat.HeightDepth, layer.VertexFormat);
        Assert.Equal(2, layer.Width);
        Assert.Equal(2, layer.Height);
        Assert.Equal(1, layer.XOffset);
        Assert.Equal(2, layer.YOffset);
        Assert.Equal(4, layer.VisibleTileCount);
        Assert.NotNull(layer.Heights);
        Assert.Equal(9, layer.Heights!.Length);
        Assert.Equal(42f, layer.Heights[0]);
        Assert.Equal(50f, layer.Heights[^1]);
        Assert.NotNull(layer.Depths);
        Assert.Equal(9, layer.Depths!.Length);
        Assert.Equal((byte)8, layer.Depths[0]);
        Assert.Equal((byte)16, layer.Depths[^1]);
        Assert.Equal((ulong)0x0123456789ABCDEF, chunk.FishableMask);
        Assert.Equal((ulong)0x0FEDCBA987654321, chunk.DeepMask);
    }

    [Fact]
    public void Read_MhdrReferencesMh2o_WhenSummaryMissesTopLevelChunk_UsesMhdrFallback()
    {
        byte[] mhdrPayload = new byte[64];
        byte[] mver = MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18));
        byte[] mhdr = MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", mhdrPayload);
        byte[] mh2o = MapFileSummaryReaderTestsAccessor.CreateChunk("MH2O", CreateMh2oPayload());
        byte[] mcnk = MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateMinimalRootMcnkPayload(indexX: 0, indexY: 0));

        int mhdrDataOffset = mver.Length + 8;
        int mh2oHeaderOffset = mver.Length + mhdr.Length;
        BinaryPrimitives.WriteInt32LittleEndian(mhdrPayload.AsSpan(40, 4), mh2oHeaderOffset - mhdrDataOffset);
        mhdr = MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", mhdrPayload);

        byte[] bytes =
        [
            .. mver,
            .. mhdr,
            .. mh2o,
            .. mcnk,
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fullSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        MapFileSummary summaryWithoutMh2o = new(
            fullSummary.SourcePath,
            fullSummary.Kind,
            fullSummary.Version,
            fullSummary.Chunks.Where(static chunk => chunk.Id != MapChunkIds.Mh2o).ToArray());

        AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, summaryWithoutMh2o, AdtFormatProfiles.AdtProfile3018303);

        AdtLiquidChunk chunk = liquidFile.Chunks[5];
        AdtLiquidLayer layer = Assert.Single(chunk.Layers);
        Assert.Equal((ushort)17, layer.LiquidTypeId);
        Assert.NotNull(layer.Heights);
        Assert.Equal(42f, layer.Heights![0]);
        Assert.Equal(50f, layer.Heights[^1]);
    }

    [Theory]
    [InlineData((ushort)17, AdtLiquidBasicType.Ocean)]
    [InlineData((ushort)19, AdtLiquidBasicType.Magma)]
    [InlineData((ushort)20, AdtLiquidBasicType.Slime)]
    [InlineData((ushort)1, AdtLiquidBasicType.Water)]
    [InlineData((ushort)9999, AdtLiquidBasicType.Water)]
    public void Read_WithDbcTable_ResolvesBasicTypeFromDbc(ushort liquidTypeId, AdtLiquidBasicType expected)
    {
        DbcLiquidTypeTable dbcTable = DbcLiquidTypeTable.LoadFromJsonString("""
        {
          "rows": [
            { "id": 1,  "type": 1 },
            { "id": 17, "type": 1 },
            { "id": 19, "type": 2 },
            { "id": 20, "type": 3 }
          ]
        }
        """);

        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MH2O", CreateMh2oPayload(liquidTypeId)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateMinimalRootMcnkPayload(indexX: 0, indexY: 0)),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, summary, profile: null, dbcTable);

        AdtLiquidLayer layer = Assert.Single(liquidFile.Chunks[5].Layers);
        Assert.Equal(liquidTypeId, layer.LiquidTypeId);
        Assert.Equal(expected, layer.BasicType);
    }

    [Fact]
    public void Read_WithDbcTable_RecordsMissingLiquidTypeIds()
    {
        DbcLiquidTypeTable dbcTable = DbcLiquidTypeTable.LoadFromJsonString("""
        { "rows": [ { "id": 1, "type": 1 } ] }
        """);

        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MH2O", CreateMh2oPayload(9999)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateMinimalRootMcnkPayload(indexX: 0, indexY: 0)),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        AdtLiquidReader.Read(stream, summary, profile: null, dbcTable);

        Assert.Single(dbcTable.MissingLiquidTypeIds);
        Assert.Contains(9999, dbcTable.MissingLiquidTypeIds);
    }

    [Fact]
    public void Read_WithoutDbcTable_FallsBackToHardcoded17_19_20Mapping()
    {
        byte[] bytes17 =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MH2O", CreateMh2oPayload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCNK", CreateMinimalRootMcnkPayload(indexX: 0, indexY: 0)),
        ];

        using MemoryStream stream17 = new(bytes17);
        MapFileSummary summary17 = MapFileSummaryReader.Read(stream17, "synthetic_0_0.adt");
        AdtLiquidFile file17 = AdtLiquidReader.Read(stream17, summary17);

        Assert.Equal(AdtLiquidBasicType.Ocean, Assert.Single(file17.Chunks[5].Layers).BasicType);
    }

    private static byte[] CreateMh2oPayload(ushort liquidTypeId = 17)
    {
        const int chunkCount = 256;
        const int headerSize = 12;
        const int attributesSize = 16;
        const int layerSize = 24;
        const int width = 2;
        const int height = 2;
        const int vertexCount = (width + 1) * (height + 1);

        int headersSize = chunkCount * headerSize;
        int attributesOffset = headersSize;
        int layerOffset = attributesOffset + attributesSize;
        int vertexOffset = layerOffset + layerSize;
        int depthOffset = vertexOffset + (vertexCount * sizeof(float));

        byte[] payload = new byte[depthOffset + vertexCount];

        int headerOffset = 5 * headerSize;
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(headerOffset, 4), (uint)layerOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(headerOffset + 4, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(headerOffset + 8, 4), (uint)attributesOffset);

        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(attributesOffset, 8), 0x0123456789ABCDEFUL);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(attributesOffset + 8, 8), 0x0FEDCBA987654321UL);

        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(layerOffset, 2), liquidTypeId);
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(layerOffset + 2, 2), 0);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(layerOffset + 4, 4), 42f);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(layerOffset + 8, 4), 50f);
        payload[layerOffset + 12] = 1;
        payload[layerOffset + 13] = 2;
        payload[layerOffset + 14] = width;
        payload[layerOffset + 15] = height;
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(layerOffset + 16, 4), 0u);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(layerOffset + 20, 4), (uint)vertexOffset);

        for (int index = 0; index < vertexCount; index++)
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(vertexOffset + (index * sizeof(float)), sizeof(float)), 42f + index);

        for (int index = 0; index < vertexCount; index++)
            payload[depthOffset + index] = (byte)(8 + index);

        return payload;
    }

    private static byte[] CreateMinimalRootMcnkPayload(uint indexX, uint indexY)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        return header;
    }
}
