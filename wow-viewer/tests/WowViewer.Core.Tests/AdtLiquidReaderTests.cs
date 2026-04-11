using System.Buffers.Binary;
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
    public void Read_DevelopmentRootAdt_ProducesStableMh2oSignals()
    {
        AdtLiquidFile liquidFile = AdtLiquidReader.Read(MapTestPaths.DevelopmentRootAdtPath);

        Assert.Equal(MapFileKind.Adt, liquidFile.Kind);
        Assert.Equal(256, liquidFile.Chunks.Count);
        Assert.True(liquidFile.Chunks.All(chunk => chunk.Layers.Count == 0));
    }

    private static byte[] CreateMh2oPayload()
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

        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(layerOffset, 2), 17);
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