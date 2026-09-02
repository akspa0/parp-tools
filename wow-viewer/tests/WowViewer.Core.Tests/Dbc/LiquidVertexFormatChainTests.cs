using System.Buffers.Binary;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Liquids;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Dbc;

/// <summary>
/// Spec 205. The values used here are the ones measured on MoP Beta 5.0.1.15464 /
/// <c>HawaiiMainLand</c>: LiquidObject 2325/2333/2372 -&gt; LiquidType 5 -&gt; LiquidMaterial 1 -&gt;
/// LVF 0, and the ocean value 42, which is <b>not</b> a LiquidObject id in that client
/// (its LiquidObject ids run 57..2390).
/// </summary>
public sealed class LiquidVertexFormatChainTests
{
    private static LiquidVertexFormatChain CreateMeasuredChain() =>
        LiquidVertexFormatChain.FromTables(
            DbcLiquidObjectTable.FromRows([(2325, 5), (2333, 5), (2372, 5), (57, 2)]),
            DbcLiquidMaterialTable.FromRows([(1, 0), (2, 1), (3, 0), (4, 1), (5, 0), (8, 0), (10, 0)]),
            new Dictionary<int, int> { [5] = 1, [2] = 2 });

    [Theory]
    [InlineData((ushort)0, AdtLiquidVertexFormat.HeightDepth)]
    [InlineData((ushort)1, AdtLiquidVertexFormat.HeightUv)]
    [InlineData((ushort)2, AdtLiquidVertexFormat.DepthOnly)]
    [InlineData((ushort)3, AdtLiquidVertexFormat.HeightUvDepth)]
    public void Resolve_DirectVertexFormat_IsUsedAsIs(ushort raw, AdtLiquidVertexFormat expected)
    {
        LiquidVertexFormatResolution resolution = CreateMeasuredChain().Resolve(raw);

        Assert.True(resolution.Resolved);
        Assert.Equal(LiquidVertexFormatSource.VertexFormatField, resolution.Source);
        Assert.Equal(expected, resolution.Format);
    }

    [Theory]
    [InlineData((ushort)2325)]
    [InlineData((ushort)2333)]
    [InlineData((ushort)2372)]
    public void Resolve_MeasuredRiverObjectIds_ResolveThroughTheChainToHeightDepth(ushort raw)
    {
        LiquidVertexFormatResolution resolution = CreateMeasuredChain().Resolve(raw);

        Assert.True(resolution.Resolved);
        Assert.Equal(LiquidVertexFormatSource.LiquidObjectChain, resolution.Source);
        Assert.Equal(AdtLiquidVertexFormat.HeightDepth, resolution.Format);
        Assert.Equal(5, resolution.LiquidTypeId);
        Assert.Equal(1, resolution.MaterialId);
    }

    [Fact]
    public void Resolve_OceanValue42_IsUnresolved_BecauseItIsNotALiquidObjectIdInThisClient()
    {
        // The wiki rule "values >= 42 are LiquidObject ids" does not hold for 5.0.1.15464, whose
        // LiquidObject ids start at 57. Treating 42 as an id would route 17,317 ocean layers into
        // the chain and decode their depth bytes as floats.
        LiquidVertexFormatChain chain = CreateMeasuredChain();

        LiquidVertexFormatResolution resolution = chain.Resolve(42);

        Assert.False(resolution.Resolved);
        Assert.Equal(LiquidVertexFormatSource.Unresolved, resolution.Source);
        Assert.NotNull(resolution.FailureReason);
        Assert.Equal(1, chain.UnresolvedCounts[42]);
    }

    [Fact]
    public void Resolve_UnresolvedValues_AreCountedPerDistinctValue()
    {
        LiquidVertexFormatChain chain = CreateMeasuredChain();

        chain.Resolve(42);
        chain.Resolve(42);
        chain.Resolve(9999);

        Assert.Equal(2, chain.UnresolvedCounts[42]);
        Assert.Equal(1, chain.UnresolvedCounts[9999]);
    }

    [Fact]
    public void Resolve_ValueBetweenVertexFormatsAndObjectIds_IsUnresolvedRatherThanGuessed()
    {
        LiquidVertexFormatResolution resolution = CreateMeasuredChain().Resolve(20);

        Assert.False(resolution.Resolved);
        Assert.Contains("neither a vertex format", resolution.FailureReason);
    }

    [Fact]
    public void Resolve_MaterialWithOutOfRangeLvf_IsUnresolvedRatherThanClamped()
    {
        // Clamping an out-of-range LVF would put a heightmap read on a depth-only block.
        LiquidVertexFormatChain chain = LiquidVertexFormatChain.FromTables(
            DbcLiquidObjectTable.FromRows([(100, 7)]),
            DbcLiquidMaterialTable.FromRows([(9, 42)]),
            new Dictionary<int, int> { [7] = 9 });

        LiquidVertexFormatResolution resolution = chain.Resolve(100);

        Assert.False(resolution.Resolved);
        Assert.Contains("outside the defined range", resolution.FailureReason);
    }

    [Fact]
    public void EmptyChain_ResolvesOnlyDirectFormats_ReproducingPreFixBehaviour()
    {
        Assert.True(LiquidVertexFormatChain.Empty.Resolve(0).Resolved);
        Assert.False(LiquidVertexFormatChain.Empty.Resolve(2325).Resolved);
    }

    /// <summary>
    /// Spec 205 FR-007: the repository has two independent MH2O decoders and only one is in the
    /// render path. They must not disagree about the same bytes, or a fix lands in the decoder that
    /// is easiest to find while the other keeps producing wrong data.
    /// </summary>
    [Theory]
    [InlineData((ushort)2325, true)]
    [InlineData((ushort)42, false)]
    public void BothDecoders_AgreeOnTheSamePayload(ushort liquidObjectOrLvf, bool expectHeights)
    {
        LiquidVertexFormatChain chain = CreateMeasuredChain();
        byte[] mh2oPayload = CreateMh2oPayload(liquidObjectOrLvf);

        // Decoder 1: the render path.
        Mh2oChunk chunk = Mh2oChunk.Parse(mh2oPayload, chain);
        Mh2oInstance instance = Assert.Single(chunk.GetInstancesForChunk(5));

        // Decoder 2: the harvest/converter path.
        byte[] adt =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(18)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MHDR", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MH2O", mh2oPayload),
        ];

        using MemoryStream stream = new(adt);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, summary, profile: null, dbcTable: null, chain);
        AdtLiquidLayer layer = Assert.Single(liquidFile.Chunks[5].Layers);

        Assert.Equal(expectHeights, instance.VertexFormatResolved);
        Assert.Equal(expectHeights, layer.Heights is not null);
        Assert.Equal(instance.HeightMap is not null, layer.Heights is not null);

        if (!expectHeights)
            return;

        Assert.Equal((int)instance.VertexFormat, (int)layer.VertexFormat);
        Assert.Equal(instance.HeightMap!.Length, layer.Heights!.Length);
        for (int index = 0; index < layer.Heights.Length; index++)
            Assert.Equal(instance.HeightMap[index], layer.Heights[index]);
    }

    [Fact]
    public void RenderPath_WithChain_RecoversTheHeightmapThatUsedToBeDiscarded()
    {
        byte[] payload = CreateMh2oPayload(2325);

        Mh2oInstance before = Assert.Single(Mh2oChunk.Parse(payload).GetInstancesForChunk(5));
        Mh2oInstance after = Assert.Single(Mh2oChunk.Parse(payload, CreateMeasuredChain()).GetInstancesForChunk(5));

        Assert.Null(before.HeightMap);
        Assert.NotNull(after.HeightMap);
        Assert.Equal(9, after.HeightMap!.Length);

        // A varying surface: this is the slope that the flat fallback was throwing away.
        Assert.Equal(42f, after.HeightMap[0]);
        Assert.Equal(50f, after.HeightMap[^1]);
        Assert.Equal((ushort)2325, after.LiquidObjectOrLvf);
    }

    /// <summary>Mirrors the payload shape used by <c>AdtLiquidReaderTests</c>.</summary>
    private static byte[] CreateMh2oPayload(ushort liquidObjectOrLvf)
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

        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(layerOffset, 2), 5);
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(layerOffset + 2, 2), liquidObjectOrLvf);
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
}
