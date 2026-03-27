using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMcnkSummaryReaderTests
{
    [Fact]
    public void Read_RootAdtBuffer_ProducesMcnkSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MHDR", new byte[64]),
            .. CreateChunk("MCNK", CreateRootMcnkPayload(indexX: 0, indexY: 0, layers: 3, areaId: 42, holes: 1, flags: 0x44, includeMcvt: true, includeMcnr: true, includeMclyEntries: 3, includeMcal: true, includeMccv: true, includeMclq: true)),
            .. CreateChunk("MCNK", CreateRootMcnkPayload(indexX: 1, indexY: 0, layers: 1, areaId: 7, holes: 0, flags: 0x00, includeMcvt: true, includeMcnr: true, includeMcsh: true)),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        AdtMcnkSummary summary = AdtMcnkSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.Equal(2, summary.McnkCount);
        Assert.Equal(0, summary.ZeroLengthMcnkCount);
        Assert.Equal(2, summary.HeaderLikeMcnkCount);
        Assert.Equal(2, summary.DistinctIndexCount);
        Assert.Equal(0, summary.DuplicateIndexCount);
        Assert.Equal(2, summary.DistinctAreaIdCount);
        Assert.Equal(1, summary.ChunksWithHoles);
        Assert.Equal(1, summary.ChunksWithLiquidFlags);
        Assert.Equal(1, summary.ChunksWithMccvFlag);
        Assert.Equal(2, summary.ChunksWithMcvt);
        Assert.Equal(2, summary.ChunksWithMcnr);
        Assert.Equal(1, summary.ChunksWithMcly);
        Assert.Equal(1, summary.ChunksWithMcal);
        Assert.Equal(1, summary.ChunksWithMcsh);
        Assert.Equal(1, summary.ChunksWithMccv);
        Assert.Equal(1, summary.ChunksWithMclq);
        Assert.Equal(0, summary.ChunksWithMcrd);
        Assert.Equal(0, summary.ChunksWithMcrw);
        Assert.Equal(4, summary.TotalLayerCount);
        Assert.Equal(3, summary.MaxLayerCount);
        Assert.Equal(1, summary.ChunksWithMultipleLayers);
        Assert.Equal(0, summary.MccvFlagWithoutPayloadCount);
        Assert.Equal(0, summary.LiquidFlagWithoutPayloadCount);
    }

    [Fact]
    public void Read_TexAdtBuffer_ProducesMcnkSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MAMP", CreateUInt32Payload(1)),
            .. CreateChunk("MTEX", CreateStringBlock("a.blp")),
            .. CreateChunk("MCNK", CreateSplitMcnkPayload(("MCLY", new byte[16]), ("MCAL", new byte[32]))),
            .. CreateChunk("MCNK", CreateSplitMcnkPayload(("MCLY", new byte[32]), ("MCSH", new byte[64]))),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0_tex0.adt");
        AdtMcnkSummary summary = AdtMcnkSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.AdtTex, summary.Kind);
        Assert.Equal(2, summary.McnkCount);
        Assert.Equal(0, summary.ZeroLengthMcnkCount);
        Assert.Equal(0, summary.HeaderLikeMcnkCount);
        Assert.Equal(0, summary.DistinctIndexCount);
        Assert.Equal(0, summary.DistinctAreaIdCount);
        Assert.Equal(0, summary.ChunksWithMcvt);
        Assert.Equal(0, summary.ChunksWithMcnr);
        Assert.Equal(2, summary.ChunksWithMcly);
        Assert.Equal(1, summary.ChunksWithMcal);
        Assert.Equal(1, summary.ChunksWithMcsh);
        Assert.Equal(3, summary.TotalLayerCount);
        Assert.Equal(2, summary.MaxLayerCount);
        Assert.Equal(1, summary.ChunksWithMultipleLayers);
    }

    [Fact]
    public void Read_ObjAdtBuffer_ProducesMcnkSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MMDX", CreateStringBlock("a.mdx")),
            .. CreateChunk("MCNK", Array.Empty<byte>()),
            .. CreateChunk("MCNK", CreateSplitMcnkPayload(("MCRD", new byte[8]))),
            .. CreateChunk("MCNK", CreateSplitMcnkPayload(("MCRW", new byte[12]))),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0_obj0.adt");
        AdtMcnkSummary summary = AdtMcnkSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.AdtObj, summary.Kind);
        Assert.Equal(3, summary.McnkCount);
        Assert.Equal(1, summary.ZeroLengthMcnkCount);
        Assert.Equal(0, summary.HeaderLikeMcnkCount);
        Assert.Equal(1, summary.ChunksWithMcrd);
        Assert.Equal(1, summary.ChunksWithMcrw);
        Assert.Equal(0, summary.TotalLayerCount);
    }

    [Fact]
    public void Read_DevelopmentRootAdt_ProducesExpectedMcnkSummary()
    {
        AdtMcnkSummary summary = AdtMcnkSummaryReader.Read(MapTestPaths.DevelopmentRootAdtPath);

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.Equal(256, summary.McnkCount);
        Assert.Equal(0, summary.ZeroLengthMcnkCount);
        Assert.Equal(256, summary.HeaderLikeMcnkCount);
        Assert.Equal(256, summary.DistinctIndexCount);
        Assert.Equal(0, summary.DuplicateIndexCount);
        Assert.Equal(1, summary.DistinctAreaIdCount);
        Assert.Equal(10, summary.ChunksWithHoles);
        Assert.Equal(0, summary.ChunksWithLiquidFlags);
        Assert.Equal(0, summary.ChunksWithMccvFlag);
        Assert.Equal(256, summary.ChunksWithMcvt);
        Assert.Equal(256, summary.ChunksWithMcnr);
        Assert.Equal(0, summary.ChunksWithMcly);
        Assert.Equal(0, summary.ChunksWithMcal);
        Assert.Equal(0, summary.ChunksWithMcsh);
        Assert.Equal(0, summary.ChunksWithMccv);
        Assert.Equal(0, summary.ChunksWithMclq);
        Assert.Equal(0, summary.ChunksWithMcrd);
        Assert.Equal(0, summary.ChunksWithMcrw);
        Assert.Equal(0, summary.TotalLayerCount);
        Assert.Equal(0, summary.MaxLayerCount);
        Assert.Equal(0, summary.ChunksWithMultipleLayers);
    }

    [Fact]
    public void Read_DevelopmentTexAdt_ProducesExpectedMcnkSummary()
    {
        AdtMcnkSummary summary = AdtMcnkSummaryReader.Read(MapTestPaths.DevelopmentTexAdtPath);

        Assert.Equal(MapFileKind.AdtTex, summary.Kind);
        Assert.Equal(256, summary.McnkCount);
        Assert.Equal(0, summary.ZeroLengthMcnkCount);
        Assert.Equal(0, summary.HeaderLikeMcnkCount);
        Assert.Equal(256, summary.ChunksWithMcly);
        Assert.Equal(203, summary.ChunksWithMcal);
        Assert.Equal(174, summary.ChunksWithMcsh);
        Assert.Equal(775, summary.TotalLayerCount);
        Assert.Equal(4, summary.MaxLayerCount);
        Assert.Equal(203, summary.ChunksWithMultipleLayers);
    }

    [Fact]
    public void Read_DevelopmentObjAdt_ProducesExpectedMcnkSummary()
    {
        AdtMcnkSummary summary = AdtMcnkSummaryReader.Read(MapTestPaths.DevelopmentObjAdtPath);

        Assert.Equal(MapFileKind.AdtObj, summary.Kind);
        Assert.Equal(256, summary.McnkCount);
        Assert.Equal(179, summary.ZeroLengthMcnkCount);
        Assert.Equal(0, summary.HeaderLikeMcnkCount);
        Assert.Equal(9, summary.ChunksWithMcrd);
        Assert.Equal(70, summary.ChunksWithMcrw);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }

    private static byte[] CreateSplitMcnkPayload(params (string Id, byte[] Payload)[] subchunks)
    {
        using MemoryStream stream = new();
        foreach ((string id, byte[] payload) in subchunks)
        {
            stream.Write(CreateChunk(id, payload));
        }

        return stream.ToArray();
    }

    private static byte[] CreateRootMcnkPayload(
        uint indexX,
        uint indexY,
        uint layers,
        uint areaId,
        ushort holes,
        uint flags,
        bool includeMcvt = false,
        bool includeMcnr = false,
        int includeMclyEntries = 0,
        bool includeMcal = false,
        bool includeMcsh = false,
        bool includeMccv = false,
        bool includeMclq = false)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x0C, 4), layers);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), areaId);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x3C, 2), holes);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        if (includeMcvt)
            stream.Write(CreateChunk("MCVT", new byte[16]));

        if (includeMcnr)
            stream.Write(CreateChunk("MCNR", new byte[0x1C0]));

        if (includeMclyEntries > 0)
            stream.Write(CreateChunk("MCLY", new byte[includeMclyEntries * 16]));

        if (includeMcal)
            stream.Write(CreateChunk("MCAL", new byte[64]));

        if (includeMcsh)
            stream.Write(CreateChunk("MCSH", new byte[128]));

        if (includeMccv)
            stream.Write(CreateChunk("MCCV", new byte[64]));

        if (includeMclq)
            stream.Write(CreateChunk("MCLQ", new byte[32]));

        return stream.ToArray();
    }
}
