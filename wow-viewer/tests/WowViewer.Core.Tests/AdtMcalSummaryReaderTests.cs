using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMcalSummaryReaderTests
{
    [Fact]
    public void Read_RootAdtBuffer_ClassifiesPackedAndCompressedOverlayLayers()
    {
        byte[] packedAlpha = Enumerable.Repeat((byte)0x10, 2048).ToArray();
        byte[] compressedAlpha =
        [
            0x81, 0x44,
            0x01, 0x55,
        ];

        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MHDR", new byte[64]),
            .. CreateChunk("MCNK", CreateRootMcnkPayload(
                flags: 0,
                layerFlags: [0u, 0x100u, 0x300u],
                layerOffsets: [0u, 0u, (uint)packedAlpha.Length],
                mcalPayload: [.. packedAlpha, .. compressedAlpha])),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        AdtMcalSummary summary = AdtMcalSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.Equal(AdtMcalDecodeProfile.LichKingStrict, summary.DecodeProfile);
        Assert.Equal(1, summary.McnkWithLayerTableCount);
        Assert.Equal(2, summary.OverlayLayerCount);
        Assert.Equal(2, summary.DecodedLayerCount);
        Assert.Equal(0, summary.MissingPayloadLayerCount);
        Assert.Equal(0, summary.DecodeFailureCount);
        Assert.Equal(1, summary.PackedLayerCount);
        Assert.Equal(1, summary.CompressedLayerCount);
        Assert.Equal(0, summary.BigAlphaLayerCount);
        Assert.Equal(0, summary.BigAlphaFixedLayerCount);
    }

    [Fact]
    public void Read_TexAdtBuffer_ClassifiesFixedBigAlphaOverlayLayers()
    {
        byte[] fixedBigAlpha = Enumerable.Repeat((byte)0x6A, 63 * 63).ToArray();

        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MTEX", CreateStringBlock("foo.blp", "bar.blp")),
            .. CreateChunk("MCNK", CreateSplitMcnkPayload(
				CreateChunk("MCLY", CreateMclyPayload([0u, 0x100u], [0u, 0u])),
                CreateChunk("MCAL", fixedBigAlpha))),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0_tex0.adt");
        AdtMcalSummary summary = AdtMcalSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.AdtTex, summary.Kind);
        Assert.Equal(AdtMcalDecodeProfile.Cataclysm400, summary.DecodeProfile);
        Assert.Equal(1, summary.McnkWithLayerTableCount);
        Assert.Equal(1, summary.OverlayLayerCount);
        Assert.Equal(1, summary.DecodedLayerCount);
        Assert.Equal(0, summary.MissingPayloadLayerCount);
        Assert.Equal(0, summary.DecodeFailureCount);
        Assert.Equal(0, summary.CompressedLayerCount);
        Assert.Equal(0, summary.BigAlphaLayerCount);
        Assert.Equal(1, summary.BigAlphaFixedLayerCount);
        Assert.Equal(0, summary.PackedLayerCount);
    }

    [Fact]
    public void Read_DevelopmentTexAdt_ProducesStableRealDataSummary()
    {
        AdtMcalSummary summary = AdtMcalSummaryReader.Read(MapTestPaths.DevelopmentTexAdtPath);

        Assert.Equal(MapFileKind.AdtTex, summary.Kind);
        Assert.Equal(AdtMcalDecodeProfile.Cataclysm400, summary.DecodeProfile);
        Assert.Equal(256, summary.McnkWithLayerTableCount);
        Assert.Equal(519, summary.OverlayLayerCount);
        Assert.True(summary.DecodedLayerCount > 0);
        Assert.True(summary.BigAlphaLayerCount + summary.BigAlphaFixedLayerCount > 0);
        Assert.Equal(summary.OverlayLayerCount, summary.DecodedLayerCount + summary.MissingPayloadLayerCount + summary.DecodeFailureCount);
        Assert.Equal(summary.DecodedLayerCount, summary.CompressedLayerCount + summary.BigAlphaLayerCount + summary.BigAlphaFixedLayerCount + summary.PackedLayerCount);
    }

    private static byte[] CreateRootMcnkPayload(uint flags, uint[] layerFlags, uint[] layerOffsets, byte[] mcalPayload)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x28, 4), (uint)(mcalPayload.Length + 8));

        using MemoryStream stream = new();
        stream.Write(header);
        stream.Write(CreateChunk("MCLY", CreateMclyPayload(layerFlags, layerOffsets)));
        stream.Write(CreateChunk("MCAL", mcalPayload));
        return stream.ToArray();
    }

    private static byte[] CreateSplitMcnkPayload(params byte[][] subchunks)
    {
        using MemoryStream stream = new();
        foreach (byte[] subchunk in subchunks)
            stream.Write(subchunk);

        return stream.ToArray();
    }

    private static byte[] CreateMclyPayload(uint[] layerFlags, uint[] layerOffsets)
    {
        if (layerFlags.Length != layerOffsets.Length)
            throw new ArgumentException("Layer flag and offset arrays must have the same length.");

        byte[] payload = new byte[layerFlags.Length * 16];
        for (int index = 0; index < layerFlags.Length; index++)
        {
            int offset = index * 16;
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset, 4), (uint)index);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 4, 4), layerFlags[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 8, 4), layerOffsets[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 12, 4), 0u);
        }

        return payload;
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }
}