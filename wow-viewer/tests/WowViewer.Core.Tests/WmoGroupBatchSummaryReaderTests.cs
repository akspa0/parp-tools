using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupBatchSummaryReaderTests
{
    [Fact]
    public void Read_WmoGroupBatchBuffer_ProducesBatchSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(17)),
            .. CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x80,
                flags: 0,
                subchunks:
                [
                    ("MOBA", CreateMobaPayload(
                        CreateBatch(materialId: 3, firstIndex: 10, indexCount: 6, flags: 1),
                        CreateBatch(materialId: 7, firstIndex: 20, indexCount: 9, flags: 0))),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupBatchSummary summary = WmoGroupBatchSummaryReader.Read(stream, "synthetic_batches_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(48, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.True(summary.HasMaterialIds);
        Assert.Equal(2, summary.DistinctMaterialIdCount);
        Assert.Equal(7, summary.HighestMaterialId);
        Assert.Equal(15, summary.TotalIndexCount);
        Assert.Equal(10, summary.MinFirstIndex);
        Assert.Equal(20, summary.MaxFirstIndex);
        Assert.Equal(29, summary.MaxIndexEnd);
        Assert.Equal(1, summary.FlaggedBatchCount);
    }

    [Fact]
    public void Read_V16StyleBatchBuffer_ProducesMateriallessSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(16)),
            .. CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x44,
                flags: 0,
                subchunks:
                [
                    ("MOBA", CreateMobaPayload(CreateBatch(materialId: 255, firstIndex: 4, indexCount: 12, flags: 0))),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupBatchSummary summary = WmoGroupBatchSummaryReader.Read(stream, "synthetic_v16_batches_000.wmo");

        Assert.Equal((uint)16, summary.Version);
        Assert.False(summary.HasMaterialIds);
        Assert.Equal(0, summary.DistinctMaterialIdCount);
        Assert.Equal(12, summary.TotalIndexCount);
        Assert.Equal(4, summary.MinFirstIndex);
        Assert.Equal(16, summary.MaxIndexEnd);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }

    private static byte[] CreateMogpPayload(int headerSize, uint flags, params (string Id, byte[] Payload)[] subchunks)
    {
        byte[] header = new byte[headerSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), flags);
        WriteSingle(header, 0x18, 1f);
        WriteSingle(header, 0x1C, 1f);
        WriteSingle(header, 0x20, 1f);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(CreateChunk(id, payload));

        return stream.ToArray();
    }

    private static byte[] CreateMobaPayload(params byte[][] batches)
    {
        using MemoryStream stream = new();
        foreach (byte[] batch in batches)
            stream.Write(batch, 0, batch.Length);

        return stream.ToArray();
    }

    private static byte[] CreateBatch(byte materialId, ushort firstIndex, ushort indexCount, byte flags)
    {
        byte[] bytes = new byte[24];
        bytes[1] = materialId;
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(14, 2), firstIndex);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(16, 2), indexCount);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(18, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(20, 2), 0);
        bytes[22] = flags;
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
