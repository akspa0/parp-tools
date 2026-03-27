using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoDoodadPlacementSummaryReaderTests
{
    [Fact]
    public void Read_ModdBuffer_ProducesPlacementSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODD", CreateModdPayload(
                CreateModdEntry(3, new Vector3(1f, 2f, 3f), 1.25f, 0xAA112233),
                CreateModdEntry(7, new Vector3(-4f, 5f, -6f), 2.5f, 0xFF445566))),
        ];

        using MemoryStream stream = new(bytes);
        WmoDoodadPlacementSummary summary = WmoDoodadPlacementSummaryReader.Read(stream, "synthetic_placements_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(80, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.Equal(2, summary.DistinctNameIndexCount);
        Assert.Equal(7, summary.MaxNameIndex);
        Assert.Equal(1.25f, summary.MinScale);
        Assert.Equal(2.5f, summary.MaxScale);
        Assert.Equal(0xAA, summary.MinAlpha);
        Assert.Equal(0xFF, summary.MaxAlpha);
        Assert.Equal(new Vector3(-4f, 2f, -6f), summary.BoundsMin);
        Assert.Equal(new Vector3(1f, 5f, 3f), summary.BoundsMax);
    }

    private static byte[] CreateModdPayload(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);

        return stream.ToArray();
    }

    private static byte[] CreateModdEntry(uint nameIndex, Vector3 position, float scale, uint color)
    {
        byte[] bytes = new byte[40];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), nameIndex);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4, 4), BitConverter.SingleToInt32Bits(position.X));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8, 4), BitConverter.SingleToInt32Bits(position.Y));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), BitConverter.SingleToInt32Bits(position.Z));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(32, 4), BitConverter.SingleToInt32Bits(scale));
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36, 4), color);
        return bytes;
    }
}
