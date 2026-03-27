using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoFogSummaryReaderTests
{
    [Fact]
    public void Read_MfogBuffer_ProducesFogSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MFOG", CreateFogs(
                CreateFog(1, new Vector3(1f, 2f, 3f), 2f, 5f, 9f),
                CreateFog(0, new Vector3(-4f, 5f, -6f), 1f, 7f, 11f))),
        ];

        using MemoryStream stream = new(bytes);
        WmoFogSummary summary = WmoFogSummaryReader.Read(stream, "synthetic_fogs_root.wmo");

        Assert.Equal(96, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.Equal(1, summary.NonZeroFlagCount);
        Assert.Equal(1f, summary.MinSmallRadius);
        Assert.Equal(7f, summary.MaxLargeRadius);
        Assert.Equal(11f, summary.MaxFogEnd);
        Assert.Equal(new Vector3(-4f, 2f, -6f), summary.BoundsMin);
        Assert.Equal(new Vector3(1f, 5f, 3f), summary.BoundsMax);
    }

    private static byte[] CreateFogs(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);
        return stream.ToArray();
    }

    private static byte[] CreateFog(uint flags, Vector3 position, float smallRadius, float largeRadius, float fogEnd)
    {
        byte[] bytes = new byte[48];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), flags);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4, 4), BitConverter.SingleToInt32Bits(position.X));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8, 4), BitConverter.SingleToInt32Bits(position.Y));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), BitConverter.SingleToInt32Bits(position.Z));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16, 4), BitConverter.SingleToInt32Bits(smallRadius));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(20, 4), BitConverter.SingleToInt32Bits(largeRadius));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24, 4), BitConverter.SingleToInt32Bits(fogEnd));
        return bytes;
    }
}
