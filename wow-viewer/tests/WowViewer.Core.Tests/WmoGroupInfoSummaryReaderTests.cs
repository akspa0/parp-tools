using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupInfoSummaryReaderTests
{
    [Fact]
    public void Read_StandardMogiBuffer_ProducesGroupInfoSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(17)),
            .. CreateChunk("MOHD", CreateMohdPayload(groupCount: 2)),
            .. CreateChunk("MOGI", CreateMogiPayload(32,
                CreateStandardMogiEntry(0x1, new Vector3(-1f, -2f, -3f), new Vector3(4f, 5f, 6f), 12),
                CreateStandardMogiEntry(0x0, new Vector3(-7f, 1f, 2f), new Vector3(3f, 8f, 9f), 40))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupInfoSummary summary = WmoGroupInfoSummaryReader.Read(stream, "synthetic_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(64, summary.PayloadSizeBytes);
        Assert.Equal(32, summary.EntrySizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.Equal(2, summary.DistinctFlagCount);
        Assert.Equal(1, summary.NonZeroFlagCount);
        Assert.Equal(12, summary.MinNameOffset);
        Assert.Equal(40, summary.MaxNameOffset);
        Assert.Equal(new Vector3(-7f, -2f, -3f), summary.BoundsMin);
        Assert.Equal(new Vector3(4f, 8f, 9f), summary.BoundsMax);
    }

    [Fact]
    public void Read_LegacyMogiBuffer_ProducesGroupInfoSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(14)),
            .. CreateChunk("MOHD", CreateMohdPayload(groupCount: 1)),
            .. CreateChunk("MOGI", CreateMogiPayload(40,
                CreateLegacyMogiEntry(100, 200, 0x2, new Vector3(1f, 2f, 3f), new Vector3(4f, 5f, 6f), 99))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupInfoSummary summary = WmoGroupInfoSummaryReader.Read(stream, "synthetic_legacy_root.wmo");

        Assert.Equal((uint)14, summary.Version);
        Assert.Equal(40, summary.PayloadSizeBytes);
        Assert.Equal(40, summary.EntrySizeBytes);
        Assert.Equal(1, summary.EntryCount);
        Assert.Equal(1, summary.DistinctFlagCount);
        Assert.Equal(1, summary.NonZeroFlagCount);
        Assert.Equal(99, summary.MinNameOffset);
        Assert.Equal(99, summary.MaxNameOffset);
        Assert.Equal(new Vector3(1f, 2f, 3f), summary.BoundsMin);
        Assert.Equal(new Vector3(4f, 5f, 6f), summary.BoundsMax);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
    }

    private static byte[] CreateMohdPayload(uint groupCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        return bytes;
    }

    private static byte[] CreateMogiPayload(int entrySize, params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
        {
            if (entry.Length != entrySize)
                throw new InvalidOperationException($"Expected entry size {entrySize}, but got {entry.Length}.");

            stream.Write(entry, 0, entry.Length);
        }

        return stream.ToArray();
    }

    private static byte[] CreateStandardMogiEntry(uint flags, Vector3 boundsMin, Vector3 boundsMax, int nameOffset)
    {
        byte[] bytes = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), flags);
        WriteSingle(bytes, 4, boundsMin.X);
        WriteSingle(bytes, 8, boundsMin.Y);
        WriteSingle(bytes, 12, boundsMin.Z);
        WriteSingle(bytes, 16, boundsMax.X);
        WriteSingle(bytes, 20, boundsMax.Y);
        WriteSingle(bytes, 24, boundsMax.Z);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28, 4), nameOffset);
        return bytes;
    }

    private static byte[] CreateLegacyMogiEntry(uint offset, uint size, uint flags, Vector3 boundsMin, Vector3 boundsMax, int nameOffset)
    {
        byte[] bytes = new byte[40];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), offset);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), size);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), flags);
        WriteSingle(bytes, 12, boundsMin.X);
        WriteSingle(bytes, 16, boundsMin.Y);
        WriteSingle(bytes, 20, boundsMin.Z);
        WriteSingle(bytes, 24, boundsMax.X);
        WriteSingle(bytes, 28, boundsMax.Y);
        WriteSingle(bytes, 32, boundsMax.Z);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(36, 4), nameOffset);
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
