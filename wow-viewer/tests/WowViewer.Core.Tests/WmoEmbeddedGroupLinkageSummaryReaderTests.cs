using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoEmbeddedGroupLinkageSummaryReaderTests
{
    [Fact]
    public void Read_AlphaRootWithEmbeddedGroups_ProducesLinkageSummary()
    {
        byte[] momoPayload = CreateMomoPayload(
            MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(groupCount: 2)),
            MapFileSummaryReaderTestsAccessor.CreateChunk(
                "MOGI",
                CreateMogiPayload(
                    CreateLegacyMogiEntry(0x8, new Vector3(-2f, -3f, -4f), new Vector3(5f, 6f, 7f)),
                    CreateLegacyMogiEntry(0x2009, new Vector3(-7f, 1f, 2f), new Vector3(3f, 8f, 9f)))));

        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO", momoPayload),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x44, 0x8, new Vector3(-2f, -3f, -4f), new Vector3(5f, 6f, 7f))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80, 0x2008, new Vector3(-7f, 1f, 2f), new Vector3(3f, 8f, 9f))),
        ];

        using MemoryStream stream = new(bytes);
        WmoEmbeddedGroupLinkageSummary summary = WmoEmbeddedGroupLinkageSummaryReader.Read(stream, "synthetic_alpha_root.wmo");

        Assert.Equal((uint)14, summary.Version);
        Assert.Equal(2, summary.GroupInfoCount);
        Assert.Equal(2, summary.EmbeddedGroupCount);
        Assert.Equal(2, summary.CoveredPairCount);
        Assert.Equal(0, summary.MissingEmbeddedGroupCount);
        Assert.Equal(0, summary.ExtraEmbeddedGroupCount);
        Assert.Equal(1, summary.FlagMatchCount);
        Assert.Equal(2, summary.BoundsMatchCount);
        Assert.Equal(0f, summary.MaxBoundsDelta);
    }

    private static byte[] CreateMohd(uint groupCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        return bytes;
    }

    private static byte[] CreateMomoPayload(params byte[][] chunks)
    {
        using MemoryStream stream = new();
        foreach (byte[] chunk in chunks)
            stream.Write(chunk, 0, chunk.Length);

        return stream.ToArray();
    }

    private static byte[] CreateLegacyMogiEntry(uint flags, Vector3 boundsMin, Vector3 boundsMax)
    {
        byte[] bytes = new byte[40];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), flags);
        WriteSingle(bytes, 12, boundsMin.X);
        WriteSingle(bytes, 16, boundsMin.Y);
        WriteSingle(bytes, 20, boundsMin.Z);
        WriteSingle(bytes, 24, boundsMax.X);
        WriteSingle(bytes, 28, boundsMax.Y);
        WriteSingle(bytes, 32, boundsMax.Z);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(36, 4), -1);
        return bytes;
    }

    private static byte[] CreateMogiPayload(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);

        return stream.ToArray();
    }

    private static byte[] CreateMogpPayload(int headerSize, uint flags, Vector3 boundsMin, Vector3 boundsMax)
    {
        byte[] header = new byte[headerSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), flags);
        WriteSingle(header, 0x0C, boundsMin.X);
        WriteSingle(header, 0x10, boundsMin.Y);
        WriteSingle(header, 0x14, boundsMin.Z);
        WriteSingle(header, 0x18, boundsMax.X);
        WriteSingle(header, 0x1C, boundsMax.Y);
        WriteSingle(header, 0x20, boundsMax.Z);
        return header;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
