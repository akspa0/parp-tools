using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoLightSummaryReaderTests
{
    [Fact]
    public void Read_MoltV17Buffer_ProducesLightSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohdWithLightCount(2)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOLT", CreateLights(
                CreateStandardLight(1, true, new Vector3(1f, 2f, 3f), 4f, 2f, 10f),
                CreateStandardLight(2, false, new Vector3(-4f, 5f, -6f), 8f, 3f, 20f))),
        ];

        using MemoryStream stream = new(bytes);
        WmoLightSummary summary = WmoLightSummaryReader.Read(stream, "synthetic_lights_root.wmo");

        Assert.Equal(96, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.Equal(2, summary.DistinctTypeCount);
        Assert.Equal(1, summary.AttenuatedCount);
        Assert.Equal(4f, summary.MinIntensity);
        Assert.Equal(8f, summary.MaxIntensity);
        Assert.Equal(2f, summary.MinAttenStart);
        Assert.Equal(3f, summary.MaxAttenStart);
        Assert.Equal(20f, summary.MaxAttenEnd);
        Assert.Equal(new Vector3(-4f, 2f, -6f), summary.BoundsMin);
        Assert.Equal(new Vector3(1f, 5f, 3f), summary.BoundsMax);
    }

    [Fact]
    public void Read_MoltV14Buffer_ProducesLightSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO", MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohdWithLightCount(2))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOLT", CreateLights(
                CreateLegacyLight(3, true, new Vector3(10f, 20f, 30f), 6f, 1f, 12f),
                CreateLegacyLight(4, false, new Vector3(-7f, 8f, -9f), 9f, 4f, 18f))),
        ];

        using MemoryStream stream = new(bytes);
        WmoLightSummary summary = WmoLightSummaryReader.Read(stream, "synthetic_alpha_lights_root.wmo");

        Assert.Equal(64, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.Equal(2, summary.DistinctTypeCount);
        Assert.Equal(1, summary.AttenuatedCount);
        Assert.Equal(6f, summary.MinIntensity);
        Assert.Equal(9f, summary.MaxIntensity);
        Assert.Equal(1f, summary.MinAttenStart);
        Assert.Equal(4f, summary.MaxAttenStart);
        Assert.Equal(18f, summary.MaxAttenEnd);
        Assert.Equal(new Vector3(-7f, 8f, -9f), summary.BoundsMin);
        Assert.Equal(new Vector3(10f, 20f, 30f), summary.BoundsMax);
    }

    private static byte[] CreateMohdWithLightCount(uint lightCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), lightCount);
        return bytes;
    }

    private static byte[] CreateLights(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);
        return stream.ToArray();
    }

    private static byte[] CreateLegacyLight(byte type, bool atten, Vector3 position, float intensity, float attenStart, float attenEnd)
    {
        byte[] bytes = new byte[32];
        bytes[0] = type;
        bytes[1] = atten ? (byte)1 : (byte)0;
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), 0xFF00FF00);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8, 4), BitConverter.SingleToInt32Bits(position.X));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), BitConverter.SingleToInt32Bits(position.Y));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16, 4), BitConverter.SingleToInt32Bits(position.Z));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(20, 4), BitConverter.SingleToInt32Bits(intensity));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24, 4), BitConverter.SingleToInt32Bits(attenStart));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28, 4), BitConverter.SingleToInt32Bits(attenEnd));
        return bytes;
    }

    private static byte[] CreateStandardLight(byte type, bool atten, Vector3 position, float intensity, float attenStart, float attenEnd)
    {
        byte[] bytes = new byte[48];
        bytes[0] = type;
        bytes[1] = atten ? (byte)1 : (byte)0;
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), 0xFF00FF00);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8, 4), BitConverter.SingleToInt32Bits(position.X));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), BitConverter.SingleToInt32Bits(position.Y));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16, 4), BitConverter.SingleToInt32Bits(position.Z));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(20, 4), BitConverter.SingleToInt32Bits(intensity));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24, 4), BitConverter.SingleToInt32Bits(0f));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28, 4), BitConverter.SingleToInt32Bits(0f));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(32, 4), BitConverter.SingleToInt32Bits(-1f));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(36, 4), BitConverter.SingleToInt32Bits(-0.5f));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(40, 4), BitConverter.SingleToInt32Bits(attenStart));
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(44, 4), BitConverter.SingleToInt32Bits(attenEnd));
        return bytes;
    }
}
